from typing import Literal, Callable, Dict, Set, List, Iterable
from dataclasses import asdict, dataclass
from .interface import Knowledge, Concept, Intrinsic, Exp, Proposition, Expression, SExp, IExpConfig
from .specific_model import (SpecificModel, ExpData, AtomExp, ExpStructure, ConclusionSet, reduce_conserved_by_ideal)
from .diffalgservice import DiffalgService
from .object_model import ObjectModel
from .memory import Memory
from .manager import Manager
from .parsing import check_single_pos, expand_expression, generate_pos_dict, count_gradient_components, extract_expression_pattern
from .probabilistic_reducer import (
    GeneralLawRankReducer,
    ProbabilisticReducerConfig,
    RankCertificate,
)
from .dimensional_analysis import (
    check_concept_dimensions, build_known_dims_from_knowledge,
)
import sympy as sp
import itertools
import copy
import re
from functools import cmp_to_key


@dataclass(frozen=True)
class ArchivedGeneralLawRef:
    """Minimal parent node used when a revision ancestor is no longer active."""

    name: str
    father: None = None


class GeneralLaw:
    """
    Each general law is a proposition that is valid in a collection of experiments.
    Name of a general law is the same as that of its corresponding proposition, such as "R_XX".
    """
    name: str
    prop: Proposition
    valid_experiments: list[str]  # List of experiment names in which the general law is valid
    father: "GeneralLaw"  # Father general law, from which the current general law is derived

    def __init__(self,
                 name: str,
                 prop: Proposition,
                 valid_experiments: list[str] = None,
                 father: "GeneralLaw" = None,
                 father_name: str | None = None,
                 knowledge_registered: bool = True) -> None:
        self.name = name
        self.prop = prop
        self.valid_experiments = valid_experiments if valid_experiments is not None else []
        self.father = father
        # ``father`` used to be the only lineage representation.  Deleting a
        # falsified parent consequently erased the revision edge and made a
        # save/load cycle unable to reconstruct it.  Keep the identifier as
        # durable provenance even when the parent has left the active basis.
        self._father_name = father.name if father is not None else father_name
        # Extended expressions normally register through deterministic opaque
        # jets in the Rust KeyValue index.  Keep this flag for compatibility
        # with older checkpoints that persisted such laws outside that index.
        self.knowledge_registered = knowledge_registered

    @property
    def father_name(self) -> str | None:
        return self.father.name if self.father is not None else self._father_name

    def __str__(self):
        return f"GeneralLaw(name={self.name}, type={self.type}, indication={self.indication}, " \
               f"prop={self.prop}, " \
               f"valid_experiments={self.valid_experiments}, " \
               f"father={self.father_name})"

    def __repr__(self):
        return self.__str__()

    @property
    def type(self) -> Literal['IsConserved', 'IsZero']:
        """
        If type is "IsConserved", the general law is conserved(including non-zero and zero)
        in all valid experiments.
        If type is "IsZero", the general law is zero in all valid experiments.
        Type of a general law is determined once it is proposed in some experiment.
        In following cases, a general law's type can be changed:
        1. By adding some terms constructed from other conserved quantities, a IsConserved general law
        can be converted to a IsZero general law(which means that it is zero in all valid experiments).
        2. ...
        """
        match self.prop.prop_type:
            case "Conserved":
                return "IsConserved"
            case "Zero":
                return "IsZero"
            case _:
                raise ValueError("Invalid prop_type.")

    @property
    def indication(self) -> Literal['Sum', 'ForAll']:
        """
        If indication is "Sum", the general law corresponds to a proposition which has a summation form.
        If indication is "ForAll", the general law corresponds to a proposition which has a for-all form.
        E.g. Energy conservation is a "Sum" general law, while Newton's second law is a "ForAll" general law.
        """
        if len(self.prop.unwrap_concept.preids) == 0:
            return "Sum"
        else:
            return "ForAll"

    @property
    def relevant_concept(self) -> Concept:
        return self.prop.unwrap_concept

    @property
    def scope_kind(self) -> Literal['contextual', 'cross_experiment']:
        """Describe whether the law is process-specific or cross-context."""

        return 'contextual' if len(self.valid_experiments) == 1 else 'cross_experiment'

    def to_json(self) -> dict:
        return {
            "prop": str(self.prop),
            "valid_experiments": self.valid_experiments,
            "father": self.father_name,
            "knowledge_registered": self.knowledge_registered,
        }

    def is_trivial(self,
                   knowledge: Knowledge,
                   raw_defi_func: Callable[[Exp], Exp],
                   exp_name: str) -> bool:
        concept = self.prop.unwrap_concept
        exprs: list[Exp] = knowledge.specialize(concept, exp_name)
        if not exprs:
            raise ValueError(f"No specialized exprs for proposition {self.prop} and experiment {exp_name}.")
        # Judge the triviality of the law by raw definition
        if all(knowledge.sympy_of_raw_defi(raw_defi_func(expr), exp_name).is_number for expr in exprs):
            return True
        return False

    def record_valid_experiment(self, exp_name: str):
        if exp_name not in self.valid_experiments:
            self.valid_experiments.append(exp_name)


class GeneralLawBase(Manager):
    """
    The GeneralLawBase class is responsible for managing all general laws.
    """
    knowledge: Knowledge
    specific: dict[str, SpecificModel]
    memory: Memory
    objmodel: dict[str, ObjectModel]
    general_laws: dict[str, GeneralLaw]
    completion_exclude: dict[str, list[str]]
    service: "GeneralLawService"  # Service for general law management

    def __init__(self, knowledge: Knowledge, specific: dict[str, SpecificModel],
                 memory: Memory, objmodel: dict[str, ObjectModel]) -> None:
        self.knowledge = knowledge
        self.specific = specific
        self.memory = memory
        self.objmodel = objmodel
        self.general_laws = dict()
        self.completion_exclude = dict()
        self.service = GeneralLawService(self)

    def __contains__(self, item: str | GeneralLaw) -> bool:
        if isinstance(item, str):
            return item in self.general_laws
        elif isinstance(item, GeneralLaw):
            return (
                any(str(law.prop) == str(item.prop) for law in self.general_laws.values())
                or self.knowledge.find_similar(item.prop) is not None
            )
        else:
            raise ValueError("Invalid type of item. Must be a GeneralLaw object or its name.")

    @staticmethod
    def from_json(data: dict, knowledge: Knowledge, specific: dict[str, SpecificModel],
                  memory: Memory, objmodel: dict[str, ObjectModel]) -> "GeneralLawBase":
        obj = object.__new__(GeneralLawBase)
        obj.knowledge = knowledge
        obj.specific = specific
        obj.memory = memory
        obj.objmodel = objmodel
        obj.general_laws = {
            key: GeneralLaw(name=key,
                            prop=Proposition(value['prop']),
                            valid_experiments=value["valid_experiments"],
                            father=None,
                            father_name=value.get("father"),
                            knowledge_registered=value.get(
                                "knowledge_registered", True
                            ))
            for key, value in data['general_laws'].items()
        }
        for k, v in data['general_laws'].items():
            if v.get("father") in obj.general_laws:
                obj.general_laws[k].father = obj.general_laws[v["father"]]
            elif v.get("father") is not None:
                obj.general_laws[k].father = ArchivedGeneralLawRef(v["father"])
        obj.completion_exclude = data['completion_exclusion']
        obj.service = GeneralLawService(obj, data.get("general_service", None))
        return obj

    def to_json(self) -> dict:
        return {
            'general_laws': {
                key: value.to_json()
                for key, value in self.general_laws.items()
            },
            'completion_exclusion': self.completion_exclude,
            'general_service': self.service.to_json()
        }

    def register_general_law(self, prop: Proposition, 
                             father: GeneralLaw = None,
                             valid_experiments: list[str] = None) -> GeneralLaw | None:
        if any(str(law.prop) == str(prop) for law in self.general_laws.values()):
            return None
        knowledge_registered = True
        try:
            name: str | None = self.knowledge.register_conclusion(prop)
        except ValueError as error:
            # Compatibility fallback for checkpoints or external core builds
            # predating opaque-subtree keys.  Current builds register extended
            # AST nodes normally and use the numerical reducer for semantics.
            if not re.search(r"\b(?:sin|cos|exp|log)\s*\(", str(prop)):
                raise
            name = self.knowledge.auto_conclusion_name()
            knowledge_registered = False
            print(
                f"Core conclusion index skipped extended law {name}: {error}",
                flush=True,
            )
        if name is not None:
            self.general_laws[name] = GeneralLaw(
                name, prop, father=father,
                valid_experiments=valid_experiments,
                knowledge_registered=knowledge_registered,
            )
            print("\033[1m" + f"Registered new general law: {name} = {self.general_laws[name]}" + "\033[0m",
                  flush=True)
        return self.general_laws[name] if name is not None else None

    def remove_general_law(self, gl_name: str, *,
                           preserve_exclusion: bool = False,
                           preserve_lineage: bool = False):
        law = self.general_laws.pop(gl_name, None)
        if law is not None and law.knowledge_registered:
            self.knowledge.remove_conclusion(gl_name)
        if not preserve_exclusion:
            self.completion_exclude.pop(gl_name, None)
        if gl_name in self.service.marked_general_laws:
            self.service.marked_general_laws.remove(gl_name)
            self.service.diffalg_for_marked_gl.pop(gl_name, None)
        # Ordinary deletion detaches descendants.  Theory revision instead
        # retains the archived parent reference so the causal edge remains
        # visible to persistence and AdaptiveTheoryReward.
        if not preserve_lineage:
            for gl in self.general_laws.values():
                if gl.father is not None and gl.father.name == gl_name:
                    gl._father_name = None
                    gl.father = None

    def record_exclusion(self, law_name: str, exp_name: str):
        """
        Record that the law is excluded from completion in experiment `exp_name`.
        """
        if law_name not in self.completion_exclude:
            self.completion_exclude[law_name] = []
        if exp_name not in self.completion_exclude[law_name]:
            self.completion_exclude[law_name].append(exp_name)

    def check_general_laws(self,
                           exp_name: str,
                           gl_dict: dict[str, GeneralLaw] = None) -> tuple[dict[str, bool],
                                                                           dict[str, ExpData | None]]:
        """
        Given a dictionary of general laws,
        check if the general laws are valid in experiment `exp_name`.
        Return a dictionary of check results and a dictionary of general laws that need completion.

        Args:
            - exp_name(str): The name of experiment to be checked.
            - gc_dict(dict[str, GeneralLaw]): The dictionary of general laws to be checked.
                Keys are the names of general laws, values are GeneralLaw objects.
                If None, all general laws will be checked.

        Returns:
            - dict_check_results(dict[str, bool]): A dictionary of check results.
                Keys are the names of general laws,
                values are bool values indicating if the general law is valid.
            - dict_need_completion(dict[str, ExpData | None]): A dictionary of general laws that need completion.
                Keys are the names of general laws,
                values are ExpData objects that need to be completed(for "Sum" general laws),
                or None(for "ForAll" general laws).
        """
        spm: SpecificModel = self.specific[exp_name]
        dict_check_results: dict[str, bool] = {}  # Record the check results of general laws
        dict_need_completion: dict[str, ExpData | None] = {}  # Record the general laws that need completion
        # For "ForAll" general laws, laws need completion is only specified be its name
        # For "Sum" general laws, laws need completion is specified by its name and the ExpData that need to be completed
        if gl_dict is None:
            # If no general law is provided, check all general laws
            gl_dict: dict[str, GeneralLaw] = self.general_laws
        for name, gl in gl_dict.items():
            gl: GeneralLaw
            # If an experiment is already recorded as valid, there's no need to check it again
            if exp_name in gl.valid_experiments:
                dict_check_results[name] = True
                continue
            # If an experiment is already recorded as invalid and should be excluded from completion,
            # there's no need to check it again, either
            elif exp_name in self.completion_exclude.get(name, []):
                dict_check_results[name] = False
                continue
            # Specialize general laws in the current experiment
            exprs: list[Exp] = self.knowledge.specialize(gl.relevant_concept, exp_name)
            # If no specialized expressions are found, consider the general law as invalid in this experiment
            # and [no need] to complete it
            if not exprs:
                dict_check_results[name] = False
                continue
            if gl.indication == 'Sum' and len(exprs) > 1:
                raise ValueError("Sum indication should not have multiple specialized expressions")
            # Check if all specialized expressions are conserved or zero
            flag = True
            for expr in exprs:
                data: ExpData = self.knowledge.eval(spm.expand_exp(expr), spm.experiment)
                if (gl.type == 'IsConserved' and not data.is_conserved)\
                        or (gl.type == 'IsZero' and not data.is_zero):
                    flag = False
                    dict_need_completion[name] = data if gl.indication == 'Sum' else None
                    break
            if flag:
                gl.record_valid_experiment(exp_name)
            dict_check_results[name] = flag

        return dict_check_results, dict_need_completion

    def is_gl_zero(self, gl: GeneralLaw) -> bool:
        # Check if the general law is now zero in all valid experiments
        flag_zero: bool = True
        for exp_name in gl.valid_experiments:
            spm: SpecificModel = self.specific[exp_name]
            # Specialize general laws in the current experiment
            exprs: list[Exp] = self.knowledge.specialize(gl.relevant_concept, exp_name)
            if not exprs:
                flag_zero = False
                break
            # Check if all specialized expressions are conserved or zero
            flag = True
            for expr in exprs:
                data: ExpData = self.knowledge.eval(spm.expand_exp(expr), spm.experiment)
                if not data.is_zero:
                    flag = False
                    break
            if not flag:
                # print(f"General law {gl.name} is not zero in experiment {exp_name}.", flush=True)
                flag_zero = False
                break

        return flag_zero

    def is_gl_conserved(self, gl: GeneralLaw) -> bool:
        # Check if the general law is now conserved in all valid experiments
        flag_conserved: bool = True
        for exp_name in gl.valid_experiments:
            spm: SpecificModel = self.specific[exp_name]
            # Specialize general laws in the current experiment
            exprs: list[Exp] = self.knowledge.specialize(gl.relevant_concept, exp_name)
            if not exprs:
                flag_conserved = False
                break
            # Check if all specialized expressions are conserved or zero
            flag = True
            for expr in exprs:
                data: ExpData = self.knowledge.eval(spm.expand_exp(expr), spm.experiment)
                if not data.is_conserved:
                    flag = False
                    break
            if not flag:
                print(f"General law {gl.name} is not conserved in experiment {exp_name}.", flush=True)
                flag_conserved = False
                break

        return flag_conserved

    def try_complete_general_law(self,
                                 gl_name: str,
                                 gl_data: ExpData | None,
                                 exprs: list[AtomExp],
                                 exp_name: str) -> tuple[str, bool]:
        """
        Try to complete an invalid general law in experiment `exp_name` using `exprs`.
        """
        if exp_name in self.completion_exclude.get(gl_name, []):
            return gl_name, False

        gl: GeneralLaw = self.general_laws[gl_name]
        spm: SpecificModel = self.specific[exp_name]
        # Evaluate the expressions
        if gl.indication == 'ForAll':
            if gl_data is not None:
                raise ValueError("ForAll indication should not have specifc data for completion")
            return gl_name, self.search_forall_completion(gl, spm, exprs, exp_name)
        elif gl.indication == 'Sum':
            if not isinstance(gl_data, ExpData):
                raise ValueError("Sum indication should have specific data for completion")
            if len(self.knowledge.specialize(gl.relevant_concept, exp_name)) > 1:
                raise ValueError("Sum indication should not have multiple specialized expressions")
            return gl_name, self.search_sum_completion(gl, spm, gl_data, exprs, exp_name)
        else:
            raise NotImplementedError("Unknown indication type")

    def search_forall_completion(self,
                                 gl: GeneralLaw,
                                 spm: SpecificModel,
                                 exprs: list[AtomExp],
                                 exp_name: str,
                                 for_convertion: str | bool = False) -> bool:
        relevant_concept = gl.relevant_concept
        exprs = [itm for itm in exprs if len(itm.vec_ids) <= 1]

        # Now consider only construct the completion term from spatial derivatives for the sake of simplicity.
        # More cases can be considered in the future.
        for i in range(len(exprs)):
            for j in range(i):
                comb: tuple[str, str] = (str(exprs[i]), str(exprs[j]))
                if comb in self.memory.general_conclusion_attempts.get(exp_name, {}).get(gl.name, set()):
                    continue
                self.memory.record_gc_attempts(exp_name, gl.name, comb)
                # Try to use spatial derivative
                for operation in [(0, 'sd'), ('sd', 0), (1, 'sd'), ('sd', 1), (-1, 'sd'), ('sd', -1)]:
                    if operation[0] == 'sd':
                        if exprs[i] in self.specific[exp_name].experiment.original_data:
                            continue
                        exp_for_partial: Exp = Exp.Atom(exprs[i])
                        exp_pow: Exp = Exp.Atom(exprs[j]).__powi__(operation[1])
                    elif operation[1] == 'sd':
                        if exprs[j] in self.specific[exp_name].experiment.original_data:
                            continue
                        exp_for_partial: Exp = Exp.Atom(exprs[j])
                        exp_pow: Exp = Exp.Atom(exprs[i]).__powi__(operation[0])
                    if spm.is_spatial_partial_diff_trivial(exp_for_partial):
                        continue
                    exp_sd: dict[AtomExp, Exp] = spm.spatial_partial_diff(exp_for_partial)
                    # Remove all zero spatial derivatives
                    exp_sd = {k: v for k, v in exp_sd.items()
                              if v.type != 'Number'}
                    if not exp_sd:
                        continue
                    # For each combination of spatial derivative and power,
                    # try to define a new concept.
                    for spatial_exp, spatial_diff in exp_sd.items():
                        val: ExpData = self.knowledge.eval(spm.expand_exp(Exp.Partial(exp_for_partial, spatial_exp) * exp_pow),
                                                           spm.experiment)
                        if not for_convertion and ((val.is_conserved and gl.type == 'IsConserved') or (val.is_zero and gl.type == 'IsZero')):
                            continue
                        possible_concept: Concept | None = \
                            self.knowledge.generalize_to_normal_concept(exp_name,
                                                                        Exp.Partial(exp_for_partial,
                                                                                    spatial_exp) * exp_pow)
                        if possible_concept is None:
                            continue
                        # First try simple ansatz
                        allows_coes = [(1, 1), (1, -1), (1, 2), (1, -2), (2, 1), (2, -1)]
                        possible_concept_final: dict[tuple[int, ...], Concept] = \
                            {coe: coe[0] * relevant_concept + coe[1] * possible_concept
                                for coe in allows_coes}
                        possible_exprs_final: dict[tuple[int, ...], list[Exp]] = \
                            {coe: self.knowledge.specialize(possible_concept, exp_name)
                             if possible_concept is not None else []
                                for coe, possible_concept in possible_concept_final.items()}
                        possible_exprs_final = {coe: exprs for coe, exprs in possible_exprs_final.items()
                                                if exprs}
                        if not possible_exprs_final:
                            continue
                        possible_exprs_final_raw: dict[tuple[int, ...], list[Exp]] = \
                            {coe: [spm.expand_exp(expr) for expr in exprs]
                                for coe, exprs in possible_exprs_final.items()}
                        # Evaluate the expressions
                        data_final: dict[tuple[int, ...], list[ExpData]] = \
                            {coe: [self.knowledge.eval(expr, spm.experiment)
                                   for expr in exprs]
                                for coe, exprs in possible_exprs_final_raw.items()}
                        # Remove the results that have incorrectly evaluated expressions
                        data_final = {coe: data
                                      for coe, data in data_final.items()
                                      if not any([d.is_err for d in data])}
                        if not data_final:
                            continue
                        # For each possible completion, check if the completion is conserved
                        # or zero under all possible index mappings
                        for coe, data in data_final.items():
                            if not for_convertion and ((all([d.is_conserved for d in data]) and gl.type == 'IsConserved') or
                                                       (all([d.is_zero for d in data]) and gl.type == 'IsZero')):
                                if not self.confirm_validation(exp_name, possible_exprs_final_raw[coe], gl.type):
                                    continue
                                create_success: bool = \
                                    self.create_general_law_from_old(exp_name, gl,
                                                                     possible_concept_final[coe])
                                if create_success:
                                    return True
                            elif for_convertion and all([d.is_zero for d in data]):
                                if not self.confirm_validation(exp_name, possible_exprs_final_raw[coe], "IsZero"):
                                    continue
                                create_success: bool = \
                                    self.create_general_law_from_old(exp_name, gl,
                                                                     possible_concept_final[coe],
                                                                     for_convertion="Zero")
                                if create_success:
                                    return True
                        # TODO: More complex ansatz via differentiation
        return False

    def search_sum_completion(self,
                              gl: GeneralLaw,
                              spm: SpecificModel,
                              gl_data: ExpData,
                              exprs: list[AtomExp],
                              exp_name: str,
                              for_convertion: bool = False) -> bool:
        # Consider all possible combinations of two expressions
        for i in range(len(exprs)):
            for j in range(i):
                comb: tuple[str, str] = (str(exprs[i]), str(exprs[j]))
                if comb in self.memory.general_conclusion_attempts.get(exp_name, {}).get(gl.name, set()):
                    continue
                self.memory.record_gc_attempts(exp_name, gl.name, comb)

                # Now consider only construct the completion term from combinations of some low power
                # of the two expressions for the sake of simplicity.
                # More cases can be considered in the future.
                for pow in [(-1, 0), (0, -1), (-1, 1), (1, -1), (-1, 2), (2, -1),
                            (1, 0), (0, 1), (1, 1), (1, 2), (2, 1)]:
                    if not exprs[i].vec_ids:
                        concept_temp = self.knowledge.fetch_concept_by_name(exprs[i].name)
                        if concept_temp is None:  # This is the case for universal constants
                            continue
                        possible_to_sum_concept: Concept = concept_temp.__powi__(pow[0])
                    elif not exprs[j].vec_ids:
                        concept_temp = self.knowledge.fetch_concept_by_name(exprs[j].name)
                        if concept_temp is None:
                            continue
                        possible_to_sum_concept: Concept = concept_temp.__powi__(pow[1])
                    else:
                        possible_to_sum_concept: Concept | None = self.make_expr_to_sum(
                            Exp.Atom(exprs[i]).__powi__(pow[0]) * Exp.Atom(exprs[j]).__powi__(pow[1]), exp_name)
                    if possible_to_sum_concept is None:
                        continue
                    if len(possible_to_sum_concept.objtype_id_map.keys()) != 1:
                        continue
                    possible_to_sum_objtype: str = list(possible_to_sum_concept.objtype_id_map.keys())[0]
                    possible_to_sum_ids: set[int] = exprs[i].allids.union(exprs[j].allids)
                    possible_to_sum: list[Exp] = self.knowledge.specialize(possible_to_sum_concept, exp_name)
                    # For sum type concepts, the specialized expressions should be unique
                    if not possible_to_sum or len(possible_to_sum) > 1:
                        continue
                    possible_to_sum_data: ExpData = self.knowledge.eval(possible_to_sum[0],
                                                                        spm.experiment)
                    if possible_to_sum_data.is_conserved or possible_to_sum_data.is_err:
                        continue  # Do not try to complete using a conserved quantity
                    # First try simple ansatz
                    allows_coes = [1, -1, 2, -2]
                    possible_results = [gl_data + coe * possible_to_sum_data
                                        for coe in allows_coes]
                    for res, coe in zip(possible_results, allows_coes):
                        if (res.is_conserved and not res.is_zero and gl.type == 'IsConserved') or \
                                (res.is_zero and gl.type == 'IsZero'):
                            if not self.confirm_validation(exp_name,
                                                           self.knowledge.specialize(gl.relevant_concept, exp_name)[0] + coe * possible_to_sum[0],
                                                           gl.type):
                                continue
                            create_success: bool = \
                                self.create_general_law_from_old_sum(exp_name, gl,
                                                                     possible_to_sum_concept,
                                                                     (1, coe))
                            if create_success:
                                return True
                    # Try more complex ansatz via differentiation
                    diff_res: ExpData = gl_data.__diff__(possible_to_sum_data)
                    diff_expr: Exp = self.knowledge.specialize(gl.relevant_concept,
                                                               exp_name)[0].__diff__(possible_to_sum[0])
                    if not diff_res.is_conserved:
                        continue
                    try_to_sum_expr: Exp | None = self.intrinsic_regression(diff_res,
                                                                            diff_expr,
                                                                            possible_to_sum_objtype,
                                                                            possible_to_sum_ids,
                                                                            exp_name)
                    if try_to_sum_expr is not None:
                        possible_to_sum_concept: Concept | None = self.make_expr_to_sum(
                            Exp.Atom(exprs[i]).__powi__(pow[0]) * Exp.Atom(exprs[j]).__powi__(pow[1]) * try_to_sum_expr,
                            exp_name)
                        if possible_to_sum_concept is None:
                            continue
                        possible_to_sum: list[Exp] = self.knowledge.specialize(possible_to_sum_concept, exp_name)
                        if not possible_to_sum or len(possible_to_sum) > 1:
                            continue
                        possible_to_sum_data: ExpData = self.knowledge.eval(possible_to_sum[0],
                                                                            spm.experiment)
                        if possible_to_sum_data.is_conserved:
                            continue
                        res: ExpData = gl_data - possible_to_sum_data
                        if (res.is_conserved and not res.is_zero and gl.type == 'IsConserved') or \
                                (res.is_zero and gl.type == 'IsZero'):
                            if not self.confirm_validation(exp_name,
                                                           self.knowledge.specialize(gl.relevant_concept, exp_name)[0] - possible_to_sum[0],
                                                           gl.type):
                                continue
                            create_success: bool = \
                                self.create_general_law_from_old_sum(exp_name, gl,
                                                                     possible_to_sum_concept,
                                                                     (1, -1))
                            if create_success:
                                return True
        return False

    def compensate_gradient(self,
                            spm: SpecificModel,
                            exp_name: str) -> bool:
        gl_name_diff: set[str] = set(self.general_laws.keys()) - self.service.marked_general_laws
        gl_partials: dict[str, list[str]] = {}
        for name in gl_name_diff:
            gl: GeneralLaw = self.general_laws[name]
            if 'Partial' not in str(gl.prop):
                continue
            gl_terms: list[str] = expand_expression(str(gl.prop.unwrap_concept.exp))
            gl_partials[name] = [term for term in gl_terms if 'Partial' in term]
        possible_partial_dict: dict[str, list[str]] = generate_pos_dict({term
                                                                         for terms in gl_partials.values()
                                                                         for term in terms})

        valid_gls = {name: gl for name, gl in self.general_laws.items()
                     if (name not in gl_name_diff) and (exp_name in gl.valid_experiments) and (gl.indication == 'ForAll')}
        possible_gls_dict: dict[str, list[str]] = {'posx': [], 'posy': [], 'posz': []}
        for name, gl in valid_gls.items():
            atom_expanded = {str(spm.expand_exp(Exp.Atom(atom)))
                             for item in expand_expression(str(gl.prop.unwrap_concept.exp)) if "[" in str(item)
                             for atom in Exp(item).all_atoms}
            if not check_single_pos(atom_expanded):
                continue
            pos = check_single_pos(atom_expanded)
            if pos not in possible_partial_dict:
                print(f"Warning: {pos} not found in possible partials.")
            possible_gls_dict[pos].append(name)

        if all([(not terms) for terms in possible_partial_dict.values()]):
            return False

        for pos, partials in possible_partial_dict.items():
            if not partials:
                continue
            for partial in partials:
                if any([partial in str(self.general_laws[name].prop) for name in self.general_laws]):
                    continue
                for old_gl in possible_gls_dict[pos]:
                    if old_gl not in self.general_laws:
                        continue
                    if self.general_laws[old_gl].father is None:
                        convertion = False
                    else:
                        convertion = "Zero" if self.general_laws[old_gl].type == 'IsZero' else "Conserved"
                    self.create_general_law_from_old(exp_name, self.general_laws[old_gl],
                                                     (self.knowledge.generalize_to_normal_concept(exp_name, Exp(partial)) +
                                                     self.general_laws[old_gl].prop.unwrap_concept),
                                                     for_convertion=convertion)

    def compensate_general_laws(self, exp_name: str):
        gl_list: list[GeneralLaw] = list(self.general_laws.values())
        for gl in gl_list:
            gl_exp_str: str = str(gl.relevant_concept.exp) if gl.indication == "ForAll" \
                else str(gl.relevant_concept).split("|- ")[1]
            if gl.indication == "Sum":
                gl_exp_expand_str: list[str] = expand_expression(gl_exp_str)
                if len(gl_exp_expand_str) != 2 or any(["*" in exp for exp in gl_exp_expand_str]):
                    continue
                gl_exp_expand_defi: list[str] = [str(self.knowledge.fetch_concept_by_name(name).exp)
                                                 for name in gl_exp_expand_str]
                sum_info_list: list[dict[str, set[int]]] = [self.knowledge.fetch_concept_by_name(name).objtype_id_map
                                                            for name in gl_exp_expand_str]
                sum_info: dict[str, set[int]] = sum_info_list[0]
                if not all([info == sum_info for info in sum_info_list]):
                    continue
                patt_matched = extract_expression_pattern(gl_exp_expand_defi,
                                                          {patt: set(direc.keys())
                                                           for patt, direc in self.memory.concept_clusters.items()},
                                                          minimal_terms=2)
                patt_matched: tuple[str, dict[str, set[str]]]
                if patt_matched[0] is None:
                    continue
                if any([len(con_set) != 2 for con_set in patt_matched[1].values()]):
                    continue
                template: str = patt_matched[0]
                concept_to_append_str: str = template
                for sym, con_set in patt_matched[1].items():
                    concept_to_append_str = concept_to_append_str.replace(sym,
                                                                          list(set(self.memory.concept_clusters[sym].keys()) - con_set)[0])
                concept_append: Concept = self.knowledge.generalize_to_normal_concept(exp_name, (Exp(concept_to_append_str)))
                if concept_append is None:
                    continue
                for obj_type in sum_info:
                    concept_append = Concept.Mksum(obj_type, concept_append)
                concept_name_to_append: str = self.register_concept_or_return_name(concept_append, self.specific[exp_name])
                if concept_name_to_append is None:
                    continue
                atom_concept_append: Concept = self.knowledge.gen_atom_concept(concept_name_to_append)
                if atom_concept_append is None:
                    continue
                new_concept = atom_concept_append + gl.relevant_concept
                if new_concept is None:
                    continue
                self.create_general_law_from_old(exp_name, gl,
                                                 new_concept,
                                                 reset_father=True)
            else:
                # TODO: For "ForAll" general laws
                pass

    def create_general_law_from_old_sum(self,
                                        exp_name: str,
                                        old_gl: GeneralLaw,
                                        possible_concept: Concept,
                                        coe: tuple[int, int],
                                        reset: bool = False,
                                        reset_father: bool = False) -> bool:
        assert old_gl.indication == 'Sum'
        concept_name: str | None = self.register_concept_or_return_name(
            possible_concept,
            self.specific[exp_name]
        )
        if concept_name is None or any([concept_name in expand_expression(str(old_gl.relevant_concept).split("|- ")[1])]):
            return False
        new_relevant_concept: Concept = coe[1] * self.knowledge.gen_atom_concept(concept_name) + coe[0] * old_gl.relevant_concept
        if new_relevant_concept is None:
            raise ValueError(f"Failed to create new concept from general law {old_gl.name}")
        return self.create_general_law_from_old(exp_name, old_gl, new_relevant_concept, reset,
                                                reset_father=reset_father)

    def create_general_law_from_old(self,
                                    exp_name: str,
                                    old_gl: GeneralLaw,
                                    possible_concept: Concept,
                                    reset: bool = False,
                                    for_convertion: str | bool = False,
                                    reset_father: bool = False) -> bool:
        # Check if the new possible concept is trivial
        check_concept_exprs: list[Exp] = self.knowledge.specialize(possible_concept, exp_name)
        if not check_concept_exprs:
            return False
        check_concept_raws: list[sp.Expr] = [self.specific[exp_name]._sympy_of_raw_defi(self.specific[exp_name].expand_exp(expr)).doit()
                                             for expr in check_concept_exprs]
        if all([raw.is_number for raw in check_concept_raws]):
            return False
        if not for_convertion:
            # Register the conclusion in knowledge as a general law
            new_prop: Proposition = (Proposition.Conserved
                                     if old_gl.type == 'IsConserved'
                                     else Proposition.Zero)(possible_concept)
            new_gl: GeneralLaw | None = self.register_general_law(new_prop,
                                                                  father=old_gl if not reset_father else None,
                                                                  valid_experiments=[exp_name])
            if new_gl is None:
                return False
            # Check new general law in all experiments
            for experiment in self.specific:
                if experiment == exp_name:
                    continue
                if reset:
                    self.specific[experiment].experiments_reset()
                _ = self.check_general_laws(experiment, {new_gl.name: new_gl})
            # Update completion_exclude
            self.record_exclusion(old_gl.name, exp_name)
            self.record_exclusion(new_gl.name, exp_name)
        else:
            if for_convertion == 'Zero':
                new_prop: Proposition = Proposition.Zero(possible_concept)
                new_gl: GeneralLaw = GeneralLaw(
                    name="R_temp",
                    prop=new_prop,
                    valid_experiments=old_gl.valid_experiments,
                    father=old_gl.father if not reset_father else None
                )
            elif for_convertion == 'Conserved':
                new_prop: Proposition = Proposition.Conserved(possible_concept)
                new_gl: GeneralLaw = GeneralLaw(
                    name="R_temp",
                    prop=new_prop,
                    valid_experiments=old_gl.valid_experiments,
                    father=old_gl.father if not reset_father else None
                )
            # print(f"Possible new general law: {new_gl}", flush=True)
            if not ((for_convertion == 'Zero' and self.is_gl_zero(new_gl)) or
                    (for_convertion == 'Conserved' and self.is_gl_conserved(new_gl))):
                # print("Conversion failed. Skip.", flush=True)
                return False
            new_gl: GeneralLaw | None = \
                self.register_general_law(new_prop,
                                          father=old_gl.father if not reset_father else None,
                                          valid_experiments=old_gl.valid_experiments)
            if new_gl is None:
                return False
            # Check new general law in other experiments
            for experiment in self.specific:
                if experiment in new_gl.valid_experiments:
                    continue
                if reset:
                    self.specific[experiment].experiments_reset()
                _ = self.check_general_laws(experiment, {new_gl.name: new_gl})
            # Obliviate the old general law
            self.remove_general_law(old_gl.name)

        return True

    def make_expr_to_sum(self, expr: Exp | AtomExp, exp_name) -> Concept | None:
        if isinstance(expr, AtomExp):
            expr = Exp.Atom(expr)
        concept: Concept | None = self.knowledge.generalize(exp_name, expr)
        if concept is None:
            return None
        if 'Sum' in str(concept):
            return concept
        else:
            obj_types = list(set(concept.objtype_id_map.keys()))
            # NOTE: Bug should be fixed here
            if len(obj_types) > 1:
                return None
            for tp in obj_types:
                concept = Concept.Mksum(objtype=tp, concept=concept)
            return concept

    def intrinsic_regression(self,
                             diff_res: ExpData,
                             diff_expr: Exp,
                             objtype: str,
                             ids: set[int],
                             exp_name: str) -> Exp | None:
        """
        Find if the expression specified by `diff_expr` can be expressed via some intrinsic concepts
        and universal constant.
        """
        # Get all relevant intrinsic concepts' names
        relevant_intrinsic_names: list[str] = list(self.objmodel.get(objtype, ObjectModel(objtype, self.knowledge)).attr.keys())
        if not relevant_intrinsic_names:
            return None
        relevant_intrinsic_concepts: list[Intrinsic] = [self.knowledge.fetch_intrinsic_by_name(name)
                                                        for name in relevant_intrinsic_names]
        if None in relevant_intrinsic_concepts:
            raise ValueError("Some intrinsic concept is not found")
        relevant_intrinsic_exprs: list[Exp] = []
        for con in relevant_intrinsic_names:
            relevant_intrinsic_exprs.extend(self.knowledge.specialize(self.knowledge.gen_atom_concept(con), exp_name))
        if not relevant_intrinsic_exprs:
            return None
        max_item = 3
        for i in range(1, max_item + 1):
            if i > len(relevant_intrinsic_exprs):
                break
            if i != len(ids):
                continue
            # Generate all possible combinations of i items without repetition
            for comb in itertools.combinations(relevant_intrinsic_exprs, i):
                ids_comb: set[int] = set.union(*[expr.allids for expr in comb])
                if len(ids_comb) != len(ids):
                    continue
                # Generate the multiplication expression
                mul_expr: Exp = comb[0]
                for j in range(1, i):
                    mul_expr = mul_expr * comb[j]
                # Evaluate the multiplication expression
                mul_data: ExpData = self.knowledge.eval(mul_expr,
                                                        self.specific[exp_name].experiment)
                if not (diff_res / mul_data).is_const:
                    continue
                # Re-evaluate the multiplication expression in testing experiments
                self.specific[exp_name].complement_experiments_test(num=20)
                experiments: list[ExpStructure] = self.specific[exp_name].experiment_test
                lst_expdata: list[ExpData] = [self.knowledge.eval(diff_expr / mul_expr, exprt) for exprt in experiments]
                if not all([d.is_const for d in lst_expdata]):
                    continue
                wrapped_data: ExpData = ExpData.wrapped_list_of_const_data([expdata.const_data for expdata in lst_expdata], 20)
                if wrapped_data.is_const:
                    iexp_config = IExpConfig.From(exp_name)
                    uc: Intrinsic = Intrinsic.From(SExp.Mk(iexp_config, diff_expr / mul_expr))
                    name: str | None = self.register_universal_constant(uc,
                                                                        diff_expr / mul_expr,
                                                                        self.specific[exp_name])
                    if name is None:
                        continue
                    new_const_exprs: list[Exp] = self.knowledge.specialize(self.knowledge.gen_atom_concept(name),
                                                                           exp_name)
                    if not new_const_exprs or len(new_const_exprs) > 1:
                        continue
                    res_expr: Exp = new_const_exprs[0] * mul_expr
                    return res_expr
        return None

    def validate_general_law_in_others(self, gl: GeneralLaw, exp_name: str,
                                       reset: bool = False) -> bool:
        num_valid = 0
        for experiment in self.specific:
            if experiment == exp_name:
                continue
            if reset:
                self.specific[experiment].experiments_reset()
            check_result, _ = self.check_general_laws(experiment, {gl.name: gl})
            check_result: dict[str, bool]
            if check_result[gl.name]:
                num_valid += 1

        return num_valid > 1 and len(gl.valid_experiments) > 1

    def _discard_new_candidate_concept(self, concept_name: str) -> None:
        """Remove a just-created concept when no law references it.

        Law proposal used to register the concept before learning that a
        process-specific proposition failed the old multi-experiment gate.
        Those orphan concepts remained selectable forever and were the main
        source of vocabulary explosion.
        """

        if concept_name not in self.knowledge.fetch_concepts:
            return
        self.knowledge.remove_concept(concept_name)
        self.memory.action_bandit.actions.discard(concept_name)
        self.memory.redundant_concepts.discard(concept_name)
        self.memory.concept_signatures = {
            key: value
            for key, value in self.memory.concept_signatures.items()
            if value != concept_name
        }

    def propose_new_general_law(self,
                                law_name: str = None,
                                specific_model: SpecificModel = None,
                                exp_name: str = None,
                                concept: Concept | None = None,
                                law_type: Literal['IsConserved', 'IsZero'] = None) -> None:
        if law_name is not None and specific_model is not None:
            law_exp: Exp = specific_model.conclusions.get(law_name).unwrap_exp
            law_type: str = specific_model.conclusions.get(law_name).prop_type  # IsConserved or IsZero
            if law_type not in ['IsConserved', 'IsZero']:
                raise NotImplementedError("Unknown conclusion type")
            if self.possess_large_coes(law_exp, specific_model):
                return
            concept: Concept | None = self.knowledge.generalize(exp_name, law_exp)
            if (concept is None) or ('Sum' not in str(concept) and len(concept.preids) > 1):
                return  # For "ForAll" indication general laws, only consider one index cases
            if self.is_geometric(concept, exp_name):
                return  # Do not consider pure geometric concepts
            # ── Dimensional check for general laws ──
            try:
                concept_str = str(concept)
                known_dims = build_known_dims_from_knowledge(self.knowledge)
                is_dim_valid, dims, dim_msg = check_concept_dimensions(concept_str, known_dims)
                if not is_dim_valid:
                    print(f"[DIM REJECT LAW] {concept_str[:100]} — {dim_msg}", flush=True)
                    return
            except Exception:
                pass
        is_general = self.is_general_in_current_experiment(concept, exp_name, law_type)
        if is_general:
            print(f"Possible general law: {str(concept)} is {law_type}. "
                  "Register it as both concept and law.", flush=True)
            concepts_before = set(self.knowledge.fetch_concepts)
            new_concept_name: str | None = self.register_concept_or_return_name(concept, self.specific[exp_name])
            if new_concept_name is None:
                return
            created_concept = new_concept_name not in concepts_before
            if self.is_geometric(self.knowledge.fetch_concept_by_name(new_concept_name),
                                 exp_name):
                print("Pure geometric concept. Skip.", flush=True)
                if created_concept:
                    self._discard_new_candidate_concept(new_concept_name)
                return
            prop_concept: Concept = self.knowledge.gen_atom_concept(new_concept_name)
            prop: Proposition = (Proposition.Conserved if law_type == 'IsConserved' else Proposition.Zero)(prop_concept)
            if new_concept_name is not None:
                # Register the law in general model
                gl = GeneralLaw(
                    name="R_temp", prop=prop,
                    valid_experiments=[exp_name]
                )
                # If the general law already exists in general model, return
                if gl in self or (specific_model is not None and gl.is_trivial(self.knowledge, specific_model.expand_exp, exp_name)):
                    print("Trivial or existing general law. Skip.", flush=True)
                    if created_concept:
                        self._discard_new_candidate_concept(new_concept_name)
                    return
                self.validate_general_law_in_others(gl, exp_name)
                # A proposition supported only by the current process is a
                # contextual law, not a failed universal-law proposal.  Its
                # explicit valid_experiments scope prevents it from leaking
                # into other regimes; later experiments may broaden that
                # scope through ordinary validation.
                gl: GeneralLaw | None = self.register_general_law(
                    prop,
                    valid_experiments=gl.valid_experiments,
                )
                if gl is not None:
                    self.completion_exclude[gl.name] = copy.deepcopy(
                        gl.valid_experiments
                    )
                elif created_concept:
                    self._discard_new_candidate_concept(new_concept_name)

    def is_geometric(self, concept: Concept, exp_name: str) -> bool:
        lst_atom_names: list[str] = [atomexpr.name
                                     for atomexpr in concept.exp.all_atoms]
        if 't' in lst_atom_names:
            # Remove the time variable
            lst_atom_names.remove('t')
        geo_vars: list[str] = [i.name for i in self.specific[exp_name].fetch_geometry_variables]
        if not lst_atom_names:
            return True
        if all([name in geo_vars for name in lst_atom_names]):
            return True
        return False

    def is_general_in_current_experiment(self,
                                         concept: Concept,
                                         current_exp_name: str,
                                         law_type: str = Literal['IsConserved', 'IsZero']) -> bool:
        # Generate specialized expressions
        specific_exprs: list[Exp] = self.knowledge.specialize(concept=concept, exp_name=current_exp_name)
        if not specific_exprs:
            return False
        # NOTE: What if the concept can't be specialized in some mappings while conserved in others?
        specific_exprs_data: list[ExpData] = [self.calc_expr(expr, current_exp_name) for expr in specific_exprs]
        for ele in specific_exprs_data:
            if law_type == 'IsConserved' and not ele.is_conserved:
                return False
            if law_type == 'IsZero' and not ele.is_zero:
                return False
        return True

    def convert_conserved_to_zero(self, exp_name: str, exprs: list[AtomExp]):
        spm: SpecificModel = self.specific[exp_name]
        gls: list[str] = [gl for gl in self.general_laws.keys()
                          if (self.general_laws[gl].type == 'IsConserved' and
                          exp_name in self.general_laws[gl].valid_experiments)]
        for gl in gls:
            if self.general_laws[gl].indication == 'ForAll':
                self.search_forall_completion(self.general_laws[gl], spm, exprs, exp_name, for_convertion="Zero")
            elif self.general_laws[gl].indication == 'Sum':
                # TODO: Implement the conversion for sum type general laws
                pass
            else:
                raise NotImplementedError("Unknown indication type")

    def fix_general_laws(self):
        """
        If there is a zero-type general law, remove the corresponding conserved-type general law.
        For all general laws recorded,
        check if all their valid experiments are recorded in completion_exclude.
        """
        print("Fixing general laws.", flush=True)
        # For a zero-type general law, remove the corresponding conserved-type general law
        zero_law_concepts: list[Concept] = [gl.relevant_concept for gl in self.general_laws.values()
                                            if gl.type == 'IsZero']
        conserved_gl_to_remove: list[str] = [name for name, gl in self.general_laws.items()
                                             if gl.type == 'IsConserved' and gl.relevant_concept in zero_law_concepts]
        while conserved_gl_to_remove:
            son_gl_to_remove: list[str] = []
            for name in conserved_gl_to_remove:
                son_gl_to_remove.extend([son for son, son_gl in self.general_laws.items()
                                         if son_gl.father is not None and son_gl.father.name == name])
                self.remove_general_law(name)
            conserved_gl_to_remove = copy.deepcopy(son_gl_to_remove)

        for _, gl in self.general_laws.items():
            if gl.name not in self.completion_exclude:
                self.completion_exclude[gl.name] = copy.deepcopy(gl.valid_experiments)
            else:
                for exp_name in gl.valid_experiments:
                    self.record_exclusion(gl.name, exp_name)

    def confirm_validation(self, exp_name: str, exprs: list[Exp], gc_type: str) -> bool:
        """
        Confirm if the expressions are really valid in the experiment.
        """
        if not isinstance(exprs, list):
            exprs: list[Exp] = [exprs]

        flag = True
        for expr in exprs:
            if not self.specific[exp_name].test_on_test_experiment(exp=expr,
                                                                   type='conserved' if gc_type == 'IsConserved' else 'zero'):
                flag = False
                break
        return flag


@dataclass(frozen=True)
class GeneralLawReplacementResult:
    """Audit record for one strict, atomic active-basis replacement."""

    status: str
    old_law: str
    new_law: str | None = None
    certificate: dict | None = None
    validated_experiments: tuple[str, ...] = ()
    contradiction_experiments: tuple[str, ...] = ()
    reason: str = ""


class GeneralLawService:
    def __init__(self, general_law_base: GeneralLawBase, json_data: dict | None = None):
        self.general_law_base: GeneralLawBase = general_law_base
        self.marked_general_laws: Set[str] = set()
        # Useful general laws that are marked
        self.diffalg_for_marked_gl: Dict[str, Dict[str, DiffalgService]] = {}
        self.derived_general_laws: Dict[str, dict] = {}
        self.derived_concepts: Set[str] = set()
        self.independence_certificates: Dict[str, dict] = {}
        self.probabilistic_reducer_config = ProbabilisticReducerConfig()
        if json_data is not None:
            self.marked_general_laws = set(json_data["marked_general_laws"])
            self.diffalg_for_marked_gl = {
                key: {
                    exp_name: DiffalgService.from_json(self.general_law_base.knowledge, value)
                    for exp_name, value in json_data["diffalg_for_marked_gl"][key].items()
                }
                for key in json_data["diffalg_for_marked_gl"]
            }
            self.derived_general_laws = copy.deepcopy(
                json_data.get("derived_general_laws", {})
            )
            self.derived_concepts = set(json_data.get("derived_concepts", []))
            self.independence_certificates = copy.deepcopy(
                json_data.get("independence_certificates", {})
            )
            self.probabilistic_reducer_config = ProbabilisticReducerConfig.from_json(
                json_data.get("probabilistic_reducer_config")
            )
        self.rank_reducer = GeneralLawRankReducer(
            self.general_law_base,
            self.probabilistic_reducer_config,
        )

    def to_json(self):
        return {
            "marked_general_laws": list(self.marked_general_laws),
            "diffalg_for_marked_gl": {
                key: {
                    exp_name: value.to_json()
                    for exp_name, value in diffalg_dict.items()
                }
                for key, diffalg_dict in self.diffalg_for_marked_gl.items()
            },
            "derived_general_laws": self.derived_general_laws,
            "derived_concepts": sorted(self.derived_concepts),
            "independence_certificates": self.independence_certificates,
            "probabilistic_reducer_config": asdict(self.probabilistic_reducer_config),
        }

    def replace_general_law_atomically(
        self,
        old_law_name: str,
        new_prop: Proposition,
        *,
        validation_experiments: Iterable[str],
        contradiction_experiments: Iterable[str] = (),
        validate_against_specific_models: bool = True,
    ) -> GeneralLawReplacementResult:
        """Replace a falsified generator after validation and rank admission.

        The candidate is first checked without mutating the knowledge base.
        It is then registered temporarily so the real ``GeneralLawRankReducer``
        can expand it through ``Knowledge`` and ``SpecificModel``.  Only an
        explicit ``independent`` certificate commits the replacement; both a
        dependent and an inconclusive candidate are removed again.

        On commit the old proposition leaves the active basis, but its
        proposition, exclusions and lineage remain in the service archive.
        This makes the operation visible as a true revision (rather than an
        unrelated deletion/addition) to checkpoint persistence and reward.
        """

        base = self.general_law_base
        if old_law_name not in base.general_laws:
            return GeneralLawReplacementResult(
                "missing_old_law", old_law_name,
                reason="the requested parent is not in the active theory",
            )
        old_law = base.general_laws[old_law_name]
        requested_validation = tuple(dict.fromkeys(validation_experiments))
        contradictions = tuple(dict.fromkeys(contradiction_experiments))
        unknown = [name for name in requested_validation if name not in base.specific]
        if unknown:
            return GeneralLawReplacementResult(
                "validation_unavailable", old_law_name,
                validated_experiments=(),
                contradiction_experiments=contradictions,
                reason=f"unknown validation experiments: {unknown}",
            )

        validated: list[str] = []
        if validate_against_specific_models:
            probe = GeneralLaw(
                "R_revision_candidate", new_prop, father=old_law,
            )
            for exp_name in requested_validation:
                try:
                    check, _ = base.check_general_laws(exp_name, {probe.name: probe})
                except Exception as error:
                    return GeneralLawReplacementResult(
                        "validation_error", old_law_name,
                        validated_experiments=tuple(validated),
                        contradiction_experiments=contradictions,
                        reason=f"{exp_name}: {type(error).__name__}: {error}",
                    )
                if not check.get(probe.name, False):
                    return GeneralLawReplacementResult(
                        "validation_failed", old_law_name,
                        validated_experiments=tuple(validated),
                        contradiction_experiments=contradictions,
                        reason=f"candidate failed held-out experiment {exp_name}",
                    )
                validated.append(exp_name)
        else:
            # This mode is for an adapter whose raw observations were already
            # held out by TheoryRevisionController.  The named experiments
            # still have to exist, preventing fabricated scope identifiers.
            validated.extend(requested_validation)

        if not validated:
            return GeneralLawReplacementResult(
                "validation_required", old_law_name,
                contradiction_experiments=contradictions,
                reason="strict revision requires at least one validation experiment",
            )

        candidate = base.register_general_law(
            new_prop, father=old_law, valid_experiments=list(validated)
        )
        if candidate is None:
            return GeneralLawReplacementResult(
                "registration_rejected", old_law_name,
                validated_experiments=tuple(validated),
                contradiction_experiments=contradictions,
                reason="knowledge base rejected a duplicate or invalid proposition",
            )

        retained_basis = set(self.marked_general_laws) - {old_law_name}
        try:
            certificate = self.rank_reducer.classify(candidate.name, retained_basis)
        except Exception as error:
            base.remove_general_law(candidate.name)
            return GeneralLawReplacementResult(
                "reducer_error", old_law_name,
                validated_experiments=tuple(validated),
                contradiction_experiments=contradictions,
                reason=f"{type(error).__name__}: {error}",
            )
        certificate_json = certificate.to_json()
        self.independence_certificates[candidate.name] = certificate_json
        if certificate.status != "independent":
            base.remove_general_law(candidate.name)
            self.independence_certificates.pop(candidate.name, None)
            return GeneralLawReplacementResult(
                "rejected_" + certificate.status,
                old_law_name,
                certificate=certificate_json,
                validated_experiments=tuple(validated),
                contradiction_experiments=contradictions,
                reason=certificate.reason,
            )

        for exp_name in contradictions:
            base.record_exclusion(old_law_name, exp_name)
        old_exclusions = sorted(base.completion_exclude.get(old_law_name, []))
        self.derived_general_laws[old_law_name] = {
            "prop": str(old_law.prop),
            "valid_experiments": sorted(old_law.valid_experiments),
            "father": old_law.father_name,
            "certificate": self.independence_certificates.get(old_law_name),
            "archive_reason": "falsified_and_revised",
            "replaced_by": candidate.name,
            "contradiction_experiments": old_exclusions,
            "top_concepts": sorted(set(re.findall(r"C_\d+", str(old_law.prop)))),
        }

        self.marked_general_laws.discard(old_law_name)
        self.diffalg_for_marked_gl.pop(old_law_name, None)
        self.independence_certificates.pop(old_law_name, None)
        self.marked_general_laws.add(candidate.name)
        # Rank certificates are the primary admission gate.  Differential
        # ideals are rebuilt lazily when a later symbolic reduction needs
        # them, avoiding the old multi-minute commit path.
        self.diffalg_for_marked_gl[candidate.name] = {}
        base.completion_exclude[candidate.name] = sorted(set(validated))
        base.remove_general_law(
            old_law_name,
            preserve_exclusion=True,
            preserve_lineage=True,
        )
        self.rank_reducer._complexity_cache.clear()
        self.exchange_heavier_basis_laws(candidate.name)
        self._refresh_derived_concept_actions()
        if set(self.marked_general_laws) != set(base.general_laws):
            raise RuntimeError("atomic replacement violated the active-basis invariant")
        return GeneralLawReplacementResult(
            "revised", old_law_name, candidate.name, certificate_json,
            tuple(validated), contradictions,
        )

    def _record_derived_law(self, gl_name: str, certificate: RankCertificate):
        """Archive a dependent theorem and remove it from the theory basis.

        Top-level concepts of a rejected law are also removed from the inner
        concept-action pool when no accepted law needs them.  They remain in
        the knowledge/specific models for auditability and can be garbage
        collected by the normal obliviate pass later.
        """

        law = self.general_law_base.general_laws[gl_name]
        self.derived_general_laws[gl_name] = {
            "prop": str(law.prop),
            "valid_experiments": sorted(law.valid_experiments),
            "father": law.father_name,
            "certificate": certificate.to_json(),
        }
        candidate_atoms = set(re.findall(r"C_\d+", str(law.prop)))
        self.derived_general_laws[gl_name]["top_concepts"] = sorted(candidate_atoms)
        other_dependencies: set[str] = set()
        for other_name, other_law in self.general_law_base.general_laws.items():
            if other_name == gl_name:
                continue
            other_dependencies |= self._concept_dependencies(str(other_law.prop))
        for concept_name in candidate_atoms - other_dependencies:
            self.derived_concepts.add(concept_name)
            self.general_law_base.memory.action_bandit.actions.discard(concept_name)
        self.general_law_base.remove_general_law(gl_name)

    def _concept_dependencies(self, expression: str) -> set[str]:
        """Resolve concept dependencies without traversing Rust sum-expression atoms.

        Calling ``all_atoms`` on the expression placeholder of a sum concept
        can underflow an allocation size in the current core extension.  The
        persisted concept definitions are sufficient for this lightweight
        reachability query and avoid that unsafe boundary.
        """

        pending = list(set(re.findall(r"C_\d+", expression)))
        dependencies: set[str] = set()
        while pending:
            name = pending.pop()
            if name in dependencies:
                continue
            dependencies.add(name)
            try:
                concept = self.general_law_base.knowledge.fetch_concept_by_name(name)
            except Exception:
                concept = None
            if concept is None:
                continue
            pending.extend(
                child for child in re.findall(r"C_\d+", str(concept))
                if child not in dependencies
            )
        return dependencies

    def _refresh_derived_concept_actions(self):
        """Keep dependent theorem concepts out of future search actions."""

        basis_dependencies: set[str] = set()
        for law in self.general_law_base.general_laws.values():
            basis_dependencies |= self._concept_dependencies(str(law.prop))
        derived_top = {
            concept
            for record in self.derived_general_laws.values()
            for concept in record.get("top_concepts", [])
        }
        self.derived_concepts = derived_top - basis_dependencies
        self.general_law_base.memory.action_bandit.actions.difference_update(
            self.derived_concepts
        )

    def audit_independence(self, law_names: Iterable[str] | None = None) -> dict[str, dict]:
        """Return a non-mutating incremental basis audit for a checkpoint."""

        names = list(law_names or self.general_law_base.general_laws.keys())
        names = sorted(names, key=self.rank_reducer.priority_key)
        accepted: list[str] = []
        results: dict[str, dict] = {}
        for name in names:
            certificate = self.rank_reducer.classify(name, accepted)
            results[name] = certificate.to_json()
            if certificate.status != "dependent":
                accepted.append(name)
        return results

    def prune_dependent_basis(self) -> dict[str, dict]:
        """Migrate an existing checkpoint to the independent local basis.

        The method is intentionally explicit rather than running during file
        loading.  Callers can audit first, then opt into the mutation and save
        under a new checkpoint name.
        """

        results = self.audit_independence()
        for name, record in results.items():
            if record["status"] != "dependent" or name not in self.general_law_base.general_laws:
                continue
            certificate = RankCertificate(**{
                key: value for key, value in record.items()
                if key in RankCertificate.__dataclass_fields__
            })
            self.independence_certificates[name] = certificate.to_json()
            self._record_derived_law(name, certificate)
        for name, record in results.items():
            if name in self.general_law_base.general_laws:
                self.independence_certificates[name] = record
        self._refresh_derived_concept_actions()
        return results

    def exchange_heavier_basis_laws(self, new_law_name: str) -> list[str]:
        """Perform the greedy basis-exchange step induced by a new generator.

        The active set before insertion is assumed to be a minimum-weight
        basis under ``priority_key``.  A new law cannot displace a cheaper
        generator, so only more expensive active laws need to be reconsidered.
        Processing them in increasing weight order is exact matroid-style
        exchange for conserved functional independence and a deterministic
        irredundant-basis heuristic for localized zero ideals.  It avoids a
        full checkpoint audit after every discovery.
        """

        if new_law_name not in self.general_law_base.general_laws:
            return []
        new_priority = self.rank_reducer.priority_key(new_law_name)
        candidates = sorted(
            (
                name for name in self.marked_general_laws
                if name != new_law_name
                and self.rank_reducer.priority_key(name) > new_priority
            ),
            key=self.rank_reducer.priority_key,
        )
        removed: list[str] = []
        for name in candidates:
            if name not in self.general_law_base.general_laws:
                continue
            basis = set(self.marked_general_laws) - {name}
            certificate = self.rank_reducer.classify(name, basis)
            self.independence_certificates[name] = certificate.to_json()
            if certificate.status != "dependent":
                continue
            print(
                f"--- basis exchange: {new_law_name} displaces {name} "
                f"using {certificate.basis_laws}",
                flush=True,
            )
            self._record_derived_law(name, certificate)
            removed.append(name)
        return removed

    def verify_minimum_basis(self) -> dict[str, dict]:
        """Audit the active basis and return any greedy-minimality violations."""

        audit = self.audit_independence()
        return {
            name: certificate
            for name, certificate in audit.items()
            if certificate["status"] == "dependent"
        }

    def refresh_scope_changed_basis_laws(self) -> list[str]:
        """Revalidate active generators whose experimental scope changed."""

        changed = [
            name for name in self.marked_general_laws
            if name in self.general_law_base.general_laws
            and set(
                self.independence_certificates.get(name, {}).get(
                    "candidate_experiments", []
                )
            ) != set(self.general_law_base.general_laws[name].valid_experiments)
        ]
        changed.sort(key=self.rank_reducer.priority_key)
        removed: list[str] = []
        for name in changed:
            if name not in self.general_law_base.general_laws:
                continue
            certificate = self.rank_reducer.classify(
                name,
                set(self.marked_general_laws) - {name},
            )
            self.independence_certificates[name] = certificate.to_json()
            if certificate.status == "dependent":
                print(
                    f"--- scope refresh makes {name} dependent on "
                    f"{certificate.basis_laws}",
                    flush=True,
                )
                self._record_derived_law(name, certificate)
                removed.append(name)
                continue
            removed.extend(self.exchange_heavier_basis_laws(name))
        return removed

    def contains_direct_subexpr(self, prop_name: str, sub_prop_name: str) -> bool:
        general_law_base: GeneralLawBase = self.general_law_base
        knowledge: Knowledge = general_law_base.knowledge
        expr_concept: Concept = general_law_base.general_laws[prop_name].prop.unwrap_concept
        sub_expr_concept: Concept = general_law_base.general_laws[sub_prop_name].prop.unwrap_concept
        if general_law_base.general_laws[prop_name].type != general_law_base.general_laws[sub_prop_name].type:
            return False
        if general_law_base.general_laws[prop_name].indication != general_law_base.general_laws[sub_prop_name].indication:
            return False
        if general_law_base.general_laws[prop_name].indication == "ForAll":
            if set(expand_expression(str(sub_expr_concept.exp))).issubset(set(expand_expression(str(expr_concept.exp)))):
                return True
        else:
            prop_terms: list[str] = expand_expression(str(expr_concept).split('|- ')[1])
            sub_prop_terms: list[str] = expand_expression(str(sub_expr_concept).split('|- ')[1])
            ext_prop_terms: list[str] = list(set(prop_terms
                                             + [('-' + item) for item in prop_terms if not item.startswith('-') and not item.startswith('C_')]
                                             + [('-1 * ' + item) for item in prop_terms if not item.startswith('-') and item.startswith('C_')]
                                             + [item.split("-")[1] for item in prop_terms if item.startswith('-') and not item.startswith('-1')]
                                             + [item.split("-1 * ")[1] for item in prop_terms if item.startswith('-1')]))
            if set(sub_prop_terms).issubset(set(ext_prop_terms)):
                return True
        for exp_name in general_law_base.general_laws[sub_prop_name].valid_experiments:
            spm: SpecificModel = general_law_base.specific[exp_name]
            exprs: list[Exp] = [spm._sympy_of_raw_defi(spm.expand_exp(exp))
                                for exp in knowledge.specialize(expr_concept, exp_name)]
            sub_exprs: list[Exp] = [spm._sympy_of_raw_defi(spm.expand_exp(exp))
                                    for exp in knowledge.specialize(sub_expr_concept, exp_name)]
            if len(exprs) != len(sub_exprs):
                return False
            for sub_expr in sub_exprs:
                rest = sp.Wild('rest')
                pattern = sub_expr + rest
                if not any([expr.match(pattern) is not None for expr in exprs]):
                    return False
        return True

    def compare(self, gl_name_1: str, gl_name_2: str) -> bool:
        """
        Compare the relative generality and complexity of two general laws.
        If gl_name_1 is more general than gl_name_2, return True.
        """
        gl1 = self.general_law_base.general_laws[gl_name_1]
        gl2 = self.general_law_base.general_laws[gl_name_2]
        if set(gl2.valid_experiments) < set(gl1.valid_experiments):
            return True
        if set(gl2.valid_experiments) == set(gl1.valid_experiments):
            if gl1.indication == "ForAll" and gl2.indication == "ForAll":
                exp1: Exp = gl1.prop.unwrap_concept.exp
                exp2: Exp = gl2.prop.unwrap_concept.exp
                if count_gradient_components(expand_expression(str(exp1))) < count_gradient_components(expand_expression(str(exp2))):
                    return True
                elif count_gradient_components(expand_expression(str(exp1))) > count_gradient_components(expand_expression(str(exp2))):
                    return False
            complexity_1 = self.general_law_base.knowledge.raw_complexity(gl1.prop)
            complexity_2 = self.general_law_base.knowledge.raw_complexity(gl2.prop)
            return complexity_1 < complexity_2
        return False

    def compare_for_sort(self, x1, x2):
        if self.compare(x1, x2):
            return 1
        else:
            return -1

    def can_reduce(self, ideal_service: DiffalgService, general_prop: Proposition,
                   timeout: float = 20.0) -> bool:
        exprs = self.general_law_base.knowledge.specialize(general_prop.unwrap_concept, ideal_service.exp_name)
        exprs = [self.general_law_base.specific[ideal_service.exp_name].expand_exp(expr) for expr in exprs]
        match general_prop.prop_type:
            case 'Conserved':
                func = Proposition.IsConserved
            case 'Zero':
                func = Proposition.IsZero
            case _:
                raise NotImplementedError("Invalid proposition type")
        props = [func(expr) for expr in exprs]
        return all(ideal_service.can_reduce(prop, timeout) for prop in props)

    def insert_to_diffalg(self, gl_name: str) -> bool:
        """
        Try to insert a new general law to the marked general laws.
        """
        valid_experiments: list[str] = self.general_law_base.general_laws[gl_name].valid_experiments

        # Collect all marked gl_name that is dominated by gl_name
        collect_names: set[str] = {old_gl_name
                                   for old_gl_name in self.marked_general_laws
                                   if self.compare(old_gl_name, gl_name)}
        if any([self.contains_direct_subexpr(old_gc_name, gl_name)
                for old_gc_name in collect_names]):
            if self.general_law_base.general_laws[gl_name].father is not None:
                print(f"--- {gl_name} is a subexpr of some marked general laws in {collect_names}", flush=True)
                self.general_law_base.remove_general_law(gl_name)
                return False

        # Build diffalg for collect_names
        diffalg_for_collect_names: dict[str, DiffalgService] = {}  # key: exp_name, value: ideal_service
        is_useful = False  # If the general conclusion can't be reduced in any of its valid experiments, then it is useful
        for exp_name in sorted(valid_experiments, key=lambda x: len(self.general_law_base.knowledge.fetch_expstruct(x).original_data)):
            ideal_service: DiffalgService = DiffalgService(self.general_law_base.knowledge, exp_name)
            props: list[Proposition] = []
            # In each valid experiment, iterate over all more generalized general laws
            # Specialize them in the current experiment and insert them to the ideal service
            for name in collect_names:
                relevant_concept: Concept = self.general_law_base.general_laws[name].relevant_concept
                exprs: list[Exp] = self.general_law_base.knowledge.specialize(relevant_concept, exp_name)
                exprs = [self.general_law_base.specific[exp_name].expand_exp(expr) for expr in exprs]
                func: Callable[[Exp], Proposition] = Proposition.IsConserved \
                    if self.general_law_base.general_laws[name].type == 'IsConserved' \
                    else Proposition.IsZero
                props.extend([func(expr) for expr in exprs])
            for prop in props:
                ideal_service.register_law(prop)
            diffalg_for_collect_names[exp_name] = ideal_service  # Finish building diffalg in current experiment
            if not is_useful:
                # If the prop of current general law can't be reduced, it is useful
                try:
                    is_useful = not self.can_reduce(ideal_service, self.general_law_base.general_laws[gl_name].prop,
                                                    timeout=20.0)
                except Exception:
                    continue

        # If the prop is useful, then insert it to marked_general_laws
        if is_useful:
            self.marked_general_laws.add(gl_name)
            self.diffalg_for_marked_gl[gl_name] = diffalg_for_collect_names
            return True
        self.general_law_base.remove_general_law(gl_name)
        return False

    def update_diffalg(self, gl_name: str):
        # Initialize specialized law set for gc_name
        relevant_concept = self.general_law_base.general_laws[gl_name].relevant_concept
        props: Dict[str, List[Proposition]] = {}
        for exp_name in self.general_law_base.general_laws[gl_name].valid_experiments:
            exprs = self.general_law_base.knowledge.specialize(relevant_concept, exp_name)
            exprs = [self.general_law_base.specific[exp_name].expand_exp(expr) for expr in exprs]
            func = Proposition.IsConserved if self.general_law_base.general_laws[gl_name].type == 'IsConserved' else Proposition.IsZero
            props[exp_name] = [func(expr) for expr in exprs]

        # Update diffalg for all marked smaller general laws
        copy_marked = copy.deepcopy(self.marked_general_laws)
        for old_gl_name in copy_marked:
            # Update the diffalg of those marked general laws whose scope is covered by gc_name
            if self.compare(gl_name, old_gl_name):
                print(f"Check if {old_gl_name} should be removed", flush=True)
                if self.general_law_base.general_laws[old_gl_name].father is not None and \
                        self.general_law_base.general_laws[old_gl_name].father.father is not None and \
                        self.contains_direct_subexpr(gl_name, old_gl_name):
                    print(f"--- {old_gl_name} is a subexpr of {gl_name}", flush=True)
                    is_useful = False
                else:
                    if self.general_law_base.general_laws[old_gl_name].father is None and \
                            self.general_law_base.general_laws[old_gl_name].type == 'IsZero':
                        is_useful = True
                    else:
                        diffalg_for_old_gl: dict[str, DiffalgService] = self.diffalg_for_marked_gl[old_gl_name]
                        for exp_name in diffalg_for_old_gl:
                            ideal_service: DiffalgService = diffalg_for_old_gl[exp_name]
                            # Append the new general law to the ideal service
                            for prop in props[exp_name]:
                                ideal_service.register_law(prop)
                        is_useful = False
                        for exp_name in diffalg_for_old_gl:
                            ideal_service = diffalg_for_old_gl[exp_name]
                            if not is_useful:  # NOTE: need break if is_useful is True?
                                try:
                                    is_useful = not self.can_reduce(ideal_service, self.general_law_base.general_laws[old_gl_name].prop,
                                                                    timeout=5.0)
                                except Exception:
                                    is_useful = True
                if not is_useful:
                    print(f"--- {old_gl_name} is removed", flush=True)
                    certificate = RankCertificate(
                        status="dependent",
                        method="symbolic_differential_ideal",
                        reason=f"reduced after inserting {gl_name}",
                        basis_laws=[gl_name],
                        candidate_scope_size=len(
                            self.general_law_base.general_laws[old_gl_name].valid_experiments
                        ),
                        candidate_experiments=sorted(
                            self.general_law_base.general_laws[old_gl_name].valid_experiments
                        ),
                        dependent_experiments=sorted(
                            self.general_law_base.general_laws[old_gl_name].valid_experiments
                        ),
                        confidence_lower_bound=1.0,
                    )
                    self.independence_certificates[old_gl_name] = certificate.to_json()
                    self._record_derived_law(old_gl_name, certificate)

    def reduce_general_laws(self) -> int:
        """
        Try to reduce new unmarked general laws.
        If a genral law can be reduced in all its valid experiments by other general laws
        whose scope covers it, then it is regarded as redundant and removed.
        Otherwise, it is marked and its diffalg is built.
        Other general laws whose scope are covered by it should be checked and updated.
        """
        old_marked = copy.deepcopy(self.marked_general_laws)
        self.refresh_scope_changed_basis_laws()
        gl_name_diff = set(self.general_law_base.general_laws.keys()) - self.marked_general_laws
        # Simpler expanded generators must enter the basis before their powers
        # and products; raw DSL complexity sees every ``R := C_x`` as equally
        # small and therefore cannot provide this ordering.
        gl_name_diff = sorted(list(gl_name_diff), key=self.rank_reducer.priority_key)
        for gl_name in gl_name_diff:
            print(f"Try to reduce {gl_name}", flush=True)
            certificate = self.rank_reducer.classify(
                gl_name,
                self.marked_general_laws,
            )
            self.independence_certificates[gl_name] = certificate.to_json()
            if certificate.status == "dependent":
                print(
                    f"--- {gl_name} is locally dependent on "
                    f"{certificate.basis_laws}; archive as a derived theorem",
                    flush=True,
                )
                self._record_derived_law(gl_name, certificate)
                continue
            if certificate.status == "independent":
                # The typed local differential-ideal certificate is the
                # admission authority.  The legacy symbolic reducer compares
                # conserved quantities at their observed constant values and
                # can therefore (incorrectly) erase a zero-order relation such
                # as PV/(nT)-R=0 merely because both sides are conserved.
                self.marked_general_laws.add(gl_name)
                self.diffalg_for_marked_gl[gl_name] = {}
                print(f"--- {gl_name} is marked as useful", flush=True)
                self.exchange_heavier_basis_laws(gl_name)
                continue
            # Unsupported/transcendental cases retain the bounded legacy
            # symbolic fallback rather than being admitted without evidence.
            if self.insert_to_diffalg(gl_name):
                print(f"--- {gl_name} is marked as useful (symbolic fallback)", flush=True)
        self._refresh_derived_concept_actions()
        assert set(self.marked_general_laws) == set(self.general_law_base.general_laws.keys())
        reward: int = len((self.marked_general_laws - old_marked) | (old_marked - self.marked_general_laws))
        return reward
