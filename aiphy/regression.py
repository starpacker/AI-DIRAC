"""
This module contains various functions for symbolic regression,
which is accelerated through the utilization of parallel computing techniques.

The capabilities of the symbolic regression methods
— search_trivial_relations
- search_relations_ver1
- search_relations_ver2
- search_relations_ver3
are ranked in ascending order. The search_relations_ver3 method is the most powerful one,
while it takes a long time to run.
"""

from typing import List, Tuple, Dict, Callable
import os
from aiphy.core import (
    search_trivial_relations,
    search_binary_relations,
    search_relations_ver1,
    search_relations_ver2,
    search_relations_ver3,
    DataStruct,
    Exp,
    ExpData
)

from .pca import search_relations_by_pca

def search_relations_ver3_and_pca(ds: DataStruct, debug: bool = False, cpu_num: int = 50) -> List[Tuple[Exp, ExpData]]:
    res = search_relations_ver3(ds, debug, cpu_num)
    res_pca = search_relations_by_pca(ds)
    res_dict = {i[0]: i[1] for i in res}
    res_pca_dict = {i[0]: i[1] for i in res_pca}
    res_dict.update(res_pca_dict)
    return [(i, res_dict[i]) for i in res_dict]


def search_relations_pysr(
    ds: DataStruct,
    debug: bool = False,
    cpu_num: int = 50,
    *,
    exp_name: str | None = None,
    specifications=None,
    old_formula: str | dict[str, str] | None = None,
) -> List[Tuple[Exp, ExpData]]:
    """Run the proven baseline and add only holdout-validated PySR proposals."""
    baseline = search_relations_ver3_and_pca(ds, debug, cpu_num)
    try:
        # Lazy import keeps PySR and Julia optional for baseline/checkpoint use.
        from .pysr_regression import propose_relations_pysr
        if specifications is None:
            from .sr_metadata import infer_generic_specifications
            specifications = infer_generic_specifications(
                ds, exp_name=exp_name,
                max_targets=int(os.getenv("AI_NEWTON_PYSR_TARGETS", "2")),
            )
        proposals = propose_relations_pysr(
            ds,
            debug,
            cpu_num,
            specification=specifications or None,
            old_formula=old_formula,
        )
    except (ImportError, ModuleNotFoundError) as error:
        if debug:
            print(f"[PySR] optional dependency unavailable: {error}", flush=True)
        proposals = []
    result = {str(expression): (expression, data) for expression, data in baseline}
    for expression, data in proposals:
        result.setdefault(str(expression), (expression, data))
    return list(result.values())

search_type: Dict[str, Callable[[DataStruct, bool, int], List[Tuple[Exp, ExpData]]]] = {
    None: search_relations_ver1,
    'trivial': search_trivial_relations,
    'ver1': search_relations_ver1,
    'ver2': search_relations_ver2,
    'ver3': search_relations_ver3_and_pca,
    'pysr': search_relations_pysr,
}
