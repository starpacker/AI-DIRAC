import argparse
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import yaml


class DummyMemory:
    def register_action(self, *args, **kwargs):
        return None

    def register_experiment(self, *args, **kwargs):
        return None


def bootstrap_real_aiphy(root: Path):
    deprecated_stub = types.ModuleType("deprecated")
    deprecated_stub.deprecated = lambda *args, **kwargs: (lambda fn: fn)
    sys.modules["deprecated"] = deprecated_stub
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import aiphy  # noqa: F401


def bootstrap_aiphy(root: Path):
    base = root / "aiphy"
    pkg = types.ModuleType("aiphy")
    pkg.__path__ = [str(base)]
    sys.modules["aiphy"] = pkg

    core_spec = importlib.util.spec_from_file_location("aiphy.core", base / "core.so")
    core_mod = importlib.util.module_from_spec(core_spec)
    sys.modules["aiphy.core"] = core_mod
    core_spec.loader.exec_module(core_mod)
    for attr in dir(core_mod):
        if not attr.startswith("_"):
            setattr(pkg, attr, getattr(core_mod, attr))

    memory_stub = types.ModuleType("aiphy.memory")
    memory_stub.dict_to_json = lambda d: {k: str(v) for k, v in d.items()}
    memory_stub.Memory = DummyMemory
    sys.modules["aiphy.memory"] = memory_stub

    deprecated_stub = types.ModuleType("deprecated")
    deprecated_stub.deprecated = lambda *args, **kwargs: (lambda fn: fn)
    sys.modules["deprecated"] = deprecated_stub

    interface_spec = importlib.util.spec_from_file_location("aiphy.interface", base / "interface.py")
    interface_mod = importlib.util.module_from_spec(interface_spec)
    sys.modules["aiphy.interface"] = interface_mod
    interface_spec.loader.exec_module(interface_mod)
    for attr in dir(interface_mod):
        if not attr.startswith("_"):
            setattr(pkg, attr, getattr(interface_mod, attr))

    for name in [
        "parsing",
        "symbolic",
        "pca",
        "regression",
        "diffalgservice",
        "object_model",
        "manager",
        "specific_model",
        "general_model",
    ]:
        module_path = base / (name.replace(".", "/") + ".py")
        spec = importlib.util.spec_from_file_location(f"aiphy.{name}", module_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"aiphy.{name}"] = mod
        spec.loader.exec_module(mod)

    experiment_pkg = types.ModuleType("aiphy.experiment")
    experiment_pkg.__path__ = [str(base / "experiment")]
    sys.modules["aiphy.experiment"] = experiment_pkg

    core_spec = importlib.util.spec_from_file_location(
        "aiphy.experiment.core",
        base / "experiment" / "core.py",
    )
    core_mod = importlib.util.module_from_spec(core_spec)
    sys.modules["aiphy.experiment.core"] = core_mod
    core_spec.loader.exec_module(core_mod)

    init_spec = importlib.util.spec_from_file_location(
        "aiphy.experiment",
        base / "experiment" / "__init__.py",
        submodule_search_locations=[str(base / "experiment")],
    )
    init_mod = importlib.util.module_from_spec(init_spec)
    sys.modules["aiphy.experiment"] = init_mod
    init_spec.loader.exec_module(init_mod)


def load_baseline_real(checkpoint_prefix: Path):
    from aiphy import Theorist

    theorist = Theorist.read_from_file(str(checkpoint_prefix), reset_experiment=True, num_threads=4)
    return theorist


def load_baseline(root: Path, checkpoint_prefix: Path):
    from aiphy.interface import Knowledge
    from aiphy.object_model import ObjectModel
    from aiphy.specific_model import SpecificModel
    from aiphy.general_model import GeneralLawBase

    knowledge = Knowledge.read_from_file(str(checkpoint_prefix) + "_knowledge.txt")

    with open(str(checkpoint_prefix) + "_specific_model.json", "r") as f:
        specific_json = json.load(f)
    specific = {}
    for exp_name in knowledge.fetch_exps:
        spm = SpecificModel(exp_name, knowledge, reset=False)
        spm.load_json(specific_json[exp_name])
        specific[exp_name] = spm

    with open(str(checkpoint_prefix) + "_object_model.json", "r") as f:
        object_json = json.load(f)
    objmodel = {
        obj_type: ObjectModel.from_json(data, knowledge)
        for obj_type, data in object_json.items()
    }

    with open(str(checkpoint_prefix) + "_general_model.json", "r") as f:
        general_json = json.load(f)
    general = GeneralLawBase.from_json(general_json, knowledge, specific, DummyMemory(), objmodel)
    return knowledge, specific, objmodel, general


def register_experiment_real(theorist, config_path: str):
    module_name, fn_name = config_path.split(":")
    config_fn = getattr(importlib.import_module(module_name), fn_name)
    exp_structure = config_fn()
    theorist.register_experiment(exp_structure)
    return exp_structure.exp_name


def register_experiment(knowledge, specific, general, config_path: str):
    module_name, fn_name = config_path.split(":")
    config_fn = getattr(importlib.import_module(module_name), fn_name)
    exp_structure = config_fn()
    exp_name = exp_structure.exp_name
    knowledge.register_expstruct(exp_name, exp_structure)
    spm = sys.modules["aiphy.specific_model"].SpecificModel(exp_name, knowledge, reset=True)
    specific[exp_name] = spm
    general.specific = specific
    for concept in spm.experiment.original_concept:
        knowledge.register_basic_concept(concept)
    return exp_name


def validate_experiment_real(theorist, exp_name: str):
    checks, need_completion = theorist.general.check_general_laws(exp_name)
    valid_laws = sorted([name for name, ok in checks.items() if ok])
    invalid_laws = sorted([name for name, ok in checks.items() if not ok])
    return {
        "experiment": exp_name,
        "valid_count": len(valid_laws),
        "invalid_count": len(invalid_laws),
        "need_completion_count": len(need_completion),
        "valid_laws": valid_laws,
        "invalid_laws": invalid_laws,
        "need_completion": sorted(list(need_completion.keys())),
    }


def validate_experiment(general, exp_name: str):
    checks, need_completion = general.check_general_laws(exp_name)
    valid_laws = sorted([name for name, ok in checks.items() if ok])
    invalid_laws = sorted([name for name, ok in checks.items() if not ok])
    return {
        "experiment": exp_name,
        "valid_count": len(valid_laws),
        "invalid_count": len(invalid_laws),
        "need_completion_count": len(need_completion),
        "valid_laws": valid_laws,
        "invalid_laws": invalid_laws,
        "need_completion": sorted(list(need_completion.keys())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-config",
        default="/data/users/yjh/ai_dirac/configs/ai_newton_transfer_suites.yaml",
    )
    parser.add_argument(
        "--domain",
        default="mechanics",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["auto", "real", "lite"],
        default="auto",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    with open(args.suite_config, "r") as f:
        suite = yaml.safe_load(f)

    checkpoint_prefix = Path(suite["baseline_checkpoint"]["prefix"])
    mode = args.mode
    if mode == "auto":
        try:
            bootstrap_real_aiphy(root)
            theorist = load_baseline_real(checkpoint_prefix)
            mode = "real"
        except Exception:
            mode = "lite"

    if mode == "real":
        if "aiphy" not in sys.modules:
            bootstrap_real_aiphy(root)
        theorist = load_baseline_real(checkpoint_prefix)
    else:
        bootstrap_aiphy(root)
        knowledge, specific, objmodel, general = load_baseline(root, checkpoint_prefix)

    results = []
    for family in suite["domains"][args.domain]["families"]:
        for experiment in family["experiments"]:
            if not args.all and experiment.get("status") != "implemented":
                continue
            config_path = experiment.get("config")
            if not config_path:
                continue
            if mode == "real":
                exp_name = register_experiment_real(theorist, config_path)
                result = validate_experiment_real(theorist, exp_name)
            else:
                exp_name = register_experiment(knowledge, specific, general, config_path)
                result = validate_experiment(general, exp_name)
            result["family_id"] = family["family_id"]
            result["suite_id"] = experiment["id"]
            result["suite_name"] = experiment["name"]
            result["expected_transfer"] = experiment.get("expected_transfer", {})
            result["validation_mode"] = mode
            results.append(result)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
