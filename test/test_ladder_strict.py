from aiphy import MeasureType

from aiphy.experiment.column_policy import ColumnRole, apply_role_overrides
from aiphy.experiment.ladder_strict import (
    LADDER_STRICT_FACTORIES,
    MODE,
    _specs,
    make_ladder_strict_config,
)


def _atom_for_spec(exp, spec):
    ids = [exp.obj_info[obj_name][1] for obj_name in spec.key[1]]
    return spec.key[0].subst_by_vec(ids)


def test_ladder_strict_factories_collect_only_open_columns():
    for factory in LADDER_STRICT_FACTORIES:
        exp = factory()
        exp.random_settings()
        data = exp.collect_expdata(MeasureType.default())
        data_keys = data.datastruct.data_keys

        assert len(exp.original_data) == len(data_keys), exp.exp_name

        specs = _specs(MODE[exp.exp_name])
        hidden_atoms = {
            _atom_for_spec(exp, spec)
            for spec in specs
            if spec.role is ColumnRole.HIDDEN
        }
        assert hidden_atoms.isdisjoint(data_keys), exp.exp_name


def test_ladder_strict_unlocked_columns_follow_certified_registry():
    exp = make_ladder_strict_config("strict_1_2_collision_reference")
    exp.random_settings()
    data = exp.collect_expdata(MeasureType.default())
    locked_key_names = {str(key) for key in data.datastruct.data_keys}

    assert "mass[1]" not in locked_key_names

    exp = make_ladder_strict_config("strict_1_2_collision_reference", unlocked_concepts={"mass"})
    exp.random_settings()
    data = exp.collect_expdata(MeasureType.default())
    unlocked_key_names = {str(key) for key in data.datastruct.data_keys}

    assert "mass[1]" in unlocked_key_names
    assert "mass[2]" not in unlocked_key_names
    assert len(data.datastruct.data_keys) == len(exp.original_data) == 14


def test_ladder_strict_column_role_override_still_requires_certification():
    exp = make_ladder_strict_config(
        "strict_1_1_collision_pair",
        column_roles={"mass_a": ColumnRole.UNLOCKED, "mass_b": "UNLOCKED"},
    )
    assert "mass[1]" not in {str(key) for key in exp.original_data}

    exp = make_ladder_strict_config(
        "strict_1_1_collision_pair",
        unlocked_concepts={"mass"},
        column_roles={"mass_a": ColumnRole.UNLOCKED, "mass_b": "UNLOCKED"},
    )
    assert {"mass[1]", "mass[2]"}.issubset({str(key) for key in exp.original_data})


def test_column_role_override_rejects_unknown_column():
    specs = _specs(MODE["strict_1_1_collision_pair"])
    try:
        apply_role_overrides(specs, {"not_a_column": "UNLOCKED"})
    except KeyError:
        pass
    else:
        raise AssertionError("unknown column override should fail fast")
