from aiphy import Theorist

from run_ladder_strict import LEVEL_BY_NAME, _assert_strict_only_theorist, run_ladder


def test_l1_level_unlocks_mass_for_second_experiment(capsys):
    registered, certified = run_ladder(
        [LEVEL_BY_NAME["L1"]],
        theorist=None,
        smoke=True,
        trust_declared_certifications=True,
    )
    output = capsys.readouterr().out

    assert registered == 0
    assert "mass" in certified
    assert "1.1 strict_1_1_collision_pair: 8 open column(s)" in output
    assert "1.2 strict_1_2_collision_reference: 14 open column(s)" in output


def test_ladder_runner_starts_from_empty_theorist():
    theorist = Theorist(include_default_experiments=False)
    registered, certified = run_ladder(
        [LEVEL_BY_NAME["L1"], LEVEL_BY_NAME["L2"]],
        theorist=theorist,
        smoke=False,
        trust_declared_certifications=False,
    )

    _assert_strict_only_theorist(theorist)
    assert registered == 5
    assert certified == set()
    assert sorted(theorist.knowledge.fetch_exps) == [
        "strict_1_1_collision_pair",
        "strict_1_2_collision_reference",
        "strict_2_1_free_fall_time",
        "strict_2_2_incline_slide",
        "strict_2_3_atwood",
    ]
    assert not any(
        str(key).startswith("mass[")
        for key in theorist.specific["strict_1_2_collision_reference"].experiment.original_data
    )
    assert not any(
        str(key).startswith("mass[")
        for key in theorist.specific["strict_2_3_atwood"].experiment.original_data
    )
