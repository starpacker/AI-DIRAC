from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_optics.common import (
    finalize_structure,
    fixed_parastructure,
    insert_optics_data,
    linear_scan,
    make_single_particle_expconfig,
    ranged_parastructure,
    time_grid,
)


BASE_EXP_PARA = {
    "mode": fixed_parastructure(1.0),
    "f": ranged_parastructure(1.2, 3.0),
    "smin": ranged_parastructure(2.5, 3.5),
    "smax": ranged_parastructure(5.5, 8.0),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def thin_element_image(s, f):
    return f * s / (s - f)


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    mode = int(round(exp_config.para("mode")))
    t = time_grid(t_end, t_num)

    s = linear_scan(exp_config.para("smin"), exp_config.para("smax"), t_num)
    f = exp_config.para("f")

    if mode == 1:
        img = thin_element_image(s, f)
        mag = -img / s
    elif mode == 2:
        near_s = linear_scan(1.25 * f, 2.0 * f, t_num)
        s = near_s
        img = thin_element_image(s, f)
        mag = -img / s
    elif mode == 3:
        f = -abs(f)
        img = thin_element_image(s, f)
        mag = -img / s
    elif mode == 4:
        img = thin_element_image(s, f)
        mag = -img / s
    elif mode == 5:
        f = -abs(f)
        img = thin_element_image(s, f)
        mag = -img / s
    else:
        raise ValueError(f"Unsupported imaging mode: {mode}")

    posx, posy, posz = s, img, mag
    return insert_optics_data(exp_config, t, error, posx, posy, posz)


def convex_lens_real_image_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "convex_lens_real_image",
        3,
        make_exp_para(mode=fixed_parastructure(1.0)),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1] * posx[o1] + posy[o1]")])


def convex_lens_near_focus_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "convex_lens_near_focus",
        3,
        make_exp_para(mode=fixed_parastructure(2.0)),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1] * posx[o1] + posy[o1]")])


def concave_lens_virtual_image_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "concave_lens_virtual_image",
        3,
        make_exp_para(mode=fixed_parastructure(3.0)),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1] * posx[o1] + posy[o1]")])


def concave_mirror_real_image_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "concave_mirror_real_image",
        3,
        make_exp_para(mode=fixed_parastructure(4.0)),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1] * posx[o1] + posy[o1]")])


def convex_mirror_virtual_image_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "convex_mirror_virtual_image",
        3,
        make_exp_para(mode=fixed_parastructure(5.0)),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1] * posx[o1] + posy[o1]")])
