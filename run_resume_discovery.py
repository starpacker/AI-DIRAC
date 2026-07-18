"""
AI-Newton: 恢复连续迁移发现脚本
从上次中断的检查点继续运行
"""
import importlib
import json
import os
import sys
import time
from pathlib import Path

# ============ 配置参数 ============
# 上次中断时的检查点路径
LAST_CHECKPOINT = "data/test_cases/continuous_discovery/advanced_mechanics"
TARGET_EPOCHS = 986  # 586 + 400 (力学进阶目标轮次)
NUM_THREADS = 64
MUL = 3
MAX_ACTIONS = 7
BIAS = 5

OUTPUT_BASE = "data/test_cases/continuous_discovery"

# 各域实验配置
DOMAINS = [
    {
        "id": "advanced_mechanics",
        "name": "进阶力学",
        "save_name": "advanced_mechanics",
        "module": "aiphy.experiment.challenge_mechanics",
        "configs": [
            "damped_hanging_spring_1_config",
            "damped_hanging_spring_2_config",
            "driven_damped_hanging_spring_1_config",
            "driven_damped_hanging_spring_2_config",
            "damped_pendulum_1_config",
            "damped_pendulum_2_config",
            "damped_elastic_pendulum_1_config",
            "variable_mass_rocket_1_config",
            "variable_mass_rocket_2_config",
            "two_stage_rocket_1_config",
        ],
    },
    {
        "id": "electromagnetism",
        "name": "电磁学",
        "save_name": "electromagnetism",
        "module": "aiphy.experiment.challenge_electromagnetism",
        "configs": [
            "charged_particle_electric_x_config",
            "charged_particle_electric_z_config",
            "charged_particle_electric_xy_config",
            "charged_particle_electric_xz_config",
            "charged_particle_electric_3d_config",
            "charged_particle_magnetic_xy_circle_config",
            "charged_particle_magnetic_xy_orbit_config",
            "charged_particle_magnetic_helix_z_config",
            "charged_particle_magnetic_yz_circle_config",
            "charged_particle_magnetic_tilted_helix_config",
            "charged_particle_crossed_EB_trochoid_config",
            "charged_particle_crossed_EB_drift_line_config",
            "charged_particle_crossed_EB_helix_config",
            "charged_particle_crossed_EB_xz_config",
            "charged_particle_crossed_EB_tilted_config",
            "charged_particle_damped_cyclotron_config",
            "charged_particle_driven_cyclotron_config",
            "charged_particle_electrostatic_oscillator_config",
            "charged_particle_paul_trap_1d_config",
            "charged_particle_penning_like_trap_config",
        ],
    },
    {
        "id": "optics",
        "name": "光学",
        "save_name": "optics",
        "module": "aiphy.experiment.challenge_optics",
        "configs": [
            "homogeneous_path_scan_config",
            "plane_mirror_equal_angle_config",
            "snell_air_to_glass_config",
            "snell_glass_to_air_config",
            "total_internal_reflection_phase_config",
            "convex_lens_real_image_config",
            "convex_lens_near_focus_config",
            "concave_lens_virtual_image_config",
            "concave_mirror_real_image_config",
            "convex_mirror_virtual_image_config",
            "prism_deviation_scan_config",
            "grin_ray_oscillation_config",
            "spherical_aberration_focus_shift_config",
            "two_lens_telescope_scan_config",
            "thick_lens_lensmaker_scan_config",
            "young_double_slit_basic_config",
            "young_double_slit_envelope_config",
            "michelson_fringes_config",
            "fabry_perot_airy_config",
            "unequal_amplitude_interference_config",
            "single_slit_fraunhofer_config",
            "diffraction_grating_multiorder_config",
            "airy_disk_cross_section_config",
            "fresnel_edge_diffraction_config",
            "partial_coherence_diffraction_config",
            "photoelectric_threshold_config",
            "photoelectric_intensity_scan_config",
            "photon_antibunching_g2_config",
            "single_photon_double_slit_counts_config",
            "complementarity_tradeoff_config",
        ],
    },
    {
        "id": "thermodynamics",
        "name": "热力学",
        "save_name": "thermodynamics",
        "module": "aiphy.experiment.challenge_thermodynamics",
        "configs": [
            "lumped_cooling_static_env_config",
            "lumped_heating_static_env_config",
            "lumped_cooling_dynamic_env_config",
            "lumped_joule_heating_config",
            "lumped_radiative_cooling_config",
            "two_body_contact_equal_config",
            "two_body_contact_unequal_config",
            "two_body_contact_with_ambient_config",
            "three_body_chain_relaxation_config",
            "three_body_ring_relaxation_config",
            "driven_ambient_oscillation_config",
            "periodic_power_heating_config",
            "thermostat_limit_cycle_config",
            "exothermic_self_heating_config",
            "quench_recovery_config",
            "latent_heat_heating_plateau_config",
            "latent_heat_cooling_plateau_config",
            "latent_heat_contact_two_body_config",
            "rod_three_node_diffusion_config",
            "rod_four_node_diffusion_config",
        ],
    },
]


def load_config_fn(module_name: str, fn_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))

    from aiphy import Theorist

    # ============ 加载检查点 ============
    print("=" * 70)
    print("AI-Newton 恢复连续迁移发现")
    print("=" * 70)
    print(f"\n[阶段0] 加载检查点...")
    print(f"  检查点路径: {LAST_CHECKPOINT}")

    checkpoint_abs = str(root / LAST_CHECKPOINT)
    theorist = Theorist.read_from_file(checkpoint_abs, reset_experiment=True,
                                        num_threads=NUM_THREADS)
    current_epoch = theorist.memory.epoch

    print(f"  当前轮次: {current_epoch}")
    print(f"  当前实验数: {len(theorist.knowledge.fetch_exps)}")
    print(f"  当前通用定律数: {len(theorist.general.general_laws)}")

    output_dir = root / OUTPUT_BASE
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 判断当前阶段
    # 力学进阶阶段: epoch 586→986 (目标986)
    # 电磁学阶段: epoch 986→1386
    # 光学阶段: epoch 1386→1786
    # 热力学阶段: epoch 1786→2186
    phase_boundaries = [586, 986, 1386, 1786, 2186]
    phase_names = ["基线", "进阶力学", "电磁学", "光学", "热力学"]

    # 找到当前所处的阶段
    current_phase = 0
    for i in range(len(phase_boundaries) - 1):
        if phase_boundaries[i] <= current_epoch < phase_boundaries[i + 1]:
            current_phase = i + 1
            break
    if current_epoch >= phase_boundaries[-1]:
        current_phase = len(phase_boundaries) - 1

    print(f"  当前阶段: {phase_names[current_phase]}")
    print(f"  目标轮次: {phase_boundaries[current_phase]}")

    # 检查当前阶段是否需要注册新实验
    remaining_epochs = TARGET_EPOCHS - current_epoch
    if remaining_epochs <= 0:
        # 力学进阶已完成，进入下一域
        current_phase += 1

    # 注册当前阶段的新实验（如果尚未注册）
    for phase_idx in range(current_phase, len(phase_names)):
        domain_idx = phase_idx - 1  # 域索引从0开始
        if domain_idx < 0 or domain_idx >= len(DOMAINS):
            continue

        domain_info = DOMAINS[domain_idx]
        phase_start = phase_boundaries[phase_idx]
        phase_end = phase_boundaries[phase_idx + 1]
        save_path = str(output_dir / domain_info["save_name"])

        # 检查该域的实验是否已经注册
        module_name = domain_info["module"]
        config_names = domain_info["configs"]
        test_config = load_config_fn(module_name, config_names[0])
        test_exp_name = test_config().exp_name

        if test_exp_name in theorist.specific:
            print(f"\n  [跳过] {domain_info['name']} 实验已注册，跳过注册步骤")
        else:
            print(f"\n{'=' * 70}")
            print(f"[阶段] {domain_info['name']} ({len(config_names)}个实验)")
            print(f"{'=' * 70}")
            print(f"\n  [1/3] 注册 {domain_info['name']} 实验...")
            registered_count = 0
            for config_name in config_names:
                try:
                    config_fn = load_config_fn(module_name, config_name)
                    exp_structure = config_fn()
                    exp_name = exp_structure.exp_name
                    if exp_name in theorist.specific:
                        continue
                    theorist.register_experiment(exp_structure)
                    registered_count += 1
                    print(f"    - 注册成功: {exp_name}")
                except Exception as e:
                    print(f"    - 注册失败: {config_name} - {e}")

            print(f"\n  共注册 {registered_count} 个新实验")
            print(f"  当前总实验数: {len(theorist.knowledge.fetch_exps)}")

            theorist.save_to_file(save_path + "_before_discovery")

        # 运行发现过程
        if current_epoch < phase_end:
            print(f"\n  [2/3] 开始发现过程 (轮次 {current_epoch + 1} → {phase_end})...")
            print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            theorist.perform_discovery(
                exp_select_mode='contextual',
                cpt_lim_mul=MUL,
                max_actions=MAX_ACTIONS,
                max_cpt_lim=3600.,
                cpt_lim_bias=BIAS,
                max_epochs=phase_end,
                save_path=save_path,
            )

            current_epoch = phase_end

            print(f"\n  [3/3] 保存 {domain_info['name']} 发现结果...")
            theorist.save_to_file(save_path)
            print(f"  结果已保存至: {save_path}")

            # 打印统计
            print(f"\n  --- {domain_info['name']} 发现统计 ---")
            print(f"  总通用定律数: {len(theorist.general.general_laws)}")

            domain_result = {
                "domain": domain_info["name"],
                "experiments_registered": len(config_names),
                "total_experiments": len(theorist.knowledge.fetch_exps),
                "total_general_laws": len(theorist.general.general_laws),
                "epochs": current_epoch,
            }
            all_results[domain_info["id"]] = domain_result

            # 打印迁移验证结果
            print(f"\n  --- 迁移验证 ---")
            for law_name, law in theorist.general.general_laws.items():
                valid_count = len(law.valid_experiments)
                total = len(theorist.knowledge.fetch_exps)
                coverage = valid_count / total * 100
                marker = " ⚠️" if coverage < 100 else " ✅"
                print(f"    {law_name}: {law.prop}{marker}")
                print(f"      适用: {valid_count}/{total} ({coverage:.0f}%)")

            with open(str(output_dir / "progress.json"), "w") as f:
                json.dump(all_results, f, indent=2)
        else:
            print(f"\n  [跳过] {domain_info['name']} 阶段已完成")

    # ============ 最终总结 ============
    print(f"\n{'=' * 70}")
    print("连续迁移发现完成！")
    print(f"{'=' * 70}")
    print(f"\n完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n最终通用物理定律 ({len(theorist.general.general_laws)} 条):")
    print("=" * 70)
    for name, law in theorist.general.general_laws.items():
        valid_count = len(law.valid_experiments)
        total = len(theorist.knowledge.fetch_exps)
        coverage = valid_count / total * 100
        print(f"  {name}: {law.prop}")
        print(f"    类型: {law.type}, 适用范围: {valid_count}/{total} ({coverage:.0f}%)")

    summary = {
        "baseline": str(LAST_CHECKPOINT),
        "total_epochs": current_epoch,
        "total_experiments": len(theorist.knowledge.fetch_exps),
        "total_general_laws": len(theorist.general.general_laws),
        "domains": all_results,
        "completed_at": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(str(output_dir / "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n摘要已保存至: {output_dir / 'summary.json'}")
    print("完成!")


if __name__ == "__main__":
    main()
