"""
AI-Newton: 连续迁移发现脚本
从力学基线（46个实验、572轮）出发，逐步加入新域实验并继续发现

域顺序（每个域400轮）：
  1. 力学进阶 (10个实验)
  2. 电磁学   (20个实验)
  3. 光学     (30个实验)
  4. 热力学   (20个实验)
"""
import importlib
import json
import os
import sys
import time
from pathlib import Path

# ============ 配置参数 ============
BASELINE_PREFIX = "data/test_cases/example_1/example_1-5m3a7p5/example_1-5m3a7p5-1"
EPOCHS_PER_DOMAIN = 400
NUM_THREADS = 64
MUL = 3
MAX_ACTIONS = 7
BIAS = 5

OUTPUT_BASE = "data/test_cases/continuous_discovery"

# 各域实验配置: (域ID, 显示名, 保存名, 模块名, 配置函数名列表)
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
    """动态加载实验配置函数"""
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))

    from aiphy import Theorist

    # ============ 加载基线 ============
    print("=" * 70)
    print("AI-Newton 连续迁移发现")
    print("=" * 70)
    print(f"\n[阶段0] 加载基线模型...")
    print(f"  基线路径: {BASELINE_PREFIX}")

    baseline_abs = str(root / BASELINE_PREFIX)
    theorist = Theorist.read_from_file(baseline_abs, reset_experiment=True,
                                        num_threads=NUM_THREADS)
    baseline_epoch = theorist.memory.epoch

    print(f"  基线轮次: {baseline_epoch}")
    print(f"  基线实验数: {len(theorist.knowledge.fetch_exps)}")
    print(f"  基线通用定律数: {len(theorist.general.general_laws)}")

    # 创建输出目录
    output_dir = root / OUTPUT_BASE
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============ 逐域运行 ============
    all_results = {}
    current_epoch = baseline_epoch

    for domain_info in DOMAINS:
        domain_id = domain_info["id"]
        domain_name = domain_info["name"]
        save_name = domain_info["save_name"]
        module_name = domain_info["module"]
        config_names = domain_info["configs"]
        save_path = str(output_dir / save_name)

        print(f"\n{'=' * 70}")
        print(f"[阶段] {domain_name} ({len(config_names)}个实验)")
        print(f"{'=' * 70}")

        # 注册新实验
        print(f"\n  [1/3] 注册 {domain_name} 实验...")
        registered_count = 0
        for config_name in config_names:
            try:
                config_fn = load_config_fn(module_name, config_name)
                exp_structure = config_fn()
                exp_name = exp_structure.exp_name
                if exp_name in theorist.specific:
                    print(f"    - 跳过已存在: {exp_name}")
                    continue
                theorist.register_experiment(exp_structure)
                registered_count += 1
                print(f"    - 注册成功: {exp_name}")
            except Exception as e:
                print(f"    - 注册失败: {config_name} - {e}")

        print(f"\n  共注册 {registered_count} 个新实验")
        print(f"  当前总实验数: {len(theorist.knowledge.fetch_exps)}")

        # 保存注册快照
        theorist.save_to_file(save_path + "_before_discovery")
        print(f"  状态快照已保存: {save_path}_before_discovery")

        # 运行发现过程
        target_epochs = current_epoch + EPOCHS_PER_DOMAIN
        print(f"\n  [2/3] 开始发现过程 (轮次 {current_epoch + 1} → {target_epochs})...")
        print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        theorist.perform_discovery(
            exp_select_mode='contextual',
            cpt_lim_mul=MUL,
            max_actions=MAX_ACTIONS,
            max_cpt_lim=3600.,
            cpt_lim_bias=BIAS,
            max_epochs=target_epochs,
            save_path=save_path,
        )

        current_epoch = target_epochs

        # 保存最终结果
        print(f"\n  [3/3] 保存 {domain_name} 发现结果...")
        theorist.save_to_file(save_path)
        print(f"  结果已保存至: {save_path}")

        # 打印统计
        print(f"\n  --- {domain_name} 发现统计 ---")
        print(f"  总通用定律数: {len(theorist.general.general_laws)}")

        # 记录结果
        domain_result = {
            "domain": domain_name,
            "experiments_registered": registered_count,
            "total_experiments": len(theorist.knowledge.fetch_exps),
            "total_general_laws": len(theorist.general.general_laws),
            "epochs": current_epoch,
        }
        all_results[domain_id] = domain_result

        # 保存进度
        with open(str(output_dir / "progress.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        # 打印迁移验证结果
        print(f"\n  --- 迁移验证: 已有定律在新实验上的有效性 ---")
        for law_name, law in theorist.general.general_laws.items():
            valid_count = len(law.valid_experiments)
            total = len(theorist.knowledge.fetch_exps)
            coverage = valid_count / total * 100
            if coverage < 100:
                marker = " ⚠️"
            else:
                marker = " ✅"
            print(f"    {law_name}: {law.prop}{marker}")
            print(f"      适用: {valid_count}/{total} ({coverage:.0f}%)")

    # ============ 最终总结 ============
    print(f"\n{'=' * 70}")
    print("连续迁移发现完成！")
    print(f"{'=' * 70}")
    print(f"\n完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 打印最终通用定律
    print(f"\n最终通用物理定律 ({len(theorist.general.general_laws)} 条):")
    print("=" * 70)
    for name, law in theorist.general.general_laws.items():
        valid_count = len(law.valid_experiments)
        total = len(theorist.knowledge.fetch_exps)
        coverage = valid_count / total * 100
        print(f"  {name}: {law.prop}")
        print(f"    类型: {law.type}, 适用范围: {valid_count}/{total} ({coverage:.0f}%)")

    # 保存最终摘要
    summary = {
        "baseline": str(BASELINE_PREFIX),
        "baseline_epochs": baseline_epoch,
        "epochs_per_domain": EPOCHS_PER_DOMAIN,
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
