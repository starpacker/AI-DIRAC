"""
AI-Newton: 从零发现物理定律的完整运行脚本
基于 test_example_1 的配置，但使用更强大的硬件设置
"""
import os
import sys
import time
from aiphy import Theorist
from aiphy.experiment.basic import (
    motion0_3d_config, motion_3d_config,
    collision_elastic_3body_config, collision_y_config,
    collision_2d_config, collision_nh_config,
)
from aiphy.experiment.ground_laboratory import (
    gravity_config, free_fall_config,
    projectile_config, oblique_projectile_x_config, oblique_projectile_y_config,
    oblique_projectile_config, vertical_projectile_config,
    hanging_spring_1_config, hanging_spring_2_config, hanging_spring_3_config,
    pendulum_oscillation_1_config, pendulum_oscillation_2_config,
    slope_oscillation_1_config, slope_oscillation_2_config,
    projectile_oscillation_1_config, projectile_oscillation_2_config,
)
from aiphy.experiment.oscillation import (
    oscillation_x_1_config, oscillation_x_2_config, oscillation_x_3_config,
    oscillation_2d_1_config, oscillation_2d_2_config, oscillation_2d_3_config,
    oscillation_r_3_config, oscillation_r_4_config,
    oscillation_rot_1_config, oscillation_rot_2_config, oscillation_rot_3_config,
)
from aiphy.experiment.cosmos import (
    celestial_2_xy_config, celestial_2_xz_config, celestial_2_yz_config,
    celestial_3_config, celestial_4_config,
)

# ============ 配置参数 ============
EXAMPLE_ID = 1
NUMBER_OF_ITERATIONS = 1200  # 完整发现需要1200个epoch
NUM_THREADS = 64             # 256核CPU，用64线程平衡负载
INIT_TIME_LIMIT = 5          # 初始时间限制（秒）
MUL = 3                      # 计算限制增长率
MAX_ACTIONS = 7              # 每次最大动作数
BIAS = 5                     # 时间偏置
ID = 1                       # 运行ID

EXAMPLE_PATH = "data/test_cases/example_1"
FILE_NAME = f"example_{EXAMPLE_ID}-{INIT_TIME_LIMIT}m{MUL}a{MAX_ACTIONS}p{BIAS}".replace(".", "-")
FILE_PATH_PARENT = os.path.join(EXAMPLE_PATH, FILE_NAME)
if ID is None:
    FILE_PATH = os.path.join(FILE_PATH_PARENT, FILE_NAME)
else:
    FILE_PATH = os.path.join(FILE_PATH_PARENT, f"{FILE_NAME}-{ID}")
os.makedirs(FILE_PATH_PARENT, exist_ok=True)

# ============ 初始化理论家 ============
print("=" * 60)
print("AI-Newton 物理定律发现系统")
print(f"配置: 线程={NUM_THREADS}, 初始时间={INIT_TIME_LIMIT}s, 最大轮次={NUMBER_OF_ITERATIONS}")
print(f"保存路径: {FILE_PATH}")
print("=" * 60)

theorist = Theorist(init_time_limit=INIT_TIME_LIMIT, num_threads=NUM_THREADS)

# ============ 注册所有实验 ============
print("\n[1/4] 注册实验...")

# 基础物理实验 (运动学)
print("  - 注册基础运动学实验...")
theorist.register_experiment(motion0_3d_config())
theorist.register_experiment(motion_3d_config())
theorist.register_experiment(collision_elastic_3body_config())
theorist.register_experiment(collision_y_config())
theorist.register_experiment(collision_2d_config())
theorist.register_experiment(collision_nh_config())

# 地面实验室实验
print("  - 注册地面实验室实验...")
theorist.register_experiment(gravity_config())
theorist.register_experiment(free_fall_config())
theorist.register_experiment(projectile_config())
theorist.register_experiment(oblique_projectile_config())
theorist.register_experiment(oblique_projectile_x_config())
theorist.register_experiment(oblique_projectile_y_config())
theorist.register_experiment(vertical_projectile_config())
theorist.register_experiment(hanging_spring_1_config())
theorist.register_experiment(hanging_spring_2_config())
theorist.register_experiment(hanging_spring_3_config())
theorist.register_experiment(pendulum_oscillation_1_config())
theorist.register_experiment(pendulum_oscillation_2_config())
theorist.register_experiment(slope_oscillation_1_config())
theorist.register_experiment(slope_oscillation_2_config())
theorist.register_experiment(projectile_oscillation_1_config())
theorist.register_experiment(projectile_oscillation_2_config())

# 振荡实验
print("  - 注册振荡实验...")
theorist.register_experiment(oscillation_x_1_config())
theorist.register_experiment(oscillation_x_2_config())
theorist.register_experiment(oscillation_x_3_config())
theorist.register_experiment(oscillation_2d_1_config())
theorist.register_experiment(oscillation_2d_2_config())
theorist.register_experiment(oscillation_2d_3_config())
theorist.register_experiment(oscillation_r_3_config())
theorist.register_experiment(oscillation_r_4_config())
theorist.register_experiment(oscillation_rot_1_config())
theorist.register_experiment(oscillation_rot_2_config())
theorist.register_experiment(oscillation_rot_3_config())

# 天体物理实验
print("  - 注册天体物理实验...")
theorist.register_experiment(celestial_2_xy_config())
theorist.register_experiment(celestial_2_xz_config())
theorist.register_experiment(celestial_2_yz_config())
theorist.register_experiment(celestial_3_config())
theorist.register_experiment(celestial_4_config())

print(f"  共 {len(theorist.knowledge.fetch_exps)} 个实验注册完成")

# ============ 初始保存 ============
print("\n[2/4] 保存初始状态...")
theorist.save_to_file(FILE_PATH)
print("  初始状态已保存")

# ============ 开始发现过程 ============
print("\n[3/4] 开始物理定律发现过程...")
print(f"  最大轮次: {NUMBER_OF_ITERATIONS}")
print(f"  保存路径: {FILE_PATH}")
print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

theorist.perform_discovery(
    exp_select_mode='contextual',
    cpt_lim_mul=MUL,
    max_actions=MAX_ACTIONS,
    max_cpt_lim=3600.,  # 最大单次时间限制1小时
    cpt_lim_bias=BIAS,
    max_epochs=NUMBER_OF_ITERATIONS,
    save_path=FILE_PATH,
)

# ============ 最终保存 ============
print("\n[4/4] 保存最终结果...")
theorist.save_to_file(FILE_PATH)
print(f"  发现过程完成于: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 打印发现的通用定律
print("\n" + "=" * 60)
print("发现的通用物理定律:")
print("=" * 60)
for name, law in theorist.general.general_laws.items():
    print(f"  {name}: {law}")
    print(f"    类型: {law.type}, 适用范围: {law.valid_experiments}")
print(f"\n共发现 {len(theorist.general.general_laws)} 个通用定律")
print(f"共发现 {len(theorist.knowledge.fetch_concepts)} 个物理概念")
print(f"共发现 {len(theorist.knowledge.fetch_conclusions)} 个结论")
