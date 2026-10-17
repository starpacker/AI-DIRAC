# 实验阶梯：按“概念解锁顺序”组织（规范化版 v6）

> 起点：仅七大 SI 基本单位 + 原始传感通道。
> 终点：`final_experiments_judge.md` 的目标公式。

## 0. 三条铁律

1. **“解锁”的严格定义**：一个量只有**在前置实验里被发现并认证（CERTIFIED）**，
   才叫“解锁后开放”。**直接可测量的几何/运动学量（位置、时间、距离、倾角、摆长、
   弹簧长度、形状尺寸）一律算“默认开放”，不算解锁。**
2. **禁止 cheat**：不能把变量预先组合成以目标命名的列。
3. **definition ≠ validation**；失败永不删除。
4. **开放/隐藏由阶梯顺序决定**：同一个代码库通道可以在后续实验开放，但在发现它自己的前置
   实验中必须隐藏。运行程序必须按层注册/解锁实验，不能把所有实验一次性丢进同一个池。

## 0.1 数据列标注规范

```
【数据列】
  · 默认开放：<实验里直接可测的原始通道>
  · 解锁后开放：<只在前置实验被发现并认证过的 concepts>
  · 隐藏（禁止喂入）：<simulator hidden>

【预期发现】
  <框架应从上述数据列中得出什么物理结论>

【注册 concepts】
  <本实验认证后新增的、供后续使用的 concepts；若无可写「无」>

【实现】
  路径：<现有 .py 路径 或 “需额外实现”>
```

## 0.2 直接可测原始通道（L0 默认开放全集）

`concept_posx/posy/posz`（位置）、`concept_t`（时间）、`concept_dist`（两点距离）、
`concept_distfix`（点到固定点距离）、`concept_length`（弹簧长度）、
`concept_cx/cy/cz`（斜面法向几何，可导出倾角 θ）。

`concept_mass` 不是 L0 默认开放量。它在代码库中可以作为传感/对象通道存在，但在 L1 质量发现
实验中必须隐藏；只有 L1 认证 `m` 后，后续实验才可把质量列作为“解锁后开放”或等价已知属性使用。

控制设定量（如截面积 `A`、驱动频率 `ω`、驱动力幅值 `F0`、热输入功率、光源频率）若实验者可直接
设定或读数，可在对应实验中列为默认开放；但不能预先组合成目标公式左端或以目标 concept 命名。

## 0.3 列角色切换规范（open/hidden switch）

每个实验的数据列必须先声明为 `ColumnSpec(name, key, role)`，其中 `role ∈ {DEFAULT, CONTROL,
UNLOCKED, HIDDEN}`：

- `DEFAULT`：位置、时间、几何、传感器原始读数等 L0 量。
- `CONTROL`：实验者直接设定或读数的控制量，如 `u`, `F0`, `omega`, `power`, `frequency`。
- `UNLOCKED`：候选解锁列；只有前置实验已认证对应 concept 后才真正开放，如 L1 后的
  `mass`、L3 后的 `kspring`。
- `HIDDEN`：simulator 内部真值或当前实验待发现量，不得进入 `data_info` 或返回数据。

同一个底层通道允许在不同阶段切换角色：例如 `mass` 在 1.1 中是 `HIDDEN`，在 L1 认证后可在
2.3 及后续实验中作为 `UNLOCKED`；`k/kspring` 在 3.2 中隐藏，在 6.1 中开放；
`lambda/wavelength` 在 O.2 中隐藏，在 O.3/O.4/O.6 中开放。实现上由
`aiphy/experiment/column_policy.py` 的 `resolve_column_specs()` 查询 `unlocked_concepts`
动态解析；消融/调试仍可用 `column_roles` 把某个 hidden 通道声明为候选，但仍必须有认证
registry 才能真正开放：

```python
make_ladder_strict_config(
    "strict_1_1_collision_pair",
    unlocked_concepts={"mass"},
    column_roles={"mass_a": "UNLOCKED", "mass_b": "UNLOCKED"},
)
```

硬约束：构造时的开放列集合必须写入 `ExpConfig`，运行时 `do_experiment` 只能插入这些开放列；
hidden 列即使在 simulator 内部生成，也不能出现在 `data_info` 或 `DataStruct`。

---

## 已解锁量寄存器

| 符号 | 含义 | 来源 | 层 |
|---|---|---|---|
| `posx,posy,posz` | 位置 | 默认开放 | L0 |
| `t` | 时间 | 默认开放 | L0 |
| `dist` | 两点距离 | 默认开放 | L0 |
| `length` | 弹簧长度 | 默认开放 | L0 |
| `cx,cy,cz`→`θ` | 倾角 | 默认开放 | L0 |

（随各层增长）

---

# L1 — 质量（匿名对象权重）【基石】

### 实验 1.1 `collision_pair` — 两体碰撞

```
【数据列】
  · 默认开放：posx/posy/posz[a](t), posx/posy/posz[b](t), dist[a,b](t), t
  · 解锁后开放：无
  · 隐藏：m_a, m_b（对象质量真值）

【预期发现】
  ① 由 pos 数值微分得 v[a], v[b]（**v 在此涌现**）；
  ② 对每场碰撞取 Δv_i，求 Σ w_i·Δv_i ≈ 0 的零空间 → 匿名权重 w_i；
  ③ 参考对象固定 w_ref=1（单位约定）。

【注册 concepts】
  · m（质量，暂名 weight_0）—— 对象内在权重，供后续所有动力学实验使用
  · v（速度）—— 本实验顺带解锁

【实现】
  路径：aiphy/experiment/basic/collision_circle.py 或 collision_y.py（现有，只给 pos/dist/t）
```

### 实验 1.2 `collision_reference` — 参考对象传递性

```
【数据列】
  · 默认开放：同上，一侧为固定参考对象
  · 解锁后开放：m（实验 1.1）
  · 隐藏：其余对象质量真值

【预期发现】
  参考对象传递性：w_{A/B}·w_{B/C} ≈ w_{A/C}。

【注册 concepts】
  无（本实验用于验证 m）

【实现】
  路径：aiphy/experiment/basic/collision_elastic_3body.py（现有，三体碰撞）
```

**本层解锁**：`m`, `v`
**可解性**：✓ 只需 pos+t。对应 `docs/operational_concept_discovery.md` 第 8 节里程碑。
从 L2 起，`m` 可作为已解锁 concept 对后续实验开放；这不是全局先验，而是 L1 的实验产物。

---

# L2 — 运动学（加速度）

### 实验 2.1 `free_fall_time` — 自由落体

```
【数据列】
  · 默认开放：posx/posy/posz[o1](t), t
  · 解锁后开放：v（L1）, a（本实验由 pos 二阶微分得）
  · 隐藏：无（g 是待发现结论，不是喂入量）

【预期发现】
  posz(t) 二次增长 → a = dv/dt = const。

【注册 concepts】
  · a（加速度）

【实现】
  路径：aiphy/experiment/ground_laboratory/free_fall.py（现有）
        aiphy/experiment/challenge_mechanics/free_fall_1.py（现有）
```

### 实验 2.2 `incline_slide` — 斜面下滑

```
【数据列】
  · 默认开放：posx/posy/posz[o1](t), t, cx/cy/cz[slope]（斜面几何，⇒ 可导出倾角 θ）
  · 解锁后开放：v（L1）, a（实验 2.1）
  · 隐藏：g（待发现）

【预期发现】
  a = g·sinθ；改变 θ（直接可测的几何量）→ 得 sinθ 的比例关系。

【注册 concepts】
  无
  （注：θ 是默认开放的几何量，直接测得，**不注册**，也不属于“解锁后开放”。）

【实现】
  路径：aiphy/experiment/ground_laboratory/slope_oscillation_1/2.py（现有，已开放 cx/cy/cz）
        需额外实现：aiphy/experiment/ground_laboratory/incline_slide.py
        （纯斜面下滑，只开放 pos + slope 几何 + t）
```

### 实验 2.3 `atwood` — 滑轮

```
【数据列】
  · 默认开放：posz[o1](t), posz[o2](t), distfix, t
  · 解锁后开放：m（L1）, v, a（L2）
  · 隐藏：滑轮转动惯量（理想滑轮为 0）

【预期发现】
  a = (m1−m2)g/(m1+m2) —— 质量差首次进入运动学。

【注册 concepts】
  无

【实现】
  路径：aiphy/experiment/ground_laboratory/pulley_1/2/3.py（现有）
```

**本层解锁**：`a`
**可解性**：✓ 输入 pos+t 全默认开放；v,m 来自 L1。

---

# L3 — 力

### 实验 3.1 `mass_under_force` — 定量施力于已知质量

```
【数据列】
  · 默认开放：pos(t), t, actuator setting u（可控施力档位/推力控制量，不能命名为 F）
  · 解锁后开放：m（L1）, a（L2）
  · 隐藏：F（待发现）

【预期发现】
  改变 u 与 m，测 a → 存在可复用作用量 F(u)，且 F = m·a
  （比例常数由单位约定或独立校准固定）。

【注册 concepts】
  · F（力）

【实现】
  路径：aiphy/experiment/challenge_mechanics/thrust_horizontal.py（现有，水平推力）
        aiphy/experiment/challenge_mechanics/thrust_vertical.py（现有）
        需 strict 改造：开放匿名控制量 u 或 mdot/exhaust_v；隐藏目标命名列 F。
```

### 实验 3.2 `hanging_spring` — 悬挂弹簧（f = k·Δx）

```
【数据列】
  · 默认开放：length[s1](t), posz[o1](t), t
  · 解锁后开放：m（L1）, a, F（实验 3.1）
  · 隐藏：k（待发现）

【预期发现】
  静止时 F = k·Δlength（胡克定律），Δlength = length − 静止长度。

【注册 concepts】
  · k（劲度系数）
  · Δlength（弹簧形变量）

【实现】
  路径：aiphy/experiment/ground_laboratory/hanging_spring_1/2/3.py（现有，含 concept_length）
        aiphy/experiment/ground_laboratory/dynamic_spring.py（现有）
```

**本层解锁**：`F`, `k`, `Δlength`
**可解性**：✓ m,a 已解锁；`length`、`pos` 默认开放。

---

# L4 — 重力 g 与平方反比

### 实验 4.1 `drop_repeat` — 改变质量的自由落体【修正】

```
【数据列】
  · 默认开放：posz[o1](t), t（不同质量的对象各测一次）
  · 解锁后开放：m（L1）, a（L2）
  · 隐藏：g（待发现）

【预期发现】
  自由落体的加速度与对象质量无关 → a = g。
  （**不测 F**——自由落体不可直接测力；直接从 a 得 g。）

【注册 concepts】
  · g（重力加速度）

【实现】
  路径：aiphy/experiment/ground_laboratory/free_fall.py（现有，内部 g=9.801234567，框架看不到）
        需额外实现：aiphy/experiment/ground_laboratory/drop_mass_vary.py
        （扫对象质量参数，验证 a 与 m 无关）
```

### 实验 4.2 `orbit_two_body` — 轨道时间序列

```
【数据列】
  · 默认开放：posx/posy[o1](t), posx/posy[o2](t), dist(t), t
  · 解锁后开放：v, a（微分）, m（L1）
  · 隐藏：M, G

【预期发现】
  ① a ∝ r⁻²（平方反比）
  ② T² ∝ a_orbit³（开普勒第三定律）
  ③ 结合 F=ma → F ∝ 1/r²，常数即 GM

【注册 concepts】
  · GM（引力参数组合，尺度不可分 —— 同 L1 逻辑）
  · a_orbit（半长轴）
  · T（周期）
  · r（轨道半径）

【实现】
  路径：aiphy/experiment/cosmos/celestial_2d*.py（现有）
        需核查：这些文件是否只开放 pos+t（不泄漏 M/G）
```

**本层解锁**：`g`, `GM`, `a_orbit`, `T`, `r`
**可解性**：✓ 落体需 pos+t+m；轨道只给位置+时间。
- ⚠️ `G`与`M`分离需独立实验（Cavendish 或已知参考天体），否则用 `GM` 组合。
- ✓ 活力公式 $v^2=GM(2/r-1/a)$：v,r,a_orbit,GM 齐。

---

# L5 — 介质属性：密度与压强

### 实验 5.1 `volume_by_displacement` — 排水法测体积【修正】

```
【数据列】
  · 默认开放：容器液面高度通道（由 posz 测）、t、物体几何尺寸（如直径 d）
  · 解锁后开放：m（L1）
  · 隐藏：V（待发现）

【预期发现】
  ① 排水法：浸入物体后液面上升 Δh → V = A_容器·Δh；
  ② 或规则物体：由尺寸（d）→ V = (4/3)π r³。
  两者交叉验证体积的测量协议。

【注册 concepts】
  · V（体积）
  · A_容器（容器截面积，几何）

【实现】
  需额外实现：aiphy/experiment/ground_laboratory/volume_by_displacement.py
  （记录液面 posz 随放入物体变化）
```

### 实验 5.2 `density_emergence` — 密度

```
【数据列】
  · 默认开放：几何尺寸（⇒ V）, 物体身份, t
  · 解锁后开放：m（L1）, V（实验 5.1）
  · 隐藏：ρ（待发现）

【预期发现】
  ρ = m/V。

【注册 concepts】
  · ρ（密度）

【实现】
  与 5.1 共享数据（同批物体测 m 与 V → ρ）
```

### 实验 5.3 `pressure_discovery` — 压强【修正：A 只作默认开放】

```
【数据列】
  · 默认开放：posz[piston](t), 截面积 A 或截面几何尺寸（直接可测）, t
  · 解锁后开放：m（L1）, a（L2）, F（L3）
  · 隐藏：P / gas_pressure（待发现）

【预期发现】
  柱塞受力/面积 = 压强 → P = F/A。
  （A 由几何直接测得，属于默认开放；只有 P 是新发现的 concept。）

【注册 concepts】
  · P（压强）

【实现】
  路径：aiphy/experiment/challenge_mechanics/pressure_discovery.py（现有，专为 strict 设计）
        需 strict 改造：`area` 必须进入 data_info 或等价几何列；不能只藏在 exp_para。
```

### 实验 5.4 `hydrostatic_pressure` — 静水压

```
【数据列】
  · 默认开放：探头深度（由 posz 得）、液面位置、t
  · 解锁后开放：P（实验 5.3）, g（L4）,
                （流体密度 ρ_f 在此涌现）
  · 隐藏：无

【预期发现】
  P = ρ_f·g·h（h 为深度，默认开放几何量）。

【注册 concepts】
  · ρ_f（流体密度）
  · h（深度）

【实现】
  需额外实现：aiphy/experiment/ground_laboratory/hydrostatic_pressure.py
```

### 实验 5.5 `sphere_terminal_fall` — 球在粘性液体中沉降

```
【数据列】
  · 默认开放：posz[ball](t), t
  · 解锁后开放：m（L1）, r（几何）, ρ_p（5.2）, ρ_f（5.4）, g（L4）,
                v_t（终端速度，由 v(t) 平台读出）
  · 隐藏：μ（待发现）

【预期发现】
  v_t ∝ r², v_t ∝ (ρ_p−ρ_f), v_t ∝ 1/μ → Stokes 终端速度。

【注册 concepts】
  · μ（粘度）
  · v_t（终端速度）

【实现】
  需额外实现：aiphy/experiment/ground_laboratory/sphere_terminal_fall.py
```

### 实验 5.6 `pipe_flow` — 泊肃叶流动

```
【数据列】
  · 默认开放：管几何（r, L，直接可测）、压差测量 ΔP（由 P 测点差）、体积-时间通道、t
  · 解锁后开放：V（5.1）, P（5.3）, μ（5.5）
  · 隐藏：Q（待发现）

【预期发现】
  Q ∝ r⁴, Q ∝ ΔP, Q ∝ 1/L, Q ∝ 1/μ → 泊肃叶定律。

【注册 concepts】
  · Q（流量）
  · ΔP（压差）

【实现】
  需额外实现：aiphy/experiment/ground_laboratory/pipe_flow.py
  （Q 由体积对时间求导/求积得到 —— OperatorMiner 的 DIFFERENTIATE）
```

**本层解锁**：`V`, `ρ`(对象), `P`, `ρ_f`, `h`, `μ`, `v_t`, `Q`, `ΔP`
**可解性**：
- ✓ 密度：`m`(L1)+`V`(5.1)；体积用排水法/几何，符合常理。
- ✓ 压强：`F`(L3)+`A`(默认开放几何)；`A` 不再是注册项。
- ✓ Stokes/Poiseuille 所有量齐。
- ⚠️ `r⁴`：MapMiner 支持到 5 阶 ✓。

---

# L6 — 守恒量、转动惯量、振子

### 实验 6.1 `spring_oscillator` — 水平弹簧振子【修正】

```
【数据列】
  · 默认开放：posx[o1](t), t
  · 解锁后开放：m（L1）, k（L3）, v（L1）
  · 隐藏：无

【预期发现】
  Δ(½mv²)+Δ(½kx²) ≈ 0 → E = ½mv²+½kx² 守恒。
  （**只注册 E；不凭空“发现 ω₀”**——ω₀ 可由周期 T 或 k/m 得到，
   如需 ω₀，应作为 T 或 k,m 的导出量，而不是独立注册的新 concept。）

【注册 concepts】
  · E（能量）

【实现】
  路径：aiphy/experiment/challenge_mechanics/spring_horizontal.py（现有，strict 设计）
        aiphy/experiment/oscillation/oscillation_x_1/2/3.py（现有）
        需 strict 改造：不要把 k 仅作为 hidden exp_para；用 L3 已解锁的 k 列/弹簧对象属性开放。
```

### 实验 6.2 `rolling_shapes` — 不同几何体斜面滚动【转动惯量】

```
【数据列】
  · 默认开放：pos(t), t, cx/cy/cz（⇒ θ）, 几何体身份
  · 解锁后开放：m（L1）, g（L4）, v, a（L2）
  · 隐藏：I/(mr²)（待发现）

【预期发现】
  a = g·sinθ/(1+I/(mr²))；
  I/(mr²) 是每种几何体的内在常数（球 2/5，圆柱 1/2，圆环 1）。
  发现方式与 L1 质量同构：在滚动能量守恒（½mv²+½Iω²，ω=v/r）中解出 I/(mr²)。

【注册 concepts】
  · I/(mr²)（滚动惯量比；如需 I 本身，再乘 m·r²）

【实现】
  需额外实现：aiphy/experiment/ground_laboratory/rolling_shapes.py
```

### 实验 6.3 `pendulum_small` — 小角度单摆【修正：T、L 不注册】

```
【数据列】
  · 默认开放：posx/posz[o1](t), t, length（摆长，直接可测）
  · 解锁后开放：m（L1）, g（L4）
  · 隐藏：无

【预期发现】
  T = 2π√(L/g)（T 由 pos(t) 自相关读出；L 是默认开放的几何量）。

【注册 concepts】
  无
  （T 是可直接从轨迹读出的量，L 是默认开放的几何量，均**不需要注册**。）

【实现】
  路径：aiphy/experiment/ground_laboratory/pendulum.py, pendulum_1/2/3.py（现有）
        aiphy/experiment/challenge_mechanics/pendulum_undamped_1.py（现有）
```

### 实验 6.4 `damped_oscillator` — 阻尼振子

```
【数据列】
  · 默认开放：pos(t), t
  · 解锁后开放：m、k、v（L1/L3）
  · 隐藏：γ（待发现）

【预期发现】
  振幅指数衰减 e^(−γt/2) → 提取 γ。

【注册 concepts】
  · γ（阻尼率）

【实现】
  路径：aiphy/experiment/ground_laboratory/damped_pendulum_exp.py（现有）
        aiphy/experiment/challenge_mechanics/damped_pendulum_1/2.py（现有）
        aiphy/experiment/challenge_mechanics/damped_hanging_spring_1/2.py（现有）
```

**本层解锁**：`E`, `I/(mr²)`, `γ`
**可解性**：
- ✓ 能量：m,v,k,x 全解锁。
- ✓ 转动惯量：复用 L1 机制。
- ✓ 斜面滚动公式所有量齐（g, θ, I/(mr²)）。

---

# L7 — 特殊函数（tanh, ln, erf）【最难层】

### 实验 7.1 `projectile_no_drag` — 无阻力抛体（基准）

```
【数据列】
  · 默认开放：posx/posy/posz[o1](t), t
  · 解锁后开放：v, θ（抛射角，由初速度分量得；也可视为几何）, g（L4）
  · 隐藏：无

【预期发现】
  抛物线 y = x·tanθ − g·x²/(2v₀²cos²θ)。

【注册 concepts】
  无（tanθ 是几何导出）

【实现】
  路径：aiphy/experiment/ground_laboratory/oblique_projectile.py（现有）
        aiphy/experiment/ground_laboratory/projectile.py（现有）
```

### 实验 7.2 `projectile_linear_drag` — 线性阻尼抛体

```
【数据列】
  · 默认开放：posx/posy/posz[o1](t), t
  · 解锁后开放：v₀, θ, g, m（L1/L2/L4）
  · 隐藏：k_drag（待发现）

【预期发现】
  y(x) = (tanθ + g/(k·v₀cosθ))·x + (g/k²)·ln(1 − kx/(v₀cosθ))；
  极限一致性：k→0 退化为抛物线。

【注册 concepts】
  · k_drag（线性阻尼系数）
  · ln（对数算子解锁）

【实现】
  需额外实现：aiphy/experiment/ground_laboratory/projectile_linear_drag.py
```

### 实验 7.3 `vertical_fall_quadratic` — 二次阻力下落

```
【数据列】
  · 默认开放：posz[ball](t), t, 球的几何（⇒ A=πr²）
  · 解锁后开放：m（L1）, g（L4）, ρ（5.2）
  · 隐藏：C_d、v_t（待发现）

【预期发现】
  v_t = √(2mg/(ρ·C_d·A))；v(t) = v_t·tanh(g·t/v_t)。

【注册 concepts】
  · C_d（阻力系数）
  · tanh（双曲正切算子解锁）
  · v_t（若 5.5 未给）

【实现】
  需额外实现：aiphy/experiment/ground_laboratory/vertical_fall_quadratic.py
  （MapMiner 模板已有 {a}*tanh({b}*x) ✓）
```

### 实验 7.4 `forced_oscillator` — 受迫阻尼振子

```
【数据列】
  · 默认开放：pos(t), t, 驱动力参数 F₀、驱动频率 ω（均为可直接设定的控制量）
  · 解锁后开放：m（L1）, k（L3）, γ（6.4）
  · 隐藏：无

【预期发现】
  A(ω) = (F₀/m)/√((ω₀²−ω²)²+(γω)²)；tanφ = γω/(ω₀²−ω²)。
  （ω₀ 由 k,m 或自由振荡周期得，不作为独立注册项。）

【注册 concepts】
  · φ（相位，若不可由轨迹直接读出）
  （F₀、ω 是控制设定量，默认开放，不注册）

【实现】
  路径：aiphy/experiment/challenge_mechanics/driven_damped_hanging_spring_1/2.py（现有）
        需 strict 改造：F₀ 与 ω 必须进入 data_info/控制列；稳态 A、φ 由轨迹提取，不直接喂入。
```

**本层解锁**：`k_drag`, `C_d`, `v_t`, `φ`；算子 `ln`, `tanh`
**可解性**：
- ⚠️ **最脆弱层**：tanh/ln 必须经 SymbolicDistiller 或模板扩展获得。
  先在 `aiphy/capability_experiments.py` 加 tanh/ln 拟合能力测试。
- ✓ 其余物理变量在本层前已解锁。

---

# L6.5（插入前置）— 弹性接触（Hertz 前置）

### 实验 X.1 `elastic_contact` — 球压平面

```
【数据列】
  · 默认开放：压头位移 δ（由 posz 得）、t、压头几何（⇒ R）
  · 解锁后开放：F（L3）
  · 隐藏：E*（待发现）

【预期发现】
  F ∝ δ^(3/2) → F = (4/3)E*√R·δ^(3/2)。
  发现“倒数相加”串联结构 1/E* = (1−ν₁²)/E₁ + (1−ν₂²)/E₂。

【注册 concepts】
  · E*（等效弹性模量；串联结构需 OperatorMiner 复合）
  （δ、R 是默认开放的位移与几何量，不注册）

【实现】
  需额外实现：aiphy/experiment/ground_laboratory/elastic_contact.py
  ⚠️ 倒数相加串联结构可能需新结构算符
```

---

# L7.5（插入前置）— 高速运动（相对论火箭前置）

### 实验 Y.1 `high_speed_kinematics` — 高速运动

```
【数据列】
  · 默认开放：pos(t), t, 能量投入参数
  · 解锁后开放：v（微分）, m（L1）
  · 隐藏：c（待发现）

【预期发现】
  速度存在上限 c；动量不再是 m·v 线性。

【注册 concepts】
  · c（速度上限/光速）

【实现】
  需额外实现：aiphy/experiment/ground_laboratory/high_speed_kinematics.py
```

### 实验 Y.2 `relativistic_rocket` — 相对论火箭

```
【数据列】
  · 默认开放：posz[t](t), t, 燃料消耗通道
  · 解锁后开放：v, m₀, m₁（L1）, c（Y.1）, tanh（7.3）, ln（7.2）
  · 隐藏：v_e（待发现）

【预期发现】
  Δv = c·tanh((v_e/c)·ln(m₀/m₁))；低速极限 → 齐奥尔科夫斯基。

【注册 concepts】
  · v_e（有效排气速度）
  （m₀, m₁ 由 L1 质量 + 燃料通道得）

【实现】
  路径：aiphy/experiment/challenge_mechanics/variable_mass_rocket_1/2.py（现有，经典）
        aiphy/experiment/challenge_mechanics/two_stage_rocket_1.py（现有）
        需额外实现：relativistic_rocket.py
```

---

# L5T — 热学基础支线（P/V/T、气体状态量、热容）

> 热学支线可在 L5 之后并行启动：`P` 来自 L5.3，`V` 来自 L5.1，温度 `T_abs`
> 是 Kelvin 基本单位的直接传感通道。若运行程序尚未解锁 `P/V`，则只能使用
> `thermodynamics_state.py` 作为“热学传感器能力测试”，不能把其中的 pressure/volume
> 当作力学链已认证的压强/体积。

### 实验 T.1 `temperature_sensor` — 温度读数与热平衡

```
【数据列】
  · 默认开放：temperature/temp[o](t), t（温度计读数，Kelvin 基本单位）
  · 解锁后开放：无
  · 隐藏：热容 C、导热系数 κ

【预期发现】
  接触足够久后温度趋同；温度是可复用状态变量。

【注册 concepts】
  · T_abs（绝对温度/温度读数）

【实现】
  路径：aiphy/experiment/challenge_thermodynamics/contact_networks.py（现有）
        aiphy/experiment/ground_laboratory/two_body_thermal.py（现有）
```

### 实验 T.2 `ideal_gas_state` — 理想气体状态方程

```
【数据列】
  · 默认开放：P(t), V(t), T_abs(t), t, samplemass, molarmass（或 amount n 作为可控装样量）
  · 解锁后开放：P（L5）, V（L5）, T_abs（T.1）
  · 隐藏：R、γ

【预期发现】
  n = samplemass/molarmass；P·V/T 与 n 成正比 → P V = n R T。

【注册 concepts】
  · n（物质的量，若未直接开放）
  · R（气体常数）
  · nR（当 n 与 R 不可分时的组合量）

【实现】
  路径：aiphy/experiment/thermodynamics_epistemic.py（现有，含 samplemass/molarmass）
        aiphy/experiment/thermodynamics_state.py（现有，amount 隐藏时只发现 nR）
```

### 实验 T.3 `adiabatic_gas` — 绝热指数

```
【数据列】
  · 默认开放：P(t), V(t), T_abs(t), t
  · 解锁后开放：n/R 或 nR（T.2）
  · 隐藏：γ

【预期发现】
  P V^γ ≈ const；同时 T V^(γ−1) ≈ const。

【注册 concepts】
  · γ_gas（绝热指数）

【实现】
  路径：aiphy/experiment/thermodynamics_state.py: ideal_gas_adiabatic_config（现有）
        aiphy/experiment/thermodynamics_epistemic.py: epistemic_gas_adiabatic_config（现有）
```

### 实验 T.4 `heat_capacity_calibration` — 热容

```
【数据列】
  · 默认开放：T_abs(t), t, heat_input/power(t)（可控热输入）, sample mass
  · 解锁后开放：m（L1）
  · 隐藏：C、c_p/c_v

【预期发现】
  ΔQ = ∫power dt；C = ΔQ/ΔT，单位质量热容 c = C/m。

【注册 concepts】
  · Q_heat（热量/热输入积分）
  · C_heat（热容）
  · c_heat（比热容）

【实现】
  路径：aiphy/experiment/challenge_thermodynamics/lumped_relaxation.py（现有）
        aiphy/experiment/challenge_thermodynamics/driven_nonlinear.py（现有）
        需 strict 改造：热输入功率/开关必须作为控制列开放，而非只藏在 exp_para。
```

### 实验 T.5 `cooling_and_conduction` — 牛顿冷却与热扩散

```
【数据列】
  · 默认开放：T_abs_i(t), 环境温度 T_env(t), 位置 x_i/杆几何, t
  · 解锁后开放：C_heat（T.4）
  · 隐藏：h_conv、α、κ

【预期发现】
  lumped: dT/dt ∝ −(T−T_env)；
  rod: ∂T/∂t = α ∂²T/∂x²；κ = α ρ c。

【注册 concepts】
  · h_conv（对流换热系数或 k/C 组合）
  · α_thermal（热扩散率）
  · κ_thermal（导热率，若 ρ,c 已齐）

【实现】
  路径：aiphy/experiment/challenge_thermodynamics/lumped_relaxation.py（现有）
        aiphy/experiment/challenge_thermodynamics/phase_diffusion.py: rod_*_diffusion_config（现有）
```

### 实验 T.6 `van_der_waals_gas` — 真实气体

```
【数据列】
  · 默认开放：P(t), V(t), T_abs(t), t
  · 解锁后开放：n, R（T.2）
  · 隐藏：a_vdw, b_vdw

【预期发现】
  (P + a n²/V²)(V − n b) = n R T；临界点关系可作后验验证。

【注册 concepts】
  · a_vdw
  · b_vdw
  · T_c, P_c（若做临界扫描）

【实现】
  需额外实现：aiphy/experiment/challenge_thermodynamics/van_der_waals.py
```

### 实验 T.7 `solid_heat_capacity` — 爱因斯坦/德拜热容

```
【数据列】
  · 默认开放：C_V(T) 或 power(t)+T_abs(t), t, 样品原子数 N/物质的量 n
  · 解锁后开放：T_abs, Q_heat, C_heat, k_B（T.8 或 T.2 中 R/N_A）
  · 隐藏：Θ_E、Θ_D

【预期发现】
  高温极限 → 3Nk_B；低温 Debye → C_V ∝ T³；中温曲线需 exp/integral 模板。

【注册 concepts】
  · Θ_E
  · Θ_D
  · integral_Debye（若表达系统支持积分算子）

【实现】
  需额外实现：aiphy/experiment/challenge_thermodynamics/solid_heat_capacity.py
```

### 实验 T.8 `entropy_expansion` — 熵与 Sackur-Tetrode 前置

```
【数据列】
  · 默认开放：P,V,T_abs,n,t，可逆路径控制量
  · 解锁后开放：R, Q_heat
  · 隐藏：S（待发现）

【预期发现】
  dS = δQ_rev/T；理想气体熵差 ΔS = nR ln(V2/V1)+nC_V ln(T2/T1)。
  若 `h`、`k_B` 已由光学/量子支线解锁，再拟合 Sackur-Tetrode 绝对熵。

【注册 concepts】
  · S_entropy
  · k_B（若通过 R/N_A 建立）

【实现】
  需额外实现：aiphy/experiment/challenge_thermodynamics/entropy_gas.py
```

**本支线解锁**：`T_abs`, `n/nR`, `R`, `γ_gas`, `Q_heat`, `C_heat`, `c_heat`,
`α_thermal`, `κ_thermal`, `a_vdw`, `b_vdw`, `S_entropy`。

---

# L5O — 光学与量子支线（λ、ν、c、h、黑体）

> 光学支线的原始读数是 detector position 与 detector intensity。强度读数是合法传感器输出；
> phase、visibility、fringe spacing 等不得作为单独数据列喂入，只能从图样拟合得到。

### 实验 O.1 `light_time_of_flight` — 光速

```
【数据列】
  · 默认开放：路径长度 L_path（几何）, 到达时间 t_arrival, t
  · 解锁后开放：无
  · 隐藏：c

【预期发现】
  L_path/t_arrival = const → c。

【注册 concepts】
  · c_light（光速；可与 Y.1 的速度上限合并验证）

【实现】
  需额外实现：aiphy/experiment/challenge_optics/time_of_flight.py
```

### 实验 O.2 `interference_wavelength` — 干涉测波长

```
【数据列】
  · 默认开放：detector position x, intensity I(x), 几何 L/d, t
  · 解锁后开放：无
  · 隐藏：λ、phase、visibility

【预期发现】
  双缝条纹间距 Δx = λ L/d；Michelson 条纹位移 ΔN = 2ΔL/λ。

【注册 concepts】
  · λ（波长）

【实现】
  路径：aiphy/experiment/challenge_optics/interference.py
        需 strict 改造：L/d/扫描几何若要发现 λ，必须作为几何/控制列开放；phase 继续隐藏。
```

### 实验 O.3 `frequency_relation` — 频率与波长

```
【数据列】
  · 默认开放：源频率 ν（可设定控制量）, detector intensity/beat signal, t
  · 解锁后开放：λ（O.2）, c_light（O.1）
  · 隐藏：无

【预期发现】
  c = λν。

【注册 concepts】
  · ν（频率，若不是直接控制列）

【实现】
  需额外实现：aiphy/experiment/challenge_optics/frequency_relation.py
```

### 实验 O.4 `diffraction` — 衍射与 sinc/Airy 结构

```
【数据列】
  · 默认开放：detector position x, intensity I(x), aperture/slit geometry a,D,L, t
  · 解锁后开放：λ（O.2）
  · 隐藏：phase

【预期发现】
  single slit: I ∝ sinc²(π a x/(λL))；
  circular aperture: Airy disk with J1 structure。

【注册 concepts】
  · aperture transfer function（可选）
  · Bessel/J1 或 sinc 算子能力（若表达系统需要显式解锁）

【实现】
  路径：aiphy/experiment/challenge_optics/diffraction.py（现有）
```

### 实验 O.5 `photoelectric_effect` — 普朗克常数

```
【数据列】
  · 默认开放：源频率 ν, 光强 intensity, 截止电压 V_stop/电子动能读数, t
  · 解锁后开放：无；若需要电荷 e，则来自电磁支线或直接电量标定
  · 隐藏：h、work_function Φ

【预期发现】
  e V_stop = hν − Φ；强度改变电流不改变截止电压。

【注册 concepts】
  · h（普朗克常数，或 h/e 组合）
  · Φ（逸出功）

【实现】
  路径：aiphy/experiment/challenge_optics/wave_particle.py（现有）
        需 strict 核查：ν、intensity、V_stop 是否作为可读/可控列开放。
```

### 实验 O.6 `blackbody_spectrum` — 普朗克黑体辐射

```
【数据列】
  · 默认开放：wavelength λ 或 detector wavelength bin, spectral intensity B_λ, T_abs, t
  · 解锁后开放：c_light（O.1）, h（O.5）, k_B（T.8 或 R/N_A）
  · 隐藏：无（或只隐藏仪器响应参数，需独立校准）

【预期发现】
  B_λ(T)=2hc²/λ⁵ · 1/(exp(hc/(λk_B T))−1)；
  同时推出 Wien 位移与 Stefan-Boltzmann 积分律。

【注册 concepts】
  · exp 算子（若尚未认证）
  · Wien constant b
  · σ_SB（Stefan-Boltzmann 常数）

【实现】
  需额外实现：aiphy/experiment/challenge_optics/blackbody_spectrum.py
```

**本支线解锁**：`c_light`, `λ`, `ν`, `h/h_over_e`, `Φ`, `exp`, `b`, `σ_SB`。

---

# L8 — 目标公式（前置全解锁）

| # | 目标公式 | 需要的量 | 何时解锁 |
|---|---|---|---|
| 1 | Kepler III $T^2=\frac{4\pi^2a^3}{G(M+m)}$ | `T,a_orbit,GM,m` | L1,L4 |
| 2 | Stokes $v_t=\frac{2(\rho_p-\rho_f)gr^2}{9\mu}$ | `ρ_p,ρ_f,g,r,μ` | L4,L5 |
| 3 | Poiseuille $Q=\frac{\pi\Delta P r^4}{8\mu L}$ | `ΔP,r,μ,L` | L5 |
| 4 | 斜面滚动 $a=\frac{g\sin\theta}{1+I/(mr^2)}$ | `g,θ,I/(mr²)` | L4,L6 |
| 5 | 线性阻尼抛体（对数式） | `θ,g,k_drag,v₀` | L7 |
| 6 | 二次阻力下落 $v=v_t\tanh(gt/v_t)$ | `v_t,g,t` | L7 |
| 7 | 受迫振子 $A,\varphi$ | `F₀,m,k,ω,γ` | L6,L7 |
| 8 | 相对论火箭 | `c,v_e,m₀,m₁` | L7.5 |
| 9 | 赫兹接触 | `E*,R,δ` | L6.5 |
| 10 | 椭圆积分单摆（级数近似） | `L,g,θ₀` | L6 |
| 11 | 范德瓦尔斯方程 | `P,V,T_abs,n,R,a_vdw,b_vdw` | L5,L5T |
| 12 | 麦克斯韦-玻尔兹曼速率分布 | `v,m,T_abs,k_B` | L1,L5T |
| 13 | Curzon-Ahlborn 效率 | `T_h,T_c` | L5T |
| 14 | 半无限体瞬态导热 `erf(x/2√αt)` | `x,t,T_abs,α_thermal` | L5T,L7 |
| 15 | 大角单摆椭圆积分/级数 | `L,g,θ₀` | L6,L7 |
| 16 | 普朗克黑体辐射 | `λ,T_abs,h,c_light,k_B` | L5T,L5O |
| 17 | 固体热容 Einstein/Debye | `T_abs,N,k_B,Θ_E/Θ_D` | L5T,L5O |
| 18 | Sackur-Tetrode 熵 | `V,N,m,U,h,k_B` | L1,L5T,L5O |
| 19 | Taylor-Sedov 爆炸波 | `E,t,ρ,R` | L5,L6 |
| 20 | Kapitza 摆稳定判据 | `a_drive,Ω,g,L` | L6,L7 |

---

# L9 — 运行规范（替代 `run_fast.py`）

```
【运行入口】
  已实现：run_ladder_strict.py

【硬约束】
  1. 按 L1 → L2 → ... 顺序注册实验；每层只注册该层 allowlist。
  2. 每个实验结束后，只把该实验 CERTIFIED concepts 写入 unlocked registry。
  3. 后续实验构造 data_info 时，只允许：
     - L0 直接测量/几何/控制量；
     - unlocked registry 里的 concepts；
     - 当前实验声明的 hidden 变量绝不进入 data_info。
  4. discovery 必须从空 `Knowledge` 启动，再只注册 strict ladder 实验；禁止让 `Theorist()` 默认
     内置实验进入候选池或知识库。
  5. 禁止使用 `run_fast.py` 的全量实验池作为 strict 结果来源。
  6. 对每个实验保存 provenance：列名、来源 RAW_SENSOR/GEOMETRY/CONTROL/UNLOCKED、依赖、是否目标命名。

【建议实现】
  · `LADDER_LEVELS = [(level_name, [(experiment, newly_certified), ...])]`
  · discovery 模式下，`LADDER_LEVELS` 里的 newly_certified 只作为目标声明；实际 unlock 必须来自
    framework certification registry。
  · 每个实验构造前执行 dynamic unlock audit：未认证 concept 对应列若进入 data_info 则 fail fast。
  · 每个实验采集后验证 DataStruct keys 与 data_info 完全一致。
```

【当前实现】

- 列策略：`aiphy/experiment/column_policy.py`
- 严格实验全集：`aiphy/experiment/ladder_strict.py`
- 运行入口：`run_ladder_strict.py`
- 回归测试：`test/test_ladder_strict.py`, `test/test_ladder_levels.py`

当前 `ladder_strict.py` 已覆盖 1.1 → O.6 共 40 个 strict 实验；默认配置只开放本阶梯允许的数据列。
`make_ladder_strict_config(name, unlocked_concepts=...)` 按 registry 动态开放候选列。
`run_ladder_strict.py --levels L1,L2` 可逐级验证每级声明认证的 concepts 和对应 open/hidden 行为。
`run_ladder_strict.py` 的注册/发现路径使用 `Theorist(include_default_experiments=False)`，并在 discovery
前断言所有实验名均为 `strict_` 前缀，避免 `motion/collision/oscillation` 等内置 legacy 实验污染候选池。
默认 discovery 模式不信任 ladder 表里的 `certifies` 声明；只有 `operational_registry` 中状态为
`CERTIFIED` 的 symbolic concept 才会进入 `unlocked_concepts`。`--trust-declared-certifications` 仅用于
schema/scaffold 检查，不作为 strict discovery 结果。

---

# 附：现有实验 ↔ strict ladder 实现对照表

所有行均已在 `aiphy/experiment/ladder_strict.py` 中实现为 strict 版本；旧路径列仅表示可参考的历史/legacy
实验来源，不再作为 strict 结果来源。

| 阶梯实验 | legacy 参考路径 | strict 实现状态 |
|---|---|---|
| 1.1 两体碰撞 | `basic/collision_circle.py`, `collision_y.py` | ✓ `strict_1_1_collision_pair` |
| 1.2 三体碰撞 | `basic/collision_elastic_3body.py` | ✓ `strict_1_2_collision_reference` |
| 2.1 自由落体 | `ground_laboratory/free_fall.py`, `challenge_mechanics/free_fall_1.py` | ✓ `strict_2_1_free_fall_time` |
| 2.2 斜面下滑 | `ground_laboratory/slope_oscillation_1/2.py` | ✓ `strict_2_2_incline_slide` |
| 2.3 滑轮 | `ground_laboratory/pulley_1/2/3.py` | ✓ `strict_2_3_atwood` |
| 3.1 施力 | `challenge_mechanics/thrust_horizontal.py`, `thrust_vertical.py` | ✓ `strict_3_1_mass_under_force` |
| 3.2 悬挂弹簧 | `ground_laboratory/hanging_spring_1/2/3.py`, `dynamic_spring.py` | ✓ `strict_3_2_hanging_spring` |
| 4.1 落体 g | `ground_laboratory/free_fall.py`, `gravity.py` | ✓ `strict_4_1_drop_repeat` |
| 4.2 轨道 | `cosmos/celestial_2d*.py` | ✓ `strict_4_2_orbit_two_body` |
| 5.1 排水法体积 | — | ✓ `strict_5_1_volume_by_displacement` |
| 5.2 密度 | — | ✓ `strict_5_2_density_emergence` |
| 5.3 压强 | `challenge_mechanics/pressure_discovery.py` | ✓ `strict_5_3_pressure_discovery` |
| 5.4 静水压 | — | ✓ `strict_5_4_hydrostatic_pressure` |
| 5.5 球沉降 | — | ✓ `strict_5_5_sphere_terminal_fall` |
| 5.6 管流 | — | ✓ `strict_5_6_pipe_flow` |
| 6.1 弹簧振子 | `challenge_mechanics/spring_horizontal.py`, `oscillation/oscillation_x_1/2/3.py` | ✓ `strict_6_1_spring_oscillator` |
| 6.2 滚动 | — | ✓ `strict_6_2_rolling_shapes` |
| 6.3 单摆 | `ground_laboratory/pendulum*.py` | ✓ `strict_6_3_pendulum_small` |
| 6.4 阻尼 | `ground_laboratory/damped_pendulum_exp.py`, `challenge_mechanics/damped_*` | ✓ `strict_6_4_damped_oscillator` |
| 7.1 无阻力抛体 | `ground_laboratory/oblique_projectile.py`, `projectile.py` | ✓ `strict_7_1_projectile_no_drag` |
| 7.2 线性阻尼抛体 | — | ✓ `strict_7_2_projectile_linear_drag` |
| 7.3 二次阻力下落 | — | ✓ `strict_7_3_vertical_fall_quadratic` |
| 7.4 受迫振子 | `challenge_mechanics/driven_damped_hanging_spring_1/2.py` | ✓ `strict_7_4_forced_oscillator` |
| X.1 弹性接触 | — | ✓ `strict_x_1_elastic_contact` |
| Y.1 高速运动 | — | ✓ `strict_y_1_high_speed_kinematics` |
| Y.2 相对论火箭 | `challenge_mechanics/variable_mass_rocket_1/2.py`, `two_stage_rocket_1.py` | ✓ `strict_y_2_relativistic_rocket` |
| T.1 温度/热平衡 | `challenge_thermodynamics/contact_networks.py`, `ground_laboratory/two_body_thermal.py` | ✓ `strict_t_1_temperature_sensor` |
| T.2 理想气体 | `thermodynamics_state.py`, `thermodynamics_epistemic.py` | ✓ `strict_t_2_ideal_gas_state` |
| T.3 绝热指数 | `thermodynamics_state.py`, `thermodynamics_epistemic.py` | ✓ `strict_t_3_adiabatic_gas` |
| T.4 热容 | `challenge_thermodynamics/lumped_relaxation.py`, `driven_nonlinear.py` | ✓ `strict_t_4_heat_capacity_calibration` |
| T.5 冷却/热扩散 | `challenge_thermodynamics/lumped_relaxation.py`, `phase_diffusion.py` | ✓ `strict_t_5_cooling_and_conduction` |
| T.6 范德瓦尔斯 | — | ✓ `strict_t_6_van_der_waals_gas` |
| T.7 固体热容 | — | ✓ `strict_t_7_solid_heat_capacity` |
| T.8 熵 | — | ✓ `strict_t_8_entropy_expansion` |
| O.1 光速飞行时间 | — | ✓ `strict_o_1_light_time_of_flight` |
| O.2 干涉测波长 | `challenge_optics/interference.py` | ✓ `strict_o_2_interference_wavelength` |
| O.3 频率-波长 | — | ✓ `strict_o_3_frequency_relation` |
| O.4 衍射 | `challenge_optics/diffraction.py` | ✓ `strict_o_4_diffraction` |
| O.5 光电效应 | `challenge_optics/wave_particle.py` | ✓ `strict_o_5_photoelectric_effect` |
| O.6 黑体辐射 | — | ✓ `strict_o_6_blackbody_spectrum` |

## 附：验证结果

- `python -m pytest test/test_ladder_strict.py test/test_ladder_levels.py -q`：strict 实验均能构造并采集；
  返回列数等于开放列数。
- hidden 列和未认证的 `UNLOCKED` 候选列默认不进入 `data_info` / `DataStruct`。
- `python run_ladder_strict.py --levels L1 --skip-register --discover-epochs 0`：1.1 不泄漏 `mass`，
  1.1 认证后 1.2 动态开放 `mass[1]`。
- `python run_ladder_strict.py --levels L1,L2 --skip-smoke --discover-epochs 0 --save-path ...`：从空
  `Knowledge` 注册 5 个 strict 实验，实验池恰好为
  `strict_1_1_collision_pair`, `strict_1_2_collision_reference`, `strict_2_1_free_fall_time`,
  `strict_2_2_incline_slide`, `strict_2_3_atwood`。
  由于当前 framework 尚未认证 `mass` operational concept，pure strict discovery 模式下 1.2 与 2.3
  的 `mass[...]` 列保持隐藏；若要演示声明式解锁，必须显式加 `--trust-declared-certifications`。

---

*文档状态：v7 —— 明确开放/隐藏由实验顺序、`ColumnSpec` 角色和动态认证 registry 共同决定；
实现 strict ladder 1.1 → O.6 全链条，支持 `--levels` 逐级验证；新增 `run_ladder_strict.py`，
从空 `Knowledge` 启动 discovery，禁止用全量 `run_fast.py` 或默认内置实验作为 strict 结果来源。*
