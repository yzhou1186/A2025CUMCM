import math
import numpy as np
import matplotlib.pyplot as plt

# 基础参数
np.random.seed(1331)  # 固定随机种子，结果可复现
# 初始解
PROB1_PARAMS = {
    "vx": -120.0,  # 朝向假目标（原点）的x方向速度（FY1→原点x负向）
    "vy": 0.0,  # 问题1无y方向速度（朝向原点）
    "vz": 0.0,  # 等高飞行（z轴速度=0）
    "t_release": 1.5,  # 投放时刻（受领任务后1.5s）
    "t_delay": 3.6  # 起爆延迟（投放后3.6s）
}


# 工具函数
def vector_length(v):
    """计算向量长度（用于M1位置、速度大小计算）"""
    x, y, z = v
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


def dot_product(v1, v2):
    """计算向量点积（线段与球体相交判断）"""
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return x1 * x2 + y1 * y2 + z1 * z2


def vector_subtract(v1, v2):
    """计算向量差（v1 - v2，位置差计算）"""
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return (x1 - x2, y1 - y2, z1 - z2)


def get_pt_lt_pos(t):
    """计算t时刻M1位置，朝原点飞行，速度300m/s）"""
    dir_ec = (-20000.0, 0.0, -2000.0)  # M1→原点方向向量
    dir_length = vector_length(dir_ec)  # M1到原点总距离≈20099.75m
    unit_dir = (dir_ec[0] / dir_length, dir_ec[1] / dir_length, dir_ec[2] / dir_length)
    dis = 300.0 * t  # t时刻飞行距离
    if dis >= dir_length:
        return (0.0, 0.0, 0.0)  # 到达原点后停止
    return (20000.0 + unit_dir[0] * dis, 0.0 + unit_dir[1] * dis, 2000.0 + unit_dir[2] * dis)


def get_sphe_pos(t, t_det, x_det, y_det, z_det):
    """计算t时刻烟幕云团位置（起爆后匀速下沉3m/s，有效20s）"""
    if t < t_det - 1e-9 or t > t_det + 20 + 1e-9:
        return None  # 云团失效
    z_pos = z_det - 3.0 * (t - t_det)  # 下沉速度3m/s
    return (float(x_det), float(y_det), float(z_pos))


def is_sme_it_sphe_ac(L, P, S, R):
    """判断线段LP（M1→真目标）是否与云团（球心S，半径R=10m）相交"""
    LP = vector_subtract(P, L)
    LS = vector_subtract(L, S)
    a = dot_product(LP, LP)
    b = 2 * dot_product(LS, LP)
    c = dot_product(LS, LS) - R ** 2
    dis_crnt = b ** 2 - 4 * a * c

    # 处理线段长度为0的特殊情况（L=P）
    if a < 1e-12:
        return vector_length(LS) <= R + 1e-9

    if dis_crnt < -1e-9:
        return False
    dis_crnt = max(dis_crnt, 0.0)
    sqrt_d = math.sqrt(dis_crnt)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    # 线段相交：t∈[0,1]
    return (t1 >= -1e-9 and t1 <= 1.0 + 1e-9) or (t2 >= -1e-9 and t2 <= 1.0 + 1e-9)


def is_cr_ful_oced(t, t_det, x_det, y_det, z_det):
    """判断t时刻真目标（圆柱）是否被完全遮蔽（底心(0,200,0)，r=7m，h=10m）"""
    light_pos = get_pt_lt_pos(t)
    sphe_pos = get_sphe_pos(t, t_det, x_det, y_det, z_det)
    if sphe_pos is None:
        return False  # 云团未生效

    # 真目标与云团参数
    cyl_rus = 7.0
    cyl_y = 200.0
    cyl_z_range = [0.0, 10.0]  # 圆柱z轴范围（底0，顶10）
    sphe_rus = 10.0

    # 快速判断：M1在云团内→完全遮蔽
    LS = vector_subtract(light_pos, sphe_pos)
    if vector_length(LS) <= sphe_rus + 1e-9:
        return True

    # 增加关键射线数目，覆盖更多圆柱表面位置
    # 采用分层采样策略，在不同高度和角度设置采样点
    key_points = []

    # 顶部圆周采样（4个方向）
    key_points.extend([
        (0.0, cyl_y, cyl_z_range[1]),  # 顶部中心
        (cyl_rus, cyl_y, cyl_z_range[1]),  # 顶部右
        (-cyl_rus, cyl_y, cyl_z_range[1]),  # 顶部左
        (0.0, cyl_y, cyl_z_range[1] + cyl_rus),  # 顶部前
        (0.0, cyl_y, cyl_z_range[1] - cyl_rus)  # 顶部后
    ])

    # 底部圆周采样（4个方向）
    key_points.extend([
        (0.0, cyl_y, cyl_z_range[0]),  # 底部中心
        (cyl_rus, cyl_y, cyl_z_range[0]),  # 底部右
        (-cyl_rus, cyl_y, cyl_z_range[0]),  # 底部左
        (0.0, cyl_y, cyl_z_range[0] + cyl_rus),  # 底部前
        (0.0, cyl_y, cyl_z_range[0] - cyl_rus)  # 底部后
    ])

    # 中间高度采样（3个高度层级，每个层级4个方向）
    hts = [cyl_z_range[0] + 2.5, cyl_z_range[0] + 5.0, cyl_z_range[0] + 7.5]
    for h in hts:
        key_points.extend([
            (cyl_rus, cyl_y, h),  # 右
            (-cyl_rus, cyl_y, h),  # 左
            (0.0, cyl_y, h + cyl_rus),  # 前
            (0.0, cyl_y, h - cyl_rus)  # 后
        ])

    # 对角线方向采样（45度角）
    dg_for = math.cos(math.pi / 4)  # 对角线位置因子
    key_points.extend([
        (cyl_rus * dg_for, cyl_y, 5.0 + cyl_rus * dg_for),  # 右上前
        (-cyl_rus * dg_for, cyl_y, 5.0 + cyl_rus * dg_for),  # 左上前
        (cyl_rus * dg_for, cyl_y, 5.0 - cyl_rus * dg_for),  # 右上后
        (-cyl_rus * dg_for, cyl_y, 5.0 - cyl_rus * dg_for)  # 左上后
    ])

    # 半半径位置采样（检测中间区域）
    half_r = cyl_rus * 0.5
    key_points.extend([
        (half_r, cyl_y, 2.5),  # 右下1/4高度
        (-half_r, cyl_y, 7.5),  # 左上3/4高度
        (0.0, cyl_y, 2.5 + half_r),  # 前1/4高度
        (0.0, cyl_y, 7.5 - half_r)  # 后3/4高度
    ])

    # 所有关键射线需被遮挡才视为完全遮蔽
    for p in key_points:
        if not is_sme_it_sphe_ac(light_pos, p, sphe_pos, sphe_rus):
            return False
    return True


def cate_det_pos(vx, vy, t_release, t_delay, g=-9.8):
    """计算烟幕弹投放点+起爆点（适配等高飞行：vz=0）"""
    s0_x, s0_y, s0_z = 17800.0, 0.0, 1800.0
    vz = 0.0  # 等高飞行，z轴速度固定为0

    # 投放点位置（t_release时刻无人机位置）
    release_x = s0_x + vx * t_release
    release_y = s0_y + vy * t_release
    release_z = s0_z + vz * t_release

    # 起爆点位置（投放后t_delay内运动：水平匀速，竖直受重力）
    det_x = release_x + vx * t_delay
    det_y = release_y + vy * t_delay
    det_z = release_z + vz * t_delay + 0.5 * g * t_delay ** 2
    t_det = t_release + t_delay

    return (release_x, release_y, release_z), (t_det, det_x, det_y, det_z)


def find_trans(start, end, t_det, x_det, y_det, z_det, target_state):
    """二分法查找遮蔽状态转换时间"""
    eps = 1e-9
    for _ in range(50):
        mid = (start + end) / 2
        current_state = is_cr_ful_oced(mid, t_det, x_det, y_det, z_det)
        if current_state == target_state:
            end = mid
        else:
            start = mid
        if end - start < eps:
            break
    return (start + end) / 2


def cate_tol_ocion_t(t_det, x_det, y_det, z_det):
    """计算总有效遮蔽时间"""
    cloud_start = t_det
    cloud_end = t_det + 20.0
    total_time = 0.0
    ocion_inval = []

    # 快速判断无遮蔽场景
    sample_t = (cloud_start + cloud_end) / 2
    if not is_cr_ful_oced(sample_t, t_det, x_det, y_det, z_det):
        if not is_cr_ful_oced(cloud_start, t_det, x_det, y_det, z_det) and \
                not is_cr_ful_oced(cloud_end, t_det, x_det, y_det, z_det):
            return 0.0, []

    # 粗查+精修遮蔽区间
    pre_oced = is_cr_ful_oced(cloud_start, t_det, x_det, y_det, z_det)
    invl_start = cloud_start if pre_oced else None

    t = cloud_start + 0.1
    while t <= cloud_end:
        cur_oced = is_cr_ful_oced(t, t_det, x_det, y_det, z_det)
        if cur_oced != pre_oced:
            transition_t = find_trans(t - 0.1, t, t_det, x_det, y_det, z_det, cur_oced)
            if pre_oced:
                ocion_inval.append((round(invl_start, 2), round(transition_t, 2)))
                total_time += transition_t - invl_start
                invl_start = None
            else:
                invl_start = transition_t
            pre_oced = cur_oced
        t += 0.1

    if pre_oced and invl_start is not None:
        ocion_inval.append((round(invl_start, 2), round(cloud_end, 2)))
        total_time += cloud_end - invl_start

    return max(total_time, 0.0), ocion_inval


# PSO优化类
class PSO_Optimizer:
    def __init__(self, pop_size=40, max_iter=150, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0):
        """初始化PSO"""
        self.bounds = np.array([
            [-140.0, -80.0],  # vx
            [0.0, 140.0],  # vy
            [0.000001, 8.0],  # t_release
            [0.000001, 10.0]  # t_delay
        ])
        self.dim = 4
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1

        # 初始化种群
        self.pos = self._init_population()
        self.vel = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop_size, self.dim))

        # 初始化最优解
        self.p_best_pos = self.pos.copy()
        self.p_best_fit = np.array([self.calculate_fitness(p) for p in self.pos])
        self.g_best_idx = np.argmax(self.p_best_fit)
        self.g_best_pos = self.p_best_pos[self.g_best_idx].copy()
        self.g_best_fit = self.p_best_fit[self.g_best_idx]
        self.fit_history = [self.g_best_fit]

    def _init_population(self):
        """种群初始化"""
        pop = np.zeros((self.pop_size, self.dim))
        base_params = [PROB1_PARAMS["vx"], PROB1_PARAMS["vy"], PROB1_PARAMS["t_release"], PROB1_PARAMS["t_delay"]]
        for i in range(self.pop_size):
            pop[i] = [
                base_params[0] + np.random.uniform(-10.0, 10.0),
                base_params[1] + np.random.uniform(-15.0, 15.0),
                base_params[2] + np.random.uniform(-1.0, 1.0),
                base_params[3] + np.random.uniform(-1.5, 1.5)
            ]
        return np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])

    def calculate_fitness(self, particle):
        """计算适应度（总遮蔽时间，违反约束则惩罚）"""
        vx, vy, t_release, t_delay = particle
        vz = 0.0
        v_mag = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

        # 约束1：速度70~140m/s
        if v_mag < 70.0 - 1e-9 or v_mag > 140.0 + 1e-9:
            return 1e-6
        # 约束2：投放/起爆时间非负
        if t_release < 0 or t_delay < 0:
            return 1e-6

        # 计算起爆参数
        (_, _, _), (t_det, dx, dy, dz) = cate_det_pos(vx, vy, t_release, t_delay)
        # 约束3：云团z坐标合理
        if dz < 50.0 or dz > 2000.0:
            return 1e-6

        # 计算遮蔽时间
        total_ocion, _ = cate_tol_ocion_t(t_det, dx, dy, dz)
        return total_ocion if total_ocion >= 1e-3 else 1e-6

    def update_velocity(self, iter_idx):
        """更新粒子速度"""
        w = self.w_start - (self.w_start - self.w_end) * (iter_idx / self.max_iter)
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        self.vel = w * self.vel + \
                   self.c1 * r1 * (self.p_best_pos - self.pos) + \
                   self.c2 * r2 * (self.g_best_pos - self.pos)
        self.vel = np.clip(self.vel, -self.v_max, self.v_max)

    def update_position(self):
        """更新粒子位置"""
        self.pos = self.pos + self.vel
        self.pos = np.clip(self.pos, self.bounds[:, 0], self.bounds[:, 1])

    def update_best(self):
        """更新最优解"""
        cur_fit = np.array([self.calculate_fitness(p) for p in self.pos])
        # 更新个体最优
        mask = cur_fit > self.p_best_fit
        self.p_best_pos[mask] = self.pos[mask].copy()
        self.p_best_fit[mask] = cur_fit[mask]
        # 更新全局最优
        cur_g_best_idx = np.argmax(self.p_best_fit)
        cur_g_best_fit = self.p_best_fit[cur_g_best_idx]
        if cur_g_best_fit > self.g_best_fit:
            self.g_best_fit = cur_g_best_fit
            self.g_best_pos = self.p_best_pos[cur_g_best_idx].copy()
        self.fit_history.append(self.g_best_fit)

    def optimize(self):
        """执行PSO优化"""
        print(f"PSO优化：种群{self.pop_size} | 迭代{self.max_iter} | 速度70~140m/s")
        for iter_idx in range(self.max_iter):
            self.update_velocity(iter_idx)
            self.update_position()
            self.update_best()
            if (iter_idx + 1) % 10 == 0:
                print(f"迭代{iter_idx + 1:3d}/{self.max_iter} | 全局最优：{self.g_best_fit:.2f}s")
        print(f"优化结束！最终最优：{self.g_best_fit:.6f}s")
        return self.g_best_pos, self.g_best_fit, self.fit_history


# 结果可视化函数
def plot_convergence_curve(fit_history, prob1_fitness, best_fitness):
    """绘制PSO收敛曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = range(len(fit_history))
    ax.plot(iterations, fit_history, color='#2E86AB', linewidth=2.5, marker='o', markersize=3, label='全局最优遮蔽时间')
    ax.axhline(y=best_fitness, color='#A23B72', linestyle='--', linewidth=2, label=f'最终最优值：{best_fitness:.2f}s')
    ax.axhline(y=prob1_fitness, color='#F18F01', linestyle=':', linewidth=2, label=f'问题1初始值：{prob1_fitness:.2f}s')

    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('全局最优遮蔽时间（s）', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(iterations) - 1)
    ax.set_ylim(0, max(fit_history) * 1.1)

    plt.tight_layout()
    plt.savefig('问题2_PSO收敛曲线.png', dpi=300, bbox_inches='tight')
    print("\n收敛曲线已保存为「问题2_PSO收敛曲线.png」")
    plt.show()


def plot_fitness_comparison(prob1_fitness, best_fitness):
    """绘制初始解vs最优解对比图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['问题1初始解', 'PSO最优解']
    values = [prob1_fitness, best_fitness]
    colors = ['#F18F01', '#2E86AB']

    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.2,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('遮蔽时间（s）', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(values) * 1.2)

    plt.tight_layout()
    plt.savefig('问题2_遮蔽时间对比图.png', dpi=300, bbox_inches='tight')
    print("对比图已保存为「问题2_遮蔽时间对比图.png」")
    plt.show()


# 灵敏性分析
def sensitivity_analysis():
    """灵敏性分析：仅分析4个核心参数"""
    # 1. 定义待分析参数
    params_config = {
        "初始惯性权重(w_start)": [0.7, 0.8, 0.9, 1.0],
        "学习因子(c1=c2)": [1.5, 2.0, 2.5, 3.0],
        "投放时刻上限(t_r_max, s)": [5.0, 8.0, 10.0, 12.0],
        "起爆延迟上限(t_d_max, s)": [8.0, 10.0, 12.0, 14.0]
    }

    # 2. 计算基准值T₀（默认参数：pop_size=40，max_iter=150）
    print("灵敏性分析：计算基准值（默认参数）")
    default_pso = PSO_Optimizer(pop_size=40, max_iter=150, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0)
    default_params, _, _ = default_pso.optimize()
    vx0, vy0, tr0, td0 = default_params
    (_, _, _), (t_det0, dx0, dy0, dz0) = cate_det_pos(vx0, vy0, tr0, td0)
    T0, _ = cate_tol_ocion_t(t_det0, dx0, dy0, dz0)
    print(f"\n基准遮蔽时间 T₀ = {T0:.2f} s\n")

    # 3. 遍历每个核心参数，计算灵敏性
    sensty_results = {}
    for param_name, param_levels in params_config.items():
        print(f"分析参数：{param_name}")
        param_results = []
        for level in param_levels:
            # 控制变量：仅修改当前参数，其余用默认值
            if param_name == "初始惯性权重(w_start)":
                pso = PSO_Optimizer(pop_size=40, max_iter=150, w_start=level, w_end=0.4, c1=2.0, c2=2.0)
            elif param_name == "学习因子(c1=c2)":
                pso = PSO_Optimizer(pop_size=40, max_iter=150, w_start=0.9, w_end=0.4, c1=level, c2=level)
            elif param_name == "投放时刻上限(t_r_max, s)":
                pso = PSO_Optimizer(pop_size=40, max_iter=150, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0)
                pso.bounds[2] = [0.001, level]  # 调整投放时刻上限
            elif param_name == "起爆延迟上限(t_d_max, s)":
                pso = PSO_Optimizer(pop_size=40, max_iter=150, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0)
                pso.bounds[3] = [0.01, level]  # 调整起爆延迟上限

            # 重复3次取平均，降低随机波动
            T_list = []
            for _ in range(3):
                best_params, _, _ = pso.optimize()
                vx, vy, tr, td = best_params
                (_, _, _), (t_det, dx, dy, dz) = cate_det_pos(vx, vy, tr, td)
                T, _ = cate_tol_ocion_t(t_det, dx, dy, dz)
                T_list.append(T)
            avg_T = np.mean(T_list)
            eta = (avg_T - T0) / T0 * 100  # 相对变化率（%）
            param_results.append((level, avg_T, eta))
            print(f"  水平{level}：平均遮蔽时间={avg_T:.2f}s，相对变化率={eta:.1f}%")
        sensty_results[param_name] = param_results
        print()

    # 4. 输出灵敏性总结
    print(f"灵敏性分析总结（基准T₀={T0:.2f}s）")
    ranking = []
    for param_name, results in sensty_results.items():
        max_eta_abs = max([abs(eta) for (_, _, eta) in results])
        # 划分敏感等级
        if max_eta_abs > 10:
            level = "高敏感"
        elif max_eta_abs > 5:
            level = "中敏感"
        else:
            level = "低敏感"
        ranking.append((param_name, max_eta_abs, level))
        print(f"{param_name:20s} | 最大相对变化率：{max_eta_abs:.1f}% | 敏感等级：{level}")

    # 5. 可视化灵敏性结果
    plot_sensitivity_2x2(sensty_results, T0)
    return sensty_results, ranking


def plot_sensitivity_2x2(sensitivity_results, T0):
    """灵敏性分析可视化"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建2×2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # 展平为1维列表，便于索引
    param_names = list(sensitivity_results.keys())

    # 遍历4个参数，填充子图
    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        results = sensitivity_results[param_name]
        levels = [r[0] for r in results]
        avg_T = [r[1] for r in results]
        eta = [r[2] for r in results]

        # 双Y轴设计：左轴=遮蔽时间，右轴=相对变化率
        ax2 = ax.twinx()
        # 左轴：遮蔽时间（蓝色折线+圆点）
        ax.plot(levels, avg_T, color='#2E86AB', linewidth=2.5, marker='o', markersize=6, label='平均遮蔽时间')
        ax.axhline(y=T0, color='#F18F01', linestyle='--', linewidth=2, label=f'基准值{T0:.2f}s')
        # 右轴：相对变化率（红色柱状图）
        bars = ax2.bar([str(l) for l in levels], eta, color='#A23B72', alpha=0.6, label='相对变化率(%)', width=0.6)

        # 子图样式配置
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel('平均遮蔽时间（s）', fontsize=8, color='#2E86AB')
        ax2.set_ylabel('相对变化率（%）', fontsize=8, color='#A23B72')
        ax.tick_params(axis='y', labelcolor='#2E86AB')
        ax2.tick_params(axis='y', labelcolor='#A23B72')
        ax.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    # 总标题
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('问题2_PSO灵敏性分析.png', dpi=300, bbox_inches='tight')
    print("\n灵敏性分析图表已保存为「问题2_PSO灵敏性分析.png」")
    plt.show()


# 求解问题+灵敏性分析
def solve_problem2_with_pso():
    """求解问题2并执行灵敏性分析"""
    # 问题1初始解遮蔽时间
    prob1_fitness = 1.38
    print(f"问题1初始解遮蔽时间：{prob1_fitness:.2f}s\n")

    # 步骤1：PSO优化
    pso = PSO_Optimizer(pop_size=40, max_iter=150, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0)
    best_params, best_fit, fit_history = pso.optimize()

    # 步骤2：解析最优参数
    vx_opt, vy_opt, t_release_opt, t_delay_opt = best_params
    vz_opt = 0.0
    v_mag_opt = math.sqrt(vx_opt ** 2 + vy_opt ** 2 + vz_opt ** 2)
    (release_x, release_y, release_z), (t_det_opt, det_x, det_y, det_z) = cate_det_pos(
        vx_opt, vy_opt, t_release_opt, t_delay_opt
    )
    total_occlusion, occlusion_intervals = cate_tol_ocion_t(t_det_opt, det_x, det_y, det_z)

    # 飞行方向角度
    dir_unit_x = vx_opt / v_mag_opt if v_mag_opt > 1e-9 else 0.0
    dir_unit_y = vy_opt / v_mag_opt if v_mag_opt > 1e-9 else 0.0
    flight_angle = abs(np.arctan(dir_unit_y / dir_unit_x) * 180 / np.pi) if abs(dir_unit_x) > 1e-9 else 90.0
    flight_angle = 180 - flight_angle if dir_unit_x < 0 else flight_angle

    # 步骤3：输出最优策略
    print("问题2 最优投放策略（等高飞行+PSO优化）")
    print("1. 无人机FY1飞行参数：")
    print(f"   - 速度大小：{v_mag_opt:.2f} m/s（70~140m/s）")
    print(f"   - 速度向量：(vx={vx_opt:.2f}, vy={vy_opt:.2f}, vz={vz_opt:.2f}) m/s")
    print(f"   - 飞行角度：{flight_angle:.1f}°（相对于x负向，逆时针为正）")
    print("\n2. 烟幕弹投放参数：")
    print(f"   - 投放时刻：{t_release_opt:.2f} s")
    print(f"   - 投放点：({release_x:.2f}, {release_y:.2f}, {release_z:.2f}) m")
    print("\n3. 烟幕弹起爆参数：")
    print(f"   - 起爆延迟：{t_delay_opt:.2f} s")
    print(f"   - 起爆时刻：{t_det_opt:.2f} s")
    print(f"   - 起爆点：({det_x:.2f}, {det_y:.2f}, {det_z:.2f}) m")
    print("\n4. 遮蔽效果：")
    print(f"   - 云团有效期：[{t_det_opt:.2f}, {t_det_opt + 20:.2f}] s")
    print(f"   - 遮蔽区间：")
    for i, (s, e) in enumerate(occlusion_intervals, 1):
        print(f"     区间{i}：{s:.2f}~{e:.2f}s（持续{e - s:.2f}s）")
    improvement = ((total_occlusion - prob1_fitness) / prob1_fitness) * 100
    print(f"   - 总遮蔽时间：{total_occlusion:.2f}s（比问题1提升{improvement:.1f}%）")

    # 步骤4：结果可视化
    try:
        plot_convergence_curve(fit_history, prob1_fitness, total_occlusion)
        plot_fitness_comparison(prob1_fitness, total_occlusion)
    except ImportError:
        print("\n未安装matplotlib，跳过可视化（安装：pip install matplotlib）")

    # 步骤5：执行灵敏性分析
    print("问题2 PSO优化策略灵敏性分析")
    sensitivity_results, sensitivity_ranking = sensitivity_analysis()

    return {
        "最优策略": {"无人机速度": v_mag_opt, "投放时刻": t_release_opt, "起爆延迟": t_delay_opt,
                     "总遮蔽时间": total_occlusion},
        "灵敏性结果": sensitivity_results,
        "敏感排序": sensitivity_ranking
    }


if __name__ == "__main__":
    opt_strate = solve_problem2_with_pso()
