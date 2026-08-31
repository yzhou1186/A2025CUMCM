import math
import numpy as np
import matplotlib.pyplot as plt

# 固定种子+基础参数
np.random.seed(314159265)  # 固定随机种子，结果可复现
PROB1_PARAMS = {
    "vx": -120.0,  # 朝向假目标的x方向速度
    "vy": 0.0,  # 初始y方向速度
    "vz": 0.0,  # 等高飞行
    "t_release": 1.5,  # 初始投放时刻
    "t_delay": 3.6  # 初始起爆延迟
}
# 烟幕基础参数（敏感性分析目标）
SMOKE_BASE_PARAMS = {
    "R": 10.0,  # 烟幕云团半径（m）
    "v_sink": 3.0  # 烟幕下沉速度（m/s）
}


# 工具函数
def vector_length(v):
    """计算向量长度"""
    x, y, z = v
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


def dot_product(v1, v2):
    """计算向量点积"""
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return x1 * x2 + y1 * y2 + z1 * z2


def vector_subtract(v1, v2):
    """计算向量差（v1 - v2）"""
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return (x1 - x2, y1 - y2, z1 - z2)


def get_pt_lt_pos(t):
    """计算t时刻M1位置（初始(20000,0,2000)，速度300m/s朝原点）"""
    dir_en = (-20000.0, 0.0, -2000.0)
    dir_length = vector_length(dir_en)
    unit_dir = (dir_en[0] / dir_length, dir_en[1] / dir_length, dir_en[2] / dir_length)
    dis = 300.0 * t
    if dis >= dir_length:
        return (0.0, 0.0, 0.0)
    return (20000.0 + unit_dir[0] * dis, 0.0 + unit_dir[1] * dis, 2000.0 + unit_dir[2] * dis)


def sphe_pos(t, t_det, x_det, y_det, z_det, v_sink):
    """计算t时刻烟幕云团位置"""
    if t < t_det - 1e-9 or t > t_det + 20 + 1e-9:
        return None  # 云团20s后失效
    z_pos = z_det - v_sink * (t - t_det)  # 动态下沉速度
    return (float(x_det), float(y_det), float(z_pos))


def is_smen_it_sphe_anic(L, P, S, R):
    """判断线段LP是否与球（S,R）相交"""
    LP = vector_subtract(P, L)
    LS = vector_subtract(L, S)
    a = dot_product(LP, LP)
    b = 2 * dot_product(LS, LP)
    c = dot_product(LS, LS) - R ** 2
    dis_crnt = b ** 2 - 4 * a * c

    if a < 1e-12:  # L=P的特殊情况
        return vector_length(LS) <= R + 1e-9
    if dis_crnt < -1e-9:
        return False
    dis_crnt = max(dis_crnt, 0.0)
    sqrt_d = math.sqrt(dis_crnt)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    return (t1 >= -1e-9 and t1 <= 1.0 + 1e-9) or (t2 >= -1e-9 and t2 <= 1.0 + 1e-9)


def is_cyer_ful_oced(t, t_det, x_det, y_det, z_det, R, v_sink):
    """判断t时刻真目标是否被完全遮蔽（根据第一个代码优化的版本）"""
    light_pos = get_pt_lt_pos(t)
    sphe_pos_ans = sphe_pos(t, t_det, x_det, y_det, z_det, v_sink)

    # 球体不存在时，圆柱不可能被完全遮挡
    if sphe_pos_ans is None:
        return False

    # 真目标参数（固定：底心(0,200,0)，r=7m，h=10m）
    cyl_radius = 7.0
    cyl_y = 200.0
    cyl_z_range = [0.0, 10.0]
    cyl_ht = cyl_z_range[1] - cyl_z_range[0]

    # 快速判断：M1在云团内→完全遮蔽
    LS = vector_subtract(light_pos, sphe_pos_ans)
    if vector_length(LS) <= R + 1e-9:
        return True

    # 圆柱关键特征点
    C_bottom = (0.0, cyl_y, cyl_z_range[0])  # 底部中心
    C_top = (0.0, cyl_y, cyl_z_range[1])  # 顶部中心

    # 增加照射锥采样点 - 在不同高度和半径位置采样
    key_rays = []

    # 顶部圆周采样（4个方向）
    key_rays.extend([
        (cyl_radius, cyl_y, C_top[2]),  # 顶部右
        (-cyl_radius, cyl_y, C_top[2]),  # 顶部左
        (0, cyl_y, C_top[2] + cyl_radius),  # 顶部前
        (0, cyl_y, C_top[2] - cyl_radius)  # 顶部后
    ])

    # 底部圆周采样（4个方向）
    key_rays.extend([
        (cyl_radius, cyl_y, C_bottom[2]),  # 底部右
        (-cyl_radius, cyl_y, C_bottom[2]),  # 底部左
        (0, cyl_y, C_bottom[2] + cyl_radius),  # 底部前
        (0, cyl_y, C_bottom[2] - cyl_radius)  # 底部后
    ])

    # 中间高度采样（3个高度层级，每个层级4个方向）
    heights = [
        C_bottom[2] + cyl_ht * 0.25,
        C_bottom[2] + cyl_ht * 0.5,
        C_bottom[2] + cyl_ht * 0.75
    ]
    for h in heights:
        key_rays.extend([
            (cyl_radius, cyl_y, h),  # 右
            (-cyl_radius, cyl_y, h),  # 左
            (0, cyl_y, h + cyl_radius),  # 前
            (0, cyl_y, h - cyl_radius)  # 后
        ])

    # 对角线方向采样（45度角）
    diag_factor = math.cos(math.pi / 4)  # 对角线位置因子
    key_rays.extend([
        (cyl_radius * diag_factor, cyl_y, C_bottom[2] + cyl_ht * 0.5 + cyl_radius * diag_factor),  # 右上前
        (-cyl_radius * diag_factor, cyl_y, C_bottom[2] + cyl_ht * 0.5 + cyl_radius * diag_factor),  # 左上前
        (cyl_radius * diag_factor, cyl_y, C_bottom[2] + cyl_ht * 0.5 - cyl_radius * diag_factor),  # 右上后
        (-cyl_radius * diag_factor, cyl_y, C_bottom[2] + cyl_ht * 0.5 - cyl_radius * diag_factor)  # 左上后
    ])

    # 轴线点采样
    key_rays.extend([
        C_bottom,  # 底部中心
        C_top,  # 顶部中心
        (0.0, cyl_y, C_bottom[2] + cyl_ht * 0.25),  # 1/4高度轴线
        (0.0, cyl_y, C_bottom[2] + cyl_ht * 0.5),  # 1/2高度轴线
        (0.0, cyl_y, C_bottom[2] + cyl_ht * 0.75)  # 3/4高度轴线
    ])

    # 锥轴向量（L→C_bottom）与长度
    cone_axis = vector_subtract(C_bottom, light_pos)
    cone_axis_len = vector_length(cone_axis)
    if cone_axis_len < 1e-12:  # 光源与底部中心重合，特殊处理
        return is_smen_it_sphe_anic(light_pos, C_top, sphe_pos_ans, R)

    # 判断所有关键射线是否都被球体遮挡
    for target_point in key_rays:
        if not is_smen_it_sphe_anic(light_pos, target_point, sphe_pos_ans, R):
            return False  # 存在未被遮挡的射线，不是完全遮挡

    # 所有关键射线均被遮挡 → 圆柱完全被遮挡
    return True


def cate_det_pos(vx, vy, t_release, t_delay, g=-9.8):
    """计算烟幕弹投放点+起爆点"""
    s0_x, s0_y, s0_z = 17800.0, 0.0, 1800.0
    vz = 0.0  # 等高飞行

    # 投放点位置
    release_x = s0_x + vx * t_release
    release_y = s0_y + vy * t_release
    release_z = s0_z + vz * t_release

    # 起爆点位置（水平匀速，竖直重力）
    det_x = release_x + vx * t_delay
    det_y = release_y + vy * t_delay
    det_z = release_z + vz * t_delay + 0.5 * g * t_delay ** 2
    t_det = t_release + t_delay

    return (release_x, release_y, release_z), (t_det, det_x, det_y, det_z)


def find_trans(start, end, t_det, x_det, y_det, z_det, R, v_sink, target_state):
    """二分法查找遮蔽状态转换时间"""
    eps = 1e-9
    for _ in range(50):
        mid = (start + end) / 2
        cur_state = is_cyer_ful_oced(mid, t_det, x_det, y_det, z_det, R, v_sink)
        if cur_state == target_state:
            end = mid
        else:
            start = mid
        if end - start < eps:
            break
    return (start + end) / 2


def cate_se_ocion_t(t_det, x_det, y_det, z_det, R, v_sink):
    """计算单个烟幕云团的有效遮蔽时间"""
    cud_start = t_det
    cud_end = t_det + 20.0
    tol_t = 0.0
    ocion_ival = []

    # 快速判断：中间采样点无遮蔽则跳过
    sample_t = (cud_start + cud_end) / 2
    if not is_cyer_ful_oced(sample_t, t_det, x_det, y_det, z_det, R, v_sink):
        if not is_cyer_ful_oced(cud_start, t_det, x_det, y_det, z_det, R, v_sink) and \
                not is_cyer_ful_oced(cud_end, t_det, x_det, y_det, z_det, R, v_sink):
            return 0.0, []

    # 粗查遮蔽区间（步长0.1s）
    pre_oced = is_cyer_ful_oced(cud_start, t_det, x_det, y_det, z_det, R, v_sink)
    inval_start = cud_start if pre_oced else None

    t = cud_start + 0.1
    while t <= cud_end:
        cur_oced = is_cyer_ful_oced(t, t_det, x_det, y_det, z_det, R, v_sink)
        if cur_oced != pre_oced:
            # 精确查找转换时间
            tron_t = find_trans(t - 0.1, t, t_det, x_det, y_det, z_det, R, v_sink, cur_oced)
            if pre_oced:
                ocion_ival.append((inval_start, tron_t))
                tol_t += tron_t - inval_start
                inval_start = None
            else:
                inval_start = tron_t
            pre_oced = cur_oced
        t += 0.1

    # 处理云团结束时仍遮蔽的情况
    if pre_oced and inval_start is not None:
        ocion_ival.append((inval_start, cud_end))
        tol_t += cud_end - inval_start

    return max(tol_t, 0.0), ocion_ival


def merge_intervals(intervals):
    """合并重叠时间区间"""
    if not intervals:
        return 0.0, []
    sd_inval = sorted(intervals, key=lambda x: x[0])
    merged = [sd_inval[0]]
    for cur in sd_inval[1:]:
        last = merged[-1]
        if cur[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], cur[1]))
        else:
            merged.append(cur)
    total_time = sum(end - start for start, end in merged)
    return total_time, merged


# 输出烟幕干扰弹详细信息
def print_smoke_bomb_info(decision_params):
    """输出每枚烟幕弹的详细参数信息"""
    vx, vy, t1, d1, t2, d2, t3, d3 = decision_params
    vz = 0.0
    v_mag = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    print("烟幕干扰弹详细参数信息")
    print(f"无人机速度：{v_mag:.2f}m/s,无人机方向：{180 + np.arctan(vy / vx) * 180 / np.pi:.2f}°")
    print("\n各烟幕弹参数：")

    # 计算第一枚烟幕弹信息
    (release_x1, release_y1, release_z1), (t_det1, x1, y1, z1) = cate_det_pos(vx, vy, t1, d1)
    _, intervals1 = cate_se_ocion_t(t_det1, x1, y1, z1, SMOKE_BASE_PARAMS["R"], SMOKE_BASE_PARAMS["v_sink"])
    smoke_time1 = sum(end - start for start, end in intervals1) if intervals1 else 0

    print("\n【第1枚烟幕弹】")
    print(f"投放时间：{t1:.4f}s")
    print(f"投放位置：x={release_x1:.2f}m, y={release_y1:.2f}m, z={release_z1:.2f}m")
    print(f"起爆延迟：{d1:.4f}s")
    print(f"起爆时间：{t_det1:.4f}s")
    print(f"起爆位置：x={x1:.2f}m, y={y1:.2f}m, z={z1:.2f}m")
    print(f"有效遮蔽时间：{smoke_time1:.2f}s")
    if intervals1:
        print(f"遮蔽时间段：{[(round(s, 2), round(e, 2)) for s, e in intervals1]}")

    # 计算第二枚烟幕弹信息
    (release_x2, release_y2, release_z2), (t_det2, x2, y2, z2) = cate_det_pos(vx, vy, t2, d2)
    _, intervals2 = cate_se_ocion_t(t_det2, x2, y2, z2, SMOKE_BASE_PARAMS["R"], SMOKE_BASE_PARAMS["v_sink"])
    smoke_time2 = sum(end - start for start, end in intervals2) if intervals2 else 0

    print("\n【第2枚烟幕弹】")
    print(f"投放时间：{t2:.4f}s")
    print(f"与前一枚投放间隔：{t2 - t1:.4f}s")
    print(f"投放位置：x={release_x2:.2f}m, y={release_y2:.2f}m, z={release_z2:.2f}m")
    print(f"起爆延迟：{d2:.4f}s")
    print(f"起爆时间：{t_det2:.4f}s")
    print(f"起爆位置：x={x2:.2f}m, y={y2:.2f}m, z={z2:.2f}m")
    print(f"有效遮蔽时间：{smoke_time2:.2f}s")
    if intervals2:
        print(f"遮蔽时间段：{[(round(s, 2), round(e, 2)) for s, e in intervals2]}")

    # 计算第三枚烟幕弹信息
    (release_x3, release_y3, release_z3), (t_det3, x3, y3, z3) = cate_det_pos(vx, vy, t3, d3)
    _, intervals3 = cate_se_ocion_t(t_det3, x3, y3, z3, SMOKE_BASE_PARAMS["R"], SMOKE_BASE_PARAMS["v_sink"])
    smoke_time3 = sum(end - start for start, end in intervals3) if intervals3 else 0

    print("\n【第3枚烟幕弹】")
    print(f"投放时间：{t3:.4f}s")
    print(f"与前一枚投放间隔：{t3 - t2:.4f}s")
    print(f"投放位置：x={release_x3:.2f}m, y={release_y3:.2f}m, z={release_z3:.2f}m")
    print(f"起爆延迟：{d3:.4f}s")
    print(f"起爆时间：{t_det3:.4f}s")
    print(f"起爆位置：x={x3:.2f}m, y={y3:.2f}m, z={z3:.2f}m")
    print(f"有效遮蔽时间：{smoke_time3:.2f}s")
    if intervals3:
        print(f"遮蔽时间段：{[(round(s, 2), round(e, 2)) for s, e in intervals3]}")

    # 总遮蔽时间
    all_intervals = intervals1 + intervals2 + intervals3
    total_time, _ = merge_intervals(all_intervals)
    print(f"总有效遮蔽时间：{total_time:.2f}s")


# PSO优化类
class PSO_Optimizer_ThreeBombs:
    def __init__(self, pop_size=100, max_iter=200, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0):
        self.bounds = np.array([
            [-140.0, 0.0],  # vx
            [0.0, 140.0],  # vy
            [0.001, 15.0],  # t1
            [0.01, 15.0],  # d1
            [1.001, 15.0],  # t2（≥t1+1）
            [0.01, 15.0],  # d2
            [2.001, 20.0],  # t3（≥t2+1）
            [0.01, 15.0]  # d3
        ])
        self.dim = self.bounds.shape[0]
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1

        # 初始化种群
        self.pos = self._init_population()
        self.vel = np.random.uniform(-self.v_max, self.v_max, (self.pop_size, self.dim))

        # 初始化最优解
        self.p_best_pos = self.pos.copy()
        self.p_best_fit = np.array([self.calculate_fitness(p) for p in self.pos])
        self.g_best_idx = np.argmax(self.p_best_fit)
        self.g_best_pos = self.p_best_pos[self.g_best_idx].copy()
        self.g_best_fit = self.p_best_fit[self.g_best_idx]
        self.fit_history = [self.g_best_fit]

    def _init_population(self):
        """种群初始化（以PROB1_PARAMS为中心）"""
        pop = np.zeros((self.pop_size, self.dim))
        base_vx = PROB1_PARAMS["vx"]
        base_vy = PROB1_PARAMS["vy"]
        base_t = PROB1_PARAMS["t_release"]
        base_d = PROB1_PARAMS["t_delay"]

        for i in range(self.pop_size):
            t1 = base_t + np.random.uniform(-0.5, 0.5)
            t2 = t1 + 1.0 + np.random.uniform(0, 1.0)
            t3 = t2 + 1.0 + np.random.uniform(0, 1.0)
            pop[i] = [
                base_vx + np.random.uniform(-10.0, 10.0),
                base_vy + np.random.uniform(-15.0, 15.0),
                t1, base_d + np.random.uniform(-1.0, 1.0),
                t2, base_d + np.random.uniform(-1.0, 1.0),
                t3, base_d + np.random.uniform(-1.0, 1.0)
            ]
        return np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])

    def calculate_fitness(self, particle):
        """计算适应度（使用基础烟幕参数）"""
        vx, vy, t1, d1, t2, d2, t3, d3 = particle
        vz = 0.0
        R = SMOKE_BASE_PARAMS["R"]
        v_sink = SMOKE_BASE_PARAMS["v_sink"]

        # 约束1：速度70~140m/s
        v_mag = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        if v_mag < 70.0 - 1e-9 or v_mag > 140.0 + 1e-9:
            return 1e-6

        # 约束2：投放间隔≥1s
        if t2 - t1 < 1.0 - 1e-9 or t3 - t2 < 1.0 - 1e-9:
            return 1e-6

        # 计算起爆参数
        try:
            _, (t_det1, x1, y1, z1) = cate_det_pos(vx, vy, t1, d1)
            _, (t_det2, x2, y2, z2) = cate_det_pos(vx, vy, t2, d2)
            _, (t_det3, x3, y3, z3) = cate_det_pos(vx, vy, t3, d3)
        except:
            return 1e-6

        # 约束3：起爆点z坐标50~2000m
        for z in [z1, z2, z3]:
            if z < 50.0 or z > 2000.0:
                return 1e-6

        # 计算遮蔽时间
        _, intervals1 = cate_se_ocion_t(t_det1, x1, y1, z1, R, v_sink)
        _, intervals2 = cate_se_ocion_t(t_det2, x2, y2, z2, R, v_sink)
        _, intervals3 = cate_se_ocion_t(t_det3, x3, y3, z3, R, v_sink)
        all_intervals = intervals1 + intervals2 + intervals3
        total_time, _ = merge_intervals(all_intervals)

        return total_time if total_time >= 1e-3 else 1e-6

    def update_velocity(self, iter_idx):
        """更新速度（惯性权重线性递减）"""
        w = self.w_start - (self.w_start - self.w_end) * (iter_idx / self.max_iter)
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        self.vel = w * self.vel + self.c1 * r1 * (self.p_best_pos - self.pos) + self.c2 * r2 * (
                self.g_best_pos - self.pos)
        self.vel = np.clip(self.vel, -self.v_max, self.v_max)

    def update_position(self):
        """更新位置（确保约束）"""
        self.pos = self.pos + self.vel
        self.pos = np.clip(self.pos, self.bounds[:, 0], self.bounds[:, 1])
        # 确保投放间隔≥1s
        for i in range(self.pop_size):
            if self.pos[i, 4] - self.pos[i, 2] < 1.0:
                self.pos[i, 4] = self.pos[i, 2] + 1.0 + 1e-9
            if self.pos[i, 6] - self.pos[i, 4] < 1.0:
                self.pos[i, 6] = self.pos[i, 4] + 1.0 + 1e-9

    def update_best(self):
        """更新最优解"""
        cur_fit = np.array([self.calculate_fitness(p) for p in self.pos])
        mask = cur_fit > self.p_best_fit
        self.p_best_pos[mask] = self.pos[mask].copy()
        self.p_best_fit[mask] = cur_fit[mask]
        current_g_best_idx = np.argmax(self.p_best_fit)
        current_g_best_fit = self.p_best_fit[current_g_best_idx]
        if current_g_best_fit > self.g_best_fit:
            self.g_best_fit = current_g_best_fit
            self.g_best_pos = self.p_best_pos[current_g_best_idx].copy()
        self.fit_history.append(self.g_best_fit)

    def plot_convergence(self, save_path="问题3_PSO收敛曲线.png"):
        """绘制PSO收敛曲线"""
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.fit_history)), self.fit_history, 'b-', linewidth=2, label='全局最优遮蔽时间')
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('全局最优遮蔽时间 (s)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        final_fitness = self.fit_history[-1]
        plt.axhline(y=final_fitness, color='r', linestyle='--', linewidth=2, label=f'最终最优{final_fitness:.2f}s',
                    alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def optimize(self):
        """执行PSO优化"""
        print("PSO优化：3个烟幕弹投放策略")
        print(f"种群规模：{self.pop_size} | 最大迭代：{self.max_iter} | 速度约束：70~140m/s")
        print(f"基础烟幕参数：R={SMOKE_BASE_PARAMS['R']}m, v_sink={SMOKE_BASE_PARAMS['v_sink']}m/s")

        for iter_idx in range(self.max_iter):
            self.update_velocity(iter_idx)
            self.update_position()
            self.update_best()
            if (iter_idx + 1) % 30 == 0:
                print(f"迭代{iter_idx + 1:3d}/{self.max_iter} | 全局最优遮蔽时间：{self.g_best_fit:.2f}s")

        print(f"优化结束！最终最优遮蔽时间：{self.g_best_fit:.6f}s")
        self.plot_convergence()
        return self.g_best_pos, self.g_best_fit


# 敏感性分析核心模块
def calculate_total_smoke_time(decision_params, R, v_sink):
    """根据决策参数、R、v_sink计算总遮蔽时间"""
    vx, vy, t1, d1, t2, d2, t3, d3 = decision_params
    vz = 0.0

    # 1. 计算3个烟幕弹的起爆参数
    _, (t_det1, x1, y1, z1) = cate_det_pos(vx, vy, t1, d1)
    _, (t_det2, x2, y2, z2) = cate_det_pos(vx, vy, t2, d2)
    _, (t_det3, x3, y3, z3) = cate_det_pos(vx, vy, t3, d3)

    # 2. 计算每个烟幕弹的遮蔽区间
    _, intervals1 = cate_se_ocion_t(t_det1, x1, y1, z1, R, v_sink)
    _, intervals2 = cate_se_ocion_t(t_det2, x2, y2, z2, R, v_sink)
    _, intervals3 = cate_se_ocion_t(t_det3, x3, y3, z3, R, v_sink)

    # 3. 合并区间并计算总时间
    all_intervals = intervals1 + intervals2 + intervals3
    total_time, _ = merge_intervals(all_intervals)
    return max(total_time, 0.0)  # 确保非负


def sensitivity_analysis(decision_params, base_R, base_v_sink, perturb_ratios=[-0.1, -0.05, 0.05, 0.1]):
    """
    敏感性分析：计算R和v_sink的扰动对遮蔽时间的影响
    """
    # 1. 计算基准遮蔽时间（原始R和v_sink）
    base_total_time = calculate_total_smoke_time(decision_params, base_R, base_v_sink)
    print(f"\n【敏感性分析基准】")
    print(f"基准烟幕半径 R0 = {base_R}m | 基准下沉速度 v0 = {base_v_sink}m/s")
    print(f"基准总遮蔽时间 F0 = {base_total_time:.2f}s")

    # 2. 初始化结果存储
    result = {
        "R": {"perturb_ratio": [], "perturb_value": [], "total_time": [], "sensitivity": []},
        "v_sink": {"perturb_ratio": [], "perturb_value": [], "total_time": [], "sensitivity": []}
    }

    # 3. 烟幕半径R的敏感性分析
    print("\n【1. 烟幕半径R敏感性分析】")
    for ratio in perturb_ratios:
        perturb_R = base_R * (1 + ratio)
        if perturb_R <= 0:  # 半径不能为负
            print(f"扰动比例 {ratio:.1%}：R={perturb_R:.2f}m（无效，跳过）")
            continue
        # 计算扰动后的遮蔽时间
        perturb_time = calculate_total_smoke_time(decision_params, perturb_R, base_v_sink)
        # 计算相对敏感性系数：S = (ΔF/F0) / (ΔR/R0)
        delta_F = perturb_time - base_total_time
        delta_R = perturb_R - base_R
        sensitivity = (delta_F / base_total_time) / (delta_R / base_R) if (
                base_total_time > 1e-9 and delta_R > 1e-9) else 0.0
        # 存储结果
        result["R"]["perturb_ratio"].append(ratio)
        result["R"]["perturb_value"].append(perturb_R)
        result["R"]["total_time"].append(perturb_time)
        result["R"]["sensitivity"].append(sensitivity)
        # 打印中间结果
        print(f"扰动比例 {ratio:.1%}：R={perturb_R:.2f}m | 遮蔽时间={perturb_time:.2f}s | 敏感性系数={sensitivity:.3f}")

    # 4. 烟幕下沉速度v_sink的敏感性分析
    print("\n【2. 烟幕下沉速度v_sink敏感性分析】")
    for ratio in perturb_ratios:
        perturb_v = base_v_sink * (1 + ratio)
        if perturb_v <= 0:  # 下沉速度不能为负
            print(f"扰动比例 {ratio:.1%}：v_sink={perturb_v:.2f}m/s（无效，跳过）")
            continue
        # 计算扰动后的遮蔽时间
        perturb_time = calculate_total_smoke_time(decision_params, base_R, perturb_v)
        # 计算相对敏感性系数：S = (ΔF/F0) / (Δv/v0)
        delta_F = perturb_time - base_total_time
        delta_v = perturb_v - base_v_sink
        sensitivity = (delta_F / base_total_time) / (delta_v / base_v_sink) if (
                base_total_time > 1e-9 and delta_v > 1e-9) else 0.0
        # 存储结果
        result["v_sink"]["perturb_ratio"].append(ratio)
        result["v_sink"]["perturb_value"].append(perturb_v)
        result["v_sink"]["total_time"].append(perturb_time)
        result["v_sink"]["sensitivity"].append(sensitivity)
        # 打印中间结果
        print(
            f"扰动比例 {ratio:.1%}：v_sink={perturb_v:.2f}m/s | 遮蔽时间={perturb_time:.2f}s | 敏感性系数={sensitivity:.3f}")

    return result, base_total_time


def plot_sensitivity_result(result, base_total_time, base_R, base_v_sink):
    """绘制敏感性分析可视化结果"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1：扰动幅度 vs 总遮蔽时间
    # 烟幕半径R
    R_ratios = [r * 100 for r in result["R"]["perturb_ratio"]]  # 转为百分比
    R_times = result["R"]["total_time"]
    ax1.plot(R_ratios, R_times, 'o-', color='darkred', linewidth=2, markersize=6, label=f'烟幕半径R（基准={base_R}m）')
    # 烟幕下沉速度v_sink
    v_ratios = [r * 100 for r in result["v_sink"]["perturb_ratio"]]
    v_times = result["v_sink"]["total_time"]
    ax1.plot(v_ratios, v_times, 's-', color='darkblue', linewidth=2, markersize=6,
             label=f'下沉速度v_sink（基准={base_v_sink}m/s）')
    # 基准线
    ax1.axhline(y=base_total_time, color='gray', linestyle='--', linewidth=1,
                label=f'基准遮蔽时间={base_total_time:.2f}s')
    # 标签
    ax1.set_xlabel('参数扰动幅度（%）', fontsize=12)
    ax1.set_ylabel('总遮蔽时间（s）', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # 子图2：参数 vs 敏感性系数
    # 烟幕半径R
    R_sens = result["R"]["sensitivity"]
    ax2.bar([f'R_{r * 100:.0f}%' for r in result["R"]["perturb_ratio"]], R_sens, color='darkred', alpha=0.7,
            label='烟幕半径R')
    # 烟幕下沉速度v_sink
    v_sens = result["v_sink"]["sensitivity"]
    ax2.bar([f'v_{r * 100:.0f}%' for r in result["v_sink"]["perturb_ratio"]], v_sens, color='darkblue', alpha=0.7,
            label='下沉速度v_sink')
    # 敏感阈值线（|S|>1为敏感，|S|<0.3为稳健）
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='敏感阈值（S=1）')
    ax2.axhline(y=-1.0, color='red', linestyle='--', linewidth=1)
    ax2.axhline(y=0.3, color='green', linestyle='--', linewidth=1, label='稳健阈值（S=0.3）')
    ax2.axhline(y=-0.3, color='green', linestyle='--', linewidth=1)
    # 标签
    ax2.set_xlabel('参数及扰动幅度', fontsize=12)
    ax2.set_ylabel('相对敏感性系数', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax2.legend()
    # 旋转x轴标签
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("问题3_PSO敏感性分析图（烟幕半径R与下沉速度）.png", dpi=300, bbox_inches='tight')
    plt.show()


# 执行优化+敏感性分析
def main():
    # 步骤1：执行PSO优化，获取最优决策参数
    pso = PSO_Optimizer_ThreeBombs(
        pop_size=30,  # 种群规模
        max_iter=150,  # 迭代次数
        w_start=0.9,  # 初始惯性权重
        w_end=0.3,  # 终止惯性权重
        c1=1.7,  # 个体学习因子
        c2=1.7  # 全局学习因子
    )
    best_decision_params, best_base_time = pso.optimize()

    # 新增步骤：输出完整的烟幕干扰弹信息
    print_smoke_bomb_info(best_decision_params)

    # 步骤2：执行敏感性分析（扰动比例±5%、±10%）
    base_R = SMOKE_BASE_PARAMS["R"]
    base_v_sink = SMOKE_BASE_PARAMS["v_sink"]
    sens_result, base_total_time = sensitivity_analysis(
        decision_params=best_decision_params,
        base_R=base_R,
        base_v_sink=base_v_sink,
        perturb_ratios=[-0.1, -0.05, 0.05, 0.1]  # ±10%、±5%扰动
    )

    # 步骤3：绘制敏感性分析结果
    plot_sensitivity_result(sens_result, base_total_time, base_R, base_v_sink)

    # 步骤4：输出最终结论
    print("【敏感性分析结论】")
    # 烟幕半径R结论
    R_sens_avg = np.mean(np.abs(sens_result["R"]["sensitivity"]))
    R_level = "高度敏感" if R_sens_avg > 1.0 else "中等敏感" if R_sens_avg > 0.3 else "低敏感"
    print(f"1. 烟幕半径R：平均敏感性系数={R_sens_avg:.3f} → {R_level}")
    print(f"   （R增大1%，遮蔽时间约增加{R_sens_avg:.3f}%")
    # 下沉速度v_sink结论
    v_sens_avg = np.mean(np.abs(sens_result["v_sink"]["sensitivity"]))
    v_level = "高度敏感" if v_sens_avg > 1.0 else "中等敏感" if v_sens_avg > 0.3 else "低敏感"
    print(f"2. 烟幕下沉速度v_sink：平均敏感性系数={v_sens_avg:.3f} → {v_level}")
    print(f"   （v_sink增大1%，遮蔽时间约减少{abs(v_sens_avg):.3f}%")


if __name__ == "__main__":
    main()
