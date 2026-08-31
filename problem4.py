import math
import numpy as np
import matplotlib.pyplot as plt

# 基础参数
np.random.seed(31415)
# 无人机初始位置
UAV_INIT_POS = {
    "FY1": (17800.0, 0.0, 1800.0),
    "FY2": (12000.0, 1400.0, 1400.0),
    "FY3": (6000.0, -3000.0, 700.0)
}
# 初始解参数
INITIAL_SOLUTIONS = {
    "FY2": {
        "vx": -32.00,
        "vy": -83.25,
        "tr": 9.21,
        "td": 6.76
    },  # 旧优化结果
    "FY3": {
        "vx": -47.40,
        "vy": 119.04,
        "tr": 19.10,
        "td": 6.27
    }  # 旧优化结果
}
# 旧优化结果
PROB1_PARAMS = {
    "vx": -129.32,
    "vy": 2.69,
    "t_release": 0.001,
    "t_delay": 3.40
}
# M1导弹与目标参数
M1_PARAMS = {
    "init_pos": (20000.0, 0.0, 2000.0), "speed": 300.0,
    "direction": (-20000.0, 0.0, -2000.0)
}
TARGET_PARAMS = {
    "cyl_radius": 7.0, "cyl_y": 200.0, "cyl_z_range": [0.0, 10.0],
    "sphere_radius": 10.0
}

# 无人机速度约束参数
UAV_SPEED_CONSTRAINTS = {
    "min": 70.0,
    "max": 140.0
}


# 工具函数
def vector_length(v):
    x, y, z = v
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


def dot_product(v1, v2):
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return x1 * x2 + y1 * y2 + z1 * z2


def vector_subtract(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


def get_pt_lt_pos(t, missile_speed=None):
    """计算t时刻M1位置，可指定导弹速度"""
    speed = missile_speed if missile_speed is not None else M1_PARAMS["speed"]
    dir_length = vector_length(M1_PARAMS["direction"])
    unit_dir = (
        M1_PARAMS["direction"][0] / dir_length,
        M1_PARAMS["direction"][1] / dir_length,
        M1_PARAMS["direction"][2] / dir_length
    )
    dis = speed * t
    if dis >= dir_length:
        return (0.0, 0.0, 0.0)
    return (
        M1_PARAMS["init_pos"][0] + unit_dir[0] * dis,
        M1_PARAMS["init_pos"][1] + unit_dir[1] * dis,
        M1_PARAMS["init_pos"][2] + unit_dir[2] * dis
    )


def sphe_pos(t, t_det, x_det, y_det, z_det):
    if t < t_det - 1e-9 or t > t_det + 20 + 1e-9:
        return None
    z_pos = z_det - 3.0 * (t - t_det)
    return (float(x_det), float(y_det), float(z_pos))


def is_smen_it_sphe_anic(L, P, S, R):
    LP = vector_subtract(P, L)
    LS = vector_subtract(L, S)
    a = dot_product(LP, LP)
    b = 2 * dot_product(LS, LP)
    c = dot_product(LS, LS) - R ** 2
    dis_crnt = b ** 2 - 4 * a * c

    if a < 1e-12:
        return vector_length(LS) <= R + 1e-9

    if dis_crnt < -1e-9:
        return False
    dis_crnt = max(dis_crnt, 0.0)
    sqrt_d = math.sqrt(dis_crnt)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    return (t1 >= -1e-9 and t1 <= 1.0 + 1e-9) or (t2 >= -1e-9 and t2 <= 1.0 + 1e-9)


def is_cyer_ful_oced(t, t_det, x_det, y_det, z_det, missile_speed=None):
    light_pos = get_pt_lt_pos(t, missile_speed)
    sphe_pos_ans = sphe_pos(t, t_det, x_det, y_det, z_det)

    # 球体不存在时，圆柱不可能被完全遮挡
    if sphe_pos_ans is None:
        return False

    # 圆柱与球体参数
    cyl_rd = TARGET_PARAMS["cyl_radius"]
    cyl_ht = TARGET_PARAMS["cyl_z_range"][1] - TARGET_PARAMS["cyl_z_range"][0]
    cyl_y = TARGET_PARAMS["cyl_y"]
    she_rd = TARGET_PARAMS["sphere_radius"]

    # 快速判断：光源在球体内，所有光线都被遮挡
    LS = vector_subtract(light_pos, sphe_pos_ans)
    if vector_length(LS) <= she_rd + 1e-9:
        return True

    # 圆柱关键特征点
    C_bottom = (0.0, cyl_y, TARGET_PARAMS["cyl_z_range"][0])  # 底部中心
    C_top = (0.0, cyl_y, TARGET_PARAMS["cyl_z_range"][1])  # 顶部中心

    key_rays = []

    # 顶部圆周采样（4个方向）
    key_rays.extend([
        (cyl_rd, cyl_y, C_top[2]),  # 顶部右
        (-cyl_rd, cyl_y, C_top[2]),  # 顶部左
        (0, cyl_y, C_top[2] + cyl_rd),  # 顶部前
        (0, cyl_y, C_top[2] - cyl_rd)  # 顶部后
    ])

    # 底部圆周采样（4个方向）
    key_rays.extend([
        (cyl_rd, cyl_y, C_bottom[2]),  # 底部右
        (-cyl_rd, cyl_y, C_bottom[2]),  # 底部左
        (0, cyl_y, C_bottom[2] + cyl_rd),  # 底部前
        (0, cyl_y, C_bottom[2] - cyl_rd)  # 底部后
    ])

    # 中间高度采样（3个高度层级，每个层级4个方向）
    heights = [
        C_bottom[2] + cyl_ht * 0.25,
        C_bottom[2] + cyl_ht * 0.5,
        C_bottom[2] + cyl_ht * 0.75
    ]
    for h in heights:
        key_rays.extend([
            (cyl_rd, cyl_y, h),  # 右
            (-cyl_rd, cyl_y, h),  # 左
            (0, cyl_y, h + cyl_rd),  # 前
            (0, cyl_y, h - cyl_rd)  # 后
        ])

    # 对角线方向采样（45度角）
    diag_factor = math.cos(math.pi / 4)  # 对角线位置因子
    key_rays.extend([
        (cyl_rd * diag_factor, cyl_y, C_bottom[2] + cyl_ht * 0.5 + cyl_rd * diag_factor),  # 右上前
        (-cyl_rd * diag_factor, cyl_y, C_bottom[2] + cyl_ht * 0.5 + cyl_rd * diag_factor),  # 左上前
        (cyl_rd * diag_factor, cyl_y, C_bottom[2] + cyl_ht * 0.5 - cyl_rd * diag_factor),  # 右上后
        (-cyl_rd * diag_factor, cyl_y, C_bottom[2] + cyl_ht * 0.5 - cyl_rd * diag_factor)  # 左上后
    ])

    # 轴线点采样
    key_rays.extend([
        C_bottom,  # 底部中心
        C_top,  # 顶部中心
        (0.0, cyl_y, C_bottom[2] + cyl_ht * 0.25),  # 1/4高度轴线
        (0.0, cyl_y, C_bottom[2] + cyl_ht * 0.5),  # 1/2高度轴线
        (0.0, cyl_y, C_bottom[2] + cyl_ht * 0.75)  # 3/4高度轴线
    ])

    # 判断所有关键射线是否都被球体遮挡
    for target_point in key_rays:
        if not is_smen_it_sphe_anic(light_pos, target_point, sphe_pos_ans, she_rd):
            return False  # 存在未被遮挡的射线，不是完全遮挡

    # 所有关键射线均被遮挡 → 圆柱完全被遮挡
    return True


def cate_det_pos(uav_id, vx, vy, t_release, t_delay, g=-9.8):
    s0_x, s0_y, s0_z = UAV_INIT_POS[uav_id]
    vz = 0.0

    # 投放点计算
    release_x = s0_x + vx * t_release
    release_y = s0_y + vy * t_release
    release_z = s0_z + vz * t_release

    # 起爆点计算
    det_x = release_x + vx * t_delay
    det_y = release_y + vy * t_delay
    det_z = release_z + vz * t_delay + 0.5 * g * t_delay ** 2
    t_det = t_release + t_delay

    return (release_x, release_y, release_z), (t_det, det_x, det_y, det_z)


# 遮蔽时间计算函数
def find_trans(start, end, t_det, x_det, y_det, z_det, target_state, missile_speed=None):
    eps = 1e-9
    for _ in range(50):
        mid = (start + end) / 2
        current_state = is_cyer_ful_oced(mid, t_det, x_det, y_det, z_det, missile_speed)
        if current_state == target_state:
            end = mid
        else:
            start = mid
        if end - start < eps:
            break
    return (start + end) / 2


def cate_se_ocion_t(t_det, x_det, y_det, z_det, missile_speed=None):
    cloud_start = t_det
    cloud_end = t_det + 20.0
    total_time = 0.0
    ocion_inval = []

    # 初始状态检查
    initial_check_t = max(cloud_start, t_det + 0.4)  # 稍微延迟一点检查，避开初始状态
    pre_oced = is_cyer_ful_oced(initial_check_t, t_det, x_det, y_det, z_det, missile_speed)
    inval_start = None

    # 回溯查找初始完全遮挡开始点
    if pre_oced:
        inval_start = find_trans(cloud_start, initial_check_t,
                                 t_det, x_det, y_det, z_det, True, missile_speed)

    t = initial_check_t + 0.1
    while t <= cloud_end:
        cur_oced = is_cyer_ful_oced(t, t_det, x_det, y_det, z_det, missile_speed)

        if cur_oced != pre_oced:
            # 找到状态转换点
            transition_t = find_trans(t - 0.1, t,
                                      t_det, x_det, y_det, z_det, cur_oced, missile_speed)
            if pre_oced:
                ocion_inval.append((round(inval_start, 2), round(transition_t, 2)))
                total_time += transition_t - inval_start
                inval_start = None
            else:
                inval_start = transition_t
            pre_oced = cur_oced

        t += 0.1

    # 处理结束时仍处于完全遮挡状态的情况
    if pre_oced and inval_start is not None:
        ocion_inval.append((round(inval_start, 2), round(cloud_end, 2)))
        total_time += cloud_end - inval_start

    return max(total_time, 0.0), ocion_inval


def merge_inval(intervals):
    if not intervals:
        return 0.0, []

    sorted_inval = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_inval[0]]

    for cur in sorted_inval[1:]:
        last = merged[-1]
        if cur[0] <= last[1] + 1e-9:
            merged[-1] = (last[0], max(last[1], cur[1]))
        else:
            merged.append(cur)

    total_time = sum(end - start for start, end in merged)
    return total_time, merged


# PSO优化类
class PSO_Optimizer_ThreeUAVs:
    def __init__(self, pop_size=60, max_iter=200, w_start=0.9, w_end=0.4, c1=2.0, c2=2.0,
                 uav_speed_max=None, missile_speed=None):
        # 可指定无人机速度上限和导弹速度（用于灵敏度分析）
        self.uav_speed_max = uav_speed_max if uav_speed_max is not None else UAV_SPEED_CONSTRAINTS["max"]
        self.missile_speed = missile_speed if missile_speed is not None else M1_PARAMS["speed"]

        self.bounds = np.array([
            [-140.0, -120.0],  # FY1_vx
            [0.0, 10.0],  # FY1_vy
            [0.0, 2.0],  # FY1_tr
            [2.0, 8.0],  # FY1_td
            [-50.0, -20.0],  # FY2_vx
            [-110.0, -60.0],  # FY2_vy
            [6.0, 15.0],  # FY2_tr
            [6.0, 15.0],  # FY2_td
            [-70.0, -40.0],  # FY3_vx
            [110.0, 140.0],  # FY3_vy
            [15.0, 25.0],  # FY3_tr
            [4.0, 8.0]  # FY3_td
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
        self.vel = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.pop_size, self.dim))

        self.p_best_pos = self.pos.copy()
        self.p_best_fit = np.array([self.calculate_fitness(p) for p in self.pos])
        self.g_best_idx = np.argmax(self.p_best_fit)
        self.g_best_pos = self.p_best_pos[self.g_best_idx].copy()
        self.g_best_fit = self.p_best_fit[self.g_best_idx]
        self.fit_history = [self.g_best_fit]

    def _init_population(self):
        """种群初始化，使用新的优化结果作为起点"""
        pop = np.zeros((self.pop_size, self.dim))

        # 第一个粒子使用最新优化结果
        pop[0] = [
            PROB1_PARAMS["vx"], PROB1_PARAMS["vy"], PROB1_PARAMS["t_release"], PROB1_PARAMS["t_delay"],
            INITIAL_SOLUTIONS["FY2"]["vx"], INITIAL_SOLUTIONS["FY2"]["vy"],
            INITIAL_SOLUTIONS["FY2"]["tr"], INITIAL_SOLUTIONS["FY2"]["td"],
            INITIAL_SOLUTIONS["FY3"]["vx"], INITIAL_SOLUTIONS["FY3"]["vy"],
            INITIAL_SOLUTIONS["FY3"]["tr"], INITIAL_SOLUTIONS["FY3"]["td"]
        ]

        # 剩余粒子：基于新初始解进行随机扰动
        for i in range(1, self.pop_size):
            # FY1参数（基于新优化结果）
            pop[i, 0] = PROB1_PARAMS["vx"] + np.random.uniform(-5.0, 5.0)  # 缩小扰动范围，因为已有较好解
            pop[i, 1] = PROB1_PARAMS["vy"] + np.random.uniform(-5.0, 5.0)
            pop[i, 2] = PROB1_PARAMS["t_release"] + np.random.uniform(-0.5, 0.5)
            pop[i, 3] = PROB1_PARAMS["t_delay"] + np.random.uniform(-0.5, 0.5)

            # FY2参数（基于新优化结果）
            pop[i, 4] = INITIAL_SOLUTIONS["FY2"]["vx"] + np.random.uniform(-3.0, 3.0)
            pop[i, 5] = INITIAL_SOLUTIONS["FY2"]["vy"] + np.random.uniform(-3.0, 3.0)
            pop[i, 6] = INITIAL_SOLUTIONS["FY2"]["tr"] + np.random.uniform(-0.5, 0.5)
            pop[i, 7] = INITIAL_SOLUTIONS["FY2"]["td"] + np.random.uniform(-0.3, 0.3)

            # FY3参数（基于新优化结果）
            pop[i, 8] = INITIAL_SOLUTIONS["FY3"]["vx"] + np.random.uniform(-2.0, 2.0)
            pop[i, 9] = INITIAL_SOLUTIONS["FY3"]["vy"] + np.random.uniform(-2.0, 2.0)
            pop[i, 10] = INITIAL_SOLUTIONS["FY3"]["tr"] + np.random.uniform(-0.3, 0.3)
            pop[i, 11] = INITIAL_SOLUTIONS["FY3"]["td"] + np.random.uniform(-0.2, 0.2)

        # 确保参数在边界内
        return np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])

    def calculate_fitness(self, particle):
        """计算适应度，考虑无人机速度上限和导弹速度"""
        fy1_vx, fy1_vy, fy1_tr, fy1_td, \
            fy2_vx, fy2_vy, fy2_tr, fy2_td, \
            fy3_vx, fy3_vy, fy3_tr, fy3_td = particle

        all_intervals = []
        vz = 0.0
        uav_min_speed = UAV_SPEED_CONSTRAINTS["min"]

        # FY1遮蔽区间
        fy1_vmag = math.sqrt(fy1_vx ** 2 + fy1_vy ** 2 + vz ** 2)
        if uav_min_speed - 1e-9 <= fy1_vmag <= self.uav_speed_max + 1e-9:
            try:
                _, (fy1_det_t, fy1_det_x, fy1_det_y, fy1_det_z) = cate_det_pos(
                    "FY1", fy1_vx, fy1_vy, fy1_tr, fy1_td
                )
                if 50.0 <= fy1_det_z <= 2000.0:
                    _, fy1_intervals = cate_se_ocion_t(
                        fy1_det_t, fy1_det_x, fy1_det_y, fy1_det_z, self.missile_speed
                    )
                    all_intervals.extend(fy1_intervals)
            except:
                pass

        # FY2遮蔽区间
        fy2_vmag = math.sqrt(fy2_vx ** 2 + fy2_vy ** 2 + vz ** 2)
        if uav_min_speed - 1e-9 <= fy2_vmag <= self.uav_speed_max + 1e-9:
            try:
                _, (fy2_det_t, fy2_det_x, fy2_det_y, fy2_det_z) = cate_det_pos(
                    "FY2", fy2_vx, fy2_vy, fy2_tr, fy2_td
                )
                if 50.0 <= fy2_det_z <= 2000.0:
                    _, fy2_intervals = cate_se_ocion_t(
                        fy2_det_t, fy2_det_x, fy2_det_y, fy2_det_z, self.missile_speed
                    )
                    all_intervals.extend(fy2_intervals)
            except:
                pass

        # FY3遮蔽区间
        fy3_vmag = math.sqrt(fy3_vx ** 2 + fy3_vy ** 2 + vz ** 2)
        if uav_min_speed - 1e-9 <= fy3_vmag <= self.uav_speed_max + 1e-9:
            try:
                # 验证投放点/起爆点
                (fy3_rel_x, fy3_rel_y, _), (fy3_det_t, fy3_det_x, fy3_det_y, _) = cate_det_pos(
                    "FY3", fy3_vx, fy3_vy, fy3_tr, fy3_td
                )
                z_pos = 700.0 + 0.5 * (-9.8) * fy3_td ** 2
                if 50.0 <= z_pos <= 2000.0:  # 起爆点z约束
                    _, fy3_intervals = cate_se_ocion_t(
                        fy3_det_t, fy3_det_x, fy3_det_y, z_pos, self.missile_speed
                    )
                    all_intervals.extend(fy3_intervals)
            except:
                pass

        merged_time, _ = merge_inval(all_intervals)
        return merged_time if merged_time >= 1e-3 else 1e-6

    def update_velocity(self, iter_idx):
        w = self.w_start - (self.w_start - self.w_end) * (iter_idx / self.max_iter)
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)

        self.vel = w * self.vel + \
                   self.c1 * r1 * (self.p_best_pos - self.pos) + \
                   self.c2 * r2 * (self.g_best_pos - self.pos)
        self.vel = np.clip(self.vel, -self.v_max, self.v_max)

    def update_position(self):
        self.pos = self.pos + self.vel
        self.pos = np.clip(self.pos, self.bounds[:, 0], self.bounds[:, 1])

    def update_best(self):
        current_fit = np.array([self.calculate_fitness(p) for p in self.pos])
        improved = current_fit > self.p_best_fit
        self.p_best_pos[improved] = self.pos[improved].copy()
        self.p_best_fit[improved] = current_fit[improved]

        current_best_idx = np.argmax(current_fit)
        if current_fit[current_best_idx] > self.g_best_fit:
            self.g_best_pos = self.pos[current_best_idx].copy()
            self.g_best_fit = current_fit[current_best_idx]

    def optimize(self, verbose=False):
        """执行优化，可控制是否打印进度"""
        if verbose:
            print(f"PSO优化：问题4（FY1/FY2/FY3各投1枚烟幕弹）")
            print(
                f"种群规模：{self.pop_size} | 最大迭代：{self.max_iter} | 速度约束：{UAV_SPEED_CONSTRAINTS['min']}~{self.uav_speed_max}m/s")

        for i in range(self.max_iter):
            self.update_velocity(i)
            self.update_position()
            self.update_best()
            self.fit_history.append(self.g_best_fit)

            # 每50次迭代打印进度
            if verbose and (i + 1) % 50 == 0:
                print(f"迭代 {i + 1}/{self.max_iter} | 全局最优遮蔽时间：{self.g_best_fit:.2f}s")

        if verbose:
            print(f"优化结束！最终最优总遮蔽时间：{self.g_best_fit:.2f}s")
        return self.g_best_pos, self.fit_history


# 结果解析与输出
def ana_solion(best_pos, missile_speed=None):
    """解析最优解，输出结果"""
    fy1_vx, fy1_vy, fy1_tr, fy1_td, \
        fy2_vx, fy2_vy, fy2_tr, fy2_td, \
        fy3_vx, fy3_vy, fy3_tr, fy3_td = best_pos

    # 计算FY1参数
    fy1_release, (fy1_det_t, fy1_det_x, fy1_det_y, fy1_det_z) = cate_det_pos(
        "FY1", fy1_vx, fy1_vy, fy1_tr, fy1_td
    )
    fy1_vmag = math.sqrt(fy1_vx ** 2 + fy1_vy ** 2)
    fy1_dir = math.degrees(math.atan2(-fy1_vy, -fy1_vx)) % 360
    fy1_occl_time, fy1_intervals = cate_se_ocion_t(
        fy1_det_t, fy1_det_x, fy1_det_y, fy1_det_z, missile_speed
    )

    # 计算FY2参数
    fy2_release, (fy2_det_t, fy2_det_x, fy2_det_y, fy2_det_z) = cate_det_pos(
        "FY2", fy2_vx, fy2_vy, fy2_tr, fy2_td
    )
    fy2_vmag = math.sqrt(fy2_vx ** 2 + fy2_vy ** 2)
    fy2_dir = math.degrees(math.atan2(-fy2_vy, -fy2_vx)) % 360
    fy2_occl_time, fy2_intervals = cate_se_ocion_t(
        fy2_det_t, fy2_det_x, fy2_det_y, fy2_det_z, missile_speed
    )

    # 计算FY3参数
    fy3_release, (fy3_det_t, fy3_det_x, fy3_det_y, fy3_det_z) = cate_det_pos(
        "FY3", fy3_vx, fy3_vy, fy3_tr, fy3_td
    )
    fy3_vmag = math.sqrt(fy3_vx ** 2 + fy3_vy ** 2)
    fy3_dir = math.degrees(math.atan2(-fy3_vy, -fy3_vx)) % 360
    fy3_occl_time, fy3_intervals = cate_se_ocion_t(
        fy3_det_t, fy3_det_x, fy3_det_y, fy3_det_z, missile_speed
    )

    # 合并总遮蔽区间
    all_intervals = fy1_intervals + fy2_intervals + fy3_intervals
    total_time, merged_intervals = merge_inval(all_intervals)

    # 输出结果
    print("问题4 最优投放策略（FY1/FY2/FY3各投1枚烟幕弹，干扰M1）")

    print("[FY1 详细参数]")
    print("1. 飞行参数：")
    print(f"   - 速度大小：{fy1_vmag:.2f} m/s | 方向角：{(180 + np.arctan(fy1_vy / fy1_vx) * 180 / np.pi):.1f}°")
    print(f"   - 速度向量：(vx={fy1_vx:.2f}, vy={fy1_vy:.2f}, vz=0.00)")
    print("2. 投放参数：")
    print(f"   - 投放时刻：{fy1_tr:.2f}s | 投放点：({fy1_release[0]:.2f}, {fy1_release[1]:.2f}, {fy1_release[2]:.2f})")
    print("3. 起爆参数：")
    print(f"   - 起爆延迟：{fy1_td:.2f}s | 起爆时刻：{fy1_det_t:.2f}s")
    print(f"   - 起爆点：({fy1_det_x:.2f}, {fy1_det_y:.2f}, {fy1_det_z:.2f})")
    print("4. 遮蔽效果：")
    print(f"   - 有效时间：{fy1_occl_time:.2f}s | 区间：{fy1_intervals}\n")

    print("[FY2 详细参数]")
    print("1. 飞行参数：")
    print(f"   - 速度大小：{fy2_vmag:.2f} m/s | 方向角：{(180 + np.arctan(fy2_vy / fy2_vx) * 180 / np.pi):.1f}°")
    print(f"   - 速度向量：(vx={fy2_vx:.2f}, vy={fy2_vy:.2f}, vz=0.00)")
    print("2. 投放参数：")
    print(f"   - 投放时刻：{fy2_tr:.2f}s | 投放点：({fy2_release[0]:.2f}, {fy2_release[1]:.2f}, {fy2_release[2]:.2f})")
    print("3. 起爆参数：")
    print(f"   - 起爆延迟：{fy2_td:.2f}s | 起爆时刻：{fy2_det_t:.2f}s")
    print(f"   - 起爆点：({fy2_det_x:.2f}, {fy2_det_y:.2f}, {fy2_det_z:.2f})")
    print("4. 遮蔽效果：")
    print(f"   - 有效时间：{fy2_occl_time:.2f}s | 区间：{fy2_intervals}\n")

    print("[FY3 详细参数]")
    print("1. 飞行参数：")
    print(f"   - 速度大小：{fy3_vmag:.2f} m/s | 方向角：{(180 + np.arctan(fy3_vy / fy3_vx) * 180 / np.pi):.1f}°")
    print(f"   - 速度向量：(vx={fy3_vx:.2f}, vy={fy3_vy:.2f}, vz=0.00)")
    print("2. 投放参数：")
    print(f"   - 投放时刻：{fy3_tr:.2f}s")
    print(f"   - 投放点：({fy3_release[0]:.2f}, {fy3_release[1]:.2f}, {fy3_release[2]:.2f})")
    print("3. 起爆参数：")
    print(f"   - 起爆延迟：{fy3_td:.2f}s")
    print(f"   - 起爆时刻：{fy3_det_t:.2f}s")
    print(f"   - 起爆点：({fy3_det_x:.2f}, {fy3_det_y:.2f}, {fy3_det_z:.2f})")
    print("4. 遮蔽效果：")
    print(f"   - 有效时间：{fy3_occl_time:.2f}s")
    print(f"   - 遮蔽区间：{fy3_intervals}\n")

    print("[总遮蔽效果]")
    print(f"   - 总有效遮蔽时间：{total_time:.2f}s")
    print(f"   - 合并区间：{merged_intervals}")
    print("   - 云团有效期：各弹起爆后20s")

    return {
        "fy1": {"occl_time": fy1_occl_time, "intervals": fy1_intervals},
        "fy2": {"occl_time": fy2_occl_time, "intervals": fy2_intervals},
        "fy3": {"occl_time": fy3_occl_time, "intervals": fy3_intervals},
        "total": {"occl_time": total_time, "intervals": merged_intervals}
    }


# 灵敏度分析函数
def sensitivity_analysis_missile_speed(best_params, base_speed, perturb_ratios=[-0.1, -0.05, 0.05, 0.1]):
    """对导弹速度进行灵敏度分析"""
    print("\n导弹速度灵敏度分析")

    # 创建基础优化器获取基准值
    base_optimizer = PSO_Optimizer_ThreeUAVs(
        pop_size=30, max_iter=100, c1=1.7, c2=1.7, missile_speed=base_speed
    )
    _, base_fit_history = base_optimizer.optimize(verbose=False)
    base_time = base_fit_history[-1]

    print(f"基准导弹速度: {base_speed} m/s")
    print(f"基准总遮蔽时间: {base_time:.2f} s")

    # 存储结果
    results = {
        "speed": [],
        "occlusion_time": [],
        "ratio": [],
        "sensitivity": []
    }

    # 对每个扰动比例进行分析
    for ratio in perturb_ratios:
        perturbed_speed = base_speed * (1 + ratio)
        if perturbed_speed <= 0:
            print(f"无效速度 {perturbed_speed:.2f} m/s，跳过...")
            continue

        print(f"\n分析导弹速度 {perturbed_speed:.2f} m/s ({ratio:.0%} 扰动)")

        # 使用扰动后的参数创建优化器
        optimizer = PSO_Optimizer_ThreeUAVs(
            pop_size=30, max_iter=100, c1=1.7, c2=1.7, missile_speed=perturbed_speed
        )
        _, fit_history = optimizer.optimize(verbose=False)
        occlusion_time = fit_history[-1]

        # 计算灵敏度系数
        delta_speed = perturbed_speed - base_speed
        delta_time = occlusion_time - base_time
        sensitivity = (delta_time / base_time) / (delta_speed / base_speed) if (
                base_time > 1e-9 and delta_speed != 0) else 0.0

        # 存储结果
        results["speed"].append(perturbed_speed)
        results["occlusion_time"].append(occlusion_time)
        results["ratio"].append(ratio)
        results["sensitivity"].append(sensitivity)

        print(f"扰动后总遮蔽时间: {occlusion_time:.2f} s")
        print(f"灵敏度系数: {sensitivity:.4f}")

    return results, base_time, base_speed


def sensitivity_analysis_uav_max_speed(best_params, base_max_speed, perturb_ratios=[-0.1, -0.05, 0.05, 0.1]):
    """对无人机速度上限进行灵敏度分析"""
    print("\n无人机速度上限灵敏度分析")

    # 创建基础优化器获取基准值
    base_optimizer = PSO_Optimizer_ThreeUAVs(
        pop_size=30, max_iter=100, c1=1.7, c2=1.7, uav_speed_max=base_max_speed
    )
    _, base_fit_history = base_optimizer.optimize(verbose=False)
    base_time = base_fit_history[-1]

    print(f"基准无人机速度上限: {base_max_speed} m/s")
    print(f"基准总遮蔽时间: {base_time:.2f} s")

    # 存储结果
    results = {
        "max_speed": [],
        "occlusion_time": [],
        "ratio": [],
        "sensitivity": []
    }

    # 对每个扰动比例进行分析
    for ratio in perturb_ratios:
        perturbed_max_speed = base_max_speed * (1 + ratio)
        # 确保不低于最小速度
        if perturbed_max_speed <= UAV_SPEED_CONSTRAINTS["min"]:
            print(f"无效速度上限 {perturbed_max_speed:.2f} m/s，跳过...")
            continue

        print(f"\n分析无人机速度上限 {perturbed_max_speed:.2f} m/s ({ratio:.0%} 扰动)")

        # 使用扰动后的参数创建优化器
        optimizer = PSO_Optimizer_ThreeUAVs(
            pop_size=30, max_iter=100, c1=1.7, c2=1.7, uav_speed_max=perturbed_max_speed
        )
        _, fit_history = optimizer.optimize(verbose=False)
        occlusion_time = fit_history[-1]

        # 计算灵敏度系数
        delta_speed = perturbed_max_speed - base_max_speed
        delta_time = occlusion_time - base_time
        sensitivity = (delta_time / base_time) / (delta_speed / base_max_speed) if (
                base_time > 1e-9 and delta_speed != 0) else 0.0

        # 存储结果
        results["max_speed"].append(perturbed_max_speed)
        results["occlusion_time"].append(occlusion_time)
        results["ratio"].append(ratio)
        results["sensitivity"].append(sensitivity)

        print(f"扰动后总遮蔽时间: {occlusion_time:.2f} s")
        print(f"灵敏度系数: {sensitivity:.4f}")

    return results, base_time, base_max_speed


def plot_sensitivity_results(missile_results, uav_results, base_missile_time, base_missile_speed,
                             base_uav_time, base_uav_speed):
    """绘制灵敏度分析结果图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 第一个子图：参数值 vs 遮蔽时间
    # 导弹速度
    ax1.plot(missile_results["speed"], missile_results["occlusion_time"],
             'o-', color='red', label=f'导弹速度 (基准={base_missile_speed}m/s)')
    # 无人机速度上限
    ax1.plot(uav_results["max_speed"], uav_results["occlusion_time"],
             's-', color='blue', label=f'无人机速度上限 (基准={base_uav_speed}m/s)')

    # 基准线
    ax1.axhline(y=base_missile_time, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=base_uav_time, color='blue', linestyle='--', alpha=0.5)

    ax1.set_xlabel('参数值', fontsize=12)
    ax1.set_ylabel('总遮蔽时间 (s)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # 第二个子图：扰动比例 vs 灵敏度系数
    # 导弹速度
    missile_ratios = [r * 100 for r in missile_results["ratio"]]  # 转为百分比
    ax2.bar([f'导弹+{r:.0f}%' for r in missile_ratios],
            missile_results["sensitivity"],
            color='red', alpha=0.7, width=0.4, label='导弹速度')

    # 无人机速度上限
    uav_ratios = [r * 100 for r in uav_results["ratio"]]  # 转为百分比
    ax2.bar([f'无人机+{r:.0f}%' for r in uav_ratios],
            uav_results["sensitivity"],
            color='blue', alpha=0.7, width=0.4, label='无人机速度上限')

    # 参考线
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='高敏感性阈值')
    ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='中等敏感性阈值')
    ax2.axhline(y=-1.0, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=-0.3, color='green', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax2.set_xlabel('参数扰动比例', fontsize=15)
    ax2.set_ylabel('灵敏度系数', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax2.legend(fontsize=30)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('问题4_PSO灵敏度分析(导弹与无人机速度).png', dpi=300, bbox_inches='tight')
    plt.show()


# 主函数执行
def main():
    # 首先获取基准最优解
    base_optimizer = PSO_Optimizer_ThreeUAVs(
        pop_size=50,
        max_iter=300,
        c1=1.7,
        c2=1.7,
        w_start=0.9,
        w_end=0.6
    )
    best_position, base_fit_history = base_optimizer.optimize(verbose=True)

    # 解析并输出基准最优策略
    print("\n=== 基准最优策略分析 ===")
    base_results = ana_solion(best_position)

    # 绘制基准收敛曲线
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))
    plt.plot(base_fit_history, 'b-', linewidth=2, label='全局最优遮蔽时间')
    plt.axhline(
        y=base_fit_history[-1],
        color='red',
        linestyle='--',
        linewidth=3,
        label=f'最终全局最优：{base_fit_history[-1]:.2f}s',
        alpha=0.7
    )
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('总遮蔽时间 (s)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('问题4_PSO收敛曲线.png')
    plt.show()

    # 执行导弹速度灵敏度分析
    missile_speed_results, base_missile_time, base_missile_speed = sensitivity_analysis_missile_speed(
        best_position, M1_PARAMS["speed"], perturb_ratios=[-0.1, -0.05, 0.05, 0.1]
    )

    # 执行无人机速度上限灵敏度分析
    uav_speed_results, base_uav_time, base_uav_speed = sensitivity_analysis_uav_max_speed(
        best_position, UAV_SPEED_CONSTRAINTS["max"], perturb_ratios=[-0.1, -0.05, 0.05, 0.1]
    )

    # 绘制灵敏度分析结果
    plot_sensitivity_results(
        missile_speed_results, uav_speed_results,
        base_missile_time, base_missile_speed,
        base_uav_time, base_uav_speed
    )

    # 输出灵敏度分析结论
    print("\n=== 灵敏度分析结论 ===")

    # 导弹速度分析结论
    missile_sens_avg = np.mean(np.abs(missile_speed_results["sensitivity"]))
    if missile_sens_avg > 1.0:
        missile_conclusion = "高度敏感"
    elif missile_sens_avg > 0.3:
        missile_conclusion = "中等敏感"
    else:
        missile_conclusion = "低敏感"

    print(f"1. 导弹速度：平均灵敏度系数 {missile_sens_avg:.4f}，属于{missile_conclusion}")
    print(f"   含义：导弹速度每变化1%，总遮蔽时间平均变化约{missile_sens_avg:.2f}%")

    # 无人机速度上限分析结论
    uav_sens_avg = np.mean(np.abs(uav_speed_results["sensitivity"]))
    if uav_sens_avg > 1.0:
        uav_conclusion = "高度敏感"
    elif uav_sens_avg > 0.3:
        uav_conclusion = "中等敏感"
    else:
        uav_conclusion = "低敏感"

    print(f"2. 无人机速度上限：平均灵敏度系数 {uav_sens_avg:.4f}，属于{uav_conclusion}")
    print(f"   含义：无人机速度上限每变化1%，总遮蔽时间平均变化约{uav_sens_avg:.2f}%")


if __name__ == "__main__":
    main()
