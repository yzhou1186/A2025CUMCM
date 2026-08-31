import math
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

# 固定全局随机种子，确保结果可复现
SEED = 4224
np.random.seed(SEED)

# 基础参数定义
# 无人机初始位置
DRONE_INIT = {
    "FY1": (17800, 0, 1800),  # (x, y, z)
    "FY2": (12000, 1400, 1400),
    "FY3": (6000, -3000, 700),
    "FY4": (11000, 2000, 1800),
    "FY5": (13000, -2000, 1300)
}

# 导弹初始位置与速度
MISSILE_PARAMS = {
    "M1": {"init_pos": (20000, 0, 2000), "speed": 300},
    "M2": {"init_pos": (19000, 600, 2100), "speed": 300},
    "M3": {"init_pos": (18000, -600, 1900), "speed": 300}
}

# 约束条件
CONSTRAINTS = {
    "drone_v_min": 70.0,  # 无人机最小速度(m/s)
    "drone_v_max": 140.0,  # 无人机最大速度(m/s)
    "drop_interval_min": 1.0,  # 两枚弹最小投放间隔(s)
    "smoke_radius": 10.0,  # 烟幕云团半径(m)
    "smoke_sink_speed": 3.0,  # 烟幕下沉速度(m/s)
    "smoke_valid_time": 20.0,  # 烟幕有效时长(s，起爆后)
    "det_z_min": 50.0,  # 起爆点最低z坐标(m)
    "det_z_max": 2000.0  # 起爆点最高z坐标(m)
}

# 真目标参数
TARGET_PARAMS = {
    "center": (0, 200, 0),  # 底心坐标(x,y,z)
    "radius": 7.0,  # 圆柱半径(m)
    "height": 10.0  # 圆柱高度(m)，z范围[0,10]
}

# 有效烟幕弹奖励参数
REWARD_CONFIG = {
    "valid_threshold": 1e-9,  # 有效弹判定阈值
    "per_smoke_reward": 5.0,  # 每枚有效弹基础奖励
    "multi_bonus": 3.0  # 多有效弹叠加奖励(每多1枚)
}


# 遮挡判断函数
# 基础向量工具函数
def vector_length(v):
    x, y, z = v
    return math.sqrt(x ** 2 + y ** 2 + z ** 2)


def dot_product(v1, v2):
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return x1 * x2 + y1 * y2 + z1 * z2


def vector_subtract(v1, v2):
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    return (x1 - x2, y1 - y2, z1 - z2)


# 合并重叠区间工具函数
def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= last_end + 1e-9:
            merged[-1] = (last_start, max(last_end, curr_end))
        else:
            merged.append(current)
    return merged


# 导弹位置计算
def get_pit_lit_pos(t, missile_id):
    init_pos = MISSILE_PARAMS[missile_id]["init_pos"]
    target_pos = (0.0, 0.0, 0.0)
    dir_ec = vector_subtract(target_pos, init_pos)
    dir_l = vector_length(dir_ec)
    if dir_l < 1e-9:
        return target_pos
    unit_dir = (dir_ec[0] / dir_l, dir_ec[1] / dir_l, dir_ec[2] / dir_l)
    dis_tan = MISSILE_PARAMS[missile_id]["speed"] * t
    if dis_tan >= dir_l:
        return target_pos
    return (
        init_pos[0] + unit_dir[0] * dis_tan,
        init_pos[1] + unit_dir[1] * dis_tan,
        init_pos[2] + unit_dir[2] * dis_tan
    )


# 烟幕位置计算
def get_she_pos(t, t_det, sphe_x0, sphe_y0, sphe_z0):
    if t < t_det - 1e-9 or t > t_det + CONSTRAINTS["smoke_valid_time"] + 1e-9:
        return None
    z_pos = sphe_z0 - CONSTRAINTS["smoke_sink_speed"] * (t - t_det)
    return (float(sphe_x0), float(sphe_y0), float(z_pos))


# 线段-球体相交判断
def is_sme_it_she_ana(L, P, S, R):
    LP = vector_subtract(P, L)
    LS = vector_subtract(L, S)
    a = dot_product(LP, LP)
    b = 2 * dot_product(LS, LP)
    c = dot_product(LS, LS) - R ** 2

    dis_miant = b ** 2 - 4 * a * c
    if dis_miant < -1e-9:
        return False
    dis_miant = max(dis_miant, 0.0)
    sqrt_d = math.sqrt(dis_miant)

    t1 = (-b - sqrt_d) / (2 * a) if a > 1e-12 else 0.0
    t2 = (-b + sqrt_d) / (2 * a) if a > 1e-12 else 0.0

    return (t1 >= -1e-9 and t1 <= 1.0 + 1e-9) or (t2 >= -1e-9 and t2 <= 1.0 + 1e-9)


# 真目标完全遮挡判断
def is_cyder_ful_oced(t, t_det, sphe_x0, sphe_y0, sphe_z0, missile_id):
    lit_pos = get_pit_lit_pos(t, missile_id)
    she_pos = get_she_pos(t, t_det, sphe_x0, sphe_y0, sphe_z0)
    if she_pos is None:
        return False

    LS = vector_subtract(lit_pos, she_pos)
    if vector_length(LS) <= CONSTRAINTS["smoke_radius"] + 1e-9:
        return True

    cyl_rd = TARGET_PARAMS["radius"]
    cyl_ht = TARGET_PARAMS["height"]
    cyl_y = TARGET_PARAMS["center"][1]
    key_rays = []

    # 顶部圆周采样
    key_rays.extend([
        (cyl_rd, cyl_y, cyl_ht), (-cyl_rd, cyl_y, cyl_ht),
        (0, cyl_y, cyl_ht + cyl_rd), (0, cyl_y, cyl_ht - cyl_rd)
    ])
    # 底部圆周采样
    key_rays.extend([
        (cyl_rd, cyl_y, 0.0), (-cyl_rd, cyl_y, 0.0),
        (0, cyl_y, cyl_rd), (0, cyl_y, -cyl_rd)
    ])
    # 中间高度采样
    heights = [cyl_ht * 0.25, cyl_ht * 0.5, cyl_ht * 0.75]
    for h in heights:
        key_rays.extend([
            (cyl_rd, cyl_y, h), (-cyl_rd, cyl_y, h),
            (0, cyl_y, h + cyl_rd), (0, cyl_y, h - cyl_rd)
        ])
    # 对角线采样
    diag_factor = math.cos(math.pi / 4)
    key_rays.extend([
        (cyl_rd * diag_factor, cyl_y, cyl_ht * 0.5 + cyl_rd * diag_factor),
        (-cyl_rd * diag_factor, cyl_y, cyl_ht * 0.5 + cyl_rd * diag_factor),
        (cyl_rd * diag_factor, cyl_y, cyl_ht * 0.5 - cyl_rd * diag_factor),
        (-cyl_rd * diag_factor, cyl_y, cyl_ht * 0.5 - cyl_rd * diag_factor)
    ])
    # 轴线采样
    key_rays.extend([
        (0.0, cyl_y, 0.0), (0.0, cyl_y, cyl_ht),
        (0.0, cyl_y, cyl_ht * 0.25), (0.0, cyl_y, cyl_ht * 0.5),
        (0.0, cyl_y, cyl_ht * 0.75)
    ])

    for target_point in key_rays:
        if not is_sme_it_she_ana(lit_pos, target_point, she_pos, CONSTRAINTS["smoke_radius"]):
            return False
    return True


# 遮挡时间计算
def find_trans_t(start, end, t_det, sphe_x0, sphe_y0, sphe_z0, missile_id, target_state):
    eps = 1e-10
    for _ in range(50):
        mid = (start + end) / 2
        cur_ste = is_cyder_ful_oced(mid, t_det, sphe_x0, sphe_y0, sphe_z0, missile_id)
        if cur_ste == target_state:
            end = mid
        else:
            start = mid
        if end - start < eps:
            break
    return (start + end) / 2


def cal_ful_ocion_t(t_det, sphe_x0, sphe_y0, sphe_z0, missile_id, corse_step=0.1):
    start_time = t_det
    end_time = t_det + CONSTRAINTS["smoke_valid_time"]
    ful_ocion_invl = []

    inal_ck_t = max(start_time, t_det + 0.1)
    pre_se = is_cyder_ful_oced(inal_ck_t, t_det, sphe_x0, sphe_y0, sphe_z0, missile_id)
    inal_start = None

    if pre_se:
        inal_start = find_trans_t(start_time, inal_ck_t, t_det, sphe_x0, sphe_y0, sphe_z0, missile_id, True)

    t = inal_ck_t + corse_step
    while t <= end_time:
        cur_se = is_cyder_ful_oced(t, t_det, sphe_x0, sphe_y0, sphe_z0, missile_id)
        if cur_se != pre_se:
            trans_t = find_trans_t(t - corse_step, t, t_det, sphe_x0, sphe_y0, sphe_z0, missile_id, cur_se)
            if pre_se:
                ful_ocion_invl.append((inal_start, trans_t))
                inal_start = None
            else:
                inal_start = trans_t
            pre_se = cur_se
        t += corse_step

    if pre_se and inal_start is not None:
        ful_ocion_invl.append((inal_start, end_time))

    total_time = sum(end - start for start, end in ful_ocion_invl)
    return ful_ocion_invl, round(total_time, 6)


# 工具函数
# 点到线段最短距离
def pt_to_st_dt(point, seg_p1, seg_p2):
    vec_seg = vector_subtract(seg_p2, seg_p1)
    vec_pt_seg1 = vector_subtract(point, seg_p1)
    dot_val = dot_product(vec_pt_seg1, vec_seg)
    seg_len_sq = dot_product(vec_seg, vec_seg)

    if seg_len_sq < 1e-18:
        return vector_length(vec_pt_seg1)
    t = max(0.0, min(1.0, dot_val / seg_len_sq))
    pj_pt = (
        seg_p1[0] + t * vec_seg[0],
        seg_p1[1] + t * vec_seg[1],
        seg_p1[2] + t * vec_seg[2]
    )
    return vector_length(vector_subtract(point, pj_pt))


# 无人机-导弹分配
def an_de_to_me():
    an_rt = {}
    miile_lt = ["M1", "M2", "M3"]
    dre_lt = ["FY1", "FY2", "FY3", "FY4", "FY5"]

    for de_id in dre_lt:
        drone_pos = DRONE_INIT[de_id]
        dist_dict = {}
        for missile_id in miile_lt:
            missile_seg_p1 = MISSILE_PARAMS[missile_id]["init_pos"]
            missile_seg_p2 = (0.0, 0.0, 0.0)
            dist = pt_to_st_dt(drone_pos, missile_seg_p1, missile_seg_p2)
            dist_dict[missile_id] = dist
        target_missile = min(dist_dict.items(), key=lambda x: x[1])[0]
        an_rt[de_id] = {
            "target_missile": target_missile,
            "distance": round(dist_dict[target_missile], 2)
        }
    return an_rt


# 3.3 烟幕投放/起爆参数计算
def calc_smoke_params(drone_id, vx, vy, t_release, t_delay):
    drone_x0, drone_y0, drone_z0 = DRONE_INIT[drone_id]
    g = -9.8  # 重力加速度

    # 投放点
    release_x = drone_x0 + vx * t_release
    release_y = drone_y0 + vy * t_release
    release_z = drone_z0
    release_pos = (round(release_x, 3), round(release_y, 3), round(release_z, 3))

    # 起爆点
    det_x = release_x + vx * t_delay
    det_y = release_y + vy * t_delay
    det_z = release_z + 0.5 * g * (t_delay ** 2)
    t_det = t_release + t_delay
    det_pos = (round(det_x, 3), round(det_y, 3), round(det_z, 3))

    return release_pos, (t_det, det_pos)


# 决策变量约束检查
def check_constraints(vx, vy, t1, t2, t3, d1, d2, d3, drone_id):
    # 1. 速度约束
    v_mag = math.sqrt(vx ** 2 + vy ** 2)
    if not (CONSTRAINTS["drone_v_min"] - 1e-9 <= v_mag <= CONSTRAINTS["drone_v_max"] + 1e-9):
        return False, f"速度{round(v_mag, 3)}m/s超出70~140m/s范围"

    # 2. 投放间隔约束
    if t2 - t1 < CONSTRAINTS["drop_interval_min"] - 1e-9:
        return False, f"第1-2枚投放间隔{round(t2 - t1, 3)}s<1s"
    if t3 - t2 < CONSTRAINTS["drop_interval_min"] - 1e-9:
        return False, f"第2-3枚投放间隔{round(t3 - t2, 3)}s<1s"

    # 3. 起爆点z坐标约束
    for t_release, t_delay in [(t1, d1), (t2, d2), (t3, d3)]:
        _, (_, (_, _, det_z)) = calc_smoke_params(drone_id, vx, vy, t_release, t_delay)
        if not (CONSTRAINTS["det_z_min"] - 1e-9 <= det_z <= CONSTRAINTS["det_z_max"] + 1e-9):
            return False, f"起爆点z={round(det_z, 3)}m超出50~2000m范围"

    # 4. 时间非负约束
    if any(t < 0 for t in [t1, t2, t3]):
        return False, "投放时间不能为负"
    if any(d < 0 for d in [d1, d2, d3]):
        return False, "起爆延迟不能为负"

    return True, "约束满足"


# 模拟退火算法
class SimulatedAnnealing:
    def __init__(self, drone_id, target_missile, max_iter=100, init_temp=100, cool_rate=0.95, seed_offset=0):
        self.drone_id = drone_id
        self.target_missile = target_missile
        self.max_iter = max_iter
        self.init_temp = init_temp
        self.cool_rate = cool_rate
        self.seed_offset = seed_offset  # 用于进程特定的随机种子
        self.decision_bounds = {
            "vx": (-140.0, 0.0),  # x方向速度
            "vy": (-140.0, 140.0),  # y方向速度
            "t1": (0.001, 30.0),  # 第1枚投放时间(s)
            "d1": (0.1, 15.0),  # 第1枚起爆延迟(s)
            "t2": (1.001, 35.0),  # 第2枚投放时间(s)
            "d2": (0.1, 15.0),  # 第2枚起爆延迟(s)
            "t3": (2.001, 40.0),  # 第3枚投放时间(s)
            "d3": (0.1, 15.0)  # 第3枚起爆延迟(s)
        }
        self.converge_history = []

        # 设置进程特定的随机种子，确保可复现性
        self.local_seed = SEED + self.seed_offset
        np.random.seed(self.local_seed)

    def init_solution(self):
        while True:
            vx = np.random.uniform(*self.decision_bounds["vx"])
            vy = np.random.uniform(*self.decision_bounds["vy"])
            t1 = np.random.uniform(*self.decision_bounds["t1"])
            d1 = np.random.uniform(*self.decision_bounds["d1"])
            t2 = np.random.uniform(t1 + 1.0, self.decision_bounds["t2"][1])
            d2 = np.random.uniform(*self.decision_bounds["d2"])
            t3 = np.random.uniform(t2 + 1.0, self.decision_bounds["t3"][1])
            d3 = np.random.uniform(*self.decision_bounds["d3"])

            valid, _ = check_constraints(vx, vy, t1, t2, t3, d1, d2, d3, self.drone_id)
            if valid:
                return np.array([vx, vy, t1, d1, t2, d2, t3, d3])

    def generate_neighbor(self, current_sol):
        while True:
            perturb = np.array([
                np.random.uniform(-20.0, 20.0),  # vx扰动
                np.random.uniform(-20.0, 20.0),  # vy扰动
                np.random.uniform(-3.0, 3.0),  # t1扰动
                np.random.uniform(-1.5, 1.5),  # d1扰动
                np.random.uniform(-5.0, 5.0),  # t2扰动
                np.random.uniform(-1.5, 1.5),  # d2扰动
                np.random.uniform(-8.0, 8.0),  # t3扰动
                np.random.uniform(-1.5, 1.5)  # d3扰动
            ])
            new_sol = current_sol + perturb

            # 边界裁剪
            new_sol[0] = np.clip(new_sol[0], *self.decision_bounds["vx"])
            new_sol[1] = np.clip(new_sol[1], *self.decision_bounds["vy"])
            new_sol[2] = np.clip(new_sol[2], *self.decision_bounds["t1"])
            new_sol[3] = np.clip(new_sol[3], *self.decision_bounds["d1"])
            new_sol[4] = np.clip(new_sol[4], *self.decision_bounds["t2"])
            new_sol[5] = np.clip(new_sol[5], *self.decision_bounds["d2"])
            new_sol[6] = np.clip(new_sol[6], *self.decision_bounds["t3"])
            new_sol[7] = np.clip(new_sol[7], *self.decision_bounds["d3"])

            # 确保投放间隔≥1s
            new_sol[4] = max(new_sol[4], new_sol[2] + 1.0)
            new_sol[6] = max(new_sol[6], new_sol[4] + 1.0)

            # 检查约束
            vx, vy, t1, d1, t2, d2, t3, d3 = new_sol
            valid, _ = check_constraints(vx, vy, t1, t2, t3, d1, d2, d3, self.drone_id)
            if valid:
                return new_sol

    def calculate_fitness(self, solution):
        vx, vy, t1, d1, t2, d2, t3, d3 = solution
        smoke_details = []
        valid_count = 0  # 有效烟幕弹数量
        all_intervals = []  # 收集所有有效烟幕弹的遮挡区间（用于合并）

        for idx, (t_release, t_delay) in enumerate([(t1, d1), (t2, d2), (t3, d3)], 1):
            release_pos, (t_det, (det_x, det_y, det_z)) = calc_smoke_params(
                self.drone_id, vx, vy, t_release, t_delay
            )
            shield_intervals, shield_time = cal_ful_ocion_t(
                t_det, det_x, det_y, det_z, self.target_missile
            )

            # 判断是否为有效烟幕弹（有遮挡时间）
            is_valid = shield_time > REWARD_CONFIG["valid_threshold"]
            if is_valid:
                valid_count += 1
                all_intervals.extend(shield_intervals)  # 仅收集有效弹的区间

            # 记录单枚烟幕弹详情
            smoke_details.append({
                "bomb_idx": idx,
                "release_time": round(t_release, 3),
                "release_pos": release_pos,
                "det_delay": round(t_delay, 3),
                "det_time": round(t_det, 3),
                "det_pos": (det_x, det_y, det_z),
                "shield_intervals": [(round(s, 2), round(e, 2)) for s, e in shield_intervals],  # 单弹原始区间
                "shield_time": shield_time,  # 单弹原始遮挡时间
                "is_valid": is_valid
            })

        # 合并所有有效弹的遮挡区间，计算真实总遮挡时间（去重）
        merged_intervals = merge_intervals(all_intervals)
        total_shield = sum(end - start for start, end in merged_intervals)  # 合并后的总遮挡时间

        # 计算奖励
        base_reward = valid_count * REWARD_CONFIG["per_smoke_reward"]
        bonus_reward = max(0, valid_count - 1) * REWARD_CONFIG["multi_bonus"]
        total_reward = base_reward + bonus_reward

        # 最终适应度 = 合并后的总遮挡时间 + 奖励
        total_fitness = total_shield + total_reward

        return total_fitness, smoke_details, valid_count, total_reward, merged_intervals

    def optimize(self):
        print(f"开始优化无人机{self.drone_id}（干扰{self.target_missile}）")
        print(f"SA参数：初始温度={self.init_temp}, 降温系数={self.cool_rate}, 总迭代={self.max_iter}")
        print(f"奖励机制：每枚有效弹{REWARD_CONFIG['per_smoke_reward']}s，多弹叠加{REWARD_CONFIG['multi_bonus']}s/枚")

        current_sol = self.init_solution()
        current_fitness, _, _, _, _ = self.calculate_fitness(current_sol)
        best_sol = current_sol.copy()
        best_fitness = current_fitness
        best_smoke_details = None
        best_valid_count = 0
        best_total_reward = 0
        best_merged_intervals = []  # 记录最优解的合并区间
        temp = self.init_temp

        for iter_idx in range(self.max_iter):
            for _ in range(50):
                new_sol = self.generate_neighbor(current_sol)
                new_fitness, new_smoke_details, new_valid_count, new_total_reward, new_merged_intervals = self.calculate_fitness(
                    new_sol)

                delta_f = new_fitness - current_fitness
                if delta_f > 1e-9:
                    current_sol = new_sol.copy()
                    current_fitness = new_fitness
                    if current_fitness > best_fitness:
                        best_fitness = current_fitness
                        best_sol = current_sol.copy()
                        best_smoke_details = new_smoke_details
                        best_valid_count = new_valid_count
                        best_total_reward = new_total_reward
                        best_merged_intervals = new_merged_intervals  # 保存最优合并区间
                else:
                    prob = math.exp(delta_f / temp)
                    if np.random.random() < prob:
                        current_sol = new_sol.copy()
                        current_fitness = new_fitness

            self.converge_history.append(best_fitness)
            if (iter_idx + 1) % 50 == 0:
                # 打印时使用合并后的遮挡时间（更准确）
                base_shield = best_fitness - best_total_reward
                print(
                    f"SA迭代{iter_idx + 1:3d}/{self.max_iter} | 最优适应度：{best_fitness:.3f} | "
                    f"遮挡{base_shield:.3f}s | 有效弹{best_valid_count}枚 | 当前温度：{temp:.2f}")

            temp *= self.cool_rate

        # 重新计算最优解的详细信息（确保一致性）
        _, best_smoke_details, best_valid_count, best_total_reward, best_merged_intervals = self.calculate_fitness(
            best_sol)
        vx, vy = best_sol[0], best_sol[1]
        v_mag = math.sqrt(vx ** 2 + vy ** 2)
        base_shield_time = best_fitness - best_total_reward  # 合并后的遮挡时间

        print(f"\n无人机{self.drone_id}最优策略详细信息")
        print(f"最优速度：vx={round(vx, 3)}m/s, vy={round(vy, 3)}m/s, 合速度={round(v_mag, 3)}m/s, 方向角{180+np.arctan(vy/(vx+1e-9))*180/np.pi:.2f}°")
        print(f"遮挡{base_shield_time:.3f}s")
        # 打印合并后的总区间
        print(
            f"所有有效弹合并后遮挡区间：{[(round(s, 2), round(e, 2)) for s, e in best_merged_intervals] if best_merged_intervals else '无'}")

        for smoke in best_smoke_details:
            status = "有效" if smoke["is_valid"] else "无效"
            print(f"\n第{smoke['bomb_idx']}枚烟幕弹（{status}）")
            print(f"投放时间: {smoke['release_time']}s")
            print(f"投放点坐标: {smoke['release_pos']}")
            print(f"起爆延迟: {smoke['det_delay']}s")
            print(f"起爆时间: {smoke['det_time']}s")
            print(f"起爆点坐标: {smoke['det_pos']}")
            print(f"单弹遮挡区间: {smoke['shield_intervals'] if smoke['shield_intervals'] else '无'}")
            print(f"单弹遮挡时间: {smoke['shield_time']:.3f}s")

        return {
            "drone_id": self.drone_id,
            "target_missile": self.target_missile,
            "best_velocity": {"vx": round(vx, 3), "vy": round(vy, 3), "v_mag": round(v_mag, 3)},
            "best_fitness": round(best_fitness, 3),
            "base_shield_time": round(base_shield_time, 3),  # 合并后的遮挡时间
            "total_reward": round(best_total_reward, 3),
            "valid_smoke_count": best_valid_count,
            "smoke_details": best_smoke_details,
            "merged_intervals": best_merged_intervals,  # 合并后的区间
            "converge_history": self.converge_history
        }


# 多进程优化函数
def optimize_drone(drone_id, target_missile, seed_offset):
    sa = SimulatedAnnealing(
        drone_id=drone_id,
        target_missile=target_missile,
        max_iter=1000,
        init_temp=10000,
        cool_rate=0.98,
        seed_offset=seed_offset
    )
    return sa.optimize()


# 可视化函数
def plot_converge_curve(all_results):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57"]
    for idx, result in enumerate(all_results):
        drone_id = result["drone_id"]
        target_missile = result["target_missile"]
        history = result["converge_history"]
        label = f"FY{drone_id[2:]}→M{target_missile[1:]}"
        plt.plot(
            range(1, len(history) + 1),
            history,
            color=colors[idx],
            linewidth=2,
            marker="o",
            markersize=3,
            label=label
        )

    plt.xlabel("模拟退火迭代次数", fontsize=12)
    plt.ylabel("最优适应度 (s)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()


# 导弹有效遮挡时间可视化图表
def plot_missile_shield_visualization(missile_intervals):
    """
    绘制3枚导弹有效遮挡时间的柱状图和遮挡区间时序图
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))  # 2行1列子图

    # 导弹遮挡时间柱状图
    missile_ids = sorted(missile_intervals.keys())  # 按M1→M2→M3排序
    shield_times = []
    for mid in missile_ids:
        # 计算每枚导弹的总遮挡时间
        merged = merge_intervals(missile_intervals[mid])
        total_time = round(sum(end - start for start, end in merged), 3)
        shield_times.append(total_time)

    # 绘制柱状图
    colors = ["#FF7675", "#74B9FF", "#00B894"]  # 红、蓝、绿区分3枚导弹
    bars = ax1.bar(missile_ids, shield_times, color=colors, width=0.6, edgecolor="black", linewidth=1.2)

    # 柱状图添加数值标签
    for bar, time in zip(bars, shield_times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,  # 柱子中心x坐标
            height + 0.1,  # 标签y坐标（柱子顶部+偏移）
            f"{time}s",    # 标签文本
            ha='center', va='bottom', fontsize=20, fontweight='bold'
        )

    # 子图1样式设置
    ax1.set_xlabel("导弹编号", fontsize=20)
    ax1.set_ylabel("有效遮挡时间 (s)", fontsize=20)
    ax1.grid(True, axis='y', linestyle="--", alpha=0.7)

    # 导弹遮挡区间时序图
    y_positions = [1, 2, 3]  # M1→y=1, M2→y=2, M3→y=3
    colors_seq = ["#E17055", "#0984E3", "#27AE60"]  # 时序图颜色

    for idx, (mid, y_pos) in enumerate(zip(missile_ids, y_positions)):
        merged_intervals = merge_intervals(missile_intervals[mid])
        # 绘制每个遮挡区间（水平线段）
        for (start, end) in merged_intervals:
            ax2.hlines(
                y=y_pos, xmin=start, xmax=end,
                color=colors_seq[idx], linewidth=8, alpha=0.8,
                label=f"{mid} 遮挡区间" if idx == 0 and start == merged_intervals[0][0] else ""
            )
        # 绘制导弹ID标签
        ax2.text(
            0.5, y_pos, mid,
            ha='left', va='center', fontsize=20, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors_seq[idx], alpha=0.3)
        )

    # 子图2样式设置
    ax2.set_xlabel("时间 (s)", fontsize=20)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(missile_ids)
    ax2.grid(True, axis='x', linestyle="--", alpha=0.7)

    # 调整布局，避免标签重叠
    plt.tight_layout()
    # 保存图表（高清PNG格式）
    plt.savefig("问题5_导弹有效遮挡时间可视化.png", dpi=300, bbox_inches='tight')
    plt.show()


# 主函数
def main():
    print("开始求解问题5：5架无人机干扰3枚导弹（模拟退火）")

    # 无人机-导弹分配
    assign_result = an_de_to_me()
    print("无人机-导弹干扰分配结果")
    for drone_id, data in assign_result.items():
        print(f"无人机{drone_id} → 导弹{data['target_missile']}（距离：{data['distance']}m）")
    print()

    # 优化各无人机
    drone_list = ["FY1", "FY2", "FY3", "FY4", "FY5"]
    drone_params = [(drone_id, assign_result[drone_id]["target_missile"], i)
                    for i, drone_id in enumerate(drone_list)]

    print(f"启动多进程优化，共{len(drone_list)}个进程...")
    with mp.Pool(processes=min(len(drone_list), mp.cpu_count())) as pool:
        optimize_func = partial(optimize_drone)
        all_results = pool.starmap(optimize_func, drone_params)

    # 按无人机编号排序结果
    all_results.sort(key=lambda x: x["drone_id"])

    # 汇总结果
    print("5架无人机优化结果汇总")
    total_fitness = 0.0
    total_base_shield = 0.0  # 所有无人机合并后的遮挡时间总和
    total_reward = 0.0
    total_valid_smoke = 0

    # 按导弹分类，收集所有干扰该导弹的无人机的合并区间
    missile_intervals = {}
    for result in all_results:
        missile_id = result["target_missile"]
        if missile_id not in missile_intervals:
            missile_intervals[missile_id] = []
        missile_intervals[missile_id].extend(result["merged_intervals"])

    # 打印无人机对导弹的贡献
    print("【无人机-导弹干扰贡献明细】")
    for result in all_results:
        print(
            f"无人机{result['drone_id']:>2} → 导弹{result['target_missile']} | "
            f"合并遮挡{result['base_shield_time']:>6.3f}s | "
            f"有效弹{result['valid_smoke_count']:>2}枚 | "
            f"适应度{result['best_fitness']:>7.3f}s"
        )
    print()

    # 打印导弹最终遮挡时间
    print("【导弹最终有效遮挡时间统计】")
    print(f"{'导弹ID':<8} {'总遮挡时间':<12} {'遮挡区间数量':<12} {'具体遮挡区间':<40}")
    total_missile_shield = 0.0
    for mid in sorted(missile_intervals.keys()):
        merged = merge_intervals(missile_intervals[mid])
        shield_time = round(sum(end - start for start, end in merged), 3)
        interval_count = len(merged)
        # 格式化区间显示
        intervals_str = str([(round(s,2), round(e,2)) for s,e in merged]) if merged else "无"
        if len(intervals_str) > 35:
            intervals_str = intervals_str[:32] + "..."
        print(f"{mid:<8} {shield_time:<12.3f} {interval_count:<12d} {intervals_str:<40}")
        total_missile_shield += shield_time
    print(f"{'合计':<8} {total_missile_shield:<12.3f} {'-':<12} {'-':<40}")
    print()

    # 全局汇总信息
    print("【全局汇总信息】")
    for result in all_results:
        total_fitness += result['best_fitness']
        total_base_shield += result['base_shield_time']
        total_reward += result['total_reward']
        total_valid_smoke += result['valid_smoke_count']
    print(f"总适应度:               {total_fitness:.3f}s")
    print(f"所有无人机遮挡时间总和:   {total_base_shield:.3f}s")
    print(f"所有导弹最终遮挡时间:     {total_missile_shield:.3f}s")
    print(f"总有效烟幕弹数量:         {total_valid_smoke}枚")

    # 绘制可视化图表（收敛曲线 + 导弹遮挡时间图表）
    print("正在生成可视化图表...")
    plot_converge_curve(all_results)
    plot_missile_shield_visualization(missile_intervals)
    print("图表生成完成！")


if __name__ == "__main__":
    mp.freeze_support()
    main()