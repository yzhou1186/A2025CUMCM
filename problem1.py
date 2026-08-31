import math


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


def get_pit_lit_pos(t):
    dir_ec = (-20000.0, 0.0, -2000.0)
    dir_l = vector_length(dir_ec)
    unit_dir = (dir_ec[0] / dir_l, dir_ec[1] / dir_l, dir_ec[2] / dir_l)
    dis_tan = 300.0 * t
    if dis_tan >= dir_l:
        return (0.0, 0.0, 0.0)
    return (20000.0 + unit_dir[0] * dis_tan, 0.0 + unit_dir[1] * dis_tan, 2000.0 + unit_dir[2] * dis_tan)


def get_she_pos(t, x0, z0):
    if t < 5.1 - 1e-9 or t > 25.1 + 1e-9:
        return None
    z_pos = z0 - 3.0 * (t - 5.1)
    return (float(x0), 0.0, z_pos)


# 解析判断：线段L→P是否与球体相交
def is_sme_it_she_ana(L, P, S, R):
    LP = vector_subtract(P, L)  # P - L
    LS = vector_subtract(L, S)  # L - S
    a = dot_product(LP, LP)  # |LP|²
    b = 2 * dot_product(LS, LP)  # 2*(L-S)·(P-L)
    c = dot_product(LS, LS) - R ** 2  # |L-S|² - R²

    dis_miant = b ** 2 - 4 * a * c
    if dis_miant < -1e-9:  # 无实根
        return False
    dis_miant = max(dis_miant, 0.0)
    sqrt_d = math.sqrt(dis_miant)

    t1 = (-b - sqrt_d) / (2 * a) if a > 1e-12 else 0.0
    t2 = (-b + sqrt_d) / (2 * a) if a > 1e-12 else 0.0

    return (t1 >= -1e-9 and t1 <= 1.0 + 1e-9) or (t2 >= -1e-9 and t2 <= 1.0 + 1e-9)


# 判断圆柱是否被完全遮挡
def is_cyder_ful_oced(t, sphe_x0, sphe_z0):
    lit_pos = get_pit_lit_pos(t)
    she_pos = get_she_pos(t, sphe_x0, sphe_z0)

    # 球体不存在时，圆柱不可能被完全遮挡
    if she_pos is None:
        return False

    # 圆柱与球体参数
    cyl_rd = 7.0
    cyl_ht = 10.0
    cyl_y = 200.0
    she_rd = 10.0

    # 快速判断：光源在球体内，所有光线都被遮挡
    LS = vector_subtract(lit_pos, she_pos)
    if vector_length(LS) <= she_rd + 1e-9:
        return True

    # 圆柱关键特征点
    C_bottom = (0.0, cyl_y, 0.0)  # 底部中心
    C_top = (0.0, cyl_y, cyl_ht)  # 顶部中心

    # 增加照射锥采样点 - 在不同高度和半径位置采样
    key_rays = []

    # 顶部圆周采样（4个方向）
    key_rays.extend([
        (cyl_rd, cyl_y, cyl_ht),  # 顶部右
        (-cyl_rd, cyl_y, cyl_ht),  # 顶部左
        (0, cyl_y, cyl_ht + cyl_rd),  # 顶部前
        (0, cyl_y, cyl_ht - cyl_rd)  # 顶部后
    ])

    # 底部圆周采样（4个方向）
    key_rays.extend([
        (cyl_rd, cyl_y, 0.0),  # 底部右
        (-cyl_rd, cyl_y, 0.0),  # 底部左
        (0, cyl_y, cyl_rd),  # 底部前
        (0, cyl_y, -cyl_rd)  # 底部后
    ])

    # 中间高度采样（3个高度层级，每个层级4个方向）
    hts = [cyl_ht * 0.25, cyl_ht * 0.5, cyl_ht * 0.75]
    for h in hts:
        key_rays.extend([
            (cyl_rd, cyl_y, h),  # 右
            (-cyl_rd, cyl_y, h),  # 左
            (0, cyl_y, h + cyl_rd),  # 前
            (0, cyl_y, h - cyl_rd)  # 后
        ])

    # 对角线方向采样（45度角）
    dg_for = math.cos(math.pi / 4)  # 对角线位置因子
    key_rays.extend([
        (cyl_rd * dg_for, cyl_y, cyl_ht * 0.5 + cyl_rd * dg_for),  # 右上前
        (-cyl_rd * dg_for, cyl_y, cyl_ht * 0.5 + cyl_rd * dg_for),  # 左上前
        (cyl_rd * dg_for, cyl_y, cyl_ht * 0.5 - cyl_rd * dg_for),  # 右上后
        (-cyl_rd * dg_for, cyl_y, cyl_ht * 0.5 - cyl_rd * dg_for)  # 左上后
    ])

    # 轴线点采样
    key_rays.extend([
        C_bottom,  # 底部中心
        C_top,  # 顶部中心
        (0.0, cyl_y, cyl_ht * 0.25),  # 1/4高度轴线
        (0.0, cyl_y, cyl_ht * 0.5),  # 1/2高度轴线
        (0.0, cyl_y, cyl_ht * 0.75)  # 3/4高度轴线
    ])

    # 锥轴向量（L→C_bottom）与长度
    cone_axis = vector_subtract(C_bottom, lit_pos)
    cone_axis_len = vector_length(cone_axis)
    if cone_axis_len < 1e-12:  # 光源与底部中心重合，特殊处理
        return is_sme_it_she_ana(lit_pos, C_top, she_pos, she_rd)

    # 判断所有关键射线是否都被球体遮挡
    for target_point in key_rays:
        if not is_sme_it_she_ana(lit_pos, target_point, she_pos, she_rd):
            return False  # 存在未被遮挡的射线，不是完全遮挡

    # 所有关键射线均被遮挡 → 圆柱完全被遮挡
    return True


def find_trans_t(start, end, sphere_x0, sphere_z0, target_state):
    eps = 1e-10
    for _ in range(50):
        mid = (start + end) / 2
        cur_ste = is_cyder_ful_oced(mid, sphere_x0, sphere_z0)
        if cur_ste == target_state:
            end = mid
        else:
            start = mid
        if end - start < eps:
            break
    return (start + end) / 2


def cal_ful_ocion_t(sphe_x0, sphe_z0, corse_step=0.01):
    start_time = 5.1
    end_time = 25.1
    ful_ocion_invl = []

    # 初始状态检查
    inal_ck_t = max(start_time, 5.5)
    pre_se = is_cyder_ful_oced(inal_ck_t, sphe_x0, sphe_z0)
    inal_start = None

    # 回溯查找初始完全遮挡开始点
    if pre_se:
        inal_start = find_trans_t(start_time, inal_ck_t,
                                  sphe_x0, sphe_z0, True)

    t = inal_ck_t + corse_step
    while t <= end_time:
        cur_se = is_cyder_ful_oced(t, sphe_x0, sphe_z0)

        if cur_se != pre_se:
            # 找到状态转换点
            trans_t = find_trans_t(t - corse_step, t,
                                   sphe_x0, sphe_z0, cur_se)
            if pre_se:
                ful_ocion_invl.append((inal_start, trans_t))
                inal_start = None
            else:
                inal_start = trans_t
            pre_se = cur_se

        t += corse_step

    # 处理结束时仍处于完全遮挡状态的情况
    if pre_se and inal_start is not None:
        ful_ocion_invl.append((inal_start, end_time))

    # 计算完全遮挡时间
    total_time = sum(end - start for start, end in ful_ocion_invl)
    return ful_ocion_invl, round(total_time, 6)


if __name__ == "__main__":
    she_x0 = 17188.0
    she_z0 = 1736.496
    ocon_ivs, tl_ocion_t = cal_ful_ocion_t(she_x0, she_z0)

    print("圆柱被完全遮挡的时间区间:")
    if not ocon_ivs:
        print("无完全遮挡区间（圆柱始终有部分被照射）")
    else:
        for idx, (start, end) in enumerate(ocon_ivs, 1):
            dura = round(end - start, 6)
            print(f"区间 {idx}: {start:.2f}s ~ {end:.2f}s, 持续时间: {dura:.2f}s")
        print(f"\n圆柱被完全遮挡的总时间: {tl_ocion_t:.2f}秒")
