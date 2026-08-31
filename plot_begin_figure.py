import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.switch_backend('TkAgg')
# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义所有点的坐标
# 目标点
target = (0, 200, 0)

# 导弹位置
missiles = {
    "M1": (20000, 0, 2000),
    "M2": (19000, 600, 2100),
    "M3": (18000, -600, 1900)
}

# 无人机位置
uavs = {
    "FY1": (17800, 0, 1800),
    "FY2": (12000, 1400, 1400),
    "FY3": (6000, -3000, 700),
    "FY4": (11000, 2000, 1800),
    "FY5": (13000, -2000, 1300)
}

# 创建三维图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制目标点
ax.scatter(target[0], target[1], target[2], c='red', s=200, marker='*', label='目标点')
ax.scatter(0, 0, 0, c='blue', s=200, marker='*', label='假目标')

# 绘制导弹
missile_x = [pos[0] for pos in missiles.values()]
missile_y = [pos[1] for pos in missiles.values()]
missile_z = [pos[2] for pos in missiles.values()]
ax.scatter(missile_x, missile_y, missile_z, c='blue', s=100, marker='D', label='导弹')

# 绘制无人机
uav_x = [pos[0] for pos in uavs.values()]
uav_y = [pos[1] for pos in uavs.values()]
uav_z = [pos[2] for pos in uavs.values()]
ax.scatter(uav_x, uav_y, uav_z, c='green', s=100, marker='^', label='无人机')

# 添加标签
for name, (x, y, z) in missiles.items():
    ax.text(x, y, z, name, fontsize=10)

for name, (x, y, z) in uavs.items():
    ax.text(x, y, z, name, fontsize=10)

ax.text(target[0], target[1]+200, target[2],'真目标', fontsize=10)
ax.text(0, -200, 0, '假目标', fontsize=10)

# 设置坐标轴标签
ax.set_xlabel('X坐标')
ax.set_ylabel('Y坐标')
ax.set_zlabel('Z坐标')


# 添加图例
ax.legend()

# 设置坐标轴范围，使图形更美观
ax.set_xlim([-1000, 21000])
ax.set_ylim([-4000, 3000])
ax.set_zlim([0, 2500])
ax.view_init(elev=19, azim=-143)

# 启用交互模式，允许鼠标拖动旋转
plt.tight_layout()
plt.savefig('初始位置图.png', dpi=300)
plt.show()

