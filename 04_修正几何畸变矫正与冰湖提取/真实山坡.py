import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.signal import argrelextrema

# 读取CSV文件中的散点数据
pdcsv = pd.read_json(r"C:\09_Code\Gis_Script\04_修正几何畸变矫正与冰湖提取\A.json")

# 提取散点数据
x_data = pdcsv.index
y_data = pdcsv

# 对数据进行样条插值
cs = CubicSpline(x_data, y_data, bc_type='natural')

# # 生成插值曲线的数据
x_fit = np.linspace(min(x_data), max(x_data), 122)
y_fit = cs(x_fit)

# 计算一阶导数、二阶导数和三阶导数
first_derivative = cs.derivative(nu=1)(x_fit)
second_derivative = cs.derivative(nu=2)(x_fit)
third_derivative = cs.derivative(nu=3)(x_fit)

# 找到n阶导数为0的位置
zero_first_derivative_indices = np.where(np.diff(np.sign(first_derivative)) != 0)[0] + 1
zero_second_derivative_indices = np.where(np.diff(np.sign(second_derivative)) != 0)[0] + 1
zero_third_derivative_indices = np.where(np.diff(np.sign(third_derivative)) != 0)[0] + 1

# 找到n阶导数满足要求的位置
positive_first_derivative_indices = np.where(first_derivative >= 0)[0]
positive_second_derivative_indices = np.where(second_derivative <= 0)[0]
positive_third_derivative_indices = np.where(third_derivative >= 0)[0]

# 使用集合操作符进行交集操作
positive_first_set = set(positive_first_derivative_indices)
positive_second_set = set(positive_second_derivative_indices)
positive_third_set = set(positive_third_derivative_indices)
candidate_turning_points = list(positive_first_set & positive_second_set & positive_third_set)

# 计算曲率
curvature = np.abs(second_derivative) / (1 + first_derivative**2)**(3/2)
local_maxima_indices = argrelextrema(curvature, np.greater)[0]

# 筛选candidate_turning_points中与曲率局部极大值index相同的点
filtered_turning_points = [i for i in local_maxima_indices if i in candidate_turning_points]

# 将原函数中的局部最大值添加进来
filtered_turning_points = list(set(filtered_turning_points + list(argrelextrema(y_fit, np.greater)[0])))

plt.figure(figsize=(20, 18))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14, 'figure.dpi': 300})

plt.subplot(3, 3, 1)
plt.plot(x_fit, y_fit, color='red', linewidth=1.5)
plt.scatter(x_fit, y_fit, color='green', marker='o', s=20)
plt.ylabel('Y')
plt.title('Simulated Mountain Line')
plt.grid(True)

plt.subplot(3, 3, 2)
plt.plot(x_fit, first_derivative, color='green')
for idx in zero_first_derivative_indices:
    plt.axvline(x=x_fit[idx], color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
plt.ylabel('First Derivative')
plt.title('First Derivative the Simulated Mountain Line')
plt.grid(True)

plt.subplot(3, 3, 3)
plt.plot(x_fit, second_derivative, color='blue')
for idx in zero_second_derivative_indices:
    plt.axvline(x=x_fit[idx], color='green', linestyle='--', linewidth=1.5, alpha=0.7)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # 使用科学计数法表示y轴标注  
plt.ylabel('Second Derivative')
plt.title('Second Derivative of the Simulated Mountain Line')
plt.grid(True)

plt.subplot(3, 3, 4)
plt.plot(x_fit, third_derivative, color='purple')
for idx in zero_third_derivative_indices:
    plt.axvline(x=x_fit[idx], color='darkslateblue', linestyle='--', linewidth=1.5, alpha=0.7)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # 使用科学计数法表示y轴标注
plt.xlabel('X')
plt.ylabel('Third Derivative')
plt.title('Third Derivative of the Simulated Mountain Line')
plt.grid(True)

plt.subplot(3, 3, 5)
plt.plot(x_fit, y_fit, color='red', linewidth=1.5)  # 先绘制曲线
plt.scatter(x_fit[candidate_turning_points], y_fit[candidate_turning_points], color='green', marker='o', s=30)  # 绘制转折点
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # 使用科学计数法表示y轴标注
plt.ylabel('Y')
plt.title('Selecting Points through Multi-level Derivatives')
plt.grid(True)

plt.subplot(3, 3, 6)
plt.plot(x_fit, curvature, color='darkslateblue', label='Curvature')  # 将曲线颜色改为青色
for idx in local_maxima_indices:
    plt.axvline(x=x_fit[idx], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)  # 将局部极大值点的竖线颜色改为洋红色
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # 使用科学计数法表示y轴标注
plt.ylabel('Curvature')
plt.title('Curvature of the Simulated Mountain Line')
plt.grid(True)

plt.subplot(3, 3, 9)
plt.plot(x_fit, y_fit, color='red')
plt.scatter(x_fit[filtered_turning_points], y_fit[filtered_turning_points], color='green', marker='o', s=30)
plt.ylabel('Y')
plt.title('Optimizing Point Selection through Curvature')
plt.grid(True)

plt.tight_layout()
plt.show()