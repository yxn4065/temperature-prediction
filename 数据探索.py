# coding=utf-8
# @Author : Xenon
# @File : 数据探索.py
# @Date : 2024/5/20 15:19 
# @IDE : PyCharm(2023.3) Python3.9.13

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv("./Lingcang202001-202312.csv")

# 查看数据集的前几行
print(df.head())

# 转换年月日为日期格式，并设置为索引
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df.set_index('date', inplace=True)

# 可视化平均气温的时间序列
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['actual'], label='Average Temperature')
plt.title('Average Daily Temperature over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# 可视化最高和最低气温的分布情况
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='temp_min', label='Lowest Temperature', fill=True)
sns.kdeplot(data=df, x='temp_max', label='Highest Temperature', fill=True)
plt.title('Distribution of Lowest and Highest Daily Temperatures')
plt.xlabel('Temperature (°C)')
plt.ylabel('Density')
plt.legend()
plt.show()

# 计算并可视化每月的平均气温
monthly_avg = df.resample('M')['actual'].mean()
plt.figure(figsize=(12, 6))
plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linestyle='-', color='b')
plt.title('Monthly Average Temperature')
plt.xlabel('Month')
plt.ylabel('Average Temperature (°C)')
plt.grid(True)
plt.show()

