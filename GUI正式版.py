# coding=utf-8
# @Author : Xenon
# @File : GUI2.py
# @Date : 2024/5/21 22:33 
# @IDE : PyCharm(2023.3) Python3.9.13

import tkinter as tk
from tkinter import font as tkFont  # 导入Tkinter的font模块
from tkinter import messagebox
from datetime import datetime
from key_predictions import MLTemperaturePredictor, MLPTemperaturePredictor, LSTMTemperaturePredictor, get_df, \
    get_real_temperature


def on_predict_click():
    try:
        year = int(year_entry.get())
        month_name = selected_month.get()
        day = int(day_entry.get())

        # 将月份名称转换为数字（如果需要的话）
        month_dict = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4,
            'May': 5, 'June': 6, 'July': 7, 'August': 8,
            'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        month = month_dict[month_name]

        # 创建目标日期对象
        target_date = datetime(year, month, day)

        # 使用机器学习模型进行预测
        ml_predictor = MLTemperaturePredictor()
        t1_dtr, t1_lr, t1_rfr = ml_predictor.predict_temperature(year, month, day)

        # 使用MLP模型进行预测
        mlp_predictor = MLPTemperaturePredictor()
        t2 = mlp_predictor.predict_temperature(year, month, day)

        # 使用LSTM模型进行预测
        lstm_predictor = LSTMTemperaturePredictor()
        t3 = lstm_predictor.predict_temperature(df, target_date)

        # 当天真实气温
        real_temp = get_real_temperature(year, month, day)

        # 当天平均气温
        try:
            avg_temp = (t1_lr + t1_dtr + t1_rfr + t2 + t3) / 5
        except Exception as e:
            avg_temp = "N/A"
            messagebox.showerror("Error", str(e))

        # 显示结果
        result_text = f"Decision Tree: {t1_dtr:.2f}°C\n"
        result_text += f"Linear Regression: {t1_lr:.2f}°C\n"
        result_text += f"Random Forest: {t1_rfr:.2f}°C\n"
        result_text += f"MLP: {t2:.2f}°C\n"
        result_text += f"LSTM: {t3:.2f}°C\n"
        result_text += f"Real Temperature: {real_temp}°C\n"
        result_text += f"Average Temperature: {avg_temp:.2f}°C"

        result_font = tkFont.Font(family="Arial", size=14, weight="bold")  # 创建一个字体对象
        result_label.config(text=result_text, font=result_font)
        # result_label.config(text=result_text, font=result_font,justify='left')
        # print(result_text)

    except ValueError as e:
        messagebox.showerror("Error", str(e))
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))


# 加载气温数据
df = get_df()

# 创建GUI界面
root = tk.Tk()
root.title("基于深度学习的气温预测程序 by@Xenon")

# 固定窗口大小为 500x400 像素
root.geometry("500x400")

# 输入框
year_label = tk.Label(root, text="Year:")
year_label.pack()
# year_entry = tk.Entry(root)
# year_entry.pack()
year_entry = tk.Entry(root, width=5)
year_entry.insert(0, '2023')  # 设置默认年份为2023
year_entry.pack()

month_label = tk.Label(root, text="Month:")
month_label.pack()
# month_entry = tk.Entry(root)
# month_entry.pack()

# 创建月份列表
months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

# 创建变量来存储选中的月份
selected_month = tk.StringVar(root)
selected_month.set('January')  # 设置默认月份为January

# 创建OptionMenu小部件
month_optionmenu = tk.OptionMenu(root, selected_month, *months)
month_optionmenu.pack()

day_label = tk.Label(root, text="Day:")
day_label.pack()
day_entry = tk.Entry(root)
day_entry.pack()

# 预测按钮
predict_button = tk.Button(root, text=">>开始预测<<", command=on_predict_click, fg="red", padx=10, pady=5,
                           font=('SimHei', 12))
predict_button.pack(pady=12)

# 结果标签
result_label = tk.Label(root, text="")
result_label.pack()

# 运行GUI事件循环
root.mainloop()
