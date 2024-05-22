# coding=utf-8
# @Author : Xenon
# @File : GUI.py
# @Date : 2024/5/21 22:27 
# @IDE : PyCharm(2023.3) Python3.9.13

import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from key_predictions import MLTemperaturePredictor, MLPTemperaturePredictor, LSTMTemperaturePredictor, get_df

def predict_temperature_for_date():
    try:
        df = get_df()  # 加载数据
        year = int(year_entry.get())
        month = int(month_entry.get())
        day = int(day_entry.get())
        target_date = datetime(year, month, day)

        # 调用不同的预测器并显示结果
        ml_predictor = MLTemperaturePredictor()
        ml_results = ml_predictor.predict_temperature(year, month, day)

        mlp_predictor = MLPTemperaturePredictor()
        mlp_result = mlp_predictor.predict_temperature(year, month, day)

        lstm_predictor = LSTMTemperaturePredictor()
        lstm_result = lstm_predictor.predict_temperature(df, target_date)  # 注意这里需要df作为参数

        # 假设你已经有了一个全局的df变量，它包含了所有的气温数据
        # 如果不是，你需要在GUI初始化时加载它

        # 显示结果
        result_text = f"Decision Tree: {ml_results[0]:.2f}°C\n"
        result_text += f"Linear Regression: {ml_results[1]:.2f}°C\n"
        result_text += f"Random Forest: {ml_results[2]:.2f}°C\n"
        result_text += f"MLP: {mlp_result:.2f}°C\n"
        result_text += f"LSTM: {lstm_result:.2f}°C"

        result_label.config(text=result_text)
    except ValueError as e:
        messagebox.showerror("Error", str(e))
    except Exception as e:
        messagebox.showerror("Unexpected Error", str(e))

    # 假设df已经在这里被加载，但通常你需要在GUI初始化时加载它
# df = pd.read_csv("./Lingcang202001-202312.csv")  # 这里只是一个示例，实际使用时需要取消注释并加载数据

# 创建GUI界面
root = tk.Tk()
root.title("Temperature Predictor")

# 输入框
year_label = tk.Label(root, text="Year:")
year_label.pack()
year_entry = tk.Entry(root)
year_entry.pack()

month_label = tk.Label(root, text="Month:")
month_label.pack()
month_entry = tk.Entry(root)
month_entry.pack()

day_label = tk.Label(root, text="Day:")
day_label.pack()
day_entry = tk.Entry(root)
day_entry.pack()

# 预测按钮
predict_button = tk.Button(root, text="Predict Temperature", command=predict_temperature_for_date)
predict_button.pack()

# 结果标签
result_label = tk.Label(root, text="")
result_label.pack()

# 运行GUI事件循环
root.mainloop()
