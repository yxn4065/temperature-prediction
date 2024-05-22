# coding=utf-8
# @Author : Xenon
# @File : main_predictions.py
# @Date : 2024/5/21 20:16 
# @IDE : PyCharm(2023.3) Python3.9.13

import joblib
import pandas as pd
from keras.models import load_model


class MLTemperaturePredictor:
    """机器学习气温预测类"""

    def __init__(self, models_dir='./models/'):
        self.models_dir = models_dir
        self.load_models()  # 加载模型

    def load_models(self):
        """加载机器学习模型"""
        self.dtr = joblib.load(f"{self.models_dir}decision_tree_model.pkl")
        self.lr = joblib.load(f"{self.models_dir}linear_regression_model.pkl")
        self.rfr = joblib.load(f"{self.models_dir}random_forest_model.pkl")
        # print("Machine Learning Models loaded successfully!")

    def prepare_input_data(self, year, month, day):
        """准备输入数据以匹配模型输入"""
        input_data_sklearn = pd.DataFrame({'year': [year], 'month': [month], 'day': [day]})
        return input_data_sklearn

    def predict_temperature(self, year, month, day):
        """预测气温 """
        # 构建输入数据
        input_data_sklearn = self.prepare_input_data(year, month, day)

        # 使用scikit-learn模型进行预测
        prediction_dtr = self.dtr.predict(input_data_sklearn)[0]
        prediction_lr = self.lr.predict(input_data_sklearn)[0]
        prediction_rfr = self.rfr.predict(input_data_sklearn)[0]

        # 返回预测结果
        return prediction_dtr, prediction_lr, prediction_rfr


class MLPTemperaturePredictor:
    """使用MLP神经网络进行气温预测"""

    def __init__(self, models_dir='./models/'):
        self.model_path = models_dir
        self.scaler_path = models_dir
        self.model = self.load_model()
        self.scaler = self.load_scaler()

    def load_model(self):
        """加载MLP模型"""
        return load_model(f"{self.model_path}MPL_temperature_prediction_model.h5")

    def load_scaler(self):
        """加载MinMaxScaler"""
        return joblib.load(f"{self.scaler_path}mlp_scaler.joblib")

    def preprocess_input(self, year, month, day):
        """预处理输入数据"""
        input_data = pd.DataFrame({'year': [year], 'month': [month], 'day': [day]})
        input_data_scaled = self.scaler.transform(input_data)
        return input_data_scaled

    def predict_temperature(self, year, month, day):
        """预测气温"""
        input_data_scaled = self.preprocess_input(year, month, day)
        prediction = self.model.predict(input_data_scaled)

        # 返回预测值
        return prediction.flatten()[0]


class LSTMTemperaturePredictor:
    """使用LSTM神经网络进行气温预测"""

    def __init__(self, model_path='./models/',):
        self.model = load_model(f"{model_path}LSTM_temperature_prediction_model.h5")
        # self.scaler = MinMaxScaler()
        self.scaler = joblib.load(f"{model_path}lstm_scaler.joblib")

    def prepare_input(self, df, target_date, n_input):
        """准备LSTM模型的输入数据"""
        # 找到目标日期前n_input天的数据
        target_index = df.index.get_loc(target_date)
        if target_index < n_input:  # 如果数据不足，则无法预测
            raise ValueError(f"Not enough data to predict for {target_date}")

        # 准备输入数据
        input_data = df['average'].iloc[target_index - n_input:target_index].values.reshape(-1, 1)
        input_data_scaled = self.scaler.transform(input_data)
        return input_data_scaled

    def predict_temperature(self, df, target_date):
        """预测气温"""
        n_input = 7  # 与训练时相同的时间步长
        input_data_scaled = self.prepare_input(df, target_date, n_input)  # 准备输入数据

        # 使用模型进行预测
        prediction_scaled = self.model.predict(input_data_scaled.reshape(1, n_input, 1))

        # 反归一化预测结果
        prediction = self.scaler.inverse_transform(prediction_scaled)

        return prediction[0][0]


def get_df(path='./Lingcang202001-202312.csv'):
    # 读取CSV文件
    df = pd.read_csv(path)

    # 将year, month, day合并为日期，并转换为datetime类型
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # 删除原始的year, month, day列
    df.drop(['year', 'month', 'day'], axis=1, inplace=True)

    # 按照日期排序，确保数据的连续性
    df.sort_values('date', inplace=True)

    # 设置日期为索引
    df.set_index('date', inplace=True)

    # 划分训练集和验证集
    # 训练集：2020年1月1日至2022年11月30日
    # 验证集：2022年12月1日至2022年12月31日
    train_end_date = "2022-11-30"
    validation_start_date = "2022-12-01"
    validation_end_date = "2022-12-31"
    test_start_date = "2023-01-01"

    # 将数据划分为训练集、验证集和测试集
    train_df = df.loc[df.index <= train_end_date]
    validation_df = df.loc[(df.index >= validation_start_date) & (df.index <= validation_end_date)]
    test_df = df.loc[df.index >= test_start_date]

    return df


def get_real_temperature(year, month, day,path='./Lingcang202001-202312.csv'):
    """获取真实气温"""
    # 读取CSV文件
    df = pd.read_csv(path)
    # 使用布尔索引来查找匹配的行
    row = df[(df['year'] == year) & (df['month'] == month) & (df['day'] == day)]
    if not row.empty:
        try:
            real_temp = row['average'].iloc[0]
        except KeyError:
            real_temp = "N/A"

        return real_temp
    else:
        return "N/A"


if __name__ == '__main__':
    # 使用机器学习方法进行预测
    p1 = MLTemperaturePredictor()
    t1_dtr, t1_lr, t1_lfr = p1.predict_temperature(2023, 5, 20)
    print(f"The predicted average temperature for May 20, 2023 using Decision Tree is: {t1_dtr:.2f}°C")
    print(f"The predicted average temperature for May 20, 2023 using Linear Regression is: {t1_lr:.2f}°C")
    print(f"The predicted average temperature for May 20, 2023 using Random Forest is: {t1_lfr:.2f}°C")


    # 使用神经网络方法进行预测
    predictor = MLPTemperaturePredictor()
    t2 = predictor.predict_temperature(2023, 5, 20)
    print(f"The predicted average temperature for May 20, 2023 is: {t2:.2f}°C")

    data_path = './Lingcang202001-202312.csv'
    # 使用LSTM神经网络进行预测
    df = get_df(data_path)
    # 创建一个预测器实例
    predictor = LSTMTemperaturePredictor()

    # 预测2023年某一天的气温
    target_date = pd.Timestamp('2023-05-20')  # 使用pandas的Timestamp来确保日期格式正确
    predicted_temp = predictor.predict_temperature(df, target_date)
    print(f"Predicted average temperature for {target_date}: {predicted_temp:.2f}°C")

    # 获取某一天真实气温
    real_t = get_real_temperature(2023, 5, 20)
    print(real_t)
