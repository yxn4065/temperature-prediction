# coding=utf-8
# @Author : Xenon
# @File : 临沧天气爬取.py
# @Date : 2024/5/16 16:02 
# @IDE : PyCharm(2023.3) Python3.9.13
import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_data(url):
    # 请求网页（第三方 requests）
    resp = requests.get(url)
    # 对于获取到的 HTML 二进制文件进行 'gbk' 转码成字符串文件
    html = resp.content.decode('gbk')
    # 通过第三方库 BeautifulSoup 缩小查找范围（同样作用的包库还有re模块、xpath等）
    soup = BeautifulSoup(html, 'html.parser')
    # 获取 HTML 中所有<tr>…</tr>标签，因为我们需要的数据全部在此标签中存放
    tr_list = soup.find_all('tr')
    # 初始化日期dates、气候contains、温度temp值
    dates, contains, max_temp, min_temp, av_temp = [], [], [], [], []
    year, month, day = [], [], []
    for data in tr_list[1:]:  # 不要表头
        # 数据值拆分，方便进一步处理（这里可以将获得的列表输出[已注释]，不理解的读者可运行查看)
        sub_data = data.text.split()
        # print(sub_data)

        # 提取日期字符串
        date_str = sub_data[0]

        # 使用字符串的split方法，以'年'、'月'和'日'为分隔符来分割字符串
        y, month_day = date_str.split('年')
        m, d = month_day.split('月')
        d = d.rstrip('日')  # 去掉字符串末尾的'日'字
        # 输出结果
        # print(f"年: {year}, 月: {month}, 日: {day}")

        year.append(y)
        month.append(m)
        day.append(d)

        # 观察上一步获得的列表，这里只想要获得列表中第二个和第三个值，采用切片法获取
        # dates.append(sub_data[0])
        # contains.append(''.join(sub_data[1:3]))
        # print(contains) #天气状况
        # 同理采用切片方式获取列表中的最高、最低气温
        # max_temp.append(','.join(sub_data[3:6]))
        x1 = sub_data[3].replace('℃', '')
        x2 = sub_data[5].replace('℃', '')
        max_temp.append(x1)
        min_temp.append(x2)

        av_t = (int(x1) + int(x2)) / 2
        av_temp.append(av_t)

        # print(max_temp)

    # 使用 _data 表存放日期、天气状况、气温表头及其值
    _data = pd.DataFrame()
    # 分别将对应值传入 _data 表中
    # _data['日期'] = dates
    _data['year'] = year
    _data['month'] = month
    _data['day'] = day
    # _data['天气状况'] = contains
    _data['最低气温'] = max_temp
    _data['最高气温'] = min_temp
    _data['平均气温'] = av_temp
    print(_data)

    return _data


def get_url(start_year_month=202001, end_year_month=202212):
    uu = []

    base_url = "http://www.tianqihoubao.com/lishi/lincang/month/{}"

    # 开始和结束的年月
    # start_year_month = 202001
    # end_year_month = 202212

    # 遍历所有月份
    current_year_month = start_year_month
    while current_year_month <= end_year_month:
        # 格式化URL
        url = base_url.format(current_year_month)
        url += ".html"
        uu.append(url)
        # print(url)  # 或者在这里执行你的数据抓取函数

        # 移到下一个月份
        # 如果月份小于12，则月份加1；否则年份加1，月份重置为1
        if current_year_month % 100 < 12:
            current_year_month += 1
        else:
            current_year_month = (current_year_month // 100 + 1) * 100 + 1

    return uu


if __name__ == '__main__':
    province = "云南"
    city = "临沧"

    # 定义CSV文件的路径
    csv_file_path = './Lingcang.csv'

    # 开始和结束的年月
    start_year_month = 202010
    end_year_month = 202212

    # 得到所有url
    urls = get_url(start_year_month, end_year_month)

    for url in urls:
        # url = f"http://www.tianqihoubao.com/lishi/lincang/month/{202001}.html"
        data_month = get_data(url)

        # 拼接所有表并重新设置行索引（若不进行此步操作，可能或出现多个标签相同的值）
        data = pd.concat([data_month]).reset_index(drop=True)

        # 将 _data 表以 .csv 格式存入指定文件夹中，并设置转码格式防止乱花（注：此转码格式可与 HTML 二进制转字符串的转码格式不同）
        # data.to_csv('./LingcangMouth1.csv', encoding="utf-8", index=False)

        # 检查CSV文件是否存在
        if os.path.exists(csv_file_path):
            # 如果文件存在，则以追加模式打开文件，并写入新的数据
            with open(csv_file_path, 'a', encoding='utf-8', newline='') as f:
                # 将DataFrame转换为CSV格式的字符串，并去掉表头（因为表头已经在文件中）
                csv_str = data.to_csv(index=False, header=False)
                # 写入CSV字符串到文件
                f.write(csv_str)
        else:
            # 如果文件不存在，则正常写入DataFrame到CSV文件
            data.to_csv(csv_file_path, encoding="utf-8", index=False)

        print(f"{url} 气温爬取成功!")

        time.sleep(2)
