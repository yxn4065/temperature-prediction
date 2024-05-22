# coding=utf-8
# @Author : Xenon
# @File : 全国天气信息爬取.py
# @Date : 2024/5/19 16:18
# @IDE : PyCharm(2023.3) Python3.9.13
import json
import os
import random
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


# 发送请求获取网页内容
def get_html(url, encoding='utf-8'):
    # 使用fake_useragent随机生成请求头
    ua = UserAgent()
    headers = {'User-Agent': ua.random}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:  # 判断请求是否成功
        # 转为GBK编码
        response.encoding = encoding
        return response.text
    else:
        return None


# 保存字典到文件
def save_dict_to_file(dict_data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dict_data, f, ensure_ascii=False, indent=4)

    print(f"内容已保存到{file_name}")


# 从JSON文件中加载字典
def load_dict_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        dict_data = json.load(f)

    return dict_data


# 获取所有省名
def get_province_name():
    url = "http://www.tianqihoubao.com/lishi/"
    html = get_html(url, encoding='GBK')
    province_dict = {}  # 创建一个空字典来存储城市信息

    if html:
        soup = BeautifulSoup(html, 'html.parser')
        # print(soup)
        # 查找所有的dt标签，这些标签包含了城市名和链接
        dts = soup.find_all('dt')

        for dt in dts:
            # 在每个dt标签内查找a标签和b标签
            a_tag = dt.find('a')
            b_tag = dt.find('b')

            # 提取href属性和b标签的文本内容
            if a_tag and b_tag:
                href = a_tag['href']
                province_name = b_tag.text
                province_dict[province_name] = href

    # print(province_dict)
    print(f"省份信息获取完毕，共获取到{len(province_dict)}个省份信息")

    # 保存一级城市信息
    save_dict_to_file(province_dict, 'city_dict_1.json')

    return province_dict


# 获取所有一级城市名
def get_city_name(province_dict):
    city_dict = {}  # 创建一个空字典来存储城市信息
    n = 0
    for province_name, province_url in province_dict.items():
        url = f"{base_url}{province_url}"
        print(f"正在获取{province_name}的城市信息,当前进度{n}/{len(province_dict)} {url}")
        n += 1
        html = get_html(url, encoding='utf-8')

        if html:
            soup = BeautifulSoup(html, 'html.parser')
            # 查找所有的dt标签，这些标签包含了城市名和链接
            dts = soup.find_all('dt')

            for dt in dts:
                # 在每个dt标签内查找a标签和b标签
                a_tag = dt.find('a')
                b_tag = dt.find('b')

                # 提取href属性和b标签的文本内容
                if a_tag and b_tag:
                    href = a_tag['href']
                    city_name = b_tag.text
                    city_dict[city_name] = href

        time.sleep(0.1)

    # print(city_dict)
    print(f"城市信息获取完毕，共获取到{len(city_dict)}个城市信息")

    # 保存一级城市信息
    # save_dict_to_file(city_dict, 'city_dict_0.json')

    return city_dict


# 获取所有区县名
def get_county_name(city_dict):
    county_dict = {}  # 创建一个空字典来存储城市信息
    n = 0
    for city_name, city_url in city_dict.items():
        url = f"{base_url}{city_url}"
        html = get_html(url, encoding='utf-8')
        print(f"正在获取{city_name}的区县信息,当前进度{n}/{len(city_dict)} {url}")
        n += 1

        num = 0
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            # 查找所有的dd标签，这些标签包含了城市名和链接
            dds = soup.find_all('dd')
            # print(dds)

            for dd in dds:
                # 在每个dd标签内查找所有的a标签
                a_tags = dd.find_all('a')
                # print(a_tags)

                for a_tag in a_tags:
                    # 提取href属性和a标签的文本内容
                    if a_tag:
                        href = a_tag['href']
                        city_n = a_tag.text
                        county_dict[city_n] = href
                        num += 1

        print(f"{city_name}地区信息获取完毕，该省份共获取到{num}个次级地区信息")
        time.sleep(random.uniform(0.3, 2))

    # print(county_dict)
    print(f"全国城市信息获取完毕，共获取到{len(county_dict)}个区县信息")

    return county_dict


# 获取单个省份的城市名,测试用途
def get_city_name_one():
    city_dict = {}  # 创建一个空字典来存储城市信息

    url = "http://www.tianqihoubao.com/lishi/yunnan.htm"
    html = get_html(url, encoding='utf-8')

    if html:
        soup = BeautifulSoup(html, 'html.parser')
        # 查找所有的dd标签，这些标签包含了城市名和链接
        dds = soup.find_all('dd')
        # print(dds)

        for dd in dds:
            # 在每个dd标签内查找所有的a标签
            a_tags = dd.find_all('a')
            # print(a_tag)

            for a_tag in a_tags:
                # 提取href属性和a标签的文本内容
                if a_tag:
                    href = a_tag['href']
                    city_name = a_tag.text
                    city_dict[city_name] = href

    # print(city_dict)
    print(f"{url}城市信息获取完毕，共获取到{len(city_dict)}个城市信息")
    return city_dict


def get_url(city_url):
    """返回该日期区间url列表"""

    # 如果开始日期在2011年之前或者在当前日期之后，则抛出异常
    if start_year_month < 201101 or start_year_month > 202401:
        raise ValueError("开始日期必须在2011年1月至2024年12月之间!!")

    uu = []

    b_url = f"{base_url}/lishi/{city_url}/month/"

    # 遍历所有月份
    current_year_month = start_year_month
    while current_year_month <= end_year_month:
        # 格式化URL
        url = b_url + str(current_year_month) + ".html"
        uu.append(url)
        # print(url)

        # 移到下一个月份
        # 如果月份小于12，则月份加1；否则年份加1，月份重置为1
        if current_year_month % 100 < 12:
            current_year_month += 1
        else:
            current_year_month = (current_year_month // 100 + 1) * 100 + 1
    print(f"获取到{len(uu)}个日期区间url")
    return uu


def get_data(url):
    # 请求网页（第三方 requests）
    html = get_html(url, encoding='GBK')
    # 通过第三方库 BeautifulSoup 缩小查找范围（同样作用的包库还有re模块、xpath等）
    soup = BeautifulSoup(html, 'html.parser')
    # 获取 HTML 中所有<tr>…</tr>标签，因为我们需要的数据全部在此标签中存放
    tr_list = soup.find_all('tr')
    # 初始化日期dates、气候contains、温度值, 风力风向wind
    dates, contains, max_temp, min_temp, av_temp, wind = [], [], [], [], [], []

    for data in tr_list[1:]:  # 不要表头
        # 数据值拆分，方便进一步处理（这里可以将获得的列表输出[已注释]，不理解的可运行查看)
        sub_data = data.text.split()
        # print(sub_data)

        # 提取日期字符串
        dates.append(sub_data[0])

        # 观察上一步获得的列表，这里只想要获得列表中第二个和第三个值，采用切片法获取
        contains.append(''.join(sub_data[1:3]))
        # print(contains) #天气状况

        # 同理采用切片方式获取列表中的最高、最低气温

        max_temp.append(sub_data[3])
        min_temp.append(sub_data[5])

        # 计算平均气温
        x1 = sub_data[3].replace('℃', '')
        x2 = sub_data[5].replace('℃', '')
        av_t = (int(x1) + int(x2)) / 2
        av_temp.append(av_t)
        # print(av_temp_temp)

        # 获取风力风向
        wind.append(sub_data[6:])


    # 使用 _data 表存放日期、天气状况、气温表头及其值
    _data = pd.DataFrame()
    # 分别将对应值传入 _data 表中
    # _data['日期'] = dates
    _data['日期'] = dates
    _data['天气状况'] = contains
    _data['最低气温'] = max_temp
    _data['最高气温'] = min_temp
    _data['平均气温'] = av_temp
    _data['风力风向(夜间/白天)'] = wind

    print(_data)

    return _data


def get_all_url():  # 获取所有城市信息到本地
    # 获取省份信息
    # province_dict = get_province_name()
    # 获取一级城市信息
    # city_dict = get_city_name(province_dict)
    # get_city_name_one()

    # 读取一级城市信息
    city_dict = load_dict_from_file('city_dict_1.json')

    # 获取所有信息
    county_dict = get_county_name(city_dict)

    # 保存所有城市信息
    save_dict_to_file(county_dict, 'all_county_dict.json')


def main():
    """主函数"""
    # 获取所有城市信息# get_all_url()
    # 读取所有城市信息
    county_dict = load_dict_from_file('all_county_dict.json')

    # 根据提供的county信息获取url
    try:
        city_url = county_dict[county]
        print(f"查询到当前城市：{county}  {base_url}{city_url}")
        city_url = city_url.split('/')[-1]
        city_url = city_url.split('.')[0]
    except KeyError:
        print("该区县不存在，请重新输入,或者检查输入是否正确！")
        return

    # 获取该地区日期区间url列表
    url_list = get_url(city_url)

    # 根据url获取数据
    # url = "http://www.tianqihoubao.com/lishi/lufeng/month/202402.html"
    # get_data(url)

    for url in url_list:
        # 获取数据
        data_month = get_data(url)

        # 拼接所有表并重新设置行索引（若不进行此步操作，可能或出现多个标签相同的值）
        data = pd.concat([data_month]).reset_index(drop=True)

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

        print(f"{url} 气温爬取成功! 当前进度{url_list.index(url) + 1}/{len(url_list)}")

        time.sleep(random.uniform(1, 3))

    print(f"所有数据爬取完毕！数据已保存到{csv_file_path}")

if __name__ == '__main__':
    base_url = "http://www.tianqihoubao.com"
    # 选择爬取的城市信息
    # provice = "云南"
    # city = "楚雄"
    county = "禄丰 "  # 后面一定要加一个空格,因为我保存的的数据中有一个空格

    # 开始和结束的年月
    start_year_month = 202401
    end_year_month = 202403

    # 定义CSV文件的路径
    csv_file_path = './WeatherData.csv'

    main()
