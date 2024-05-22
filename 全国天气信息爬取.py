# coding=utf-8
# @Author : Xenon
# @File : 全国天气信息爬取.py
# @Date : 2024/5/22 16:18 
# @IDE : PyCharm(2023.3) Python3.9.13
import json

import requests
from bs4 import BeautifulSoup


# 发送请求获取网页内容
def get_html(url, encoding='GBK'):
    response = requests.get(url)
    if response.status_code == 200:  # 判断请求是否成功
        # 转为GBK编码
        response.encoding = encoding
        return response.text
    else:
        return None

# 保存字典到文件
def save_dict_to_file(dict_data, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dict_data, f)

    print(f"字典已保存到{file_name}")

# 从JSON文件中加载字典
with open('data.json', 'r') as file:
    loaded_data = json.load(file)
    print(loaded_data)

# 获取所有省名
def get_province_name():
    url = "http://www.tianqihoubao.com/lishi/"
    html = get_html(url)
    province_dict = {}  # 创建一个空字典来存储城市信息

    if html:
        soup = BeautifulSoup(html, 'html.parser')
        print(soup)
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
    return province_dict


# 获取所有城市名
def get_city_name(province_dict):
    city_dict = {}  # 创建一个空字典来存储城市信息

    for province_name, province_url in province_dict.items():
        print(f"正在获取{province_name}的城市信息")
        url = f"http://www.tianqihoubao.com{province_url}"
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

    # print(city_dict)
    print(f"城市信息获取完毕，共获取到{len(city_dict)}个城市信息")

    # 保存城市信息
    save_dict_to_file(city_dict, 'city_dict.json')

    return city_dict

# 获取单个省份的城市名
def get_city_name_one():
    city_dict = {}  # 创建一个空字典来存储城市信息

    url = "http://www.tianqihoubao.com/lishi/yunnan.htm"
    html = get_html(url)

    if html:
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

    print(city_dict)
    print(f"城市信息获取完毕，共获取到{len(city_dict)}个城市信息")
    return city_dict

# 获取所有区县名
def get_county_name(city_dict):
    county_dict = {}  # 创建一个空字典来存储城市信息

    for city_name, city_url in city_dict.items():
        print(f"正在获取{city_name}的区县信息")
        url = f"http://www.tianqihoubao.com{city_url}"
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
                    county_name = b_tag.text
                    county_dict[county_name] = href

    # print(county_dict)
    print(f"区县信息获取完毕，共获取到{len(county_dict)}个区县信息")

    # 保存区县信息
    save_dict_to_file(county_dict, 'all_county_dict.json')

    return county_dict

def main():
    province_dict = get_province_name()
    city_dict = get_city_name(province_dict)
    # city_dict = get_city_name_one()


    # 保存城市信息
    # with open('city_dict.txt', 'w', encoding='utf-8') as f:
    #     f.write(str(city_dict))

if __name__ == '__main__':
    # 选择爬取的城市信息
    provice = "云南"
    city = "楚雄"
    counties = "禄丰"

    # 开始和结束的年月
    start_year_month = 202401
    end_year_month = 202403

    main()
