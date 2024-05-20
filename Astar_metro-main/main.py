import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
from queue import PriorityQueue
import numpy as np

np.set_printoptions(threshold=np.inf)  # 打印的矩阵全部显示
transfer_cost = 10000  # 换乘的代价10000，全局变量


def get_metro_information():
    print('正在获取地铁信息中')  # 根据城市获取地铁信息
    url = "https://ditie.hao86.com/xuzhou/"  # 获取地铁信息的网址,如若需要修改获取城市地铁站点，在com后的//里更改城市名，需要小写且无空格,还需更改第40行的city名
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.48'
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")  # 使用BeautifulSoup解析网页
    info = soup.find_all("ul", class_="ib-mn cm-mn")
    if not info:  # 判断是否找到地铁信息
        print("没有找到地铁信息")
        return
    else:
        print("找到地铁信息")
    df = pd.DataFrame(columns=["name", "line", "longitude", "latitude"])
    for info in tqdm(info):  # 使用tqdm显示进度条
        line_all = info.find_all("li", class_="sLink")
        df_line = pd.DataFrame(columns=["name", "line", "longitude", "latitude"])  # 创建一个新的数据框来存储每条线路的站点
        for line in line_all:  # 遍历每条线路
            line_name = line.find("a", class_="cm-tt").get("title")
            station_all = line.find_all("a")
            for station in station_all:  # 遍历每个站点
                if station.get("class") != ["cm-tt"]:
                    station_name = station.get_text().strip()
                    try:  # 如果没有找到经纬度信息,抛出异常
                        locations = get_location(station_name, city="徐州")  # 在这里更改城市名，可以切换到其他城市，要与之前爬取的数据网址一起修改
                        longitude, latitude = locations[0]
                        example = {"name": station_name, "line": [line_name], "longitude": longitude,
                                   "latitude": latitude}
                        df_line = df_line._append(example, ignore_index=True)  # 将站点添加到新的数据框中
                    except TypeError:
                        print(f"没有找到{station_name}的经纬度信息")
                        continue
        df = df._append(df_line, ignore_index=True)  # 在处理完一条线路后，将新的数据框添加到总的数据框中
    print("df的内容是:", df)
    df.to_excel("./subway.xlsx", index=False)
    print("已将地铁信息保存到subway.xlsx文件中")


def check_file():
    # 检查excel文件
    if os.path.exists("./subway.xlsx"):
        data = pd.read_excel("./subway.xlsx")
        if data.empty:
            print("subway.xlsx 文件为空")
        else:
            print("subway.xlsx 文件包含以下信息:")
            print(data)
    else:
        print("subway.xlsx 文件不存在")


def get_location(address, city):
    # 定义一个获取经纬度的方法
    print(f"正在获取{address}经纬度信息")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.48"
    headers = {"User-Agent": user_agent}  # 定义一个字典变量
    url = f"https://restapi.amap.com/v3/geocode/geo?key={keynumber}&address={address}&city={city}&output=json"  # API接口
    data = requests.get(url, headers=headers)  # 使用request发送一个HTTPGET请求返回到response
    data.encoding = "utf-8"  # 设置编码为UTF-8
    data = json.loads(data.text)  # 将data转换为json格式
    if data["status"] == "1":
        geocodes = data["geocodes"]
        if geocodes:
            locations = []
            for geocode in geocodes:
                location = geocode["location"]
                longitude, latitude = location.split(",")
                print(f"找到{address}经纬度为{longitude},{latitude}")  # 获取到获取的经纬度信息
                locations.append((longitude, latitude))
            return locations  # 返回经纬度信息
    print(f"没有找到{address}经纬度信息")
    return None


def test_api_call():
    # 测试高德API的调用
    keynumber = "d0ba7c3f4602f8a00e0d50b17599ef0f"  # 高德地图API的key
    address = "徐州火车站"
    city = "徐州"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.48"
    headers = {"User-Agent": user_agent}
    url = f"https://restapi.amap.com/v3/geocode/geo?key={keynumber}&address={address}&city={city}&output=json"
    response = requests.get(url, headers=headers)
    print(response.text)


# 创建一个全局字典来存储已经计算过的距离
calculated_distances = {}


def calculate_distance(longitude1, latitude1, longitude2, latitude2):
    global calculated_distances
    key = (longitude1, latitude1, longitude2, latitude2)
    if key in calculated_distances:
        return calculated_distances[key]
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (HTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.48"
    headers = {"User-Agent": user_agent}
    url = f"https://restapi.amap.com/v3/distance?key={keynumber}&origins={str(longitude1) + ',' + str(latitude1)}&destination={str(longitude2) + "," + str(latitude2)}&output=json"
    data = requests.get(url, headers=headers)
    data.encoding = "utf-8"
    data = json.loads(data.text)
    if 'results' in data and data["results"]:
        result = data["results"][0]["distance"]
        calculated_distances[key] = int(result)
        return int(result)
    else:
        print(f"无法获取 {longitude1}, {latitude1} 和 {longitude2}, {latitude2} 之间的距离")
        return 0


def common_line(line1, line2):
    # 解决换乘站点，图无法判断换乘站点和邻居站点之间是否在一条线上的问题
    if line1 is None or line2 is None:
        return False
    # 修改这里，只要两个站点有至少一条共同的线路，就返回 True
    return any(line in line1 for line in line2)


def is_adjacent(lines, station1, station2):
    for line, stations in lines.items():
        if station1 in stations and station2 in stations:
            index1 = stations.index(station1)
            index2 = stations.index(station2)
            if abs(index1 - index2) == 1:
                return True
    return False


def process_data_neighbor():  # 处理站点之中的邻居站点，并将其保存到subway.xlsx文件中
    # 读取xlsx文件
    data = pd.read_excel("./subway.xlsx")

    # 创建一个新的DataFrame来存储处理后的数据
    processed_data = pd.DataFrame(columns=data.columns.tolist())
    processed_data['neighbor'] = None  # 创建新的 'neighbor' 列

    # 创建一个字典来存储每个站点的邻居站点
    neighbors = {}

    # 遍历每个站点
    for i in range(len(data)):
        # 找到上一位和下一位的站点
        prev_station = data.iloc[i - 1]["name"] if i > 0 and data.iloc[i - 1]["line"] == data.iloc[i]["line"] else None
        next_station = data.iloc[i + 1]["name"] if i < len(data) - 1 and data.iloc[i + 1]["line"] == data.iloc[i][
            "line"] else None

        # 将这两个站点的"name"添加到当前站点的"neighbor"列中
        row = data.iloc[i].to_dict()
        row["neighbor"] = [prev_station, next_station]

        # 如果当前站点是首站点或尾站点，只添加一个邻居站点
        if i == 0 or i == len(data) - 1:
            row["neighbor"] = [next_station] if i == 0 else [prev_station]

        # 将处理后的行添加到新的DataFrame中
        processed_data = pd.concat([processed_data, pd.DataFrame([row])], ignore_index=True)

        # 将当前站点的邻居站点添加到字典中
        if row["name"] not in neighbors:
            neighbors[row["name"]] = row["neighbor"]
        else:
            neighbors[row["name"]].extend(
                [neighbor for neighbor in row["neighbor"] if neighbor not in neighbors[row["name"]]])

    # 遍历字典，将每个站点的邻居站点添加到它的neighbor列中
    for name, neighbor in neighbors.items():
        processed_data.loc[processed_data["name"] == name, "neighbor"] = str(neighbor)  # 将列表转换为字符串

    # 将处理后的数据保存回xlsx文件
    processed_data.to_excel("./subway.xlsx", index=False)


def get_graph():  # 创建图来表示地铁站点之间的连接
    # 读取xlsx文件
    data = pd.read_excel("./subway.xlsx")

    # 创建一个字典来存储站点信息
    station_info = {}
    for i in range(len(data)):
        name = data.iloc[i]["name"]
        line = eval(data.iloc[i]["line"])
        longitude = data.iloc[i]["longitude"]
        latitude = data.iloc[i]["latitude"]
        neighbor = eval(data.iloc[i]["neighbor"])
        station_info[name] = {"line": line, "longitude": longitude, "latitude": latitude, "neighbor": neighbor}

    # 创建一个邻接矩阵来存储邻居站点
    num_stations = len(station_info)
    adjacency_matrix = np.zeros((num_stations, num_stations))

    # 遍历每个站点的 "neighbor" 列
    for name, info in station_info.items():
        i = list(station_info.keys()).index(name)
        neighbors = info["neighbor"]
        for neighbor in neighbors:
            if neighbor in station_info:
                j = list(station_info.keys()).index(neighbor)
                adjacency_matrix[i][j] = 1

    return station_info, adjacency_matrix


def check_graph(graph):  # 检查图是否正确
    station_info, adjacency_matrix = graph
    print("Station Info:")
    for node, info in station_info.items():
        print(f"Node: {node}")
        print("Info: {info}")
    print("Adjacency Matrix:")
    print(adjacency_matrix)


def check_neighbours(graph, site):  # 检查站点的邻居站点
    station_info, adjacency_matrix = graph
    if site in station_info:
        print(f"{site}的邻居站点:")
        site_index = list(station_info.keys()).index(site)
        neighbors_indices = np.where(adjacency_matrix[site_index] == 1)[0]
        neighbors = [list(station_info.keys())[i] for i in neighbors_indices]
        for neighbor in neighbors:
            print(f"{neighbor}")
    else:
        print(f"{site}不在图中")


def print_graph(graph):  # 打印图
    check_graph(graph)


def check_site_in_graph(graph, site):  # 检查站点是否在图中
    station_info, _ = graph
    if site in station_info:
        print(f"{site} is in the graph.")
    else:
        print(f"{site} is not in the graph.")


def check_path_between_sites(graph, site1, site2):  # 检查两个站点之间是否有路径
    station_info, adjacency_matrix = graph
    if site1 in station_info and site2 in station_info:
        site1_index = list(station_info.keys()).index(site1)
        site2_index = list(station_info.keys()).index(site2)
        if adjacency_matrix[site1_index][site2_index] == 1:
            print(f"There is a path between {site1} and {site2}")
        else:
            print(f"There is no path between {site1} and {site2}")
    else:
        print("One or both sites are not in the graph")


def save_graph(graph):  # 打印保存图之前的信息
    print(f"Saving graph: {graph}")
    print(f"Graph type: {type(graph)}")
    with open("graph.pkl", "wb") as file:
        pickle.dump(graph, file)


def load_graph():  # 打印加载图之后的信息
    with open("graph.pkl", "rb") as file:
        graph = pickle.load(file)
    print(f"Loaded graph: {graph}")
    print(f"Graph type: {type(graph)}")
    return graph


def Astar(graph, start, end, heuristic, transfer_cost):  # A*算法
    station_info, adjacency_matrix = graph
    if start not in station_info or end not in station_info:  # 判断合法性
        return None, None
    frontier = PriorityQueue()  # 创建一个优先队列
    frontier.put((0, start, None))  # 将起点放入优先对列
    come_from = {start: None}  # 创建一个字典用来记录路径,且起点的前一个站点为None
    cost_so_far = {start: 0}  # 创建一个字典用来记录当前代价,且起点的代价为0
    processed = set()  # 创建一个集合用来记录已经访问过的站点
    in_frontier = {start}  # 创建一个集合用来记录已经在优先队列中的站点
    while not frontier.empty():  # 当优先队列不为空时,进行循环
        _, current, previous = frontier.get()  # 从优先队列中取出一个元素
        processed.add(current)  # 将这个元素添加到已经访问过的站点的集合中
        if current == end:
            break
        if current in in_frontier:
            in_frontier.remove(current)  # 从优先队列中的站点的集合中移除这个元素
        current_index = list(station_info.keys()).index(current)
        neighbors_indices = np.where(adjacency_matrix[current_index] == 1)[0]
        neighbors = [list(station_info.keys())[i] for i in neighbors_indices]
        for next in neighbors:  # 遍历当前站点的所有邻居站点
            new_cost = cost_so_far[current] + calculate_distance(station_info[current]["longitude"],
                                                                 station_info[current]["latitude"],
                                                                 station_info[next]["longitude"],
                                                                 station_info[next]["latitude"])
            if previous and not common_line(station_info[current]["line"], station_info[previous]["line"]):
                new_cost += transfer_cost
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(station_info, next, end)
                frontier.put((priority, next, current))
                come_from[next] = current
        # 如果当前站点的邻居站点不在优先队列中，则将其添加到优先队列中
        if current not in cost_so_far:
            cost_so_far[current] = new_cost
    return come_from, cost_so_far  # 返回路径和代价


def heuristic(station_info, site1, site2):  # 定义一个函数用来计算两个站点之间的预估代价
    if site1 not in station_info or site2 not in station_info:
        print(f"站点{site1}或{site2}不在图中")
        return 0
    if "longitude" not in station_info[site1] or "latitude" not in station_info[site1]:
        print(f"站点{site1}没有经纬度信息")
        return 0
    else:
        print(
            f"站点{site1}的经纬度信息为{station_info[site1]['longitude']},{station_info[site1]['latitude']}")  # 获取站点的经纬度信息
    if "longitude" not in station_info[site2] or "latitude" not in station_info[site2]:
        print(f"站点{site2}没有经纬度信息")
        return 0
    else:
        print(
            f"站点{site2}的经纬度信息为{station_info[site2]['longitude']},{station_info[site2]['latitude']}")  # 获取站点的经纬度信息
    longitude1, latitude1 = station_info[site1]["longitude"], station_info[site1]["latitude"]
    longitude2, latitude2 = station_info[site2]["longitude"], station_info[site2]["latitude"]
    return calculate_distance(longitude1, latitude1, longitude2, latitude2)


def subway_line_astar(start, end):  # 定义地铁路线规划函数
    file = open("graph.pkl", "rb")  # 打开之前爬取数据创建的graph.pkl文件
    graph = pickle.load(file)  # 用pickle模块的load函数将graph.pkl文件中的对象加载到graph中
    station_info, _ = graph  # 解包图元组以获取站点信息字典
    shortest_path = None
    if start not in station_info or end not in station_info:  # 在站点信息字典中查找站点
        print("站点输入错误，请重新输入")
        return None, None
    shortest_distance = float('inf')
    path = Astar(graph, start, end, heuristic, transfer_cost=10000)  # 调用A*算法
    if path is None:
        return None, None
    if end in path[1] and path[1][end] < shortest_distance:
        shortest_distance = path[1][end]
        shortest_path = path
    return shortest_path if shortest_path else (None, None)  # 返回最短路径


def check_connection(graph, site1, site2):  # 检查两个站点之间是否有直接连接
    station_info, adjacency_matrix = graph
    if site1 in station_info and site2 in station_info:
        site1_index = list(station_info.keys()).index(site1)
        site2_index = list(station_info.keys()).index(site2)
        if adjacency_matrix[site1_index][site2_index] == 1:
            print(f"There is a direct connection between {site1} and {site2}")
        else:
            print(f"There is no direct connection between {site1} and {site2}")
    else:
        print("One or both sites are not in the graph")


def main():  # 打#的为测试函数，在实际运行时可以跳过。
    get_metro_information()
    global calculated_distances
    # check_file()
    # test_api_call()
    process_data_neighbor()
    graph = get_graph()
    # save_graph(graph)
    # graph = load_graph()
    # check_connection(graph, "", "")#检查两个站点之间是否有直接连接
    # print_graph(graph)
    file = open("graph.pkl", "rb")
    graph = pickle.load(file)
    # print(f"Graph: {graph}")
    # print(f"Graph type: {type(graph)}")
    station_info, _ = graph  # 解包图元组以获取站点信息字典
    while True:
        while True:
            site1 = input("请输入起始站点:")
            site2 = input("请输入终点站点:")
            if site1 in station_info and site2 in station_info:  # 判断站点是否合法
                break
            else:
                print("站点输入错误，请重新输入")
        come_from, cost_so_far = subway_line_astar(site1, site2)  # 调用地铁路线规划函数
        if come_from is None or cost_so_far is None:
            print("无法找到从{}到{}的路径".format(site1, site2))
            return
        if site2 not in come_from:
            print("无法找到从{}到{}的路径".format(site1, site2))
            return
        path = []
        current = site2
        while current is not None:
            path.append(current)
            current = come_from[current]
        path.reverse()
        print("最短路径为:{})", format(path))  # 打印最短路径
        # 询问用户是否继续查找
        continue_search = input("是否继续查找?输入y来继续,如果退出则输入任意键。 (y/n): ")
        if continue_search.lower() != 'y':
            break
    # 检查站点是否在图中
    # check_site_in_graph(graph, "")
    # 检查两个站点之间是否有路径
    # check_path_between_sites(graph, "", "")
    # 检查站点的邻居站点
    # check_neighbours(graph, '')


if __name__ == "__main__":
    global keynumber
    keynumber = ""  # 填入你的高德地图API的key
    main()
