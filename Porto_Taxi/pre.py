# 用于获取全量数据中最大的distance、radius等信息，为之后的评估做准备
import pickle
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from geopy import distance
from shapely.geometry import LineString


if __name__ == '__main__':
    geo = pd.read_csv('../../traj_gen_exp_porto/data/Porto_Taxi/cleaned-data/roadmap.cleaned.geo')
    all_traj = pd.read_csv('../../traj_gen_exp_porto/data/Porto_Taxi/raw-data/Porto_Taxi_trajectory.csv')
    with open('../../traj_gen_exp_porto/data/Porto_Taxi/cleaned-data/need_removed.pkl', 'rb') as file:
        need_removed = pickle.load(file)
    with open('../../traj_gen_exp_porto/data/Porto_Taxi/cleaned-data/oldrid2newrid.pkl', 'rb') as file:
        oldrid2newrid = pickle.load(file)

    num_roads = len(geo)

    road_gps = []
    for i, row in geo.iterrows():
        coordinates = eval(row['coordinates'])
        road_line = LineString(coordinates=coordinates)
        center_coord = road_line.centroid
        center_lon, center_lat = center_coord.x, center_coord.y
        road_gps.append((center_lon, center_lat))

    max_travel_distance = 0
    max_radius = 0
    rid_freq = np.zeros((num_roads), dtype=np.float32)
    max_time_cost = 0

    for _, row in tqdm(all_traj.iterrows(), total=len(all_traj)):
        rid_list = list(eval(row['rid_list']))
        flag = False
        for rid in rid_list:
            if rid in need_removed:
                flag = True
                break
        if flag:
            continue

        time_list = row['time_list'].split(',')
        time_list = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in time_list]
        flag = False
        for i in range(1, len(time_list)):
            if time_list[i] - time_list[i-1] > timedelta(minutes=15):
                flag = True
                break
        if flag:
            continue

        rid_list = [oldrid2newrid[x] for x in rid_list]

        travel_distance = 0
        for i in range(1, len(rid_list)):
            travel_distance += distance.distance((road_gps[rid_list[i-1]][1], road_gps[rid_list[i-1]][0]), (road_gps[rid_list[i]][1], road_gps[rid_list[i]][0])).kilometers
        max_travel_distance = max(max_travel_distance, travel_distance)

        lon_mean = np.mean([road_gps[rid][0] for rid in rid_list])
        lat_mean = np.mean([road_gps[rid][1] for rid in rid_list])
        rad = []
        for rid in rid_list:
            lon = road_gps[rid][0]
            lat = road_gps[rid][1]
            dis = distance.distance((lat_mean, lon_mean), (lat, lon)).kilometers
            rad.append(dis)
        max_radius = max(max_radius, np.mean(rad))

        for rid in rid_list:
            rid_freq[rid] += 1

        start_time = time_list[0]
        end_time = time_list[-1]
        time_cost = (end_time-start_time).total_seconds() / 60
        max_time_cost = max(max_time_cost, time_cost)

    print(f'max_travel_distance: {max_travel_distance}')
    print(f'max_radius: {max_radius}')
    rid_freq /= np.sum(rid_freq)
    print(f'max_rid_freq: {np.max(rid_freq)}')
    print(f'max_time_cost: {max_time_cost}')
