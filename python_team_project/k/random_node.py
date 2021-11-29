import pandas as pd
import osmnx as ox
import os
import cv2



def select(time, start, end):
    route = []

    file_path =str(start) + '_' + str(end)
    print(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isfile(os.path.dirname(os.path.abspath(__file__))+'/data/' + file_path + '.csv'):
        print('is not file')
        return route

    df = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/data/' + file_path + '.csv')
    filt = df['time'] == time
    df_sample = df[filt]
    if len(df_sample) < 130:
        return route

    df_sample = df_sample.sample()
    route = df_sample['node'].tolist()
    route = route[0]
    route = route[1:-1]
    route = route.split()

    list = []
    for i in route:
        i = i[:-1]
        i = int(i)
        list.append(i)

    # list.insert(0,start)
    # list.append(end)
    return list[:-1]


if __name__ == '__main__':
    start = (35.8956224, 128.6224266)
    end = (35.888836, 128.6102997)

    G = ox.graph_from_point(start, dist=2500, dist_type='bbox', network_type='walk')

    start_node = ox.nearest_nodes(G, start[1], start[0])
    end_node = ox.nearest_nodes(G, end[1], end[0])

    list = select(60,start_node, end_node)
    print(list)

    # route1 = [5540134344, 4634458114, 3968872959, 6077982441, 6077982439, 6077982438, 6077982448, 6077982447, 7712664005,
    #          6077982445, 6077982443, 436847411, 3969143356, 436847410, 6110920583, 3969015056, 4864256203, 4864256198,
    #          4864256205, 8247026945, 5343005483, 8247026977, 8247026978, 697232798, 5343005479, 369914144, 8246998080,
    #          4641802035, 7098099026, 4079631406, 697232797, 5343005470, 4641802061, 3969143335, 4079631475, 4079631487,
    #          5551904035, 4079631506, 4079631499, 420450813, 420450808, 5100101377, 420450800, 5551903411, 420450804,
    #          5608611406, 5608611415, 420450805, 5608611402, 436816081, 5624709473, 5635283399, 5101432040, 5101432041,
    #          5101432039, 5101432037, 5124316811, 5104520888, 5578497792, 5104520893, 7966922418, 4631924213, 5608591572,
    #          7966949127, 5091020036, 5088768411, 7966907612, 5088812286, 4641219047, 5647912227]

    # print(route1)

    # G = ox.graph_from_point((35.89576144057368, 128.6224051398348), dist=3500, dist_type='bbox', network_type='walk')
    # print('경로 그리는 중')
    fig, ax = ox.plot_graph_route(G, list, node_size=0, show=False)