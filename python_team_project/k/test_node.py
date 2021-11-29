import osmnx as ox
import SearchPathAlgorithm as al
import random
import numpy as np
import cv2

import networkx as nx

time = 60
start = (35.8956224, 128.6224266)
end = (35.888836, 128.6102997)
G = ox.graph_from_point(start, dist=2500, dist_type='bbox', network_type='walk')
g_node = len(list(G))

# start_node = ox.nearest_nodes(G, start[1],start[0])
# end_node = ox.nearest_nodes(G, end[1], end[0])
#
# length = nx.shortest_path_length(G, start_node, end_node, weight='length')
#
# print(length/66.5)

def node_division(fig):
    # 맵 데이터인 fig를 이미지화 시킴
    # 그러면서 빨간색 경로가 파랑색으로 바뀜
    img_color = np.array(fig.canvas.renderer._renderer)

    height, width = img_color.shape[:2]  # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)  # cvtColor 함수를 이용하여 hsv 색공간으로 변환

    # 추출할 파랑색의 범위를 지정하는 부분
    lower_blue = (120 - 10, 30, 30)  # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
    upper_blue = (120 + 10, 255, 255)

    img = cv2.inRange(img_hsv, lower_blue, upper_blue)  # 범위내의 픽셀들은 흰색, 나머지 검은색

    # 변환된 mask 이미지를 보통 이미지로 전환
    img_mask_compare_0 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img_mask_compare_0



# while True:
#     g_node_range = random.randrange(1, 8)
#     route_list = [list(G)[random.randrange(0, g_node - 1)] for i in range(g_node_range)]
#     length_sum, time_sum, success, route_sum, start_node, end_node = al.route(route_list, G, start, end,
#                                                                               time)
#     if success:  # 만약 성공시
#         set_route = set(route_sum)  # 너무 중복된 길인지 확인
#         if len(set_route) / len(route_sum) > 0.85:
#             fig, ax = ox.plot_graph_route(G, route_sum, node_size=0)
#             print(route_sum)
#             print(time_sum)
#             print()
#             i = int(input())
#
#
#             if (i == 0):
#                 img = node_division(fig)
#                 cv2.imwrite('./0.png', img)
#                 break
#             else:
#                 continue

start_node = ox.nearest_nodes(G, start[1],start[0])
end_node = ox.nearest_nodes(G, end[1], end[0])
print(start_node, end_node)



# [5540134344, 4634458114, 3968872959, 6077982441, 7712526521, 6077982440, 6077982442, 7712663994, 5551904069, 5551904039, 5551904042, 3969016066, 3110817066, 5551904088, 5551904090, 5551904091, 5551904093, 8237660480, 7710550207, 8237660480, 5551904093, 3969143344, 436666386, 5551904089, 5551904175, 436666385, 5551904053, 436666384, 7712664008, 436847411, 6077982443, 5095171222, 436816090, 4636709799, 4636709802, 4636709797, 4636707481, 436816089, 2964248634, 4631924268, 5102781024, 5102781023, 5102781107, 5102781108, 4631924256, 5101376663, 5102781043, 5537488444, 5102781045, 5100252921, 5624667035, 4641213982, 5091023259, 4631924241, 7967059626, 5647912223, 8018704520, 4631924245, 7902714794, 7902714793, 7902714794, 7902714792, 7902723660, 7902723679, 7902723678, 5537917661, 4631924178, 7966907595, 5647912227]

