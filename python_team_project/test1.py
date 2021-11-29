def hms(s):
    hours = s // 3600
    s = s - hours * 3600
    mu = s // 60
    ss = s - mu * 60
    print(hours, "시간", mu, "분", ss, "초 입니다.")


orig = (35.895622, 128.622427)
dest = (35.89112, 128.61220)


random_num = random.randrange(1, 6)
random_node = []

# dist 가 작으면 노드에 해당하는 그림이 다 그려지지 않음! 크게 해야함
G = ox.graph_from_point(orig, 1500, network_type="walk")

G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

edges = ox.graph_to_gdfs(G, nodes=False)
hwy_speeds = {"residential": 35, "secondary": 50, "tertiary": 60}
G = ox.add_edge_speeds(G, hwy_speeds)
G = ox.add_edge_travel_times(G)

origin_node = ox.distance.nearest_nodes(G, orig[1], orig[0])
destination_node = ox.distance.nearest_nodes(G, dest[1], dest[0])

G_list = list(G.nodes)
G_list_len = len(G_list)

for i in range(random_num):
    push_random = random.randrange(1, G_list_len)
    random_node.append(G_list[push_random])


all_route = []

random_node.insert(0, origin_node)
random_node.append(destination_node)


# print(random_node, ' random_node')

f_map = folium.Map(orig, width="100%", height="100%")

# while true :
all_time = 0
all_length = 0


for i in range(len(random_node) - 1):
    print(random_node[i], random_node[i + 1], "this route")
    route = ox.shortest_path(G, random_node[i], random_node[i + 1], weight="length")
    route = ox.shortest_path(
        G, random_node[i], random_node[i + 1], weight="travel_time"
    )

    f_map = ox.folium.plot_route_folium(G, route, f_map)
    # print(route, 'this route')
    all_route.extend(route)
    route_time = int(
        sum(ox.utils_graph.get_route_edge_attributes(G, route, "travel_time"))
    )
    route_length = int(
        sum(ox.utils_graph.get_route_edge_attributes(G, route, "length"))
    )
    # print(route_time)
    all_time += route_time
    all_length += route_length

    # f_map = f_map

print("all_length", all_length, "all_time", all_time)


# 100 => 90초
# 10 => 9초
# 1 = > 0.9
# 8323 => 7490 => 124 분

# 67미터 1분

hms(all_length * 0.9)
folium.Marker(
    orig,
    popup=folium.Popup(f"<b>origin</b>", show=True),
).add_to(f_map)
folium.Marker(
    dest,
    popup=folium.Popup(f"<b>destination</b>", show=True),
).add_to(f_map)


f_map
