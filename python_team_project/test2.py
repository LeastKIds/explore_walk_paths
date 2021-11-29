def hms(s):
    hours = s // 3600
    s = s - hours * 3600
    mu = s // 60
    ss = s - mu * 60
    print(hours, "시간", mu, "분", ss, "초 입니다.")


orig = (35.895622, 128.622427)
dest = (35.89112, 128.61220)


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

# print(random_node, ' random_node')


is_not_complete = True
all_time = 1

while is_not_complete:
    f_map = folium.Map(orig, width="100%", height="100%")
    all_length = 0
    all_route = []
    random_num = random.randrange(1, 3)
    random_node = []

    G_list = list(G.nodes)
    G_list_len = len(G_list)

    for i in range(random_num):
        push_random = random.randrange(1, G_list_len)
        random_node.append(G_list[push_random])

    random_node.insert(0, origin_node)
    random_node.append(destination_node)

    print("random_node", random_node)

    for i in range(len(random_node) - 1):
        route = ox.shortest_path(G, random_node[i], random_node[i + 1], weight="length")
        f_map = ox.folium.plot_route_folium(G, route, f_map)
        # print(route, 'this route')
        all_route.extend(route)
        route_time = int(
            sum(ox.utils_graph.get_route_edge_attributes(G, route, "travel_time"))
        )
        route_length = int(
            sum(ox.utils_graph.get_route_edge_attributes(G, route, "length"))
        )
        all_length += route_length

    d_node = {}
    d_node.clear()

    for i in all_route:
        d_node[i] = 0

    for i in all_route:
        d_node[i] = 1 if d_node[i] == 0 else d_node[i] + 1

    print("------------------------------------")
    print(all_time)
    print(d_node.values())
    print("------------------------------------")

    check_count = 0

    for i in d_node.values():
        check_count += 1
        # print(check_count)
        if i > 2:
            all_time += 1
            break
        if check_count == len(d_node.values()) - 2:
            print("ok...")
            is_not_complete = False


print("all_length", all_length, "all_time", all_time)


# print(all_route)

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
