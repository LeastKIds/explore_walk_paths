import networkx as nx
import osmnx as ox


def route(list, g, start, end, time):
    start_node = ox.nearest_nodes(g, start[1],start[0])
    end_node = ox.nearest_nodes(g, end[1], end[0])
    list.insert(0,start_node)
    route_sum=[]
    time_sum=0
    length_sum=0
    for i in range(len(list)-1):
        route = nx.shortest_path(g, list[i], list[i+1], weight='length')
        route_sum.extend(route[1:])
        length = nx.shortest_path_length(g, list[i], list[i+1], weight='length')
        length_sum += length
        time_sum += (length/66.5)
        # print('======')
        # print('time_sum : ', end='')
        # print(time_sum)
        # print('======')
        if time+10 < time_sum:
            success = False
            return length_sum, time_sum, success, route_sum, start_node, end_node

    route = nx.shortest_path(g, list[-1], end_node, weight='length')
    route_sum.extend(route[1:])
    length = nx.shortest_path_length(g, list[-1], end_node, weight='length')
    length_sum += length
    time_sum += (length / 66.5)

    if time + 10 > time_sum > time - 10:
        success = True
        return length_sum, time_sum, success, route_sum, start_node, end_node

    else:
        success = False
        return length_sum, time_sum, success, route_sum, start_node, end_node




