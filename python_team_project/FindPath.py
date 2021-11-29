from flask.helpers import flash
import networkx as nx
import osmnx as ox
import random
import math


class FindPath:
    def __init__(self, graph, orig, dest, time):
        self.G = graph
        self.origin_node = ox.distance.nearest_nodes(graph, orig[1], orig[0])
        self.destination_node = ox.distance.nearest_nodes(graph, dest[1], dest[0])
        self.time = time
        self.orig = orig

    def generate_path(self):

        is_not_complete = True
        all_time = 1

        while is_not_complete:

            
            all_length = 0
            all_route = []
            random_num = random.randrange(1, 4)
            random_node = []

            G_list = list(self.G.nodes)
            G_list_len = len(G_list)

            for i in range(random_num):
                push_random = random.randrange(1, G_list_len)
                random_node.append(G_list[push_random])

            random_node.insert(0, self.origin_node)
            random_node.append(self.destination_node)

            for i in range(len(random_node) - 1):
                route = ox.shortest_path(self.G, random_node[i], random_node[i + 1], weight="length")
                all_route.extend(route[:-1])
                route_length = int(
                    sum(ox.utils_graph.get_route_edge_attributes(self.G, route, "length"))
                )
                all_length += route_length
            
            all_route.append(self.destination_node)    
            
                
            # 시간 계산 
            mu = self.__cal_time(all_length * 0.9) 
            if mu < self.time - 3 or mu > self.time + 3 :
                all_time += 1
                continue
            
                

            d_node = {}
            d_node.clear()

            for i in all_route:
                d_node[i] = 0

            for i in all_route:
                d_node[i] = 1 if d_node[i] == 0 else d_node[i] + 1

            check_count = 0

            for i in d_node.values():
                check_count += 1
                if i > 2:
                    all_time += 1
                    break
                if check_count == len(d_node.values()) - 2:
                    is_not_complete = False
        return all_route, all_length
    
    def hms(self, m): # 전체길이에 66.5 로 나누면 -> 분
        hours = m // 60
        hours = int(hours)
        # s = s - hours * 3600
        mu = m - 60 * hours
        # ss = s - mu * 60
        mu = math.floor(mu)
        result_hms = [hours, mu]
        
        return result_hms
    
    def __cal_time(self, s) :
        return int(s // 60)
    
    
    def check_shortest_time(self) :
        route = ox.shortest_path(self.G, self.origin_node, self.destination_node, weight="length")        
        route_length = int(
            sum(ox.utils_graph.get_route_edge_attributes(self.G, route, "length"))
        )
        
        return self.__cal_time(route_length)
        
