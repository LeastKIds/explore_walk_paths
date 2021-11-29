import folium
import osmnx as ox


class GeneratorMap:
    def __init__(self, origin, destination):

        self.orig = (origin['lat'], origin['lng'])  # origin
        self.dest = (destination['lat'], destination['lng'])  # destination
        self.G = ox.graph_from_point(self.orig, 2500, network_type="walk")
        

    def f_map_marker(self, all_route):
        
        f_map = folium.Map(self.orig, width="100%", height="100%") 
        f_map = ox.folium.plot_route_folium(self.G, all_route, f_map)

        folium.Marker(self.orig, 
                      popup=folium.Popup(f"<b>origin</b>", show=True)).add_to(f_map)
        folium.Marker(self.dest, 
                      popup=folium.Popup(f"<b>destination</b>", show=True)).add_to(f_map)

        return f_map._repr_html_()

    
