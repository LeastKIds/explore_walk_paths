import enum
from flask import Flask, render_template, request, flash
from dotenv import load_dotenv

from GeneratorMap import GeneratorMap
from FindPath import FindPath
from Geocoding import Geocoding
from k import random_node
from k import first_search


app = Flask(__name__)
load_dotenv()

app.secret_key = "my secret key".encode("utf8")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/show_map", methods=["POST"])
def show_map():
    orig = str(request.form["orig"])
    dest = str(request.form["dest"])
    duration = int(request.form["duration"])
    
    geocoding  = Geocoding(orig, dest)
    origin_geocoding, destination_geocoding = geocoding.geocoding()
    
    
    origin_tuple = (origin_geocoding['lat'], origin_geocoding['lng'])
    destination_tuple = (destination_geocoding['lat'], destination_geocoding['lng'])
    

    generate_map = GeneratorMap(origin_geocoding, destination_geocoding)
    find_path = FindPath(
        generate_map.G, generate_map.orig, generate_map.dest, duration
    )   
    check_time = find_path.check_shortest_time()
    
    
    # 학습이 된 경로 탐색시
    node_list = random_node.select(duration, find_path.origin_node, find_path.destination_node)
    if node_list :
        folium_map = generate_map.f_map_marker(node_list)
        # 시간 수정!!!
        # result_hms = find_path.hms(all_length * 0.9)
        hms = [duration // 60, duration - duration // 60 * 60]
        return render_template("map.html", folium_map=folium_map, result_hms = hms, orig =  orig, dest = dest, duration = duration, all_route = 0 )
        # return render_template("home.html")
        
        

    if duration < check_time:
        flash("해당 시간에 맞는 경로를 찾을 수 없습니다(최단 시간보다 작음)")
    else:
        try:
            all_route, all_length = find_path.generate_path()
            folium_map = generate_map.f_map_marker(all_route)
            result_hms = find_path.hms(all_length / 66.5)
            
            return render_template("map.html", folium_map=folium_map, result_hms = result_hms, orig = orig, dest = dest, duration = duration, all_route = all_route )
        except Exception as error:
            flash("error occured : " + repr(error))
    return render_template("home.html")


@app.route('/learning',  methods=["POST"])
def learning() :
    params = request.get_json()
    orig = params['orig']
    dest = params['dest']
    geocoding = Geocoding(orig, dest)
    origin_geocoding, destination_geocoding = geocoding.geocoding()
    origin_tuple = (origin_geocoding['lat'], origin_geocoding['lng'])
    destination_tuple = (destination_geocoding['lat'], destination_geocoding['lng'])
    
    duration = int(params['duration'])
    all_route = params['all_route']
    
    all_route_test = all_route[1:-1]
    all_route_test2 = all_route_test.split()
    all_route = []
    len_all_route = len(all_route_test2)
    for i, val in enumerate(all_route_test2) :
        if i == len_all_route - 1 :
            all_route.append(int(val))
        else :     
            all_route.append(int(val[:-1]))
    
    
    # 첫 경로 학습시
    first_search.search(duration, origin_tuple, destination_tuple, all_route)
    return {}
    

if __name__ == "__main__":
    app.run(debug=True)
