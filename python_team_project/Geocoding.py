import googlemaps
import os

class Geocoding:
    def __init__(self, orig, dest ) :
        self.orig = orig
        self.dest = dest
    
    def geocoding(self) :
        
        API_KEY = os.environ.get('API_KEY')
        
        map_client = googlemaps.Client(API_KEY)
        
        origin_geocoding = map_client.geocode(self.orig)
        destination_geocoding = map_client.geocode(self.dest)
        
        origin_geocoding = origin_geocoding[0]['geometry']['location']
        destination_geocoding = destination_geocoding[0]['geometry']['location']
        
        return origin_geocoding, destination_geocoding
        
        
        
        
        
        
        
        