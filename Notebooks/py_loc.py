import requests

# For calculating the lat and long of places 
from geopy.geocoders import Nominatim
geolocator = Nominatim()
source = geolocator.geocode("tambaram")
print(source.longitude, source.latitude)
dest = geolocator.geocode("nungambakkam")
print (dest.longitude , dest.latitude )

source_coordinates = str(source.latitude)+","+str(source.longitude)  # if not using geopy replace with source_latitude
dest_coordinates =  str(dest.latitude)+","+str(dest.longitude)

# IF NOT USING GEOPY 
# source_latitude = ""
# source_longitude = ""

# dest_latitude = ""
# dest_longitude = "" 

#graph hopper api key 
api = "5eda2deb-7ac1-44d9-990d-8aac82e96def"
# url for given loc 
url = "https://graphhopper.com/api/1/route?point="+ source_coordinates +"&point="+ dest_coordinates +"&vehicle=car&locale=de&key="+api

response = requests.get(url)
data = response.json()
print (data)
