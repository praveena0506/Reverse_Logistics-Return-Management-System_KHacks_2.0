import pandas as pd
from geopy.geocoders import Nominatim
from math import radians, cos, sin, sqrt, atan2

# 1. Load the dataset (warehouse data)
df = pd.read_csv('data/warehouse_locations.csv')  # Your warehouse dataset

# 2. Geocoding function to convert user location to lat and lon
def geocode_location(location):
    geolocator = Nominatim(user_agent="myGeopyApp")
    location_obj = geolocator.geocode(location)
    if location_obj:
        return location_obj.latitude, location_obj.longitude
    else:
        raise ValueError("Location not found!")

# 3. Haversine function to calculate the distance between two lat-lon points
def haversine(lat1, lon1, lat2, lon2):
    # The radius of the Earth in km
    R = 6371.0
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c  # Resulting distance in kilometers
    return distance

# 4. Create the graph (distance between user location and each warehouse)
def create_graph(user_lat, user_lon, df):
    graph = {}
    for _, row in df.iterrows():
        warehouse_location = row['Warehouse Location']
        warehouse_lat = row['Latitude']
        warehouse_lon = row['Longitude']
        distance = haversine(user_lat, user_lon, warehouse_lat, warehouse_lon)
        graph[warehouse_location] = distance  # Store distance from user to each warehouse

    return graph

# 5. Calculate cost based on distance
def calculate_cost(distance, cost_per_km):
    return distance * cost_per_km

# 6. Main function to get nearest warehouses
def get_nearest_warehouses(user_location, cost_per_km):
    # 6.1 Geocode user input location
    try:
        user_lat, user_lon = geocode_location(user_location)
    except ValueError as e:
        return str(e)

    # 6.2 Create graph of warehouse distances from user location
    graph = create_graph(user_lat, user_lon, df)

    # 6.3 Get the top 5 nearest warehouses based on distance
    nearest_warehouses = sorted(graph.items(), key=lambda x: x[1])[:5]

    # 6.4 Calculate cost for each warehouse based on distance
    result = []
    for warehouse, distance in nearest_warehouses:
        cost = calculate_cost(distance, cost_per_km)
        result.append({
            "Warehouse Location": warehouse,
            "Distance (km)": distance,
            "Cost (Currency)": cost  # The calculated cost based on distance
        })

    return result

# Example usage
user_input_location = "Coimbatore"  # User enters a location name
cost_per_km = 5  # Cost per kilometer (example: 5 currency units per km)

# Get nearest warehouses and costs
nearest_warehouses = get_nearest_warehouses(user_input_location, cost_per_km)

# Output the result
if isinstance(nearest_warehouses, list):
    for warehouse in nearest_warehouses:
        print(f"Warehouse: {warehouse['Warehouse Location']}, Distance: {warehouse['Distance (km)']} km, Cost: {warehouse['Cost (Currency)']} currency units")
else:
    print(nearest_warehouses)  # If there's an error message, print it
