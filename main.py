import os
from math import radians, cos, sin, sqrt, atan2
import pickle
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import folium
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from geopy.geocoders import Nominatim
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = FastAPI()
from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

app = FastAPI()
security = HTTPBasic()

# Dummy user database (replace with a real database)
users_db = {
    "user1": {"password": "password1", "role": "user"},
    "agent1": {"password": "password1", "role": "delivery_agent"},
}

@app.get("/user_login", response_class=HTMLResponse)
def user_login(request: Request):
    return templates.TemplateResponse("user_login.html", {"request": request})

@app.post("/user_login/")
def user_login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in users_db and users_db[username]["password"] == password and users_db[username]["role"] == "user":
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("user_login.html", {"request": request, "message": "Invalid credentials"})

@app.get("/delivery_agent_login", response_class=HTMLResponse)
def delivery_agent_login(request: Request):
    return templates.TemplateResponse("delivery_agent_login.html", {"request": request})

@app.post("/delivery_agent_login/")
def delivery_agent_login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in users_db and users_db[username]["password"] == password and users_db[username]["role"] == "delivery_agent":
        return RedirectResponse(url="/view_map/1/", status_code=303)  # Redirect to view_map for delivery agents
    return templates.TemplateResponse("delivery_agent_login.html", {"request": request, "message": "Invalid credentials"})

@app.get("/signup", response_class=HTMLResponse)
def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup/")
def signup_post(request: Request, username: str = Form(...), password: str = Form(...), role: str = Form(...)):
    if username in users_db:
        return templates.TemplateResponse("signup.html", {"request": request, "message": "Username already exists"})
    users_db[username] = {"password": password, "role": role}
    if role == "user":
        return RedirectResponse(url="/user_login", status_code=303)
    else:
        return RedirectResponse(url="/delivery_agent_login", status_code=303)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load models
with open("models/barcode_model.pkl", "rb") as f:
    barcode_data = pickle.load(f)
damage_model = tf.keras.models.load_model("models/damage_model.h5")

# Load datasets
trust_score_df = pd.read_csv("data/trust_score_dataset.csv")
product_score_df = pd.read_csv("data/product_score.csv")
barcode_df = pd.read_csv("data/metadata.csv")
warehouse_df = pd.read_csv("data/warehouse_locations.csv")
repair_df = pd.read_csv("data/repair_centers.csv")
recycle_df = pd.read_csv("data/recycle_centers_dataset.csv")

# Geocode a location using geopy
def geocode_location(location):
    geolocator = Nominatim(user_agent="GETmyapi")
    try:
        location_data = geolocator.geocode(location)
        if location_data:
            return (location_data.latitude, location_data.longitude)
        else:
            raise ValueError(f"Could not geocode location: {location}")
    except Exception as e:
        raise ValueError(f"Geocoding error: {e}")

# Haversine function to calculate the distance between two lat-lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # The radius of the Earth in km
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c  # Resulting distance in kilometers
    return distance

# Create the graph (distance between user location and each warehouse)
def create_graph(user_lat, user_lon, df):
    graph = {}
    for _, row in df.iterrows():
        warehouse_location = row['Location']
        warehouse_lat = row['Latitude']
        warehouse_lon = row['Longitude']
        distance = haversine(user_lat, user_lon, warehouse_lat, warehouse_lon)
        graph[warehouse_location] = {
            "distance": distance,
            "latitude": warehouse_lat,
            "longitude": warehouse_lon
        }
    return graph

# Validate barcode function
def validate_barcode(product_id, uploaded_barcode_image):
    if product_id not in barcode_data:
        return False
    stored_barcode_image = barcode_data[product_id]
    difference = cv2.absdiff(stored_barcode_image, uploaded_barcode_image)
    return np.sum(difference) == 0

# Classify damage function
def classify_damage(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = damage_model.predict(image)
    predicted_class = np.argmax(prediction)
    return "Recycle" if predicted_class >= 2 else "Repair"

@app.get("/delivery_picker/")
def delivery_picker(request: Request):
    return templates.TemplateResponse("delivery_picker.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/check_customer/")
def check_customer(request: Request, user_id: str = Form(...), product_id: int = Form(...)):
    # Check trust score
    trust_score = trust_score_df.loc[trust_score_df["UserID"] == user_id, "TrustScore"].values
    if len(trust_score) == 0 or trust_score[0] < 2.5:
        return templates.TemplateResponse("index.html", {"request": request, "message": "Cannot return for 3 months due to low trust score."})

    product_score = product_score_df.loc[product_score_df["Product_id"] == product_id, "product_score"].values
    if len(product_score) == 0:
        return templates.TemplateResponse("index.html",
                                          {"request": request, "message": "Product not found in database."})

    warning_message = "Warning: High review score. Please reconsider your return." if product_score[0] >= 0.85 else None
    return templates.TemplateResponse("reason.html",
                                      {"request": request, "product_id": product_id,
                                       "warning_message": warning_message})

@app.post("/reason_selection/")
def reason_selection(request: Request, user_id: str = Form(...), product_id: int = Form(...), reason: str = Form(...)):
    if reason in ["Size or Fit Issue", "Color Change", "Change of Mind"]:
        return templates.TemplateResponse("barcode.html", {"request": request, "user_id": user_id, "product_id": product_id})
    else:
        return templates.TemplateResponse("damage.html", {"request": request, "user_id": user_id, "product_id": product_id})

@app.post("/process_barcode/")
def process_barcode(request: Request, user_id: str = Form(...), product_id: int = Form(...),
                    barcode_file: UploadFile = File(...), location: str = Form(...)):
    barcode_path = f"temp/{barcode_file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(barcode_path, "wb") as f:
        f.write(barcode_file.file.read())

    # Validate barcode
    uploaded_barcode_image = cv2.imread(barcode_path, cv2.IMREAD_GRAYSCALE)
    uploaded_barcode_image = cv2.resize(uploaded_barcode_image, (200, 200))
    if not validate_barcode(product_id, uploaded_barcode_image):
        return templates.TemplateResponse("barcode.html",
                                          {"request": request, "message": "Barcode does not match the product ID."})

    # Geocode location
    try:
        customer_coords = geocode_location(location)
    except ValueError as e:
        return templates.TemplateResponse("barcode.html", {"request": request, "message": str(e)})

    # Find nearest warehouses
    graph = create_graph(customer_coords[0], customer_coords[1], warehouse_df)
    nearest_warehouses = sorted(graph.items(), key=lambda x: x[1]["distance"])[:10]  # Change from 5 to 10

    # Generate Excel with Longitude and Latitude
    excel_data = []
    for i, (loc, data) in enumerate(nearest_warehouses):
        excel_data.append({
            "Product ID": product_id,
            "Customer ID": user_id,
            "Location": loc,
            "Latitude": data["latitude"],
            "Longitude": data["longitude"],
            "Cost": data["distance"]
        })
    df = pd.DataFrame(excel_data)
    output_path = f"data/nearest_warehouses_{product_id}.xlsx"
    os.makedirs("data", exist_ok=True)
    df.to_excel(output_path, index=False)

    return RedirectResponse(url=f"/view_map/{product_id}/", status_code=303)

@app.post("/process_damage/")
def process_damage(request: Request, user_id: str = Form(...), product_id: int = Form(...), damage_file: UploadFile = File(...), location: str = Form(...)):
    damage_path = f"temp/{damage_file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(damage_path, "wb") as f:
        f.write(damage_file.file.read())

    damage_type = classify_damage(damage_path)

    # Geocode location
    try:
        customer_coords = geocode_location(location)
    except ValueError as e:
        return templates.TemplateResponse("damage.html", {"request": request, "message": str(e)})

    # Choose facility based on damage type
    facility_df = warehouse_df if damage_type == "Repair" else recycle_df
    graph = create_graph(customer_coords[0], customer_coords[1], facility_df)
    nearest_facilities = sorted(graph.items(), key=lambda x: x[1]["distance"])[:10]  # Change from 5 to 10

    # Generate Excel with Longitude and Latitude
    excel_data = []
    for i, (loc, data) in enumerate(nearest_facilities):
        excel_data.append({
            "Product ID": product_id,
            "Customer ID": user_id,
            "Location": loc,
            "Latitude": data["latitude"],
            "Longitude": data["longitude"],
            "Cost": data["distance"]
        })
    df = pd.DataFrame(excel_data)
    output_path = f"data/nearest_facilities_{product_id}.xlsx"
    os.makedirs("data", exist_ok=True)
    df.to_excel(output_path, index=False)

    return RedirectResponse(url=f"/view_map/{product_id}/", status_code=303)

@app.get("/view_map/{product_id}/")
def view_map(request: Request, product_id: int):
    excel_path_warehouse = f"data/nearest_warehouses_{product_id}.xlsx"
    excel_path_facility = f"data/nearest_facilities_{product_id}.xlsx"

    # Check which file exists (warehouse or facility)
    if os.path.exists(excel_path_warehouse):
        excel_path = excel_path_warehouse
    elif os.path.exists(excel_path_facility):
        excel_path = excel_path_facility
    else:
        return templates.TemplateResponse("delivery_picker.html", {
            "request": request,
            "message": "No delivery locations found for this product."
        })

    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Ensure latitude and longitude columns exist
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return templates.TemplateResponse("delivery_picker.html", {
            "request": request,
            "message": "Latitude or Longitude missing in the Excel file."
        })

    # Get user's location (first row in the Excel file)
    user_location = {
        "latitude": df["Latitude"].iloc[0],
        "longitude": df["Longitude"].iloc[0],
        "location": df["Location"].iloc[0]
    }

    # Create map centered on user's location
    map_center = [user_location["latitude"], user_location["longitude"]]
    folium_map = folium.Map(location=map_center, zoom_start=10)

    # Add a red marker for the user's location
    folium.Marker(
        location=[user_location["latitude"], user_location["longitude"]],
        popup=f"User Location: {user_location['location']}",
        icon=folium.Icon(color="red")
    ).add_to(folium_map)

    # Add markers and arrows for each warehouse/facility
    for _, row in df.iterrows():
        warehouse_location = [row["Latitude"], row["Longitude"]]
        distance = row["Cost"]
        co2_emissions = distance * 0.2  # Example: 0.2 kg CO2 per km

        # Add a marker for the warehouse/facility
        folium.Marker(
            location=warehouse_location,
            popup=f"Location: {row['Location']}<br>Distance: {distance:.2f} km<br>CO2 Emissions: {co2_emissions:.2f} kg",
            icon=folium.Icon(color="blue")
        ).add_to(folium_map)

        # Draw an arrow from the user's location to the warehouse/facility
        folium.PolyLine(
            locations=[map_center, warehouse_location],
            color="green",
            weight=2.5,
            opacity=1,
            popup=f"Distance: {distance:.2f} km<br>CO2 Emissions: {co2_emissions:.2f} kg"
        ).add_to(folium_map)

    # Save the map to an HTML file
    map_file = f"static/maps/{product_id}_map.html"
    os.makedirs("static/maps", exist_ok=True)
    folium_map.save(map_file)

    # Calculate sustainability metrics (example: CO2 emissions based on distance)
    sustainability_metrics = []
    for _, row in df.iterrows():
        distance = row["Cost"]
        co2_emissions = distance * 0.2  # Example: 0.2 kg CO2 per km
        sustainability_metrics.append({
            "Location": row["Location"],
            "Distance": f"{distance:.2f} km",
            "CO2_Emissions": f"{co2_emissions:.2f} kg"
        })

    return templates.TemplateResponse("view_map.html", {
        "request": request,
        "product_id": product_id,
        "sustainability_metrics": sustainability_metrics
    })