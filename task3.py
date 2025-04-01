import pandas as pd
import spacy
from geopy.geocoders import Nominatim
import folium
from folium.plugins import HeatMap
import time
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

######################################################### extract locations #############################################

# Load spaCy model for place recognition
nlp = spacy.load("en_core_web_sm")

# Initialize geocoder
geolocator = Nominatim(user_agent="crisis_mapping")

# Function to extract locations using spaCy NLP
def extract_location(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE (Geopolitical Entity) is for places
            return ent.text
    return None

# Function to geocode the location (convert to lat/lon)
def geocode_location(location):
    if not location:  # Handle empty locations
        return None, None
    
    retries = 3  # Number of retries
    for _ in range(retries):
        try:
            location_obj = geolocator.geocode(location, timeout=5)  # Increased timeout
            if location_obj:
                return location_obj.latitude, location_obj.longitude
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding error for '{location}': {e}")
            time.sleep(2)  # Wait before retrying
    return None, None  # Return None if geocoding fails after retries

# Sample data: Replace this with your actual DataFrame
df = pd.read_csv("data/classified_reddit_posts.csv")  

# Extract and geocode locations
df["location"] = df["Content"].fillna("").apply(extract_location)
df["Latitude"], df["Longitude"] = zip(*df["location"].fillna("").apply(geocode_location))

# Print out the DataFrame to check results
print('✅ locations are successfully geocoded!')

######################################################### generatig heatmaps #############################################
# Filter out rows with missing coordinates
df_valid_locations = df.dropna(subset=["Latitude", "Longitude"])

# Create a base map centered around an average location (latitude, longitude)
m = folium.Map(location=[df_valid_locations["Latitude"].mean(), df_valid_locations["Longitude"].mean()], zoom_start=5)

# Add HeatMap layer
heat_data = [[row["Latitude"], row["Longitude"]] for index, row in df_valid_locations.iterrows()]
HeatMap(heat_data).add_to(m)

# Save the map as an HTML file
m.save("htmls/crisis_heatmap.html")


######################################################### desplaying the top 5 #############################################
# Count occurrences of each location
location_counts = df_valid_locations["location"].value_counts().head(5)

# Display the top 5 locations
print("Top 5 Locations with Crisis Discussions:")
print(location_counts)

# Optionally, add markers to the map for the top 5 locations
for loc, count in location_counts.items():
    location_obj = geolocator.geocode(loc)
    if location_obj:
        folium.Marker(
            location=[location_obj.latitude, location_obj.longitude],
            popup=f"{loc}: {count} posts"
        ).add_to(m)

# Save the map with markers
m.save("htmls/crisis_heatmap_with_top_5_locations.html")
