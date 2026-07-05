from pyproj import Transformer
from pyproj.transformer import AreaOfInterest
import json

def import_region(path):
    with open(path, "r") as file:
        config = json.load(file)
    min_lon = config.get("min_lon", config.get("lon_min"))
    max_lon = config.get("max_lon", config.get("lon_max"))
    min_lat = config.get("min_lat", config.get("lat_min"))
    max_lat = config.get("max_lat", config.get("lat_max"))
    region = Region(min_lon, max_lon, min_lat, max_lat)
    return region

class Region:
    def __init__(self, min_lon, max_lon, min_lat, max_lat):
        self._locked = False
        
        self._west_deg = min_lon
        self._east_deg = max_lon
        self._south_deg = min_lat
        self._north_deg = max_lat
        
        self.calculate_projections()
        self._locked = True

    def save(self, path):
        with open(path, "w") as file:
            json.dump({
                "min_lon": self.west_deg,
                "max_lon": self.east_deg,
                "min_lat": self.south_deg,
                "max_lat": self.north_deg
            }, file)
        print(f"✓ Region bounds saved to {path}")
    
    # ---  GPS Degree Getters and Setters ---
    @property
    def west_deg(self): return self._west_deg
    @west_deg.setter
    def west_deg(self, value):
        self._west_deg = value
        self.calculate_projections()

    @property
    def east_deg(self): return self._east_deg
    @east_deg.setter
    def east_deg(self, value):
        self._east_deg = value
        self.calculate_projections()

    @property
    def south_deg(self): return self._south_deg
    @south_deg.setter
    def south_deg(self, value):
        self._south_deg = value
        self.calculate_projections()

    @property
    def north_deg(self): return self._north_deg
    @north_deg.setter
    def north_deg(self, value):
        self._north_deg = value
        self.calculate_projections()

    # ---  Alias Getters and Setters (common naming) ---
    @property
    def lon_min(self): return self._west_deg
    @lon_min.setter
    def lon_min(self, value):
        self.west_deg = value

    @property
    def lon_max(self): return self._east_deg
    @lon_max.setter
    def lon_max(self, value):
        self.east_deg = value

    @property
    def lat_min(self): return self._south_deg
    @lat_min.setter
    def lat_min(self, value):
        self.south_deg = value

    @property
    def lat_max(self): return self._north_deg
    @lat_max.setter
    def lat_max(self, value):
        self.north_deg = value

    # ---  UTM Meter Getters and Setters ---
    @property
    def west_m(self): return self._west_m
    @west_m.setter
    def west_m(self, value):
        new_lon, _ = self._utm_to_gps.transform(value, self._south_m)
        self._west_deg = new_lon
        self.calculate_projections()

    @property
    def east_m(self): return self._east_m
    @east_m.setter
    def east_m(self, value):
        new_lon, _ = self._utm_to_gps.transform(value, self._north_m)
        self._east_deg = new_lon
        self.calculate_projections()

    @property
    def south_m(self): return self._south_m
    @south_m.setter
    def south_m(self, value):
        _, new_lat = self._utm_to_gps.transform(self._west_m, value)
        self._south_deg = new_lat
        self.calculate_projections()

    @property
    def north_m(self): return self._north_m
    @north_m.setter
    def north_m(self, value):
        _, new_lat = self._utm_to_gps.transform(self._east_m, value)
        self._north_deg = new_lat
        self.calculate_projections()

    # ---  Kilometer Getters and Setters (Multiplies by 1000 to convert to meters) ---
    @property
    def west_km(self): return self._west_km
    @west_km.setter
    def west_km(self, value):
        meters_val = value * 1000.0
        new_lon, _ = self._utm_to_gps.transform(meters_val, self._south_m)
        self._west_deg = new_lon
        self.calculate_projections()

    @property
    def east_km(self): return self._east_km
    @east_km.setter
    def east_km(self, value):
        meters_val = value * 1000.0
        new_lon, _ = self._utm_to_gps.transform(meters_val, self._north_m)
        self._east_deg = new_lon
        self.calculate_projections()

    @property
    def south_km(self): return self._south_km
    @south_km.setter
    def south_km(self, value):
        meters_val = value * 1000.0
        _, new_lat = self._utm_to_gps.transform(self._west_m, meters_val)
        self._south_deg = new_lat
        self.calculate_projections()

    @property
    def north_km(self): return self._north_km
    @north_km.setter
    def north_km(self, value):
        meters_val = value * 1000.0
        _, new_lat = self._utm_to_gps.transform(self._east_m, meters_val)
        self._north_deg = new_lat
        self.calculate_projections()

    def calculate_projections(self):
        """Recalculates all UTM projections, unit conversions, and corners."""
        self._locked = False  # Temporarily unlock to update
        
        center_lon = (self._west_deg + self._east_deg) / 2.0
        center_lat = (self._south_deg + self._north_deg) / 2.0
        utm_zone = int((center_lon + 180) / 6) + 1
        self.epsg = f"EPSG:326{utm_zone:02d}" if center_lat >= 0 else f"EPSG:327{utm_zone:02d}"
        
        aoi = AreaOfInterest(west_lon_degree=self._west_deg, south_lat_degree=self._south_deg, east_lon_degree=self._east_deg, north_lat_degree=self._north_deg)
        gps_to_utm = Transformer.from_crs("EPSG:4326", self.epsg, always_xy=True, area_of_interest=aoi)
        self._utm_to_gps = Transformer.from_crs(self.epsg, "EPSG:4326", always_xy=True, area_of_interest=aoi)
        
        # Calculate raw UTM boundaries
        self._west_m, self._south_m = gps_to_utm.transform(self._west_deg, self._south_deg)
        self._east_m, self._north_m = gps_to_utm.transform(self._east_deg, self._north_deg)
        
        # Calculate Kilometer limits
        self._west_km = self._west_m / 1000.0
        self._east_km = self._east_m / 1000.0
        self._south_km = self._south_m / 1000.0
        self._north_km = self._north_m / 1000.0
        
        # Calculate spatial dimensions
        self.width_deg = self._east_deg - self._west_deg
        self.height_deg = self._north_deg - self._south_deg
        self.width_m = self._east_m - self._west_m
        self.height_m = self._north_m - self._south_m
        self.width_km = self._east_km - self._west_km
        self.height_km = self._north_km - self._south_km
        
        # Calculate Areas
        self.area_m2 = self.width_m * self.height_m
        self.area_km2 = self.width_km * self.height_km
        
        # Map corner coordinates as simple tuples
        self.southwest_deg = (self._west_deg, self._south_deg)
        self.southeast_deg = (self._east_deg, self._south_deg)
        self.northwest_deg = (self._west_deg, self._north_deg)
        self.northeast_deg = (self._east_deg, self._north_deg)
        
        self.southwest_m = (self._west_m, self._south_m)
        self.southeast_m = (self._east_m, self._south_m)
        self.northwest_m = (self._west_m, self._north_m)
        self.northeast_m = (self._east_m, self._north_m)
        
        self.southwest_km = (self._west_km, self._south_km)
        self.southeast_km = (self._east_km, self._south_km)
        self.northwest_km = (self._west_km, self._north_km)
        self.northeast_km = (self._east_km, self._north_km)
        
        self._locked = True  # Re-lock

    def __setattr__(self, name, value):
        # List of calculated attributes we must protect from external overrides
        protected_attributes = [
            "epsg", 
            "width_deg", "height_deg", "width_m", "height_m", "width_km", "height_km",
            "area_m2", "area_km2", 
            "southwest_deg", "southeast_deg", "northwest_deg", "northeast_deg",
            "southwest_m", "southeast_m", "northwest_m", "northeast_m",
            "southwest_km", "southeast_km", "northwest_km", "northeast_km"
        ]
        
        if hasattr(self, "_locked") and self._locked and name in protected_attributes:
            raise AttributeError(f"'{name}' is read-only. Modify boundary coordinates to update.")
            
        super().__setattr__(name, value)

    def plot_on_map(self):
        """Renders an interactive OpenStreetMap displaying the region boundary and its four corners."""
        import folium
        center_lat = (self._south_deg + self._north_deg) / 2.0
        center_lon = (self._west_deg + self._east_deg) / 2.0
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")
        
        bounds_rect = [[self._south_deg, self._west_deg], [self._north_deg, self._east_deg]]
        folium.Rectangle(bounds_rect, color="royalblue", weight=2.5, fill=True, fill_opacity=0.15).add_to(m)
        
        folium.Marker(self.northeast_deg, popup="Northeast").add_to(m)
        folium.Marker(self.northwest_deg, popup="Northwest").add_to(m)
        folium.Marker(self.southeast_deg, popup="Southeast").add_to(m)
        folium.Marker(self.southwest_deg, popup="Southwest").add_to(m)
        
        return m