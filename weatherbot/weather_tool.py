#weather_tool.py


import requests
import os
import json
from langchain_core.tools import tool
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import math

# Simple cache for geocoding results to avoid repeated API calls
geocoding_cache = {}

@tool
def get_coordinates(location: str) -> Optional[Dict[str, Any]]:
    """Enhanced worldwide geocoding without hardcoded locations"""
    try:
        normalized_location = normalize_location_name(location)
        print(f"DEBUG: get_coordinates called with location: '{location}', normalized: '{normalized_location}'")
        
        # Only use cache for successful results
        if normalized_location in geocoding_cache and geocoding_cache[normalized_location]:
            print(f"DEBUG: Using cached result for '{normalized_location}'")
            return geocoding_cache[normalized_location]
        
        # Dynamic query strategies - no hardcoded locations
        queries = [
            normalized_location,  # Basic name
            f"{normalized_location}, Earth",  # Global context
            f"{normalized_location} city",  # Urban areas
            f"{normalized_location} town",  # Smaller settlements
            f"{normalized_location} village",  # Rural areas
        ]
        
        # If location contains comma, try different combinations
        if ',' in normalized_location:
            parts = [p.strip() for p in normalized_location.split(',')]
            if len(parts) > 1:
                # Try with first part only
                queries.append(parts[0])
                # Try with first two parts
                queries.append(f"{parts[0]}, {parts[1]}")
                # Try with last part as primary
                queries.append(f"{parts[-1]}, {parts[0]}")
        
        for query in queries:
            # URL-encode the query to handle spaces and special characters
            from urllib.parse import quote
            encoded_query = quote(query)
            search_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_query}&count=5&language=en&format=json"
            print(f"DEBUG: Trying query: '{query}' -> URL: {search_url}")
            
            response = requests.get(search_url, timeout=10)
            print(f"DEBUG: Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"DEBUG: Response data: {data}")
                
                # Check if 'results' exists and has data
                if data.get('results') and len(data['results']) > 0:
                    # Format results consistently
                    formatted_results = []
                    for result in data['results']:
                        formatted_results.append({
                            'name': result.get('name', ''),
                            'lat': result.get('latitude', 0),
                            'lon': result.get('longitude', 0),
                            'country': result.get('country', ''),
                            'region': result.get('admin1', ''),
                            'admin2': result.get('admin2', ''),
                            'admin3': result.get('admin3', ''),
                            'population': result.get('population', 0)
                        })
                    
                    # Cache only successful results
                    geocoding_cache[normalized_location] = formatted_results
                    return formatted_results
                else:
                    print(f"DEBUG: No results found for query: '{query}'")
            else:
                print(f"DEBUG: Request failed: {response.text}")
        
        print(f"DEBUG: No results found for location: '{location}'")
        return None
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
        return None

@tool
def search_locations(query: str, count: int = 10, language: str = "en") -> List[Dict[str, Any]]:
    """Search for locations worldwide using Open-Meteo Geocoding API."""
    try:
        from urllib.parse import quote
        encoded_query = quote(query)
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_query}&count={count}&language={language}&format=json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                return data['results']  # List of dicts with name, lat, lon, country, population, etc.
        return []
    except Exception as e:
        print(f"Location search error: {str(e)}")
        return []

def normalize_location_name(name: str) -> str:
    """Normalize location names for worldwide support"""
    name_lower = name.lower().strip()
    
    # Remove common administrative terms
    admin_terms = [
        "county", "province", "state", "district", "municipality", 
        "prefecture", "territory", "region", "city", "town", "village"
    ]
    
    # Remove terms from the name
    for term in admin_terms:
        if name_lower.endswith(f" {term}"):
            name_lower = name_lower.replace(f" {term}", "")
        elif name_lower.startswith(f"{term} "):
            name_lower = name_lower.replace(f"{term} ", "")
    
    # Common abbreviations
    abbreviations = {
        "st.": "saint",
        "mt.": "mount",
        "ft.": "fort",
        "co.": "county",
    }
    
    # Apply abbreviation mappings
    for abbrev, full in abbreviations.items():
        name_lower = name_lower.replace(abbrev, full)
    
    # Capitalize properly
    return ' '.join(word.capitalize() for word in name_lower.split())

# WMO Weather interpretation codes(https://open-meteo.com/en/docs)
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

def get_weather_description(code: int) -> str:
    """Convert WMO weather code to description"""
    return WMO_WEATHER_CODES.get(code, "Unknown")

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit"""
    return (celsius * 9/5) + 32

def kmh_to_mph(kmh: float) -> float:
    """Convert km/h to mph"""
    return kmh * 0.621371

def get_location_hierarchy(location_data):
    """Extract detailed location hierarchy from geocoding data"""
    if not location_data:
        return {}
    
    location_info = location_data[0]
    hierarchy = {
        'city': location_info.get('name', ''),
        'district': location_info.get('admin2', ''),
        'state': location_info.get('admin1', ''),  # Fixed: was 'region' now 'admin1'
        'country': location_info.get('country', ''),
        'latitude': location_info.get('lat', 0),
        'longitude': location_info.get('lon', 0),
        'population': location_info.get('population', 0)
    }
    
    # Clean up empty values
    return {k: v for k, v in hierarchy.items() if v}

def clear_geocoding_cache():
    """Clear the geocoding cache"""
    global geocoding_cache
    geocoding_cache = {}
    print("Geocoding cache cleared")

@tool
def get_weather_data(location: str) -> str:
    """Fetch current weather data with detailed location information"""
    try:
        print(f"DEBUG: get_weather_data called with location: '{location}'")
        
        # First, get coordinates for the location
        geo_data = get_coordinates.run(location)
        if not geo_data:
            return f"Location '{location}' not found. Please check the spelling or try a nearby major city."
        
        print(f"DEBUG: Geo data received: {geo_data}")
        
        # Get detailed location hierarchy
        location_hierarchy = get_location_hierarchy(geo_data)
        print(f"DEBUG: Location hierarchy: {location_hierarchy}")
        
        # Get coordinates
        location_info = geo_data[0]
        lat = location_info['lat']
        lon = location_info['lon']
        
        # Create a clean location name for display
        location_parts = []
        seen = set()  # To avoid duplicates
        
        # Add location parts in order of specificity
        for part in [
            location_hierarchy.get('city'),
            location_hierarchy.get('district'),
            location_hierarchy.get('state'),
            location_hierarchy.get('country')
        ]:
            if part and part not in seen:
                location_parts.append(part)
                seen.add(part)
        
        display_location = ', '.join(location_parts) if location_parts else location
        
        print(f"DEBUG: Display location: '{display_location}'")
        
        # Get weather data using Open-Meteo API
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relativehumidity_2m,visibility,windspeed_10m,pressure_msl"
        print(f"DEBUG: API URL: '{weather_url}'")
        
        weather_response = requests.get(weather_url, timeout=10)
        print(f"DEBUG: API Response Status: {weather_response.status_code}")
        
        if weather_response.status_code == 200:
            data = weather_response.json()
            
            # Extract relevant information
            current = data.get('current_weather', {})
            if not current:
                return f"No current weather data available for {display_location}"
            
            hourly = data.get('hourly', {})
            
            # Get current time index
            current_time = current.get('time', '')
            time_index = 0  # Default to first hour if current_time not found
            
            if current_time and 'time' in hourly:
                try:
                    time_index = hourly.get('time', []).index(current_time)
                except ValueError:
                    time_index = 0
            
            # Extract weather data
            temp_c = current.get('temperature', 0)
            temp_f = celsius_to_fahrenheit(temp_c)
            weather_code = current.get('weathercode', 0)
            condition = get_weather_description(weather_code)
            humidity = hourly.get('relativehumidity_2m', [0])[time_index] if 'relativehumidity_2m' in hourly else 0
            wind_kmh = current.get('windspeed', 0)
            wind_mph = kmh_to_mph(wind_kmh)
            pressure_mb = hourly.get('pressure_msl', [0])[time_index] if 'pressure_msl' in hourly else 0
            visibility_km = hourly.get('visibility', [0])[time_index] / 1000 if 'visibility' in hourly else 0  # Convert to km
            is_day = current.get('is_day', 1)
            wind_direction = current.get('winddirection', 0)
            
            # Format the response with detailed location information
            result = f"Weather in {display_location}:\n"
            result += f"Location Details:\n"
            if location_hierarchy.get('city'):
                result += f"  City: {location_hierarchy['city']}\n"
            if location_hierarchy.get('district') and location_hierarchy.get('district') != location_hierarchy.get('city'):
                result += f"  District: {location_hierarchy['district']}\n"
            if location_hierarchy.get('state'):
                result += f"  State: {location_hierarchy['state']}\n"
            if location_hierarchy.get('country'):
                result += f"  Country: {location_hierarchy['country']}\n"
            result += f"  Coordinates: {lat}, {lon}\n"
            if location_hierarchy.get('population'):
                result += f"  Population: {location_hierarchy['population']:,}\n"
            result += "\n"
            result += f"Current Conditions:\n"
            result += f"  Temperature: {temp_c}°C ({temp_f:.1f}°F)\n"
            result += f"  Condition: {condition}\n"
            result += f"  Humidity: {humidity}%\n"
            result += f"  Wind: {wind_kmh} km/h ({wind_mph:.1f} mph) from {wind_direction}°\n"
            result += f"  Pressure: {pressure_mb} hPa\n"
            result += f"  Visibility: {visibility_km:.1f} km\n"
            result += f"  Daytime: {'Yes' if is_day == 1 else 'No'}"
            
            print(f"DEBUG: Formatted result: {result}")
            return result
        else:
            error_msg = f"Error fetching weather data: {weather_response.status_code} - {weather_response.text}"
            print(f"DEBUG: Error message: {error_msg}")
            return error_msg
    except Exception as e:
        error_msg = f"Error fetching weather data: {str(e)}"
        print(f"DEBUG: Exception: {error_msg}")
        return error_msg

@tool
def get_weather_forecast(location: str, days: int = 5) -> str:
    """Fetch weather forecast for multiple days using Open-Meteo API"""
    try:
        # First, get coordinates for the location
        geo_data = get_coordinates.run(location)
        if not geo_data:
            return f"Location '{location}' not found"
        
        # Get coordinates
        location_info = geo_data[0]
        lat = location_info['lat']
        lon = location_info['lon']
        location_name = f"{location_info['name']}, {location_info['region']}, {location_info['country']}"
        
        # Get forecast data using Open-Meteo API
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weathercode,temperature_2m_max,temperature_2m_min,sunrise,sunset,precipitation_probability_max,windspeed_10m_max&timezone=auto"
        print(f"Calling Open-Meteo Forecast API: {forecast_url}")
        forecast_response = requests.get(forecast_url, timeout=10)
        
        if forecast_response.status_code == 200:
            data = forecast_response.json()
            print(f"Open-Meteo Forecast API response: {json.dumps(data, indent=2)}")
            
            # Format the forecast data in a structured way
            result = f"Weather forecast for {location_name}:\n\n"
            
            daily = data.get('daily', {})
            dates = daily.get('time', [])
            weather_codes = daily.get('weathercode', [])
            max_temps = daily.get('temperature_2m_max', [])
            min_temps = daily.get('temperature_2m_min', [])
            sunrises = daily.get('sunrise', [])
            sunsets = daily.get('sunset', [])
            precip_probs = daily.get('precipitation_probability_max', [])
            wind_speeds = daily.get('windspeed_10m_max', [])
            
            for i in range(min(days, len(dates))):
                date = dates[i]
                weather_code = weather_codes[i]
                condition = get_weather_description(weather_code)
                max_temp_c = max_temps[i]
                max_temp_f = celsius_to_fahrenheit(max_temp_c)
                min_temp_c = min_temps[i]
                min_temp_f = celsius_to_fahrenheit(min_temp_c)
                sunrise = sunrises[i].split('T')[1] if 'T' in sunrises[i] else sunrises[i]
                sunset = sunsets[i].split('T')[1] if 'T' in sunsets[i] else sunsets[i]
                precip_prob = precip_probs[i]
                wind_speed = wind_speeds[i]
                wind_speed_mph = kmh_to_mph(wind_speed)
                
                result += f"Date: {date}\n"
                result += f"  Condition: {condition}\n"
                result += f"  Max Temp: {max_temp_c}°C ({max_temp_f:.1f}°F)\n"
                result += f"  Min Temp: {min_temp_c}°C ({min_temp_f:.1f}°F)\n"
                result += f"  Chance of Rain: {precip_prob}%\n"
                result += f"  Wind: {wind_speed} km/h ({wind_speed_mph:.1f} mph)\n"
                result += f"  Sunrise: {sunrise}\n"
                result += f"  Sunset: {sunset}\n\n"
            
            return result
        else:
            error_msg = f"Error fetching forecast data: {forecast_response.status_code} - {forecast_response.text}"
            print(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"Error fetching forecast data: {str(e)}"
        print(error_msg)
        return error_msg

@tool
def get_visual_weather_data(location: str, days: int = 7) -> Dict[str, Any]:
    """Fetch structured weather data for a specified number of days, optimized for visualization."""
    try:
        # Clamp days to a reasonable forecast limit (1-14 days)
        days = max(1, min(14, days))

        # First, get coordinates for the location
        geo_data = get_coordinates.run(location)
        if not geo_data:
            return {"error": f"Location '{location}' not found", "status": "error"}
        
        # Get detailed location hierarchy
        location_hierarchy = get_location_hierarchy(geo_data)
        
        # Get coordinates
        location_info = geo_data[0]
        lat = location_info['lat']
        lon = location_info['lon']
        
        # Create detailed location name without duplicating names
        location_parts = []
        seen = set()  # To avoid duplicates
        
        # Add location parts in order of specificity
        for part in [
            location_hierarchy.get('city'),
            location_hierarchy.get('district'),
            location_hierarchy.get('state'),
            location_hierarchy.get('country')
        ]:
            if part and part not in seen:
                location_parts.append(part)
                seen.add(part)
        
        location_name = ', '.join(location_parts) if location_parts else location
        
        # Get current weather data
        current_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relativehumidity_2m,visibility,windspeed_10m,pressure_msl,precipitation"
        print(f"Calling Open-Meteo Current API: {current_url}")
        current_response = requests.get(current_url, timeout=10)
        
        # Get hourly forecast data
        hourly_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,weathercode,precipitation_probability,windspeed_10m,relativehumidity_2m&forecast_days=2"
        print(f"Calling Open-Meteo Hourly API: {hourly_url}")
        hourly_response = requests.get(hourly_url, timeout=10)
        
        # Get daily forecast data for the requested number of days
        daily_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weathercode,temperature_2m_max,temperature_2m_min,sunrise,sunset,precipitation_probability_max,windspeed_10m_max,uv_index_max&timezone=auto&forecast_days={days}"
        print(f"Calling Open-Meteo Daily API: {daily_url}")
        daily_response = requests.get(daily_url, timeout=10)
        
        if current_response.status_code == 200 and hourly_response.status_code == 200 and daily_response.status_code == 200:
            current_data = current_response.json()
            hourly_data = hourly_response.json()
            daily_data = daily_response.json()
            
            # Extract current weather data
            current = current_data.get('current_weather', {})
            hourly = current_data.get('hourly', {})
            
            # Get current time index
            current_time = current.get('time', '')
            time_index = hourly.get('time', []).index(current_time) if current_time in hourly.get('time', []) else 0
            
            # Format current weather
            current_weather = {
                "temp_c": current.get('temperature', 0),
                "temp_f": celsius_to_fahrenheit(current.get('temperature', 0)),
                "condition": get_weather_description(current.get('weathercode', 0)),
                "weather_code": current.get('weathercode', 0),
                "humidity": hourly.get('relativehumidity_2m', [0])[time_index],
                "wind_kmh": current.get('windspeed', 0),
                "wind_mph": kmh_to_mph(current.get('windspeed', 0)),
                "wind_direction": current.get('winddirection', 0),
                "pressure_mb": hourly.get('pressure_msl', [0])[time_index],
                "precip_mm": hourly.get('precipitation', [0])[time_index],
                "visibility_km": hourly.get('visibility', [0])[time_index] / 1000,
                "is_day": current.get('is_day', 1) == 1,
                "last_updated": current_time
            }
            
            # Extract hourly forecast for the next 24 hours
            hourly_forecast = []
            hourly_times = hourly_data.get('hourly', {}).get('time', [])
            
            # Find the current hour index
            current_hour_index = 0
            for i, time_str in enumerate(hourly_times):
                if current_time in time_str:
                    current_hour_index = i
                    break
            
            # Get the next 24 hours
            for i in range(current_hour_index, min(current_hour_index + 24, len(hourly_times))):
                hour_data = {
                    "time": hourly_times[i],
                    "temp_c": hourly_data.get('hourly', {}).get('temperature_2m', [])[i],
                    "temp_f": celsius_to_fahrenheit(hourly_data.get('hourly', {}).get('temperature_2m', [])[i]),
                    "condition": get_weather_description(hourly_data.get('hourly', {}).get('weathercode', [])[i]),
                    "weather_code": hourly_data.get('hourly', {}).get('weathercode', [])[i],
                    "chance_of_rain": hourly_data.get('hourly', {}).get('precipitation_probability', [])[i],
                    "wind_kmh": hourly_data.get('hourly', {}).get('windspeed_10m', [])[i],
                    "wind_mph": kmh_to_mph(hourly_data.get('hourly', {}).get('windspeed_10m', [])[i]),
                    "humidity": hourly_data.get('hourly', {}).get('relativehumidity_2m', [])[i]
                }
                hourly_forecast.append(hour_data)
            
            # Extract daily forecast for the number of days returned by the API
            daily_forecast = []
            daily_times = daily_data.get('daily', {}).get('time', [])
            
            for i in range(len(daily_times)): # MODIFIED: Loop over all returned days
                date = daily_times[i]
                try:
                    day_of_week = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
                except:
                    day_of_week = "Unknown"
                
                daily_forecast.append({
                    "date": date,
                    "day_of_week": day_of_week,
                    "condition": get_weather_description(daily_data.get('daily', {}).get('weathercode', [])[i]),
                    "weather_code": daily_data.get('daily', {}).get('weathercode', [])[i],
                    "max_temp_c": daily_data.get('daily', {}).get('temperature_2m_max', [])[i],
                    "max_temp_f": celsius_to_fahrenheit(daily_data.get('daily', {}).get('temperature_2m_max', [])[i]),
                    "min_temp_c": daily_data.get('daily', {}).get('temperature_2m_min', [])[i],
                    "min_temp_f": celsius_to_fahrenheit(daily_data.get('daily', {}).get('temperature_2m_min', [])[i]),
                    "avg_humidity": 0,  # Not available in daily data
                    "chance_of_rain": daily_data.get('daily', {}).get('precipitation_probability_max', [])[i],
                    "uv_index": daily_data.get('daily', {}).get('uv_index_max', [])[i],
                    "wind_kmh": daily_data.get('daily', {}).get('windspeed_10m_max', [])[i],
                    "wind_mph": kmh_to_mph(daily_data.get('daily', {}).get('windspeed_10m_max', [])[i]),
                    "sunrise": daily_data.get('daily', {}).get('sunrise', [])[i].split('T')[1] if 'T' in daily_data.get('daily', {}).get('sunrise', [])[i] else daily_data.get('daily', {}).get('sunrise', [])[i],
                    "sunset": daily_data.get('daily', {}).get('sunset', [])[i].split('T')[1] if 'T' in daily_data.get('daily', {}).get('sunset', [])[i] else daily_data.get('daily', {}).get('sunset', [])[i]
                })
            
            # Return structured data for visualization with detailed location info
            return {
                "location": {
                    "name": location_name,
                    "city": location_hierarchy.get('city', ''),
                    "district": location_hierarchy.get('district', ''),
                    "state": location_hierarchy.get('state', ''),
                    "country": location_hierarchy.get('country', ''),
                    "population": location_hierarchy.get('population', 0),
                    "lat": lat,
                    "lon": lon,
                    "timezone": daily_data.get('timezone', '')
                },
                "current": current_weather,
                "hourly": hourly_forecast,
                "daily": daily_forecast,
                "status": "success"
            }
        else:
            error_msg = f"Error fetching weather data: One or more API calls failed"
            print(error_msg)
            return {"error": error_msg, "status": "error"}
    except Exception as e:
        error_msg = f"Error fetching visual weather data: {str(e)}"
        print(error_msg)
        return {"error": error_msg, "status": "error"}

@tool
def get_historical_weather_data(location: str = "", start_date: str = "", end_date: str = "") -> Dict[str, Any]:
    """Get historical weather data for a location"""
    try:
        # Check if required parameters are provided
        if not location:
            return {"error": "Missing required parameter: location", "status": "error"}
        if not start_date:
            return {"error": "Missing required parameter: start_date", "status": "error"}
        if not end_date:
            return {"error": "Missing required parameter: end_date", "status": "error"}
        
        # Get coordinates
        geo_data = get_coordinates.run(location)
        if not geo_data:
            return {"error": f"Location '{location}' not found", "status": "error"}
        
        location_info = geo_data[0]
        lat = location_info['lat']
        lon = location_info['lon']
        
        # Get historical weather data
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,sunrise,sunset,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours,wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,shortwave_radiation_sum,et0_fao_evapotranspiration"
        }
        
        print(f"Fetching historical data with params: {params}")
        response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process the data
            historical_data = {
                "location": location_info,
                "daily": [],
                "status": "success"
            }
            
            # Process daily data
            dates = data.get("daily", {}).get("time", [])
            for i, date in enumerate(dates):
                day_data = {
                    "date": date,
                    "weather_code": data.get("daily", {}).get("weather_code", [])[i] if "weather_code" in data.get("daily", {}) and i < len(data.get("daily", {}).get("weather_code", [])) else None,
                    "max_temp": data.get("daily", {}).get("temperature_2m_max", [])[i] if "temperature_2m_max" in data.get("daily", {}) and i < len(data.get("daily", {}).get("temperature_2m_max", [])) else None,
                    "min_temp": data.get("daily", {}).get("temperature_2m_min", [])[i] if "temperature_2m_min" in data.get("daily", {}) and i < len(data.get("daily", {}).get("temperature_2m_min", [])) else None,
                    "precipitation_sum": data.get("daily", {}).get("precipitation_sum", [])[i] if "precipitation_sum" in data.get("daily", {}) and i < len(data.get("daily", {}).get("precipitation_sum", [])) else None,
                    "sunrise": data.get("daily", {}).get("sunrise", [])[i] if "sunrise" in data.get("daily", {}) and i < len(data.get("daily", {}).get("sunrise", [])) else None,
                    "sunset": data.get("daily", {}).get("sunset", [])[i] if "sunset" in data.get("daily", {}) and i < len(data.get("daily", {}).get("sunset", [])) else None
                }
                historical_data["daily"].append(day_data)
            
            print(f"Successfully processed historical data: {historical_data}")
            return historical_data
        
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            print(error_msg)
            return {"error": error_msg, "status": "error"}
            
    except Exception as e:
        error_msg = f"Error fetching historical weather data: {str(e)}"
        print(error_msg)
        return {"error": error_msg, "status": "error"}
    
@tool
def get_marine_weather_data(location: str = "") -> Dict[str, Any]:
    """Get marine weather data for coastal locations"""
    try:
        if not location:
            return {"error": "Missing required parameter: location", "status": "error"}
        
        # Get coordinates
        geo_data = get_coordinates.run(location)
        if not geo_data:
            return {"error": f"Location '{location}' not found", "status": "error"}
        
        location_info = geo_data[0]
        lat = location_info['lat']
        lon = location_info['lon']
        
        # Get marine weather data - using correct parameter names
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "wave_height,swell_wave_height,swell_wave_period,swell_wave_direction,wind_wave_height,wind_wave_period,wind_wave_direction",
            "daily": "wave_height_max,wind_wave_height_max,swell_wave_height_max",
            "timezone": "auto"
        }
        
        print(f"Fetching marine data with params: {params}")
        response = requests.get("https://marine-api.open-meteo.com/v1/marine", params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process the data
            marine_data = {
                "location": location_info,
                "hourly": [],
                "daily": [],
                "status": "success"
            }
            
            # Process hourly data
            times = data.get("hourly", {}).get("time", [])
            for i, time in enumerate(times):
                hour_data = {
                    "time": time,
                    "wave_height": data.get("hourly", {}).get("wave_height", [])[i] if "wave_height" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("wave_height", [])) else None,
                    "swell_height": data.get("hourly", {}).get("swell_wave_height", [])[i] if "swell_wave_height" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("swell_wave_height", [])) else None,
                    "swell_period": data.get("hourly", {}).get("swell_wave_period", [])[i] if "swell_wave_period" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("swell_wave_period", [])) else None,
                    "swell_direction": data.get("hourly", {}).get("swell_wave_direction", [])[i] if "swell_wave_direction" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("swell_wave_direction", [])) else None,
                    "wind_wave_height": data.get("hourly", {}).get("wind_wave_height", [])[i] if "wind_wave_height" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("wind_wave_height", [])) else None,
                    "wind_wave_period": data.get("hourly", {}).get("wind_wave_period", [])[i] if "wind_wave_period" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("wind_wave_period", [])) else None,
                    "wind_wave_direction": data.get("hourly", {}).get("wind_wave_direction", [])[i] if "wind_wave_direction" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("wind_wave_direction", [])) else None
                }
                marine_data["hourly"].append(hour_data)
            
            # Process daily data
            dates = data.get("daily", {}).get("time", [])
            for i, date in enumerate(dates):
                day_data = {
                    "date": date,
                    "wave_height_max": data.get("daily", {}).get("wave_height_max", [])[i] if "wave_height_max" in data.get("daily", {}) and i < len(data.get("daily", {}).get("wave_height_max", [])) else None,
                    "wind_wave_height_max": data.get("daily", {}).get("wind_wave_height_max", [])[i] if "wind_wave_height_max" in data.get("daily", {}) and i < len(data.get("daily", {}).get("wind_wave_height_max", [])) else None,
                    "swell_height_max": data.get("daily", {}).get("swell_wave_height_max", [])[i] if "swell_wave_height_max" in data.get("daily", {}) and i < len(data.get("daily", {}).get("swell_wave_height_max", [])) else None
                }
                marine_data["daily"].append(day_data)
            
            print(f"Successfully processed marine data: {marine_data}")
            return marine_data
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            print(error_msg)
            return {"error": error_msg, "status": "error"}
            
    except Exception as e:
        error_msg = f"Error fetching marine weather data: {str(e)}"
        print(error_msg)
        return {"error": error_msg, "status": "error"}

@tool
def get_air_quality_data(location: str = "") -> Dict[str, Any]:
    """Get air quality data for a location"""
    try:
        if not location:
            return {"error": "Missing required parameter: location", "status": "error"}
        
        # Get coordinates
        geo_data = get_coordinates.run(location)
        if not geo_data:
            return {"error": f"Location '{location}' not found", "status": "error"}
        
        location_info = geo_data[0]
        lat = location_info['lat']
        lon = location_info['lon']
        
        # Get air quality data
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index_clear_sky",
            "timezone": "auto",
            "past_days": 1,
            "forecast_days": 3
        }
        
        response = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Process the data
            air_quality_data = {
                "location": location_info,
                "hourly": [],
                "status": "success"
            }
            
            # Process hourly data
            times = data.get("hourly", {}).get("time", [])
            for i, time in enumerate(times):
                hour_data = {
                    "time": time,
                    "pm10": data.get("hourly", {}).get("pm10", [])[i] if "pm10" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("pm10", [])) else None,
                    "pm2_5": data.get("hourly", {}).get("pm2_5", [])[i] if "pm2_5" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("pm2_5", [])) else None,
                    "carbon_monoxide": data.get("hourly", {}).get("carbon_monoxide", [])[i] if "carbon_monoxide" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("carbon_monoxide", [])) else None,
                    "nitrogen_dioxide": data.get("hourly", {}).get("nitrogen_dioxide", [])[i] if "nitrogen_dioxide" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("nitrogen_dioxide", [])) else None,
                    "sulphur_dioxide": data.get("hourly", {}).get("sulphur_dioxide", [])[i] if "sulphur_dioxide" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("sulphur_dioxide", [])) else None,
                    "ozone": data.get("hourly", {}).get("ozone", [])[i] if "ozone" in data.get("hourly", {}) and i < len(data.get("hourly", {}).get("ozone", [])) else None
                }
                air_quality_data["hourly"].append(hour_data)
            
            return air_quality_data
        
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            print(error_msg)
            return {"error": error_msg, "status": "error"}
            
    except Exception as e:
        error_msg = f"Error fetching air quality data: {str(e)}"
        print(error_msg)
        return {"error": error_msg, "status": "error"}

@tool
def get_weather_details(location: str, detail_type: str) -> str:
    """Get specific weather details for a location using Open-Meteo API"""
    try:
        # Get the full weather data first
        weather_data = get_weather_data.run(location)
        
        if "Location" not in weather_data:
            return weather_data  # Return error message if location not found
        
        # Parse the weather data to extract the requested detail
        lines = weather_data.split('\n')
        detail_value = "Unknown"
        
        for line in lines:
            if line.startswith(f"{detail_type}:"):
                detail_value = line.replace(f"{detail_type}:", "").strip()
                break
        
        # Format the response based on the detail type
        if detail_type.lower() == "condition":
            # Check if it's raining
            if "rain" in detail_value.lower():
                return f"Yes, it is currently raining in {location}. The condition is {detail_value}."
            else:
                return f"No, it is not currently raining in {location}. The condition is {detail_value}."
        elif detail_type.lower() == "temperature":
            return f"The current temperature in {location} is {detail_value}."
        elif detail_type.lower() == "humidity":
            return f"The humidity in {location} is {detail_value}."
        elif detail_type.lower() == "wind":
            return f"The wind speed in {location} is {detail_value}."
        elif detail_type.lower() == "pressure":
            return f"The atmospheric pressure in {location} is {detail_value}."
        elif detail_type.lower() == "visibility":
            return f"The visibility in {location} is {detail_value}."
        else:
            return f"The {detail_type} in {location} is {detail_value}."
    except Exception as e:
        error_msg = f"Error getting weather details: {str(e)}"
        print(error_msg)
        return {"error": error_msg, "status": "error"}

def _process_hourly_data(hourly_data):
    """Process hourly data into a more usable format"""
    if not hourly_data:
        return []
    
    # Get all available time points
    times = hourly_data.get("time", [])
    
    # Process each hour
    hourly_list = []
    for i, time in enumerate(times):
        hour_data = {
            "time": time,
            "temperature": hourly_data.get("temperature_2m", [])[i] if "temperature_2m" in hourly_data else None,
            "humidity": hourly_data.get("relative_humidity_2m", [])[i] if "relative_humidity_2m" in hourly_data else None,
            "apparent_temp": hourly_data.get("apparent_temperature", [])[i] if "apparent_temperature" in hourly_data else None,
            "precip_prob": hourly_data.get("precipitation_probability", [])[i] if "precipitation_probability" in hourly_data else None,
            "precipitation": hourly_data.get("precipitation", [])[i] if "precipitation" in hourly_data else None,
            "weather_code": hourly_data.get("weather_code", [])[i] if "weather_code" in hourly_data else None,
            "cloud_cover": hourly_data.get("cloud_cover", [])[i] if "cloud_cover" in hourly_data else None,
            "wind_speed": hourly_data.get("wind_speed_10m", [])[i] if "wind_speed_10m" in hourly_data else None,
            "wind_direction": hourly_data.get("wind_direction_10m", [])[i] if "wind_direction_10m" in hourly_data else None,
            "uv_index": hourly_data.get("uv_index", [])[i] if "uv_index" in hourly_data else None,
            "visibility": hourly_data.get("visibility", [])[i] if "visibility" in hourly_data else None,
            "is_day": hourly_data.get("is_day", [])[i] if "is_day" in hourly_data else None
        }
        hourly_list.append(hour_data)
    
    return hourly_list

def _process_daily_data(daily_data):
    """Process daily data into a more usable format"""
    if not daily_data:
        return []
    
    # Get all available dates
    dates = daily_data.get("time", [])
    
    # Process each day
    daily_list = []
    for i, date in enumerate(dates):
        day_data = {
            "date": date,
            "weather_code": daily_data.get("weather_code", [])[i] if "weather_code" in daily_data else None,
            "max_temp": daily_data.get("temperature_2m_max", [])[i] if "temperature_2m_max" in daily_data else None,
            "min_temp": daily_data.get("temperature_2m_min", [])[i] if "temperature_2m_min" in daily_data else None,
            "max_apparent_temp": daily_data.get("apparent_temperature_max", [])[i] if "apparent_temperature_max" in daily_data else None,
            "min_apparent_temp": daily_data.get("apparent_temperature_min", [])[i] if "apparent_temperature_min" in daily_data else None,
            "sunrise": daily_data.get("sunrise", [])[i] if "sunrise" in daily_data else None,
            "sunset": daily_data.get("sunset", [])[i] if "sunset" in daily_data else None,
            "daylight_duration": daily_data.get("daylight_duration", [])[i] if "daylight_duration" in daily_data else None,
            "sunshine_duration": daily_data.get("sunshine_duration", [])[i] if "sunshine_duration" in daily_data else None,
            "precipitation_sum": daily_data.get("precipitation_sum", [])[i] if "precipitation_sum" in daily_data else None,
            "rain_sum": daily_data.get("rain_sum", [])[i] if "rain_sum" in daily_data else None,
            "showers_sum": daily_data.get("showers_sum", [])[i] if "showers_sum" in daily_data else None,
            "snowfall_sum": daily_data.get("snowfall_sum", [])[i] if "snowfall_sum" in daily_data else None,
            "precipitation_hours": daily_data.get("precipitation_hours", [])[i] if "precipitation_hours" in daily_data else None,
            "precipitation_prob_max": daily_data.get("precipitation_probability_max", [])[i] if "precipitation_probability_max" in daily_data else None,
            "wind_speed_max": daily_data.get("wind_speed_10m_max", [])[i] if "wind_speed_10m_max" in daily_data else None,
            "wind_gusts_max": daily_data.get("wind_gusts_10m_max", [])[i] if "wind_gusts_10m_max" in daily_data else None,
            "wind_direction_dominant": daily_data.get("wind_direction_10m_dominant", [])[i] if "wind_direction_10m_dominant" in daily_data else None,
            "shortwave_radiation": daily_data.get("shortwave_radiation_sum", [])[i] if "shortwave_radiation_sum" in daily_data else None,
            "et0_fao_evapotranspiration": daily_data.get("et0_fao_evapotranspiration", [])[i] if "et0_fao_evapotranspiration" in daily_data else None,
            "uv_index_max": daily_data.get("uv_index_max", [])[i] if "uv_index_max" in daily_data else None
        }
        daily_list.append(day_data)
    
    return daily_list

@tool
def get_comprehensive_weather_data(location: str) -> Dict[str, Any]:
    """Get comprehensive weather data using multiple Open-Meteo APIs"""
    try:
        # Get coordinates
        geo_data = get_coordinates.run(location)
        if not geo_data:
            return {"error": f"Location '{location}' not found", "status": "error"}
        
        location_info = geo_data[0]
        lat = location_info['lat']
        lon = location_info['lon']
        
        # Get current weather
        current_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"is_day,precipitation,rain,showers,snowfall,weather_code,cloud_cover,"
            f"pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,"
            f"wind_gusts_10m,uv_index,visibility"
            f"&hourly=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"precipitation_probability,precipitation,rain,showers,snowfall,"
            f"weather_code,cloud_cover,pressure_mse,surface_pressure,wind_speed_10m,"
            f"wind_direction_10m,wind_gusts_10m,uv_index,visibility,is_day"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min,"
            f"apparent_temperature_max,apparent_temperature_min,sunrise,sunset,"
            f"daylight_duration,sunshine_duration,precipitation_sum,rain_sum,"
            f"showers_sum,snowfall_sum,precipitation_hours,precipitation_probability_max,"
            f"wind_speed_10m_max,wind_gusts_10m_max,wind_direction_10m_dominant,"
            f"shortwave_radiation_sum,et0_fao_evapotranspiration,uv_index_max"
            f"&timezone=auto"
            f"&forecast_days=7"
        )
        
        # Get historical data (past 7 days)
        historical_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')}"
            f"&end_date={datetime.now().strftime('%Y-%m-%d')}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        )
        
        # Get air quality data (if available)
        air_quality_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
            f"sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index_clear_sky"
        )
        
        # Get elevation data
        elevation_url = (
            f"https://api.open-meteo.com/v1/elevation?"
            f"latitude={lat}&longitude={lon}"
        )
        
        # Make API calls
        current_response = requests.get(current_url, timeout=15)
        historical_response = requests.get(historical_url, timeout=15)
        air_quality_response = requests.get(air_quality_url, timeout=15)
        elevation_response = requests.get(elevation_url, timeout=15)
        
        # Process responses
        result = {
            "location": location_info,
            "status": "success",
            "current": {},
            "hourly": [],
            "daily": [],
            "historical": {},
            "air_quality": {},
            "elevation": 0
        }
        
        if current_response.status_code == 200:
            current_data = current_response.json()
            
            # Process current weather
            result["current"] = {
                "temperature": current_data.get("current", {}).get("temperature_2m"),
                "humidity": current_data.get("current", {}).get("relative_humidity_2m"),
                "apparent_temperature": current_data.get("current", {}).get("apparent_temperature"),
                "is_day": current_data.get("current", {}).get("is_day"),
                "precipitation": current_data.get("current", {}).get("precipitation"),
                "rain": current_data.get("current", {}).get("rain"),
                "showers": current_data.get("current", {}).get("showers"),
                "snowfall": current_data.get("current", {}).get("snowfall"),
                "weather_code": current_data.get("current", {}).get("weather_code"),
                "cloud_cover": current_data.get("current", {}).get("cloud_cover"),
                "pressure_msl": current_data.get("current", {}).get("pressure_msl"),
                "surface_pressure": current_data.get("current", {}).get("surface_pressure"),
                "wind_speed": current_data.get("current", {}).get("wind_speed_10m"),
                "wind_direction": current_data.get("current", {}).get("wind_direction_10m"),
                "wind_gusts": current_data.get("current", {}).get("wind_gusts_10m"),
                "uv_index": current_data.get("current", {}).get("uv_index"),
                "visibility": current_data.get("current", {}).get("visibility"),
                "time": current_data.get("current", {}).get("time")
            }
            
            # Process hourly data
            result["hourly"] = _process_hourly_data(current_data.get("hourly", {}))
            
            # Process daily data
            result["daily"] = _process_daily_data(current_data.get("daily", {}))
        
        if historical_response.status_code == 200:
            historical_data = historical_response.json()
            result["historical"] = {
                "daily": _process_daily_data(historical_data.get("daily", {}))
            }
        
        if air_quality_response.status_code == 200:
            air_quality_data = air_quality_response.json()
            result["air_quality"] = {
                "hourly": _process_hourly_data(air_quality_data.get("hourly", {}))
            }
        
        if elevation_response.status_code == 200:
            elevation_data = elevation_response.json()
            result["elevation"] = elevation_data.get("elevation", [0])[0] if elevation_data.get("elevation") else 0
        
        return result
        
    except Exception as e:
        error_msg = f"Error fetching comprehensive weather data: {str(e)}"
        print(error_msg)
        return {"error": error_msg, "status": "error"}
    
@tool
def disambiguate_location(location: str) -> List[Dict[str, Any]]:
    """Disambiguate a location name by finding possible matches"""
    try:
        # Try to get coordinates directly first
        geo_data = get_coordinates.run(location)
        if geo_data:
            return geo_data
        
        # Try some common variations if direct lookup fails
        variations = [
            location,
            f"{location}, United States",
            f"{location}, United Kingdom",
            f"{location}, Canada",
            f"{location}, Australia"
        ]
        
        for variation in variations:
            geo_data = get_coordinates.run(variation)
            if geo_data:
                return geo_data
        
        return []
    except Exception as e:
        print(f"Error disambiguating location: {str(e)}")
        return []