17/10/25


# app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import traceback
import re
from workflow import WeatherAgent
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize the weather agent
agent = WeatherAgent()

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/weather', methods=['POST', 'OPTIONS'])
def get_weather():
    """API endpoint to get weather information"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        if not data or ('message' not in data and 'query' not in data):
            return jsonify({
                'error': 'No message or query provided. Please provide a location.',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        # Accept both 'message' and 'query' fields
        user_input = data.get('message') or data.get('query')
        
        # Extract location from query if it's a full sentence
        cleaned_input = extract_location_from_query(user_input)
        print(f"Original input: {user_input}")
        print(f"Cleaned input: {cleaned_input}")
        
        # Check if the location is too short or ambiguous
        if len(cleaned_input.strip()) < 2:
            return jsonify({
                'error': 'Location name too short. Please provide a more specific location name.',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        # Get response from the agent
        try:
            result = agent.run(cleaned_input)
            print(f"Agent raw result: {result}")
            
            # Handle the case when agent.run returns more than 2 values
            if len(result) == 3:
                response, follow_up, visual_data = result
            else:
                # Fallback for backward compatibility
                response, follow_up = result
                visual_data = {}
            
            print(f"Agent response: {response}")
        except Exception as agent_error:
            print(f"Agent error: {str(agent_error)}")
            print(traceback.format_exc())
            return jsonify({
                'error': f'Error processing your request: {str(agent_error)}',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 500
        
        # Check for error responses from the agent
        error_keywords = [
            "failed to fetch weather data",
            "location not found",
            "error fetching weather data",
            "error occurred",
            "location 'ch' not found",
            "location not found. please check the spelling",
            "unknown location",
            "not found"
        ]
        
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in error_keywords):
            print(f"Error detected in agent response: {response}")
            return jsonify({
                'error': response,
                'follow_up': follow_up,
                'visual_data': visual_data,
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 500
        
        return jsonify({
            'response': response,
            'follow_up': follow_up,
            'visual_data': visual_data,
            'status': 'success'
        })
        
    except ValueError as ve:
        # Handle unpacking errors specifically
        print(f"Unpacking error: {str(ve)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error. Please try again.',
            'details': str(ve),
            'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
        }), 500
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Server error: {str(e)}',
            'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
        }), 500

def extract_location_from_query(user_input):
    """Extract location name from various query formats"""
    print(f"DEBUG: extract_location_from_query called with user_input: '{user_input}'")
    if not user_input:
        return ""
    
    user_input = user_input.strip()
    
    # Common weather query patterns
    patterns = [
        r"weather in (.+?)(?:\?|$)", 
        r"weather at (.+?)(?:\?|$)",   
        r"weather for (.+?)(?:\?|$)",  
        r"forecast in (.+?)(?:\?|$)",  
        r"forecast for (.+?)(?:\?|$)", 
        r"temperature in (.+?)(?:\?|$)", 
        r"what's the weather in (.+?)(?:\?|$)", 
        r"how is the weather in (.+?)(?:\?|$)",
        r"air quality in (.+?)(?:\?|$)",
        r"marine weather in (.+?)(?:\?|$)",
        r"historical weather in (.+?)(?:\?|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            # Remove trailing question words
            location = re.sub(r'\s+(please|\?|\.|\,)$', '', location, flags=re.IGNORECASE)
            print(f"DEBUG: extracted location via pattern: '{location}'")
            return location
    
    # If no pattern matches, check if it's just a location name
    words = user_input.split()
    if len(words) <= 4 and not any(word.lower() in ['what', 'how', 'when', 'where', 'why', 'tell', 'show', 'give', 'is', 'are', 'the', 'a', 'an'] for word in words):
        location = user_input
        print(f"DEBUG: extracted location as simple name: '{location}'")
        return location
    
    # Default: return the input as is
    location = user_input
    print(f"DEBUG: extracted location (default): '{location}'")
    return location
@app.route('/api/visual-weather', methods=['POST'])
def get_visual_weather():
    """API endpoint to get visual weather data"""
    try:
        data = request.get_json()
        if not data or 'location' not in data:
            return jsonify({
                'error': 'No location provided',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        location = data['location']
        
        # Check if the location is too short
        if len(location.strip()) < 2:
            return jsonify({
                'error': 'Location name too short. Please provide a more specific location name.',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        visual_data = agent.get_visual_weather_data(location)
        print(f"Visual weather data: {visual_data}")
        
        # Check if location was not found
        if visual_data.get('status') == 'error' or 'error' in str(visual_data.get('error', '')).lower():
            return jsonify({
                'error': visual_data.get('error', 'Failed to fetch weather data'),
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 500
        
        if 'location' not in visual_data:
            visual_data['location'] = location

        return jsonify(visual_data)
    except Exception as e:
        print(f"Error processing visual weather request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
        }), 500

@app.route('/api/test-location', methods=['POST'])
def test_location():
    """Test endpoint to check if a location can be found"""
    try:
        data = request.get_json()
        if not data or 'location' not in data:
            return jsonify({
                'error': 'No location provided',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        location = data['location']
        print(f"Testing location: {location}")
        
        # Try to get coordinates for the location
        from weather_tool import get_coordinates
        geo_data = get_coordinates.run(location)
        
        if not geo_data:
            return jsonify({
                'found': False,
                'message': f"Location '{location}' not found. Please check the spelling.",
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            })
        
        return jsonify({
            'found': True,
            'location': geo_data[0],
            'message': f"Location '{location}' found successfully"
        })
    except Exception as e:
        print(f"Error testing location: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
        }), 500

@app.route('/api/air-quality-visual', methods=['POST'])
def get_air_quality_visual():
    """API endpoint to get visual air quality data"""
    try:
        data = request.get_json()
        if not data or 'location' not in data:
            return jsonify({
                'error': 'No location provided',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        location = data['location']
        
        # Check if the location is too short
        if len(location.strip()) < 2:
            return jsonify({
                'error': 'Location name too short. Please provide a more specific location name.',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        # Get structured air quality data
        from weather_tool import get_air_quality_data
        air_quality_data = get_air_quality_data.run(location)
        print(f"Air quality data: {air_quality_data}")
        
        # Check if location was not found
        if air_quality_data.get('status') == 'error' or 'error' in str(air_quality_data.get('error', '')).lower():
            return jsonify({
                'error': air_quality_data.get('error', 'Failed to fetch air quality data'),
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 500
        
        return jsonify(air_quality_data)
    except Exception as e:
        print(f"Error processing air quality request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
        }), 500

@app.route('/api/marine', methods=['POST'])
def get_marine_weather():
    """API endpoint to get marine weather data"""
    try:
        data = request.get_json()
        if not data or 'location' not in data:
            return jsonify({
                'error': 'No location provided',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        location = data['location']
        
        # Check if the location is too short
        if len(location.strip()) < 2:
            return jsonify({
                'error': 'Location name too short. Please provide a more specific location name.',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        # Get structured marine weather data
        from weather_tool import get_marine_weather_data
        marine_data = get_marine_weather_data.run({
            "location": location
        })
        print(f"Marine weather data: {marine_data}")
        
        # Check if location was not found
        if marine_data.get('status') == 'error' or 'error' in str(marine_data.get('error', '')).lower():
            return jsonify({
                'error': marine_data.get('error', 'Failed to fetch marine weather data'),
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 500
        
        return jsonify(marine_data)
    except Exception as e:
        print(f"Error processing marine weather request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
        }), 500
    
@app.route('/api/historical-weather', methods=['POST'])
def get_historical_weather():
    """API endpoint to get historical weather data"""
    try:
        data = request.get_json()
        if not data or 'location' not in data:
            return jsonify({
                'error': 'No location provided',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        location = data['location']
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Check if the location is too short
        if len(location.strip()) < 2:
            return jsonify({
                'error': 'Location name too short. Please provide a more specific location name.',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        # Set default dates if not provided
        if not start_date or not end_date:
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Get structured historical weather data directly
        from weather_tool import get_historical_weather_data
        historical_data = get_historical_weather_data.run({
            "location": location,
            "start_date": start_date,
            "end_date": end_date
        })
        print(f"Historical weather data: {historical_data}")
        
        # Check if location was not found
        if historical_data.get('status') == 'error' or 'error' in str(historical_data.get('error', '')).lower():
            return jsonify({
                'error': historical_data.get('error', 'Failed to fetch historical weather data'),
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 500
        
        return jsonify(historical_data)
    except Exception as e:
        print(f"Error processing historical weather request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
        }), 500
# Update the /api/chat endpoint in app.py
@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for conversational weather chatbot"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'No message provided',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 400
        
        user_message = data.get('message')
        conversation_id = data.get('conversation_id', str(uuid.uuid4()))
        
        # Get response from the agent
        try:
            response, follow_up, visual_data, conversation_history, conversation_id = agent.run_conversation(
                user_message, 
                conversation_id
            )
            
            return jsonify({
                'response': response,
                'follow_up': follow_up,
                'visual_data': visual_data,
                'conversation_id': conversation_id,
                'conversation_history': conversation_history,
                'status': 'success'
            })
            
        except Exception as agent_error:
            print(f"Agent error: {str(agent_error)}")
            return jsonify({
                'error': f'Error processing your request: {str(agent_error)}',
                'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
            }), 500
            
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'suggestions': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Global Weather Assistant',
        'version': '1.0'
    })

if __name__ == '__main__':
    from database import Base, engine
    Base.metadata.create_all(engine)
    app.run(debug=True, host='0.0.0.0', port=5000)