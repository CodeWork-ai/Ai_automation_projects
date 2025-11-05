# workflow.py
import os
import re
import json
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from weather_tool import get_visual_weather_data, get_air_quality_data
from llm_manager import WeatherLLM

# --- AGENT STATE (MODIFIED) ---
class AgentState(TypedDict):
    user_input: str
    response: str
    follow_up: str # For storing generated follow-up questions
    intent: str
    locations: List[str]
    last_known_location: Optional[str]
    weather_data: Dict[str, Any]
    forecast_days: Optional[int] # ADDED: To store the requested number of forecast days

# --- HELPER FUNCTION ---
def get_aqi_interpretation(pm25_value: Optional[float]) -> tuple[str, str]:
    if pm25_value is None: return "Unknown", "PM2.5 data not available."
    if pm25_value <= 12.0: return "Good", "Air quality poses little or no risk."
    if pm25_value <= 35.4: return "Moderate", "Some may experience moderate health concern."
    return "Unhealthy", "Health risks for sensitive groups."

class WeatherAgent:
    def __init__(self):
        env_path = Path('.') / '.env'
        if env_path.exists(): load_dotenv(dotenv_path=env_path)
        try:
            self.llm = WeatherLLM()
            self.graph = self._build_graph()
            print("Intelligent Conversational WeatherAgent Initialized")
        except Exception as e:
            raise

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("determine_intent_and_location", self.determine_intent_and_location_node)
        workflow.add_node("fetch_data", self.fetch_data_node)
        workflow.add_node("reason_and_respond", self.reason_and_respond_node)
        workflow.set_entry_point("determine_intent_and_location")
        workflow.add_conditional_edges(
            "determine_intent_and_location",
            self.route_after_intent,
            {"fetch": "fetch_data", "respond_directly": "reason_and_respond"}
        )
        workflow.add_edge("fetch_data", "reason_and_respond")
        workflow.add_edge("reason_and_respond", END)
        return workflow.compile()

    def _extract_json_from_response(self, text: str) -> Optional[dict]:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except json.JSONDecodeError: return None
        return None

    # =========================================================================
    #  CORE AGENT NODES
    # =========================================================================
    def _generate_follow_up_questions(self, intent: str, location_name: str) -> str:
        """Generates context-aware follow-up questions."""
        if not location_name or intent in ["GREETING", "GENERAL_KNOWLEDGE"]:
            return "Ask me about another city?|What is the air quality?|What's the 5-day forecast?"

        prompt = f"""
        Based on the user's last action ('{intent}' for '{location_name}'), generate exactly three diverse and relevant follow-up questions.
        Output MUST be a single string with each question separated by a pipe character '|'. Add no other text.

        Example for 'GET_FULL_REPORT':
        What is the air quality like?|What's the marine forecast?|When is sunrise and sunset?

        Example for 'GET_AIR_QUALITY':
        What is the full weather forecast?|Is it safe for a walk?|How does this compare to yesterday?

        Now, generate questions for intent '{intent}'.
        """
        return self.llm(prompt, max_tokens=150, temperature=0.6).strip()
        
    def determine_intent_and_location_node(self, state: AgentState) -> AgentState:
        user_input = state['user_input']
        last_location = state.get('last_known_location')
        prompt = f"""
        Analyze the user's query to determine their intent, extract locations, and the number of forecast days requested.
        Your entire response MUST be a single, valid JSON object.

        ## Intents (Priority Order)
        1.  GET_ACTIVITY_SUGGESTION: User asks for advice on an activity (hiking, picnic, etc.).
        2.  GET_AIR_QUALITY: User asks for "air quality" or "AQI".
        3.  GET_BRIEF_DETAIL: User asks for a single, specific weather fact (e.g., "when will it rain?", "what is the sunrise time?").
        4.  GET_FULL_REPORT: User asks a general question ("how is the weather?", "10 day forecast"). This is the most common intent for forecasts.
        5.  GET_TRAVEL_ADVICE: User asks for the best month/time to visit.
        6.  GENERAL_KNOWLEDGE: User asks "why" or "how" about weather science.
        7.  GREETING: User is just saying hello.

        ## Rules
        - "locations" is a list of strings.
        - If no new location is mentioned, use the last known location: "{last_location}".
        - "forecast_days" should be an integer between 1 and 14.
        - If the user does not specify a number of days for a forecast, default "forecast_days" to 7.
        - If the user asks for more than 14 days, set "forecast_days" to 14.

        User Query: "{user_input}"
        """
        response_str = self.llm(prompt, max_tokens=200)
        response_json = self._extract_json_from_response(response_str)
        
        intent, locations, forecast_days = "GET_FULL_REPORT", [user_input], 7
        if response_json:
            intent = response_json.get("intent", intent)
            locations = response_json.get("locations", locations)
            # Get forecast_days, ensure it's an int, and clamp it between 1 and 14
            try:
                days = int(response_json.get("forecast_days", 7))
                forecast_days = max(1, min(14, days))
            except (ValueError, TypeError):
                forecast_days = 7

        print(f"Primary Intent: {intent}, Locations: {locations}, Forecast Days: {forecast_days}")
        return {**state, "intent": intent, "locations": locations, "forecast_days": forecast_days}

    def route_after_intent(self, state: AgentState) -> str:
        intent = state['intent']
        if intent in ["GET_TRAVEL_ADVICE", "GENERAL_KNOWLEDGE", "GREETING"] or not state['locations']:
            return "respond_directly"
        return "fetch"

    def fetch_data_node(self, state: AgentState) -> AgentState:
        intent, locations = state['intent'], state['locations']
        forecast_days = state.get('forecast_days', 7)  # Get days from state
        weather_data_map = {}
        new_last_location = state.get('last_known_location')
        for loc in locations:
            if not isinstance(loc, str) or not loc.strip(): continue
            if loc.lower() == "kerala": loc = "Kerala, India"

            # Decide which tool to run
            if intent == "GET_AIR_QUALITY":
                data = get_air_quality_data.run(loc)
            else:
                # For most intents, fetch the comprehensive visual data
                tool_args = {"location": loc, "days": forecast_days}
                data = get_visual_weather_data.run(tool_args)
            
            if data and data.get("status") != "error":
                proper_name = data.get('location', {}).get('name', loc)
                weather_data_map[proper_name] = data
                new_last_location = proper_name
            else:
                weather_data_map[loc] = {"error": f"Could not retrieve data for {loc}."}
        return {**state, "weather_data": weather_data_map, "last_known_location": new_last_location}

    def reason_and_respond_node(self, state: AgentState) -> AgentState:
        intent = state['intent']
        user_input = state['user_input']
        weather_data = state.get('weather_data', {})
        final_response = "I'm not quite sure how to handle that. Can you rephrase?"

        if intent == "GREETING":
            final_response = "Hello! How can I help you with the weather today?"
        elif intent == "GET_TRAVEL_ADVICE":
            final_response = self.generate_travel_advice_response(user_input)
        elif intent == "GENERAL_KNOWLEDGE":
            final_response = self.generate_knowledge_response(user_input)
        elif not state['locations']:
            final_response = "I can help with that! Please tell me the location you're interested in."
        else:
            location_name = list(weather_data.keys())[0]
            location_data = weather_data.get(location_name, {})
            if "error" in location_data:
                final_response = f"I'm sorry, but I couldn't find any data for '{location_name}'. Please check the spelling."
            elif intent == "GET_AIR_QUALITY":
                final_response = self.generate_air_quality_response(location_data)
            elif intent == "GET_BRIEF_DETAIL":
                final_response = self.generate_brief_response(user_input, location_data)
            elif intent == "GET_ACTIVITY_SUGGESTION":
                final_response = self.generate_activity_suggestion_response(user_input, location_data)
            else:
                # Pass the number of requested days to the response generator
                days = state.get('forecast_days', 7)
                final_response = self.generate_full_report_response(location_data, days)

        # Generate follow-ups separately after determining the main response
        last_known_loc = state.get('last_known_location', '')
        follow_up_text = self._generate_follow_up_questions(intent, last_known_loc)
        
        return {**state, "response": final_response, "follow_up": follow_up_text}

    # =========================================================================
    #  SPECIALIZED RESPONSE GENERATORS (Cleaned Prompts)
    # =========================================================================

    def generate_brief_response(self, user_input, data):
        prompt = f"""
        You are a weather analyst. Provide a direct, two-part answer to the user's specific question using the provided weather data.

        **User's Question:** "{user_input}"
        **Weather Data:** {json.dumps(data)}

        **Your Task:**
        1.  **Direct Answer First:** Start with a single, clear sentence that immediately answers the question.
        2.  **Supporting Details:** Briefly provide key data points that support your answer.
        """
        return self.llm(prompt)

    def generate_air_quality_response(self, data: dict) -> str:
        location_name = data.get('location', {}).get('name', 'the area')
        latest_reading = data.get('hourly', [])[-1] if data.get('hourly') else {}
        pm25 = latest_reading.get('pm2_5')
        level, description = get_aqi_interpretation(pm25)
        return f"""### 🌬️ Air Quality in {location_name}\nThe current air quality is **{level}** ({description}).\n\n**PM2.5:** {pm25 or 'N/A'} µg/m³\n**PM10:** {latest_reading.get('pm10', 'N/A')} µg/m³"""

    def generate_travel_advice_response(self, user_input: str) -> str:
        return self.llm(f"""You are a travel advisor answering: "{user_input}". Use general climate knowledge for a concise recommendation on the best months to visit. Do NOT use current weather.""")

    def generate_full_report_response(self, data, days: int):
        # The forecast title is now dynamic based on the 'days' parameter
        return self.llm(f"""Create a weather report for **{data['location']['name']}** in Markdown using this data: {json.dumps(data)}. Format: ### 🌤️ Current Weather (Temp, Conditions) ### 📅 {days}-Day Forecast (Table) ### 🧭 Travel & Outdoor Tips (2 tips)""")

    def generate_activity_suggestion_response(self, user_input, data):
        return self.llm(f"User wants to know: '{user_input}'. Based on this weather data, give a clear recommendation and why. Data: {json.dumps(data)}")
        
    def generate_knowledge_response(self, user_input):
        return self.llm(f"User has a science question: '{user_input}'. Provide a clear, concise explanation.")

    # =========================================================================
    #  RUN AND CONVERSATION MANAGEMENT
    # =========================================================================
    def run(self, state: dict) -> dict:
        return self.graph.invoke(state)

    def run_conversation(self, user_input: str, conversation_id: str = None):
        current_state = { "user_input": user_input, "last_known_location": getattr(self, '_current_session_location', None), "follow_up": "" }
        final_state = self.run(current_state)
        self._current_session_location = final_state.get('last_known_location')
        visual_data = {}
        if final_state.get('last_known_location'):
            visual_data = final_state.get("weather_data", {}).get(final_state['last_known_location'], {})
            
        follow_up_text = final_state.get("follow_up", "Is there anything else I can help with?")
        return final_state["response"], follow_up_text, visual_data, [], conversation_id