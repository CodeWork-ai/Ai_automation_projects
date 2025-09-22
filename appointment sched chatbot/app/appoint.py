import os
import sys
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
# LangChain imports
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
# Hugging Face imports
from huggingface_hub import InferenceClient
# Try to import Twilio, but don't fail if it's not available
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️  Twilio module not found. SMS functionality will be disabled.")
    print("   To enable SMS, run: pip install twilio")

# ============ DATA MODELS ============
class Patient:
    def __init__(self, name, contact_info, availability=None):
        self.name = name
        self.contact_info = contact_info
        self.availability = availability if availability else []

class Clinic:
    def __init__(self, name, available_slots=None):
        self.name = name
        self.available_slots = available_slots if available_slots else []

class Appointment:
    def __init__(self, patient, clinic, time, confirmed=False):
        self.patient = patient
        self.clinic = clinic
        self.time = time
        self.confirmed = confirmed

# ============ SCHEDULER ============
class Scheduler:
    def __init__(self, clinic):
        self.clinic = clinic
        # Define day order for finding next available day
        self.day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    def get_next_day(self, day):
        """Get the next day in the week"""
        if day in self.day_order:
            idx = self.day_order.index(day)
            return self.day_order[(idx + 1) % 7]
        return day
    
    def find_available_slot(self, patient):
        """Find an available slot, trying the next day if needed"""
        for preferred_slot in patient.availability:
            current_slot = preferred_slot
            original_slot = preferred_slot
            attempts = 0
            
            # Try to find an available slot, checking up to 7 days (a full week)
            while attempts < 7:
                if current_slot in self.clinic.available_slots:
                    return current_slot
                
                # Extract day and time from the slot
                parts = current_slot.split()
                if len(parts) >= 2:
                    day = parts[0]
                    time_part = ' '.join(parts[1:])
                    next_day = self.get_next_day(day)
                    current_slot = f"{next_day} {time_part}"
                else:
                    # If slot format is unexpected, break the loop
                    break
                
                attempts += 1
            
            # If we've tried all days for this slot and found nothing, continue to next preferred slot
        
        return None
    
    def book_appointment(self, patient):
        slot = self.find_available_slot(patient)
        if slot:
            self.clinic.available_slots.remove(slot)
            appointment = Appointment(patient, self.clinic, slot, confirmed=True)
            return appointment
        return None
    
    def cancel_appointment(self, time):
        """Cancel appointment by adding the time back to available slots"""
        if time not in self.clinic.available_slots:
            self.clinic.available_slots.append(time)
            self.clinic.available_slots.sort()

# ============ NOTIFIER ============
class Notifier:
    def __init__(self):
        # Initialize Twilio client only if available
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.twilio_phone_numbers = self._load_twilio_phone_numbers()
        
        if TWILIO_AVAILABLE and self.twilio_account_sid and self.twilio_auth_token and self.twilio_phone_number:
            self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
            self.sms_enabled = True
            print("✅ Twilio SMS service is enabled")
            self._print_verified_numbers()
        else:
            self.sms_enabled = False
            if not TWILIO_AVAILABLE:
                print("⚠️  Twilio module not installed. SMS reminders will not be sent.")
                print("   Install with: pip install twilio")
            else:
                print("⚠️  Twilio credentials not found. SMS reminders will not be sent.")
                print("   Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER")
    
    def _load_twilio_phone_numbers(self):
        """Load multiple Twilio phone numbers from environment variable"""
        phone_numbers_str = os.getenv("TWILIO_PHONE_NUMBERS", "")
        if phone_numbers_str:
            # Split by commas and strip whitespace
            phone_numbers = [num.strip() for num in phone_numbers_str.split(',') if num.strip()]
            return phone_numbers
        return []
    
    def _print_verified_numbers(self):
        """Print all verified Twilio phone numbers"""
        if self.twilio_phone_numbers:
            print("\n📱 Verified Twilio Phone Numbers:")
            for i, number in enumerate(self.twilio_phone_numbers, 1):
                print(f"   {i}. {number}")
            print()
    
    def send_confirmation(self, appointment, original_slot=None):
        print(f"\n✅ APPOINTMENT CONFIRMED")
        print(f"Patient: {appointment.patient.name}")
        print(f"Date/Time: {appointment.time}")
        if original_slot and original_slot != appointment.time:
            print(f"(Originally requested: {original_slot})")
        print(f"Clinic: {appointment.clinic.name}")
        print(f"Confirmation sent to: {appointment.patient.contact_info}")
        
        # Send SMS confirmation if enabled
        if self.sms_enabled:
            phone_numbers = self._extract_phone_numbers(appointment.patient.contact_info)
            if phone_numbers:
                message = f"Appointment confirmed for {appointment.time} at {appointment.clinic.name}."
                for phone_number in phone_numbers:
                    self._send_sms(
                        phone_number,
                        message,
                        "Confirmation"
                    )
    
    def send_reminder(self, appointment):
        print(f"\n📅 REMINDER: You have an appointment at {appointment.time}")
        print(f"Location: {appointment.clinic.name}")
        print(f"Patient: {appointment.patient.name}")
        
        # Send SMS reminder if enabled
        if self.sms_enabled:
            phone_numbers = self._extract_phone_numbers(appointment.patient.contact_info)
            if phone_numbers:
                message = f"Reminder: You have an appointment at {appointment.time} at {appointment.clinic.name}."
                for phone_number in phone_numbers:
                    self._send_sms(
                        phone_number,
                        message,
                        "Reminder"
                    )
    
    def send_cancellation_confirmation(self, name, time, contact=None):
        print(f"\n✅ APPOINTMENT CANCELLED")
        print(f"Patient: {name}")
        print(f"Cancelled appointment: {time}")
        print("Cancellation confirmation sent to your contact information.")
        
        # Send SMS cancellation confirmation if enabled and contact is provided
        if self.sms_enabled and contact:
            phone_numbers = self._extract_phone_numbers(contact)
            if phone_numbers:
                message = f"Your appointment on {time} has been cancelled."
                for phone_number in phone_numbers:
                    self._send_sms(
                        phone_number,
                        message,
                        "Cancellation"
                    )
    
    def send_reschedule_confirmation(self, appointment, original_slot=None):
        print(f"\n✅ APPOINTMENT RESCHEDULED")
        print(f"Patient: {appointment.patient.name}")
        print(f"New Date/Time: {appointment.time}")
        if original_slot and original_slot != appointment.time:
            print(f"(Originally scheduled for: {original_slot})")
        print(f"Clinic: {appointment.clinic.name}")
        print(f"Reschedule confirmation sent to: {appointment.patient.contact_info}")
        
        # Send SMS reschedule confirmation if enabled
        if self.sms_enabled:
            phone_numbers = self._extract_phone_numbers(appointment.patient.contact_info)
            if phone_numbers:
                message = f"Your appointment has been rescheduled to {appointment.time} at {appointment.clinic.name}."
                if original_slot and original_slot != appointment.time:
                    message += f" (Original time: {original_slot})"
                for phone_number in phone_numbers:
                    self._send_sms(
                        phone_number,
                        message,
                        "Reschedule"
                    )
    
    def _extract_phone_numbers(self, contact_info):
        """Extract and validate phone numbers from a string that may contain multiple numbers separated by commas."""
        if not contact_info:
            return []
        
        # Split by commas and process each part
        numbers = [num.strip() for num in contact_info.split(',')]
        valid_numbers = []
        
        for num in numbers:
            # Format the number to E.164
            formatted_num = self._format_phone_number(num)
            # Validate the formatted number
            if self._validate_phone_number(formatted_num):
                valid_numbers.append(formatted_num)
        
        return valid_numbers
    
    def _send_sms(self, to_number, message, message_type):
        """Send SMS using Twilio and display details"""
        try:
            # Validate the formatted number
            if not self._validate_phone_number(to_number):
                print(f"❌ Error: Invalid phone number format: {to_number}")
                print("   Please provide your phone number in international format (e.g., +1 for US, +91 for India).")
                print("   For example: +919551841398 for Indian numbers")
                return
            
            # Display SMS details before sending
            print("\n" + "=" * 50)
            print(f"📱 SENDING {message_type.upper()} SMS")
            print("=" * 50)
            print(f"📞 To: {to_number}")
            print(f"📝 Message: {message}")
            
            # If multiple Twilio numbers are available, select one
            from_number = self._select_twilio_number()
            print(f"📤 From: {from_number}")
            print("-" * 50)
            
            # Send the SMS
            message_obj = self.twilio_client.messages.create(
                body=message,
                from_=from_number,
                to=to_number
            )
            
            # Display SMS status after sending
            print(f"✅ SMS sent successfully!")
            print(f"🆔 Message SID: {message_obj.sid}")
            print(f"📊 Status: {message_obj.status}")
            print(f"📅 Date sent: {message_obj.date_created}")
            print("=" * 50)
            
            # Provide a link to view the message in Twilio console
            account_sid = self.twilio_account_sid
            print(f"🔗 View message: https://console.twilio.com/us1/monitor/logs/{message_obj.sid}")
            
        except TwilioRestException as e:
            print(f"❌ Twilio Error: {e}")
            if e.code == 21610:  # Unverified number error
                print("   This number is not verified in your Twilio account.")
                print("   For trial accounts, you must verify recipient numbers.")
                print("   Visit: https://www.twilio.com/console/phone-numbers/verified")
            elif e.code == 21211:  # Invalid 'To' phone number
                print("   The recipient phone number is invalid.")
                print("   Please check the number and try again.")
            else:
                print("   Please check your Twilio account settings.")
            print("=" * 50)
        except Exception as e:
            print(f"❌ Error sending SMS: {e}")
            print("   Please check your phone number format and Twilio account settings.")
            print("=" * 50)
    
    def _select_twilio_number(self):
        """Select a Twilio phone number to send from"""
        if not self.twilio_phone_numbers:
            return self.twilio_phone_number
        
        # For simplicity, we'll just use the first number
        # In a real application, you might want to implement rotation or selection logic
        return self.twilio_phone_numbers[0]
    
    def _validate_phone_number(self, phone_number):
        """Validate phone number format"""
        # Remove all non-digit characters and the + sign
        digits = re.sub(r'[^\d]', '', phone_number)
        
        # Check if it starts with a +
        if not phone_number.startswith('+'):
            return False
        
        # Check if it has enough digits (minimum 8 digits after country code)
        if len(digits) < 8:
            return False
        
        # Check for common country codes
        if phone_number.startswith('+1'):
            # US numbers should have 11 digits total (1 + 10)
            return len(digits) == 11
        elif phone_number.startswith('+91'):
            # Indian numbers should have 12 digits total (91 + 10)
            return len(digits) == 12
        elif phone_number.startswith('+44'):
            # UK numbers should have 12 digits total (44 + 10)
            return len(digits) == 12
        else:
            # For other countries, we'll be more lenient
            return len(digits) >= 8
    
    def _format_phone_number(self, phone_number):
        """Format phone number to E.164 standard"""
        # Remove all non-digit characters
        digits = re.sub(r'[^\d]', '', phone_number)
        
        # If the number starts with a +, we assume it's already in international format
        if phone_number.startswith('+'):
            return '+' + digits
        
        # Check for common country code patterns
        if digits.startswith('00'):
            # International dialing prefix, replace with +
            return '+' + digits[2:]
        elif digits.startswith('0'):
            # Local number, add default country code (US)
            return '+1' + digits[1:]
        elif len(digits) == 10:
            # 10-digit number - check if it's likely a US number
            # US numbers typically start with area codes between 200-999
            if 200 <= int(digits[:3]) <= 999:
                return '+1' + digits
            else:
                # Not a valid US area code, check if it could be an Indian number
                # Indian mobile numbers typically start with 7, 8, or 9
                if digits[0] in ['7', '8', '9']:
                    return '+91' + digits
                else:
                    # Default to US format with a warning
                    print(f"⚠️  Warning: Ambiguous phone number format for {digits}. Defaulting to US format (+1).")
                    return '+1' + digits
        elif len(digits) == 11 and digits.startswith('1'):
            # 11-digit number starting with 1, assume US
            return '+' + digits
        elif len(digits) == 12 and digits.startswith('91'):
            # 12-digit number starting with 91, assume India
            return '+' + digits
        elif len(digits) == 10 and digits[0] in ['7', '8', '9']:
            # 10-digit number starting with 7,8,9 - likely Indian mobile
            return '+91' + digits
        elif len(digits) == 9 and digits[0] in ['7', '8', '9']:
            # 9-digit number starting with 7,8,9 - likely Indian mobile (missing a digit)
            # We'll add the Indian country code but warn about the missing digit
            print(f"⚠️  Warning: Phone number {digits} appears to be missing a digit. Adding Indian country code.")
            return '+91' + digits
        else:
            # Default case: just add + and hope it's correct
            print(f"⚠️  Warning: Unrecognized phone number format for {digits}. Adding + prefix.")
            return '+' + digits

# ============ UTILITIES ============
def parse_any_time_format(time_str):
    if not time_str:
        return None
    
    time_str = time_str.strip().lower()
    
    day_map = {
        'mon': 'Mon', 'monday': 'Mon',
        'tue': 'Tue', 'tuesday': 'Tue',
        'wed': 'Wed', 'wednesday': 'Wed',
        'thu': 'Thu', 'thursday': 'Thu',
        'fri': 'Fri', 'friday': 'Fri',
        'sat': 'Sat', 'saturday': 'Sat',
        'sun': 'Sun', 'sunday': 'Sun',
        'today': 'Today',
        'tomorrow': 'Tomorrow'
    }
    
    day = None
    for day_name, day_abbr in day_map.items():
        if day_name in time_str:
            day = day_abbr
            time_str = time_str.replace(day_name, '', 1).strip()
            break
    
    if not day:
        day = "Today"
    
    time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?'
    time_match = re.search(time_pattern, time_str)
    
    if not time_match:
        return None
    
    hour = int(time_match.group(1))
    minute = int(time_match.group(2)) if time_match.group(2) else 0
    period = time_match.group(3).upper() if time_match.group(3) else None
    
    if not period:
        period = "PM"
    
    if period == "PM" and hour != 12:
        hour += 12
    elif period == "AM" and hour == 12:
        hour = 0
    
    minute_str = f"{minute:02d}"
    display_hour = hour % 12
    if display_hour == 0:
        display_hour = 12
    
    return f"{day} {display_hour}:{minute_str} {period}"

def format_available_slots(slots):
    if not slots:
        return "No available slots"
    
    formatted = []
    for i, slot in enumerate(slots, 1):
        formatted.append(f"{i}. {slot}")
    return "\n".join(formatted)

def format_appointments(appointments):
    if not appointments:
        return "No appointments found."
    
    # Remove duplicates by converting to a set of tuples and back to list
    unique_appointments = []
    seen = set()
    
    for appt in appointments:
        # Create a unique key based on patient name and time
        key = (appt['patient_name'], appt['time'])
        if key not in seen:
            seen.add(key)
            unique_appointments.append(appt)
    
    formatted = []
    for i, appt in enumerate(unique_appointments, 1):
        formatted.append(f"{i}. {appt['time']}")
    return "\n".join(formatted)

def normalize_contact(contact):
    """Normalize contact information for comparison"""
    if not contact:
        return ""
    # Remove all whitespace and convert to lowercase for comparison
    return re.sub(r'\s+', '', contact).lower()

# ============ STORAGE ============
class Storage:
    def __init__(self, file_path='appointments.json'):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)
    
    def save_appointment(self, appointment):
        appointments = self.load_appointments()
        
        appointment_data = {
            'patient_name': appointment.patient.name,
            'patient_contact': appointment.patient.contact_info,
            'clinic_name': appointment.clinic.name,
            'time': appointment.time,
            'confirmed': appointment.confirmed
        }
        
        # Check if this appointment already exists (same patient and time)
        for appt in appointments:
            if (appt['patient_name'] == appointment_data['patient_name'] and 
                appt['time'] == appointment_data['time']):
                # Appointment already exists, don't save again
                return
        
        appointments.append(appointment_data)
        with open(self.file_path, 'w') as f:
            json.dump(appointments, f, indent=2)
        
        # Debug: Print what was saved
        print(f"DEBUG: Saved appointment for {appointment_data['patient_name']} with contact {appointment_data['patient_contact']}")
    
    def load_appointments(self):
        try:
            with open(self.file_path, 'r') as f:
                appointments = json.load(f)
                # Remove duplicates by converting to a set of tuples and back to list
                unique_appointments = []
                seen = set()
                
                for appt in appointments:
                    # Create a unique key based on patient name and time
                    key = (appt['patient_name'], appt['time'])
                    if key not in seen:
                        seen.add(key)
                        unique_appointments.append(appt)
                
                # Debug: Print what was loaded
                print(f"DEBUG: Loaded {len(unique_appointments)} appointments")
                for appt in unique_appointments:
                    print(f"DEBUG: - {appt['patient_name']} at {appt['time']} with contact {appt['patient_contact']}")
                
                return unique_appointments
        except (json.JSONDecodeError, FileNotFoundError):
            print("DEBUG: No appointments file found or file is empty")
            return []
    
    def cancel_appointment(self, name, contact, time):
        appointments = self.load_appointments()
        updated_appointments = []
        cancelled = None
        
        # Normalize contact for comparison
        normalized_contact = normalize_contact(contact)
        
        print(f"DEBUG: Looking for appointment with name='{name}', contact='{contact}' (normalized: '{normalized_contact}'), time='{time}'")
        
        for appt in appointments:
            appt_normalized_contact = normalize_contact(appt['patient_contact'])
            print(f"DEBUG: Checking appointment: name='{appt['patient_name']}', contact='{appt['patient_contact']}' (normalized: '{appt_normalized_contact}'), time='{appt['time']}'")
            
            if (appt['patient_name'] == name and 
                appt_normalized_contact == normalized_contact and 
                appt['time'] == time):
                cancelled = appt  # Save the cancelled appointment details
                print(f"DEBUG: Found matching appointment to cancel")
            else:
                updated_appointments.append(appt)
        
        if cancelled:
            with open(self.file_path, 'w') as f:
                json.dump(updated_appointments, f, indent=2)
            print(f"DEBUG: Successfully cancelled appointment")
            return cancelled  # Return the cancelled appointment
        else:
            print(f"DEBUG: No matching appointment found to cancel")
            return None

# ============ CUSTOM LLM WRAPPER ============
class CustomHuggingFaceLLM(LLM):
    model_id: str
    api_token: str
    temperature: float = 0.7
    max_new_tokens: int = 1024
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the InferenceClient
        self._client = InferenceClient(model=self.model_id, token=self.api_token)
    
    @property
    def _llm_type(self) -> str:
        return "custom_huggingface"
    
    def _call(self, prompt, stop=None):
        try:
            response = self._client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                seed=42
            )
            # Clean up the response
            response = response.replace("[INST]", "").replace("[/INST]", "").strip()
            return response
        except Exception as e:
            # If the model fails, return a simple response
            return "I'm here to help with your appointment needs. Could you please tell me more about what you're looking for?"
    
    def invoke(self, prompt):
        """Invoke the LLM with the given prompt"""
        return self._call(prompt)

# ============ SEMANTIC CHATBOT ============
class SemanticAppointmentChatbot:
    def __init__(self):
        # Check if API token is set
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            raise ValueError("Hugging Face API token not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
        
        # Initialize clinic with available slots
        self.clinic = Clinic(
            name="General Practice",
            available_slots=[
                "Mon 9:00 AM", "Mon 11:00 AM", "Mon 2:00 PM",
                "Tue 10:00 AM", "Tue 1:00 PM", "Tue 3:30 PM",
                "Wed 9:30 AM", "Wed 11:30 AM", "Wed 4:00 PM",
                "Thu 10:30 AM", "Thu 2:30 PM", "Thu 4:30 PM",
                "Fri 9:00 AM", "Fri 12:00 PM", "Fri 3:00 PM"
            ]
        )
        self.scheduler = Scheduler(self.clinic)
        self.notifier = Notifier()
        self.storage = Storage()
        
        # Initialize custom LLM with a better model
        self.llm = CustomHuggingFaceLLM(
            model_id="microsoft/DialoGPT-medium",  # Better for conversations
            api_token=api_token,
            temperature=0.7,  # Higher temperature for more natural responses
            max_new_tokens=1024  # Allow longer responses
        )
        
        # Conversation memory
        self.memory = ConversationBufferMemory()
        
        # Current context
        self.current_context = {
            "intent": None,
            "name": None,
            "contact": None,
            "times": [],
            "appointment_time": None,
            "step": None,  # Track multi-step interactions
            "last_action": None,  # Track the last completed action
            "expecting_name": False,  # Flag to indicate we're expecting a name
            "expecting_contact": False,  # Flag to indicate we're expecting contact info
            "expecting_availability": False,  # Flag to indicate we're expecting availability
            "rescheduling": False,  # Flag to indicate we're in rescheduling mode
            "canceled_appointment_time": None  # Store the time of the canceled appointment
        }
    
    def extract_entities(self, text):
        entities = {
            'name': None,
            'contact': None,
            'times': []
        }
        
        # Extract name - improved patterns
        name_patterns = [
            r"(my name is|i am|i'm|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)$"  # Just a name
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['name'] = match.group(2) if match.lastindex == 2 else match.group(1)
                break
        
        # If we're expecting a name and didn't extract one, check if the input is a plausible name
        if self.current_context['expecting_name'] and not entities['name']:
            # Check if the input is a plausible name (starts with capital and consists of letters and spaces)
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', text.strip()):
                entities['name'] = text.strip()
        
        # Extract contact
        email_pattern = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
        phone_pattern = r"(\+?\d{10,})"
        
        email_match = re.search(email_pattern, text)
        if email_match:
            entities['contact'] = email_match.group(1)
        
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            entities['contact'] = phone_match.group(1)
        
        # Extract times
        time_patterns = [
            r"(today|tomorrow|mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+(today|tomorrow|mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"(today|tomorrow|mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s+(today|tomorrow|mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 4:
                    day, hour, minute, period = match
                    time_str = f"{day} at {hour}:{minute if minute else '00'} {period if period else 'pm'}"
                elif len(match) == 3:
                    if match[0].isdigit():
                        hour, minute, period, day = match[0], match[1], match[2], None
                    else:
                        day, hour, minute, period = match[0], match[1], match[2], None
                    time_str = f"{day} at {hour}:{minute if minute else '00'} {period if period else 'pm'}"
                else:
                    continue
                    
                parsed_time = parse_any_time_format(time_str)
                if parsed_time and parsed_time not in entities['times']:
                    entities['times'].append(parsed_time)
        
        return entities
    
    def determine_intent(self, text):
        text = text.lower().strip()
        
        # If we're in the middle of a multi-step process, preserve the current intent
        if (self.current_context['step'] is not None or 
            self.current_context['expecting_name'] or 
            self.current_context['expecting_contact'] or 
            self.current_context['expecting_availability']):
            return self.current_context['intent']
        
        # Check for conversational responses first
        if text in ['no', 'nope', 'nothing', "that's all", 'no thank you']:
            return "CONVERSATION_END"
        
        if text in ['yes', 'yeah', 'yep', 'sure', 'okay']:
            return "CONVERSATION_CONTINUE"
        
        # Check for rescheduling
        if any(word in text for word in ["reschedule", "rescheduling", "reschedule my appointment", "rescheduling my appointment", "change my appointment", "move my appointment"]):
            return "RESCHEDULE"
        
        # Check for view appointments - improved pattern matching
        if "view" in text and "appointment" in text:
            return "VIEW"
        
        # Check for check available slots
        if any(word in text for word in ["check", "see", "available", "slots", "when", "show"]):
            return "CHECK"
        
        # Check for cancellation
        if any(word in text for word in ["cancel", "cancel my appointment"]):
            return "CANCEL"
        
        # Check for booking
        if any(word in text for word in ["book", "schedule", "make an", "set up"]):
            return "BOOK"
        
        return "UNKNOWN"
    
    def update_context(self, text):
        entities = self.extract_entities(text)
        intent = self.determine_intent(text)
        
        # Update context with extracted information
        if entities['name']:
            self.current_context['name'] = entities['name']
            self.current_context['expecting_name'] = False
        
        if entities['contact']:
            self.current_context['contact'] = entities['contact']
            self.current_context['expecting_contact'] = False
        
        if entities['times']:
            for time in entities['times']:
                if time not in self.current_context['times']:
                    self.current_context['times'].append(time)
        
        # If a specific appointment time is mentioned for cancellation
        if intent == "CANCEL" and entities['times']:
            self.current_context['appointment_time'] = entities['times'][0]
        
        # Always update the intent if it's not UNKNOWN
        if intent != "UNKNOWN":
            self.current_context['intent'] = intent
    
    def generate_response(self, user_input):
        # Update context with user input
        self.update_context(user_input)
        
        # Handle rescheduling after cancellation
        if self.current_context['rescheduling']:
            if self.current_context['intent'] == "CONVERSATION_CONTINUE":
                # User wants to reschedule - directly transition to booking
                self.current_context['rescheduling'] = False
                self.current_context['intent'] = "BOOK"
                self.current_context['times'] = []  # Clear previous times
                # Since we already have name and contact, we can directly ask for availability
                return "Great! Let's reschedule your appointment. When are you available? Please provide your preferred day and time."
            elif self.current_context['intent'] == "CONVERSATION_END":
                # User doesn't want to reschedule
                self.reset_context()
                return "Thank you for using our service. Have a great day!"
            else:
                # Ask if they want to reschedule
                return "Would you like to reschedule your appointment? Please say 'yes' to book a new appointment or 'no' to end the conversation."
        
        # Generate response based on current context
        if self.current_context['intent'] == "BOOK":
            return self.handle_booking(user_input)
        elif self.current_context['intent'] == "CANCEL":
            return self.handle_cancellation(user_input)
        elif self.current_context['intent'] == "RESCHEDULE":
            # Rescheduling is similar to cancellation but we'll offer to book a new appointment after
            return self.handle_reschedule(user_input)
        elif self.current_context['intent'] == "CHECK":
            return self.handle_check_slots(user_input)
        elif self.current_context['intent'] == "VIEW":
            return self.handle_view_appointments(user_input)
        elif self.current_context['intent'] == "CONVERSATION_END":
            self.reset_context()
            return "Thank you for using our service. Have a great day!"
        elif self.current_context['intent'] == "CONVERSATION_CONTINUE":
            self.reset_context(partial=True)  # Keep user info but reset intent
            return "What would you like to do next? You can book, cancel, or check appointments."
        else:
            # If intent is unknown, use the LLM to generate a more conversational response
            prompt = f"""You are an appointment scheduling assistant for a medical clinic. 
            The user said: '{user_input}'
            
            Your task is to respond helpfully and conversationally. If the user is asking about appointments, booking, cancellation, or available slots, 
            try to guide them toward providing the necessary information.
            
            If the user is asking something unrelated, politely explain that you can only help with appointment scheduling.
            
            Keep your response friendly and concise.
            """
            return self.llm.invoke(prompt)
    
    def handle_booking(self, user_input):
        # Check if we have all required information
        if not self.current_context['name']:
            self.current_context['expecting_name'] = True
            return "I'd be happy to help you book an appointment. What's your name?"
        
        if not self.current_context['contact']:
            self.current_context['expecting_contact'] = True
            return f"Thanks, {self.current_context['name']}! What's your email or phone number? Please include the country code if providing a phone number (e.g., +1 for US, +91 for India). You can provide multiple phone numbers separated by commas."
        
        # Check if user is asking for available slots
        if self.current_context['intent'] == "CHECK" or any(word in user_input.lower() for word in ["show", "available", "slots"]):
            slots = format_available_slots(self.clinic.available_slots)
            self.current_context['expecting_availability'] = True
            return f"Here are our available appointment slots:\n{slots}\nPlease let me know which time you prefer."
        
        if not self.current_context['times']:
            self.current_context['expecting_availability'] = True
            return "Thank you! When are you available for an appointment? Please provide your preferred day and time."
        
        # If we have all information, try to book the appointment
        patient = Patient(
            self.current_context['name'],
            self.current_context['contact'],
            self.current_context['times']
        )
        
        appointment = self.scheduler.book_appointment(patient)
        
        if appointment:
            self.storage.save_appointment(appointment)
            
            # Check if this is a rescheduling
            is_reschedule = self.current_context.get('rescheduling', False)
            original_slot = None
            
            if patient.availability and appointment.time not in patient.availability:
                original_slot = patient.availability[0]
            
            # Send confirmation with appropriate message
            if is_reschedule:
                self.notifier.send_reschedule_confirmation(appointment, original_slot)
            else:
                self.notifier.send_confirmation(appointment, original_slot)
            
            # Reset context for next conversation but keep user info
            self.reset_context(partial=True)
            self.current_context['last_action'] = "BOOK"
            
            response = f"Great! I've booked your appointment for {appointment.time}."
            if original_slot:
                response += f" (The doctor wasn't available on {original_slot}, so I scheduled you for the next available day.)"
            response += f" A confirmation has been sent to {patient.contact_info}. Is there anything else I can help you with?"
            
            return response
        else:
            # Doctor is not available on any of the preferred days
            preferred_days = ", ".join([time.split()[0] for time in patient.availability])
            return f"I'm sorry, but the doctor is not available on any of your preferred days ({preferred_days}). Here are our available slots:\n{format_available_slots(self.clinic.available_slots)}"
    
    def handle_reschedule(self, user_input):
        # Rescheduling is essentially canceling and then booking a new appointment
        # We'll first cancel the existing appointment and then offer to book a new one
        return self.handle_cancellation(user_input, for_reschedule=True)
    
    def handle_cancellation(self, user_input, for_reschedule=False):
        # Store name and contact in local variables to prevent context changes
        name = self.current_context['name']
        contact = self.current_context['contact']
        
        # Check if we have all required information
        if not name:
            self.current_context['expecting_name'] = True
            return "I can help you cancel an appointment. What's your name?"
        
        if not contact:
            self.current_context['expecting_contact'] = True
            return f"Thanks, {name}! What's your email or phone number? Please include the country code if providing a phone number (e.g., +1 for US, +91 for India). You can provide multiple phone numbers separated by commas."
        
        # Get user's appointments
        appointments = self.storage.load_appointments()
        user_appointments = [appt for appt in appointments 
                           if appt['patient_name'] == name and normalize_contact(appt['patient_contact']) == normalize_contact(contact)]
        
        print(f"DEBUG: Found {len(user_appointments)} appointments for {name} with contact {contact}")
        
        if not user_appointments:
            self.reset_context()
            return f"I couldn't find any appointments for {name} with contact {contact}. Please check the name and contact information you provided."
        
        # If there's only one appointment, cancel it directly
        if len(user_appointments) == 1:
            appointment = user_appointments[0]
            cancelled_appointment = self.storage.cancel_appointment(
                name, 
                contact,
                appointment['time']  # Pass the time parameter
            )
            
            if cancelled_appointment:
                self.scheduler.cancel_appointment(cancelled_appointment['time'])
                # Pass contact information for SMS
                self.notifier.send_cancellation_confirmation(
                    name, 
                    cancelled_appointment['time'],
                    cancelled_appointment['patient_contact']
                )
                
                # Set context for rescheduling
                self.current_context['rescheduling'] = True
                self.current_context['canceled_appointment_time'] = cancelled_appointment['time']
                return "Your appointment has been cancelled. Would you like to reschedule for a different time?"
            else:
                return "There was an error canceling your appointment. Please try again."
        
        # If we're waiting for appointment selection
        if self.current_context['step'] == 'awaiting_appointment_selection':
            return self.process_appointment_selection(user_input, user_appointments, for_reschedule)
        
        # If a specific appointment time is mentioned
        if self.current_context['appointment_time']:
            for appt in user_appointments:
                if appt['time'].lower() == self.current_context['appointment_time'].lower():
                    cancelled_appointment = self.storage.cancel_appointment(
                        name, 
                        contact,
                        appt['time']  # Pass the time parameter
                    )
                    
                    if cancelled_appointment:
                        self.scheduler.cancel_appointment(cancelled_appointment['time'])
                        # Pass contact information for SMS
                        self.notifier.send_cancellation_confirmation(
                            name, 
                            cancelled_appointment['time'],
                            cancelled_appointment['patient_contact']
                        )
                        
                        # Set context for rescheduling
                        self.current_context['rescheduling'] = True
                        self.current_context['canceled_appointment_time'] = cancelled_appointment['time']
                        return "Your appointment has been cancelled. Would you like to reschedule for a different time?"
            
            # If we get here, the appointment time was not found
            self.reset_context()
            return f"I couldn't find an appointment at {self.current_context['appointment_time']}. Here are your appointments:\n{format_appointments(user_appointments)}\nPlease specify which one to cancel."
        
        # If no specific time mentioned and multiple appointments, show all appointments and wait for selection
        self.current_context['step'] = 'awaiting_appointment_selection'
        result = f"Here are your appointments:\n{format_appointments(user_appointments)}\n"
        result += "Which appointment would you like to cancel? Please specify the date and time or say 'the first one', 'the second one', etc."
        return result
    
    def process_appointment_selection(self, user_input, user_appointments, for_reschedule=False):
        # Store name and contact in local variables to prevent context changes
        name = self.current_context['name']
        contact = self.current_context['contact']
        user_input_lower = user_input.lower()
        
        # Check for appointment time in user input
        entities = self.extract_entities(user_input)
        if entities['times']:
            for appt in user_appointments:
                if appt['time'].lower() == entities['times'][0].lower():
                    cancelled_appointment = self.storage.cancel_appointment(
                        name, 
                        contact,
                        appt['time']  # Pass the time parameter
                    )
                    
                    if cancelled_appointment:
                        self.scheduler.cancel_appointment(cancelled_appointment['time'])
                        # Pass contact information for SMS
                        self.notifier.send_cancellation_confirmation(
                            name, 
                            cancelled_appointment['time'],
                            cancelled_appointment['patient_contact']
                        )
                        
                        # Set context for rescheduling
                        self.current_context['rescheduling'] = True
                        self.current_context['canceled_appointment_time'] = cancelled_appointment['time']
                        return "Your appointment has been cancelled. Would you like to reschedule for a different time?"
        
        # Check for ordinal references (first, second, etc.)
        if "first" in user_input_lower and len(user_appointments) >= 1:
            appt = user_appointments[0]
            return self.cancel_selected_appointment(appt, for_reschedule)
        elif "second" in user_input_lower and len(user_appointments) >= 2:
            appt = user_appointments[1]
            return self.cancel_selected_appointment(appt, for_reschedule)
        elif "third" in user_input_lower and len(user_appointments) >= 3:
            appt = user_appointments[2]
            return self.cancel_selected_appointment(appt, for_reschedule)
        elif "last" in user_input_lower and len(user_appointments) >= 1:
            appt = user_appointments[-1]
            return self.cancel_selected_appointment(appt, for_reschedule)
        
        # If no valid selection found
        return "I didn't understand which appointment you want to cancel. Please specify the date and time or say 'the first one', 'the second one', etc."
    
    def cancel_selected_appointment(self, appointment, for_reschedule=False):
        # Store name and contact in local variables to prevent context changes
        name = self.current_context['name']
        contact = self.current_context['contact']
        
        cancelled_appointment = self.storage.cancel_appointment(
            name, 
            contact,
            appointment['time']  # Pass the time parameter
        )
        
        if cancelled_appointment:
            self.scheduler.cancel_appointment(cancelled_appointment['time'])
            # Pass contact information for SMS
            self.notifier.send_cancellation_confirmation(
                name, 
                cancelled_appointment['time'],
                cancelled_appointment['patient_contact']
            )
            
            # Set context for rescheduling
            self.current_context['rescheduling'] = True
            self.current_context['canceled_appointment_time'] = cancelled_appointment['time']
            return "Your appointment has been cancelled. Would you like to reschedule for a different time?"
        else:
            return "There was an error canceling your appointment. Please try again."
    
    def handle_check_slots(self, user_input):
        slots = format_available_slots(self.clinic.available_slots)
        self.reset_context()
        return f"Here are our available appointment slots:\n{slots}\nWould you like to book one of these?"
    
    def handle_view_appointments(self, user_input):
        # Check if we have all required information
        if not self.current_context['name']:
            self.current_context['expecting_name'] = True
            return "I'd be happy to show you your appointments. What's your name?"
        
        if not self.current_context['contact']:
            self.current_context['expecting_contact'] = True
            return f"Thanks, {self.current_context['name']}! What's your email or phone number? Please include the country code if providing a phone number (e.g., +1 for US, +91 for India). You can provide multiple phone numbers separated by commas."
        
        appointments = self.storage.load_appointments()
        user_appointments = [appt for appt in appointments 
                           if appt['patient_name'] == self.current_context['name'] and normalize_contact(appt['patient_contact']) == normalize_contact(self.current_context['contact'])]
        
        if not user_appointments:
            self.reset_context()
            return f"You don't have any appointments scheduled."
        
        self.reset_context()
        return f"Here are your appointments:\n{format_appointments(user_appointments)}\nIs there anything else I can help you with?"
    
    def reset_context(self, partial=False):
        """Reset the conversation context, optionally preserving some information"""
        if partial:
            # Only reset the intent and step, keep user information
            self.current_context["intent"] = None
            self.current_context["step"] = None
            self.current_context["times"] = []
            self.current_context["appointment_time"] = None
            self.current_context["expecting_name"] = False
            self.current_context["expecting_contact"] = False
            self.current_context["expecting_availability"] = False
            # Keep rescheduling flag until explicitly reset
        else:
            # Reset everything
            self.current_context = {
                "intent": None,
                "name": None,
                "contact": None,
                "times": [],
                "appointment_time": None,
                "step": None,
                "last_action": None,
                "expecting_name": False,
                "expecting_contact": False,
                "expecting_availability": False,
                "rescheduling": False,
                "canceled_appointment_time": None
            }
    
    def start_conversation(self):
        """Start the chatbot conversation"""
        print("=" * 50)
        print("🤖 APPOINTMENT SCHEDULING CHATBOT")
        print("=" * 50)
        print("Hello! I'm your appointment scheduling assistant.")
        print("You can ask me to:")
        print("- Book an appointment")
        print("- Cancel an appointment")
        print("- Reschedule an appointment")
        print("- Check available slots")
        print("- View your appointments")
        print("- Type 'help' for assistance or 'exit' to quit")
        print("=" * 50)
        print("How can I help you today?")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("Thank you for using our service. Have a great day!")
                break
            
            # Handle help command
            if user_input.lower() in ['help', 'what can you do']:
                print("\nI can help you with the following tasks:")
                print("- Book an appointment: Tell me your name, contact info, and preferred time")
                print("- Cancel an appointment: Provide your name, contact info, and appointment time")
                print("- Reschedule an appointment: Cancel your current appointment and book a new one")
                print("- Check available slots: Just ask to see available appointment times")
                print("- View your appointments: Provide your name and contact info")
                print("- Type 'exit' to end the conversation")
                continue
            
            # Add user input to memory
            self.memory.chat_memory.add_user_message(user_input)
            
            # Generate response
            response = self.generate_response(user_input)
            
            # Add response to memory
            self.memory.chat_memory.add_ai_message(response)
            
            print(f"\nChatbot: {response}")

# ============ MAIN FUNCTION ============
def print_welcome_banner():
    print("=" * 60)
    print("🤖 APPOINTMENT SCHEDULING CHATBOT")
    print("=" * 60)
    print("Welcome to the AI-powered appointment scheduling system!")
    print("This chatbot uses LangChain and Hugging Face to help you")
    print("book, cancel, and manage your appointments.")
    print("=" * 60)
    print()

def print_instructions():
    print("\n📋 INSTRUCTIONS:")
    print("1. Make sure you have a valid Hugging Face API token")
    print("2. Set the token as an environment variable:")
    print("   - Windows: set HUGGINGFACEHUB_API_TOKEN=your_token_here")
    print("   - Linux/macOS: export HUGGINGFACEHUB_API_TOKEN=your_token_here")
    print("3. Or create a .env file with: HUGGINGFACEHUB_API_TOKEN=your_token_here")
    print("4. Get your token from: https://huggingface.co/settings/tokens")
    print("\n5. For SMS reminders (optional):")
    print("   - Install Twilio: pip install twilio")
    print("   - Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER")
    print("   - Get your credentials from: https://www.twilio.com/console")
    print("\n6. For multiple Twilio phone numbers (optional):")
    print("   - Set TWILIO_PHONE_NUMBERS with comma-separated numbers")
    print("   - Example: TWILIO_PHONE_NUMBERS=+15017122661,+15017122662")
    print("   - Verify all numbers in your Twilio account")
    print("\n7. For trial accounts, you must verify each recipient phone number in the Twilio console:")
    print("   - Visit: https://www.twilio.com/console/phone-numbers/verified")
    print("   - Add and verify each phone number you want to send SMS to")
    print("\n8. When providing a phone number, please include the country code")
    print("   (e.g., +14155552671 for US, +919551841398 for India)")
    print("\n9. To provide multiple phone numbers, separate them with commas")
    print("   (e.g., +14155552671, +919551841398)")
    print("\n10. To view sent SMS messages:")
    print("    - Check the console output after booking an appointment")
    print("    - Visit the Twilio console: https://console.twilio.com")
    print("    - Use the direct link provided in the console output")
    print()

def check_dependencies():
    missing = []
    try:
        import langchain
    except ImportError:
        missing.append("langchain")
    
    try:
        import dotenv
    except ImportError:
        missing.append("python-dotenv")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nPlease install them using:")
        print("pip install langchain python-dotenv")
        return False
    
    return True

def main():
    try:
        print_welcome_banner()
        
        if not check_dependencies():
            return
        
        load_dotenv()
        
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not api_token:
            print("❌ Error: Hugging Face API token not found.")
            print_instructions()
            return
        
        # Check for Twilio credentials (optional)
        twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        twilio_phones = os.getenv("TWILIO_PHONE_NUMBERS")
        
        if TWILIO_AVAILABLE and twilio_sid and twilio_token and (twilio_phone or twilio_phones):
            print("✅ Twilio credentials found. SMS reminders are enabled.")
            if twilio_phones:
                print(f"✅ Multiple Twilio phone numbers configured: {len(twilio_phones.split(','))}")
        else:
            print("⚠️  SMS functionality is disabled.")
            if not TWILIO_AVAILABLE:
                print("   Twilio module not installed. Install with: pip install twilio")
            else:
                print("   Twilio credentials not found.")
                print("   Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER or TWILIO_PHONE_NUMBERS")
        
        print("🔄 Initializing chatbot...")
        chatbot = SemanticAppointmentChatbot()
        
        print("\n✅ Chatbot initialized successfully!")
        print("Starting conversation...")
        print("-" * 60)
        chatbot.start_conversation()
        
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print_instructions()
    except KeyboardInterrupt:
        print("\n\n👋 Thank you for using the Appointment Scheduling Assistant!")
        print("Have a great day!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Verify your Hugging Face API token is valid")
        print("3. Make sure you have sufficient API quota")
        print("4. Try again later")
    finally:
        print("\n" + "=" * 60)

# ============ EXECUTION BLOCK ============
if __name__ == "__main__":
    main()