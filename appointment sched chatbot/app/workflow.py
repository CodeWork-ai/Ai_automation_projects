import os
import sys
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
# SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
# Twilio imports
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️  Twilio module not found. SMS functionality will be disabled.")
    print("   To enable SMS, run: pip install twilio")
# Load environment variables
load_dotenv()
# ============ DATABASE SETUP ============
SQLALCHEMY_DATABASE_URL = "sqlite:///./appointments.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
# ============ DATABASE MODELS ============
class AppointmentDB(Base):
    __tablename__ = "appointments"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, index=True)
    patient_contact = Column(String, index=True)
    clinic_name = Column(String)
    time = Column(String)
    confirmed = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Add a column to track SMS status
    sms_sent = Column(Boolean, default=False)
Base.metadata.create_all(bind=engine)
# ============ PYDANTIC MODELS ============
class AppointmentCreate(BaseModel):
    patient_name: str
    patient_contact: str
    clinic_name: str = "General Practice"
    time: str
    
    @field_validator('patient_contact')
    @classmethod
    def validate_contact(cls, v):
        # Basic validation for phone number or email
        phone_pattern = r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{4,6}$'
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not (re.match(phone_pattern, v) or re.match(email_pattern, v)):
            raise ValueError('Please provide a valid phone number or email address')
        return v
class AppointmentResponse(BaseModel):
    id: int
    patient_name: str
    patient_contact: str
    clinic_name: str
    time: str
    confirmed: bool
    created_at: datetime
    sms_sent: bool = False
    sms_status: Optional[str] = None
    
    model_config = {"from_attributes": True}
    
class RescheduleRequest(BaseModel):
    new_time: str
# ============ SMS SERVICE ============
class SMSService:
    def __init__(self):
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.twilio_phone_numbers = self._load_twilio_phone_numbers()
        
        # Trial account configuration - ALWAYS USE VERIFIED NUMBER
        self.trial_mode = True  # Force trial mode
        self.verified_number = "+919551382296"  # Your verified number hardcoded
        self.verified_number = "+919043502948"  # Your verified number hardcoded
        self.verified_number = "+91 73584 40769"  # Your verified number hardcoded
        self.sms_enabled = False
        if TWILIO_AVAILABLE:
            if self.check_twilio_configuration():
                self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
                self.sms_enabled = True
                print("✅ Twilio SMS service is enabled")
                self._print_verified_numbers()
        else:
            print("⚠️  Twilio module not installed. SMS functionality will be disabled.")
            print("   Install with: pip install twilio")
    
    def _load_twilio_phone_numbers(self):
        """Load multiple Twilio phone numbers from environment variable"""
        phone_numbers_str = os.getenv("TWILIO_PHONE_NUMBERS", "")
        if phone_numbers_str:
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
        print(f"📞 Trial Mode Recipient: {self.verified_number}")
    
    def check_twilio_configuration(self):
        """Check if Twilio is properly configured"""
        if not TWILIO_AVAILABLE:
            print("❌ Twilio module is not installed. Install with: pip install twilio")
            return False
        
        if not self.twilio_account_sid:
            print("❌ TWILIO_ACCOUNT_SID is not set in environment variables")
            return False
        
        if not self.twilio_auth_token:
            print("❌ TWILIO_AUTH_TOKEN is not set in environment variables")
            return False
        
        if not self.twilio_phone_number:
            print("❌ TWILIO_PHONE_NUMBER is not set in environment variables")
            return False
        
        print("✅ Twilio configuration is valid")
        return True
    
    def extract_phone_numbers(self, contact_info):
        """Extract and validate phone numbers from contact info"""
        if not contact_info:
            return []
        
        numbers = [num.strip() for num in contact_info.split(',')]
        valid_numbers = []
        
        for num in numbers:
            formatted_num = self._format_phone_number(num)
            if self._validate_phone_number(formatted_num):
                valid_numbers.append(formatted_num)
            else:
                print(f"⚠️  Invalid phone number format: {num}")
        
        return valid_numbers
    
    def _format_phone_number(self, phone_number):
        """Format phone number to E.164 standard"""
        digits = re.sub(r'[^\d]', '', phone_number)
        
        if phone_number.startswith('+'):
            return '+' + digits
        
        if digits.startswith('00'):
            return '+' + digits[2:]
        elif digits.startswith('0'):
            return '+1' + digits[1:]
        elif len(digits) == 10:
            if 200 <= int(digits[:3]) <= 999:
                return '+1' + digits
            else:
                if digits[0] in ['7', '8', '9']:
                    return '+91' + digits
                else:
                    print(f"⚠️  Warning: Ambiguous phone number format for {digits}. Defaulting to US format (+1).")
                    return '+1' + digits
        elif len(digits) == 11 and digits.startswith('1'):
            return '+' + digits
        elif len(digits) == 12 and digits.startswith('91'):
            return '+' + digits
        elif len(digits) == 10 and digits[0] in ['7', '8', '9']:
            return '+91' + digits
        elif len(digits) == 9 and digits[0] in ['7', '8', '9']:
            print(f"⚠️  Warning: Phone number {digits} appears to be missing a digit. Adding Indian country code.")
            return '+91' + digits
        else:
            print(f"⚠️  Warning: Unrecognized phone number format for {digits}. Adding + prefix.")
            return '+' + digits
    
    def _validate_phone_number(self, phone_number):
        """Validate phone number format"""
        digits = re.sub(r'[^\d]', '', phone_number)
        
        if not phone_number.startswith('+'):
            return False
        
        # Check for specific Twilio test numbers that will be rejected
        twilio_test_numbers = [
            '+15005550006',  # Valid test number
            '+15005550007',  # Invalid test number
            '+15005550008',  # Invalid test number
            '+15005550009',  # Invalid test number
            '+15005550001',  # Invalid test number
            '+15005550002',  # Invalid test number
            '+15005550003',  # Invalid test number
            '+15005550004',  # Invalid test number
            '+15005550005'   # Invalid test number
        ]
        
        if phone_number in twilio_test_numbers:
            print(f"⚠️  Twilio test number detected: {phone_number}")
            return True  # Allow test numbers but handle them specially in send_sms
        
        if len(digits) < 8:
            return False
        
        if phone_number.startswith('+1'):
            return len(digits) == 11
        elif phone_number.startswith('+91'):
            return len(digits) == 12
        elif phone_number.startswith('+44'):
            return len(digits) == 12
        else:
            return len(digits) >= 8
    
    def send_sms(self, to_number, message):
        """Send SMS using Twilio - ALWAYS TO VERIFIED NUMBER IN TRIAL MODE"""
        if not self.sms_enabled:
            print("SMS service is not enabled")
            return False, "SMS service is not enabled"
        
        # FOR TRIAL ACCOUNTS: ALWAYS SEND TO VERIFIED NUMBER
        if self.trial_mode:
            original_number = to_number
            to_number = self.verified_number  # 👈 YOUR VERIFIED NUMBER: +919551382296
            print(f"🔒 TRIAL MODE: Redirecting SMS from {original_number} to verified number {to_number}")
        
        # Handle test numbers specially
        twilio_test_numbers = [
            '+15005550006',  # Valid test number
            '+15005550007',  # Invalid test number
            '+15005550008',  # Invalid test number
            '+15005550009',  # Invalid test number
            '+15005550001',  # Invalid test number
            '+15005550002',  # Invalid test number
            '+15005550003',  # Invalid test number
            '+15005550004',  # Invalid test number
            '+15005550005'   # Invalid test number
        ]
        
        if to_number in twilio_test_numbers:
            print(f"⚠️  Test number detected: {to_number}")
            print(f"📝 Test message: {message}")
            print("✅ Test message processed (not actually sent)")
            return True, "Test message processed (not actually sent)"
        
        try:
            if not self._validate_phone_number(to_number):
                print(f"❌ Error: Invalid phone number format: {to_number}")
                return False, "Invalid phone number format"
            
            # 👇 THIS IS WHERE YOUR NUMBER IS USED AS RECIPIENT
            message_obj = self.twilio_client.messages.create(
                body=message,
                from_=self.twilio_phone_number,
                to=to_number  # 👈 ALWAYS YOUR VERIFIED NUMBER IN TRIAL MODE
            )
            
            print(f"✅ SMS sent successfully to {to_number}")
            print(f"🆔 Message SID: {message_obj.sid}")
            return True, "SMS sent successfully"
            
        except TwilioRestException as e:
            print(f"❌ Twilio Error: {e}")
            if e.code == 21610:
                error_msg = "This number is not verified in your Twilio account. For trial accounts, you can only send SMS to verified numbers."
                print(f"   {error_msg}")
                return False, error_msg
            elif e.code == 21211:
                error_msg = "The recipient phone number is invalid."
                print(f"   {error_msg}")
                return False, error_msg
            elif e.code == 30001:
                error_msg = "Queue overflow: You have exceeded the maximum number of queued messages."
                print(f"   {error_msg}")
                return False, error_msg
            else:
                return False, f"Twilio error: {str(e)}"
        except Exception as e:
            print(f"❌ Error sending SMS: {e}")
            return False, f"Error sending SMS: {str(e)}"
    
    def send_confirmation(self, contact, name, time, clinic_name="General Practice"):
        """Send appointment confirmation SMS"""
        print(f"Attempting to send confirmation SMS to {contact} for {name}")
        
        # Check if contact is an email
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, contact):
            print(f"ℹ️  Contact is an email ({contact}), SMS not applicable.")
            return True, "Contact is an email address, SMS not sent"
        
        phone_numbers = self.extract_phone_numbers(contact)
        if not phone_numbers:
            print(f"⚠️  No valid phone numbers found in contact information: {contact}")
            return False, "No valid phone numbers found"
        
        message = f"Hi {name}, your appointment is confirmed for {time} at {clinic_name}."
        sms_sent = False
        error_msg = ""
        
        for phone_number in phone_numbers:
            print(f"📞 Attempting to send SMS to {phone_number}")
            success, msg = self.send_sms(phone_number, message)
            if success:
                sms_sent = True
                print(f"✅ SMS sent successfully to {phone_number}")
            else:
                error_msg = msg
                print(f"❌ Failed to send SMS to {phone_number}: {msg}")
        
        return sms_sent, error_msg if not sms_sent else "SMS sent successfully"
    
    def send_cancellation(self, contact, name, time):
        """Send appointment cancellation SMS"""
        print(f"Attempting to send cancellation SMS to {contact} for {name}")
        
        # Check if contact is an email
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, contact):
            print(f"ℹ️  Contact is an email ({contact}), SMS not applicable.")
            return True, "Contact is an email address, SMS not sent"
        
        phone_numbers = self.extract_phone_numbers(contact)
        if not phone_numbers:
            print(f"⚠️  No valid phone numbers found in contact information: {contact}")
            return False, "No valid phone numbers found"
        
        message = f"Hi {name}, your appointment on {time} has been cancelled as requested."
        sms_sent = False
        error_msg = ""
        
        for phone_number in phone_numbers:
            print(f"📞 Attempting to send SMS to {phone_number}")
            success, msg = self.send_sms(phone_number, message)
            if success:
                sms_sent = True
                print(f"✅ SMS sent successfully to {phone_number}")
            else:
                error_msg = msg
                print(f"❌ Failed to send SMS to {phone_number}: {msg}")
        
        return sms_sent, error_msg if not sms_sent else "SMS sent successfully"
    
    def send_reschedule(self, contact, name, old_time, new_time, clinic_name="General Practice"):
        """Send appointment reschedule SMS"""
        print(f"Attempting to send reschedule SMS to {contact} for {name}")
        
        # Check if contact is an email
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, contact):
            print(f"ℹ️  Contact is an email ({contact}), SMS not applicable.")
            return True, "Contact is an email address, SMS not sent"
        
        phone_numbers = self.extract_phone_numbers(contact)
        if not phone_numbers:
            print(f"⚠️  No valid phone numbers found in contact information: {contact}")
            return False, "No valid phone numbers found"
        
        message = f"Hi {name}, your appointment has been rescheduled from {old_time} to {new_time} at {clinic_name}."
        sms_sent = False
        error_msg = ""
        
        for phone_number in phone_numbers:
            print(f"📞 Attempting to send SMS to {phone_number}")
            success, msg = self.send_sms(phone_number, message)
            if success:
                sms_sent = True
                print(f"✅ SMS sent successfully to {phone_number}")
            else:
                error_msg = msg
                print(f"❌ Failed to send SMS to {phone_number}: {msg}")
        
        return sms_sent, error_msg if not sms_sent else "SMS sent successfully"
# Initialize SMS service
sms_service = SMSService()
# ============ FASTAPI APP ============
app = FastAPI(title="Appointment Scheduling System")
# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
# Set up templates
templates = Jinja2Templates(directory="templates")
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============ MOUNT STATIC FILES ============
app.mount("/static", StaticFiles(directory="static"), name="static")
# ============ HELPER FUNCTIONS ============
def normalize_contact(contact):
    """Normalize contact information for comparison"""
    if not contact:
        return ""
    return re.sub(r'\s+', '', contact).lower()
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
# ============ DATABASE DEPENDENCY ============
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# ============ AVAILABLE SLOTS ============
AVAILABLE_SLOTS = [
    "Mon 9:00 AM", "Mon 11:00 AM", "Mon 2:00 PM",
    "Tue 10:00 AM", "Tue 1:00 PM", "Tue 3:30 PM",
    "Wed 9:30 AM", "Wed 11:30 AM", "Wed 4:00 PM",
    "Thu 10:30 AM", "Thu 2:30 PM", "Thu 4:30 PM",
    "Fri 9:00 AM", "Fri 12:00 PM", "Fri 3:00 PM"
]
# ============ API ENDPOINTS ============
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page at root"""
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/test-send-sms")
async def test_send_sms(request: Request):
    """Test sending SMS"""
    data = await request.json()
    
    phone_number = data.get("phone_number")
    print(f"Testing SMS to: {phone_number}")
    message = data.get("message", "This is a test message from the appointment system.")
    
    if not phone_number:
        raise HTTPException(status_code=400, detail="Phone number is required")
    
    # Format the phone number
    formatted_number = sms_service._format_phone_number(phone_number)
    if not sms_service._validate_phone_number(formatted_number):
        raise HTTPException(status_code=400, detail="Invalid phone number format")
    
    # Send the SMS
    success, msg = sms_service.send_sms(formatted_number, message)
    
    return {
        "success": success,
        "phone_number": formatted_number,
        "message": message,
        "details": msg
    }
@app.get("/slots")
def get_available_slots():
    """Get all available appointment slots"""
    return {"slots": AVAILABLE_SLOTS}
@app.post("/appointments", response_model=AppointmentResponse)
def create_appointment(appointment: AppointmentCreate, db: Session = Depends(get_db)):
    """Create a new appointment"""
    # Check if the slot is available
    if appointment.time not in AVAILABLE_SLOTS:
        raise HTTPException(status_code=400, detail="Requested time slot is not available")
    
    # Check if the patient already has an appointment at the same time
    existing_appointment = db.query(AppointmentDB).filter(
        AppointmentDB.patient_name == appointment.patient_name,
        AppointmentDB.patient_contact == appointment.patient_contact,
        AppointmentDB.time == appointment.time
    ).first()
    
    if existing_appointment:
        raise HTTPException(status_code=400, detail="You already have an appointment at this time")
    
    # Create the appointment
    db_appointment = AppointmentDB(
        patient_name=appointment.patient_name,
        patient_contact=appointment.patient_contact,
        clinic_name=appointment.clinic_name,
        time=appointment.time
    )
    
    db.add(db_appointment)
    db.commit()
    db.refresh(db_appointment)
    
    # Remove the slot from available slots
    if appointment.time in AVAILABLE_SLOTS:
        AVAILABLE_SLOTS.remove(appointment.time)
    
    # Send SMS confirmation
    print(f"Booking appointment for {appointment.patient_name}")
    sms_sent, sms_status = sms_service.send_confirmation(
        appointment.patient_contact,
        appointment.patient_name,
        appointment.time,
        appointment.clinic_name
    )
    
    # Update the appointment with SMS status
    db_appointment.sms_sent = sms_sent
    db.commit()
    db.refresh(db_appointment)
    
    # Add SMS status to response
    db_appointment.sms_status = sms_status
    
    return db_appointment
@app.get("/appointments", response_model=List[AppointmentResponse])
def get_appointments(patient_name: str = None, patient_contact: str = None, db: Session = Depends(get_db)):
    """Get appointments, optionally filtered by patient name and contact"""
    query = db.query(AppointmentDB)
    
    if patient_name:
        query = query.filter(AppointmentDB.patient_name == patient_name)
    
    if patient_contact:
        normalized_contact = normalize_contact(patient_contact)
        appointments = []
        for appt in query.all():
            if normalize_contact(appt.patient_contact) == normalized_contact:
                appointments.append(appt)
        return appointments
    
    return query.all()
@app.delete("/appointments/{appointment_id}")
def cancel_appointment(appointment_id: int, db: Session = Depends(get_db)):
    """Cancel an appointment"""
    appointment = db.query(AppointmentDB).filter(AppointmentDB.id == appointment_id).first()
    
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    # Add the slot back to available slots
    if appointment.time not in AVAILABLE_SLOTS:
        AVAILABLE_SLOTS.append(appointment.time)
        AVAILABLE_SLOTS.sort()
    
    # Send SMS cancellation
    print(f"Cancelling appointment {appointment_id}")
    print(f"Contact: {appointment.patient_contact}")
    print(f"Name: {appointment.patient_name}")
    print(f"Time: {appointment.time}")
    
    sms_sent, sms_status = sms_service.send_cancellation(
        appointment.patient_contact,
        appointment.patient_name,
        appointment.time
    )
    
    db.delete(appointment)
    db.commit()
    
    # Provide detailed SMS status
    if not sms_sent:
        if not sms_service.sms_enabled:
            sms_status = "SMS service is not configured."
        elif not TWILIO_AVAILABLE:
            sms_status = "Twilio module is not installed."
        # If we don't have a specific error message, use a generic one
        elif not sms_status or sms_status == "No valid phone numbers found":
            sms_status = "SMS could not be sent. Check the contact number."
    
    return {"message": "Appointment cancelled successfully", "sms_sent": sms_sent, "details": sms_status}
@app.put("/appointments/{appointment_id}", response_model=AppointmentResponse)
def reschedule_appointment(appointment_id: int, request: RescheduleRequest, db: Session = Depends(get_db)):
    """Reschedule an appointment to a new time"""
    new_time = request.new_time
    
    # Check if the new time is available
    if new_time not in AVAILABLE_SLOTS:
        raise HTTPException(status_code=400, detail="Requested time slot is not available")
    
    appointment = db.query(AppointmentDB).filter(AppointmentDB.id == appointment_id).first()
    
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    
    # Add the old slot back to available slots
    if appointment.time not in AVAILABLE_SLOTS:
        AVAILABLE_SLOTS.append(appointment.time)
        AVAILABLE_SLOTS.sort()
    
    # Store the old time for SMS
    old_time = appointment.time
    
    # Update the appointment time
    appointment.time = new_time
    
    db.commit()
    db.refresh(appointment)
    
    # Remove the new slot from available slots
    if new_time in AVAILABLE_SLOTS:
        AVAILABLE_SLOTS.remove(new_time)
    
    # Send SMS reschedule notification
    print(f"Rescheduling appointment {appointment_id}")
    print(f"Contact: {appointment.patient_contact}")
    print(f"Name: {appointment.patient_name}")
    print(f"Old time: {old_time}")
    print(f"New time: {new_time}")
    
    sms_sent, sms_status = sms_service.send_reschedule(
        appointment.patient_contact,
        appointment.patient_name,
        old_time,
        new_time,
        appointment.clinic_name
    )
    
    # Update the appointment with SMS status
    appointment.sms_sent = sms_sent
    db.commit()
    db.refresh(appointment)
    
    # Add SMS status to response
    appointment.sms_status = sms_status
    
    return appointment
# ============ FRONTEND ROUTES ============
@app.get("/index", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page"""
    return RedirectResponse(url="/")
@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_interface(request: Request):
    """Serve the chatbot interface"""
    return templates.TemplateResponse("chatbot.html", {"request": request})
# ============ CHATBOT ENDPOINT ============
# Store conversation state
conversation_state = {}
def extract_name_and_contact(message):
    """Extract name and contact from a message"""
    # Try different separators
    separators = ['-', '|', ',', ':', ';']
    
    for sep in separators:
        if sep in message:
            parts = message.split(sep)
            if len(parts) >= 2:
                name = parts[0].strip().title()
                contact = parts[1].strip()
                # Validate contact
                phone_pattern = r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{4,6}$'
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                
                if re.match(phone_pattern, contact) or re.match(email_pattern, contact):
                    return name, contact
    
    # If no separator found, try to extract phone number from the message
    phone_pattern = r'(\+?\d{10,15})'
    match = re.search(phone_pattern, message)
    if match:
        phone = match.group(1)
        # Remove the phone from the message to get the name
        name = message.replace(phone, '').strip().title()
        return name, phone
    
    return None, None
@app.post("/chat")
async def chat_endpoint(request: Request):
    """Process chatbot messages with improved conversation flow"""
    data = await request.json()
    user_message = data.get("message", "").lower().strip()
    session_id = data.get("session_id", "default")
    
    # Initialize session state if not exists
    if session_id not in conversation_state:
        conversation_state[session_id] = {
            "state": "idle",
            "data": {}
        }
    
    state = conversation_state[session_id]
    
    # Process message based on current state
    if state["state"] == "idle":
        if "hello" in user_message or "hi" in user_message:
            return {"response": "Hello! I'm your appointment scheduling assistant. How can I help you today?"}
        
        elif "book" in user_message or "schedule" in user_message or "appointment" in user_message:
            state["state"] = "booking_name"
            return {"response": "I can help you book an appointment. What's your name?"}
        
        elif "view" in user_message or "my appointments" in user_message or "show" in user_message:
            state["state"] = "viewing_name"
            return {"response": "I can show you your appointments. What's your name?"}
        
        elif "cancel" in user_message:
            state["state"] = "cancel_name"
            return {"response": "I can help you cancel an appointment. What's your name?"}
        
        elif "available" in user_message or "slots" in user_message:
            # Format available slots with line breaks for better readability
            slots_list = "\n".join([f"• {slot}" for slot in AVAILABLE_SLOTS])
            return {"response": f"Here are the available slots:\n{slots_list}"}
        
        elif "help" in user_message:
            return {"response": "I can help you with booking, viewing, or canceling appointments. Just let me know what you need!"}
        
        elif "bye" in user_message or "goodbye" in user_message:
            return {"response": "Goodbye! Have a great day!"}
        
        else:
            return {"response": "I'm not sure I understand. You can ask me to book, view, or cancel appointments. Type 'help' for more options."}
    
    # Booking flow - Step 1: Get name
    elif state["state"] == "booking_name":
        # Try to extract both name and contact
        name, contact = extract_name_and_contact(user_message)
        
        if name and contact:
            # Both name and contact provided
            state["data"]["name"] = name
            state["data"]["contact"] = contact
            state["state"] = "booking_time"
            
            # Format available slots with line breaks for better readability
            slots_list = "\n".join([f"• {slot}" for slot in AVAILABLE_SLOTS])
            return {"response": f"Thanks, {name}! Now please select a time slot from the available options:\n{slots_list}"}
        else:
            # Only name provided
            name = user_message.title()  # Capitalize the name
            state["data"]["name"] = name
            state["state"] = "booking_contact"
            return {"response": f"Thanks, {name}! What's your contact number?"}
    
    # Booking flow - Step 2: Get contact
    elif state["state"] == "booking_contact":
        # Validate and extract contact information
        contact = user_message
        phone_pattern = r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{4,6}$'
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not (re.match(phone_pattern, contact) or re.match(email_pattern, contact)):
            return {"response": "Please provide a valid phone number or email address."}
        
        state["data"]["contact"] = contact
        state["state"] = "booking_time"
        
        # Format available slots with line breaks for better readability
        slots_list = "\n".join([f"• {slot}" for slot in AVAILABLE_SLOTS])
        return {"response": f"Great! Now please select a time slot from the available options:\n{slots_list}"}
    
    # Booking flow - Step 3: Get time slot
    elif state["state"] == "booking_time":
        # Try to match the requested time with available slots
        requested_time = user_message
        matched_slot = None
        
        # Check for exact match first
        if requested_time in AVAILABLE_SLOTS:
            matched_slot = requested_time
        else:
            # Try to parse the time
            parsed_time = parse_any_time_format(requested_time)
            if parsed_time and parsed_time in AVAILABLE_SLOTS:
                matched_slot = parsed_time
        
        if matched_slot:
            # Create the appointment
            try:
                db = next(get_db())
                appointment_data = {
                    "patient_name": state["data"]["name"],
                    "patient_contact": state["data"]["contact"],
                    "clinic_name": "General Practice",
                    "time": matched_slot
                }
                
                # Check if the slot is available
                if matched_slot not in AVAILABLE_SLOTS:
                    state["state"] = "booking_time"
                    slots_list = "\n".join([f"• {slot}" for slot in AVAILABLE_SLOTS])
                    return {"response": f"Sorry, that time slot is no longer available. Please select another:\n{slots_list}"}
                
                # Check if the patient already has an appointment at the same time
                existing_appointment = db.query(AppointmentDB).filter(
                    AppointmentDB.patient_name == appointment_data["patient_name"],
                    AppointmentDB.patient_contact == appointment_data["patient_contact"],
                    AppointmentDB.time == appointment_data["time"]
                ).first()
                
                if existing_appointment:
                    state["state"] = "idle"
                    state["data"] = {}
                    return {"response": "You already have an appointment at this time. Please select a different time."}
                
                # Create the appointment
                db_appointment = AppointmentDB(
                    patient_name=appointment_data["patient_name"],
                    patient_contact=appointment_data["patient_contact"],
                    clinic_name=appointment_data["clinic_name"],
                    time=appointment_data["time"]
                )
                
                db.add(db_appointment)
                db.commit()
                db.refresh(db_appointment)
                
                # Remove the slot from available slots
                if matched_slot in AVAILABLE_SLOTS:
                    AVAILABLE_SLOTS.remove(matched_slot)
                
                # Send SMS confirmation
                print(f"Booking appointment for {appointment_data['patient_name']}")
                sms_sent, sms_status = sms_service.send_confirmation(
                    appointment_data["patient_contact"],
                    appointment_data["patient_name"],
                    appointment_data["time"],
                    appointment_data["clinic_name"]
                )
                
                # Update the appointment with SMS status
                db_appointment.sms_sent = sms_sent
                db.commit()
                db.refresh(db_appointment)
                
                # Reset state
                state["state"] = "idle"
                state["data"] = {}
                
                sms_response = ""
                if sms_sent:
                    sms_response = " You'll receive a confirmation message shortly."
                else:
                    if "email" in sms_status:
                        sms_response = " We couldn't send an SMS since you provided an email address."
                    else:
                        sms_response = " We couldn't send a confirmation message. " + sms_status
                
                return {"response": f"Great! Your appointment has been booked for {matched_slot}.{sms_response}"}
            
            except Exception as e:
                state["state"] = "idle"
                state["data"] = {}
                return {"response": f"Sorry, there was an error booking your appointment. Please try again."}
        else:
            slots_list = "\n".join([f"• {slot}" for slot in AVAILABLE_SLOTS])
            return {"response": f"I couldn't find that time slot. Please select from the available options:\n{slots_list}"}
    
    # Viewing appointments flow
    elif state["state"] == "viewing_info":
        # Try to extract name and contact
        name, contact = extract_name_and_contact(user_message)
        if not name and not contact:
            # Check if the message looks like a contact
            phone_pattern = r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{4,6}$'
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(phone_pattern, user_message) or re.match(email_pattern, user_message):
                contact = user_message
            else:
                name = user_message.title()
        
        try:
            db = next(get_db())
            contact = state["data"].get("contact", None)
            appointments = db.query(AppointmentDB).filter(AppointmentDB.patient_name == name).all()
            
            if name:
                query = query.filter(AppointmentDB.patient_name == name)
            
            appointments = []
            if contact:
                normalized_contact = normalize_contact(contact)
            appointments = [
                appt for appt in appointments
                if normalize_contact(appt.patient_contact) == normalized_contact
            ]
            
            if not appointments:
                state["state"] = "idle"
                state["data"] = {}
                return {"response": f"You don't have any appointments scheduled, {name}."}
            
            response = f"Here are your appointments, {name}:\n\n"
            for appt in appointments:
                response += f"• {appt.time} at {appt.clinic_name}\n"
            
            state["state"] = "idle"
            state["data"] = {}
            return {"response": response}
        
        except Exception as e:
            state["state"] = "idle"
            state["data"] = {}
            return {"response": "Sorry, there was an error retrieving your appointments. Please try again."}
    
    # Canceling appointments flow
    elif state["state"] == "cancel_name":
        name = user_message.title()
        state["data"]["name"] = name
        
        try:
            db = next(get_db())
            appointments = db.query(AppointmentDB).filter(AppointmentDB.patient_name == name).all()
            
            if not appointments:
                state["state"] = "idle"
                state["data"] = {}
                return {"response": f"You don't have any appointments scheduled, {name}."}
            
            state["data"]["appointments"] = appointments
            state["state"] = "cancel_select"
            
            response = f"Here are your appointments, {name}:\n\n"
            for i, appt in enumerate(appointments, 1):
                response += f"{i}. {appt.time} at {appt.clinic_name}\n"
            
            response += "\nPlease enter the number of the appointment you want to cancel:"
            return {"response": response}
        
        except Exception as e:
            state["state"] = "idle"
            state["data"] = {}
            return {"response": "Sorry, there was an error retrieving your appointments. Please try again."}
    
    elif state["state"] == "cancel_select":
        try:
            selection = int(user_message)
            appointments = state["data"]["appointments"]
            
            if selection < 1 or selection > len(appointments):
                return {"response": f"Please enter a number between 1 and {len(appointments)}:"}
            
            selected_appointment = appointments[selection - 1]
            
            # Cancel the appointment
            db = next(get_db())
            appointment = db.query(AppointmentDB).filter(AppointmentDB.id == selected_appointment.id).first()
            
            if appointment:
                # Add the slot back to available slots
                if appointment.time not in AVAILABLE_SLOTS:
                    AVAILABLE_SLOTS.append(appointment.time)
                    AVAILABLE_SLOTS.sort()
                
                # Send SMS cancellation
                print(f"Cancelling appointment for {appointment.patient_name}")
                sms_sent, sms_status = sms_service.send_cancellation(
                    appointment.patient_contact,
                    appointment.patient_name,
                    appointment.time
                )
                
                db.delete(appointment)
                db.commit()
                
                state["state"] = "idle"
                state["data"] = {}
                
                sms_response = ""
                if sms_sent:
                    sms_response = " You'll receive a cancellation message shortly."
                else:
                    if "email" in sms_status:
                        sms_response = " We couldn't send an SMS since you provided an email address."
                    else:
                        sms_response = " We couldn't send a cancellation message. " + sms_status
                
                return {"response": f"Your appointment on {appointment.time} has been cancelled successfully.{sms_response}"}
            else:
                state["state"] = "idle"
                state["data"] = {}
                return {"response": "Sorry, I couldn't find that appointment."}
        
        except ValueError:
            return {"response": "Please enter a valid number:"}
        except Exception as e:
            state["state"] = "idle"
            state["data"] = {}
            return {"response": "Sorry, there was an error cancelling your appointment. Please try again."}
    
    # Default response for unknown states
    state["state"] = "idle"
    state["data"] = {}
    return {"response": "I'm not sure what happened. Let's start over. How can I help you today?"}


# ============ MAIN FUNCTION ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
