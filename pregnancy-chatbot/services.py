# services.py

import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

def make_voice_call(target_phone_number, message_to_say):
    """
    Uses Twilio to make an automated voice call and speak a message.
    """
    try:
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        twilio_number = os.environ.get("TWILIO_PHONE_NUMBER")
        
        if not all([account_sid, auth_token, twilio_number, target_phone_number]):
            missing = [v for v, name in [(account_sid, "SID"), (auth_token, "Token"), (twilio_number, "Twilio #"), (target_phone_number, "Target #")] if not v]
            error_msg = f"Twilio credentials or target number are not fully configured. Missing: {', '.join(missing)}"
            print(error_msg)
            return False, error_msg

        client = Client(account_sid, auth_token)
        twiml_instruction = f'<Response><Say voice="Google.en-US-Wavenet-F">{message_to_say}</Say></Response>'

        call = client.calls.create(
            twiml=twiml_instruction,
            to=target_phone_number,
            from_=twilio_number
        )
        
        print(f"Voice call initiated with SID: {call.sid}")
        return True, f"Initiating call to {target_phone_number}."
        
    # This will catch specific API errors from Twilio (e.g., "Invalid phone number")
    except TwilioRestException as e:
        error_msg = f"Twilio API returned an error: {e.msg}"
        print(error_msg)
        return False, error_msg

    # This will catch general network errors (like the one you are seeing)
    except Exception as e:
        error_msg = f"A network error occurred. This is often caused by a firewall, VPN, or proxy. Please check your network connection. Details: {e}"
        print(error_msg)
        return False, "Call failed: Could not connect to the voice service."