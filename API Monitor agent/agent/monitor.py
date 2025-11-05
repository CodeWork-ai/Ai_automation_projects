import requests
import time
from datetime import datetime
import statistics
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class APIMonitor:
    def __init__(self, api_config):
        self.config = api_config
        self.name = api_config.get('name')
        self.primary_endpoint = api_config.get('primary_endpoint')
        self.backup_endpoint = api_config.get('backup_endpoint')
        self.api_type = api_config.get('api_type', 'external')
        self.current_endpoint = self.primary_endpoint
        self.status = "Calibrating"  # Start in "Calibrating" state
        self.response_time = 0
        self.last_check_timestamp = None
        self.logs = []
        self.config['performance_threshold_ms'] = self.config.get('performance_threshold_ms', 2000)

        # Initialize enhanced session with retry strategy
        self.session = self._create_session()

    def _create_session(self):
        """Create a requests session with retry strategy and connection pooling"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _log(self, level, message):
        log_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "level": level,
            "message": message
        }
        self.logs.insert(0, log_entry)
        if len(self.logs) > 50:
            self.logs.pop()

    def _perform_request(self):
        headers = {}
        auth_config = self.config.get('auth', {})
        auth_method = auth_config.get('method', 'none')
        if auth_method == 'bearer' and auth_config.get('key_value'):
            self._log("INFO", "Attempting request with bearer authentication.")
            headers["Authorization"] = f"Bearer {auth_config.get('key_value')}"
        elif auth_method == 'api_key' and auth_config.get('key_name') and auth_config.get('key_value'):
            self._log("INFO", f"Attempting request with api-key authentication (Header: {auth_config.get('key_name')}).")
            headers[auth_config.get('key_name')] = auth_config.get('key_value')
        else:
            self._log("INFO", "Attempting request with no authentication.")

        method = self.config.get('method', 'GET')
        if self.api_type == 'internal':
            connect_timeout, read_timeout = 5, 15
        else:
            connect_timeout, read_timeout = 15, 45

        request_kwargs = {
            'method': method,
            'url': self.current_endpoint,
            'headers': headers,
            'timeout': (connect_timeout, read_timeout),
            'verify': True,
            'allow_redirects': True
        }

        # MODIFIED: Check for request body for POST, PUT, and PATCH methods
        if method in ['POST', 'PUT', 'PATCH']:
            headers["Content-Type"] = "application/json"
            if 'data' in self.config:
                try:
                    if isinstance(self.config['data'], str):
                        request_kwargs['json'] = json.loads(self.config['data'])
                    else:
                        request_kwargs['json'] = self.config['data']
                except json.JSONDecodeError:
                    request_kwargs['data'] = self.config['data']
            elif 'groq.com' in self.current_endpoint:
                json_payload = {
                    "messages": [{"role": "user", "content": "Hello, this is a health check test."}],
                    "model": "llama-3.3-70b-versatile",
                    "max_tokens": 10,
                    "temperature": 0
                }
                request_kwargs['json'] = json_payload
                self._log("INFO", "Using default Groq health check payload.")

        start_time = time.time()
        try:
            response = self.session.request(**request_kwargs)
            response_time = round((time.time() - start_time) * 1000)
            expected_status = self.config.get('expected_status_code', 200)
            if response.status_code != expected_status:
                response.raise_for_status()
            return response, response_time
        except requests.exceptions.RequestException as e:
            self._log("ERROR", f"Request failed: {str(e)}")
            raise

    def calibrate_threshold(self):
        self._log("INFO", f"Calibrating performance threshold for {self.api_type} API...")
        timings = []
        for i in range(3):
            try:
                _, response_time = self._perform_request()
                timings.append(response_time)
                time.sleep(1)
            except Exception as e:
                self._log("WARN", f"Calibration check {i+1} failed: {e}")
        if timings:
            avg_time = statistics.mean(timings)
            multiplier = 2.5 if self.api_type == 'external' else 2.0
            buffer = 200 if self.api_type == 'external' else 100
            calculated_threshold = round(avg_time * multiplier + buffer)
            self.config['performance_threshold_ms'] = max(calculated_threshold, 500)
            self._log("INFO", f"Calibration complete. Threshold set to {self.config['performance_threshold_ms']}ms (avg: {round(avg_time)}ms).")
        else:
            self.config['performance_threshold_ms'] = 5000
            self._log("WARN", "All calibration checks failed. Using default threshold: 5000ms")

    def check_api(self):
        self.last_check_timestamp = datetime.now()
        if self.status == "Calibrating":
            self._log("INFO", "First check: starting performance calibration.")
            try:
                self.calibrate_threshold()
            except Exception as e:
                self._log("ERROR", f"Calibration failed: {e}")
                self._handle_failure(e)
                return

        try:
            response, response_time = self._perform_request()
            self.response_time = response_time
            if 'X-RateLimit-Remaining' in response.headers and int(response.headers['X-RateLimit-Remaining']) < 10:
                self.status = "Warning"
                self._log("WARN", f"Rate limit approaching: {response.headers['X-RateLimit-Remaining']} requests remaining.")
            elif self.response_time > self.config.get('performance_threshold_ms', 2000):
                self.status = "Warning"
                self._log("WARN", f"Performance degradation: Response time is {self.response_time}ms (threshold is {self.config.get('performance_threshold_ms')}ms).")
            else:
                self.status = "OK"
                self._log("INFO", "API check successful.")
        except requests.exceptions.RequestException as e:
            self.response_time = 0
            self._handle_failure(e)

    def _handle_failure(self, error):
        self.status = "Down"
        status_code = getattr(error.response, 'status_code', None)
        response_text = getattr(error.response, 'text', str(error))
        error_message = f"API check failed. Status: {status_code or 'N/A'}. Error: {str(error)}"
        self._log("ERROR", error_message)

        if self.backup_endpoint and self.current_endpoint != self.backup_endpoint:
            self._log("INFO", f"Switching to backup endpoint: {self.backup_endpoint}")
            self.current_endpoint = self.backup_endpoint
            self.check_api() # Re-check with the backup

    def generate_curl_command(self):
        method = self.config.get('method', 'GET')
        curl_parts = [f"curl -X {method}"]
        auth_config = self.config.get('auth', {})
        if auth_config.get('method') == 'bearer' and auth_config.get('key_value'):
            curl_parts.append(f'-H "Authorization: Bearer {auth_config.get("key_value")}"')
        elif auth_config.get('method') == 'api_key' and auth_config.get('key_name') and auth_config.get('key_value'):
            curl_parts.append(f'-H "{auth_config.get("key_name")}: {auth_config.get("key_value")}"')
        
        if method in ['POST', 'PUT', 'PATCH'] and 'data' in self.config:
            curl_parts.append('-H "Content-Type: application/json"')
            data_payload = self.config['data']
            if isinstance(data_payload, dict):
                data_payload = json.dumps(data_payload)
            curl_parts.append(f"-d '{data_payload}'")

        curl_parts.append(f'"{self.current_endpoint}"')
        return ' '.join(curl_parts)

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close()