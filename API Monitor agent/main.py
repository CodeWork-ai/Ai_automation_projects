import streamlit as st
import time
import json
import re
from agent.monitor import APIMonitor
from ui.dashboard import show_enhanced_dashboard
from postman_parser import parse_postman_collection

# Set page configuration
st.set_page_config(page_title="Enhanced API Monitor", layout="wide")

def initialize_session_state():
    """Initializes the session state."""
    if 'monitors' not in st.session_state:
        st.session_state.monitors = []
    if 'last_run_time' not in st.session_state:
        st.session_state.last_run_time = 0
    if 'api_selection_state' not in st.session_state:
        st.session_state.api_selection_state = {}

def parse_curl_command(curl_string):
    """
    Parses a cURL command and returns a dictionary with the extracted components,
    or None if parsing fails.
    """
    if not curl_string or not curl_string.strip().startswith("curl"):
        return None

    method = "GET"
    method_match = re.search(r'-X\s+([A-Z]+)|--request\s+([A-Z]+)', curl_string)
    if method_match:
        method = method_match.group(1) or method_match.group(2)

    url_match = re.search(r"(?:'|\")?https?://[^\s'\"]+", curl_string)
    if not url_match:
        return None
    url = url_match.group().strip('\'"')

    headers = re.findall(r"-H\s+'([^']*)'|-H\s+\"([^\"]*)\"", curl_string)
    auth_config = {'method': 'none'}
    for h in headers:
        header_str = h[0] or h[1]
        if ': ' in header_str:
            key, value = header_str.split(': ', 1)
            key, value = key.strip(), value.strip()
            if key.lower() == 'authorization' and value.lower().startswith('bearer '):
                auth_config = {
                    'method': 'bearer',
                    'key_value': value[7:].strip()
                }
                break
            elif 'api-key' in key.lower() or 'token' in key.lower() or 'auth' in key.lower():
                auth_config = {
                    'method': 'api_key',
                    'key_name': key,
                    'key_value': value
                }
                break
    
    data = None
    # Enhanced regex to capture data payload from -d, --data, or --data-raw
    data_match = re.search(r"(?:-d|--data|--data-raw)\s+(['\"])(.*?)\1", curl_string)
    if data_match:
        data = data_match.group(2)

    return {
        'primary_endpoint': url,
        'method': method,
        'auth': auth_config,
        'data': data
    }

def create_api_config_from_manual_form():
    """Creates an API configuration dictionary from the manual form inputs."""
    auth_config = {'method': st.session_state.auth_method}
    if st.session_state.auth_method == 'bearer':
        auth_config['key_value'] = st.session_state.auth_token.strip()
    elif st.session_state.auth_method == 'api_key':
        auth_config['key_name'] = st.session_state.auth_key_name.strip()
        auth_config['key_value'] = st.session_state.auth_key_value.strip()

    return {
        'name': st.session_state.api_name.strip(),
        'api_type': st.session_state.api_type,
        'primary_endpoint': st.session_state.primary_endpoint.strip(),
        'backup_endpoint': st.session_state.backup_endpoint.strip(),
        'method': st.session_state.request_method,
        'auth': auth_config,
        'expected_status_code': st.session_state.expected_status,
    }

# --- Sidebar UI ---
with st.sidebar:
    st.title("Configuration")

    st.header("🚀 Import from Postman Collection")
    with st.expander("Upload & Parse Collection", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload Postman Collection JSON",
            type=['json'],
            help="Upload your exported Postman collection JSON file"
        )
        collection_text = st.text_area(
            "Or paste collection JSON here",
            placeholder="Paste your Postman collection JSON content...",
            height=100
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Parse Collection", use_container_width=True):
                collection_data = None
                if uploaded_file is not None:
                    try:
                        collection_data = json.load(uploaded_file)
                        st.success("✅ File uploaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to parse uploaded file: {str(e)}")
                elif collection_text.strip():
                    try:
                        collection_data = json.loads(collection_text.strip())
                        st.success("✅ JSON parsed successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to parse JSON: {str(e)}")
                if collection_data:
                    parsed_apis = parse_postman_collection(collection_data)
                    if parsed_apis:
                        st.session_state.parsed_apis = parsed_apis
                        st.info(f"🔍 Found {len(parsed_apis)} API endpoints")
                        st.rerun()
                    else:
                        st.warning("⚠️ No valid API endpoints found in the collection")
        with col2:
            if hasattr(st.session_state, 'parsed_apis') and st.session_state.parsed_apis:
                if st.button("⚡ Import All", use_container_width=True, type="primary"):
                    for api_config in st.session_state.parsed_apis:
                        new_monitor = APIMonitor(api_config)
                        st.session_state.monitors.append(new_monitor)
                    st.success(f"✅ Successfully imported {len(st.session_state.parsed_apis)} APIs!")
                    del st.session_state.parsed_apis
                    if 'api_selection_state' in st.session_state:
                        del st.session_state.api_selection_state
                    st.rerun()

    if hasattr(st.session_state, 'parsed_apis') and st.session_state.parsed_apis:
        st.subheader("📋 Select APIs to Import")
        internal_count = sum(1 for api in st.session_state.parsed_apis if api['api_type'] == 'internal')
        external_count = len(st.session_state.parsed_apis) - internal_count
        st.info(f"🏠 Internal: {internal_count} | 🌐 External: {external_count}")
        apis_to_remove = []
        for i, api in enumerate(st.session_state.parsed_apis):
            with st.container():
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    st.markdown(f"**{api['name']}**")
                    endpoint_display = api['primary_endpoint'][:60] + ('...' if len(api['primary_endpoint']) > 60 else '')
                    st.caption(f"🔗 {endpoint_display}")
                    auth_info = "🔒 " + api['auth']['method'].title() if api['auth']['method'] != 'none' else "🔓 No Auth"
                    method_info = f"📋 {api['method']}"
                    type_info = f"{'🏠' if api['api_type'] == 'internal' else '🌐'} {api['api_type'].title()}"
                    st.caption(f"{auth_info} | {method_info} | {type_info}")
                with col2:
                    threshold = api.get('performance_threshold_ms', 1000)
                    st.caption(f"⏱️ {threshold}ms")
                with col3:
                    button_key = f"add_api_{i}_{hash(api['name'])}"
                    if st.button("➕ Add", key=button_key, use_container_width=True):
                        try:
                            existing_names = [monitor.name for monitor in st.session_state.monitors]
                            if api['name'] in existing_names:
                                st.warning(f"⚠️ API '{api['name']}' already exists!")
                            else:
                                new_monitor = APIMonitor(api)
                                st.session_state.monitors.append(new_monitor)
                                apis_to_remove.append(i)
                                st.success(f"✅ Added {api['name']}!")
                        except Exception as e:
                            st.error(f"❌ Failed to add {api['name']}: {str(e)}")
                st.divider()
        if apis_to_remove:
            for idx in sorted(apis_to_remove, reverse=True):
                st.session_state.parsed_apis.pop(idx)
            if not st.session_state.parsed_apis:
                del st.session_state.parsed_apis
                if 'api_selection_state' in st.session_state:
                    del st.session_state.api_selection_state
            st.rerun()
        if st.button("🗑️ Clear Selection", use_container_width=True):
            del st.session_state.parsed_apis
            if 'api_selection_state' in st.session_state:
                del st.session_state.api_selection_state
            st.rerun()

    st.markdown("---")

    st.header("📄 Add from cURL")
    with st.form("curl_import_form", clear_on_submit=True):
        curl_command = st.text_area("Paste cURL command here", height=80)
        api_name_from_curl = st.text_input("API Name")
        col1, col2 = st.columns(2)
        with col1:
            api_type_curl = st.selectbox(
                "API Type", ["internal", "external"], key="api_type_curl", index=1,
                help="Select whether this is an internal or external API"
            )
        with col2:
            method_curl = st.selectbox(
                "HTTP Method", ["AUTO", "GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"], key="method_curl", index=0,
                help="Override the HTTP method. AUTO uses the method from cURL command"
            )
        curl_submitted = st.form_submit_button("🚀 Add Monitor from cURL", use_container_width=True)
        if curl_submitted:
            if not api_name_from_curl or not curl_command:
                st.error("❌ Please provide both an API Name and a cURL command.")
            else:
                parsed_data = parse_curl_command(curl_command)
                if parsed_data is None:
                    st.error("❌ Could not parse the cURL command. Please check its format.")
                else:
                    final_method = parsed_data['method'] if method_curl == "AUTO" else method_curl
                    api_config = {
                        'name': api_name_from_curl.strip(),
                        'api_type': api_type_curl,
                        'primary_endpoint': parsed_data['primary_endpoint'],
                        'backup_endpoint': '',
                        'method': final_method,
                        'auth': parsed_data['auth'],
                        'expected_status_code': 200,
                    }
                    if parsed_data.get('data'):
                        api_config['data'] = parsed_data['data']
                        
                    new_monitor = APIMonitor(api_config)
                    st.session_state.monitors.append(new_monitor)
                    st.success(f"✅ Added '{api_name_from_curl}' ({api_type_curl}, {final_method}) from cURL!")

    st.markdown("---")

    st.header("✏️ Add Manually")
    # REMOVED the st.form wrapper to make auth selection interactive
    st.text_input("API Name", key="api_name", placeholder="e.g., GitHub Status API")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("API Type", ["internal", "external"], key="api_type", index=1)
    with col2:
        st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"], key="request_method")
    st.text_input("Primary Endpoint URL", key="primary_endpoint", placeholder="https://api.example.com/health")
    st.text_input("Backup Endpoint URL (Optional)", key="backup_endpoint")
    st.number_input("Expected Status Code", min_value=100, max_value=599, value=200, key="expected_status")
    
    st.subheader("🔐 Authentication")
    st.selectbox("Auth Method", ["none", "bearer", "api_key"], key="auth_method")
    
    # This conditional logic now works instantly because there is no form batching the state
    if st.session_state.get("auth_method") == 'bearer':
        st.text_input("Bearer Token", key="auth_token", type="password")
    elif st.session_state.get("auth_method") == 'api_key':
        st.text_input("Header Key Name", key="auth_key_name", placeholder="X-API-KEY")
        st.text_input("Header Key Value", key="auth_key_value", type="password")
        
    # REPLACED st.form_submit_button with st.button
    manual_submitted = st.button("🚀 Add and Start Monitoring", use_container_width=True)
    if manual_submitted:
        if not st.session_state.api_name or not st.session_state.primary_endpoint:
            st.error("❌ API Name and Primary Endpoint are required for manual entry.")
        else:
            new_config = create_api_config_from_manual_form()
            new_monitor = APIMonitor(new_config)
            st.session_state.monitors.append(new_monitor)
            st.success(f"✅ Added '{st.session_state.api_name}' to the monitor list!")
            # Note: Inputs are not cleared automatically, which can be useful for adding similar APIs.

def run_checks():
    """Run API checks for all monitors"""
    if not st.session_state.monitors:
        return
    for monitor in st.session_state.monitors:
        monitor.check_api()
    st.session_state.last_run_time = time.time()

def main():
    """Main application function"""
    initialize_session_state()
    dashboard_placeholder = st.empty()

    # Run checks every 15 seconds
    if time.time() - st.session_state.last_run_time > 15:
        run_checks()

    # Display dashboard
    with dashboard_placeholder.container():
        show_enhanced_dashboard(st.session_state.monitors)
    
    # Auto-refresh loop
    time.sleep(5)
    st.rerun()

if __name__ == "__main__":
    main()