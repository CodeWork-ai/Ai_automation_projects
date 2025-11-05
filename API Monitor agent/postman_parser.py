import json
import re
import streamlit as st

def parse_postman_collection(collection_json):
    """
    Advanced Postman Collection Parser - FIXED VERSION
    Extracts all API endpoints with their authentication, methods, and configurations
    Returns a list of API configurations ready for monitoring
    """
    try:
        if isinstance(collection_json, str):
            collection_data = json.loads(collection_json)
        else:
            collection_data = collection_json

        apis = []

        def extract_requests(items, folder_name=""):
            for item in items:
                if 'request' in item and item['request'] is not None:
                    # Extract request details
                    request = item['request']
                    api_name = f"{folder_name}/{item['name']}" if folder_name else item['name']

                    # Extract URL with smart parsing - FIXED VERSION
                    url_data = request.get('url', {})
                    endpoint = None

                    if isinstance(url_data, str):
                        endpoint = url_data
                    elif isinstance(url_data, dict) and url_data:
                        # Handle structured URL object
                        protocol = url_data.get('protocol', 'https')

                        # Handle host - can be array or string
                        host = url_data.get('host', [])
                        if isinstance(host, list) and host:
                            host_str = '.'.join(host)
                        elif isinstance(host, str):
                            host_str = host
                        else:
                            # Skip if no valid host
                            continue

                        port = url_data.get('port', '')

                        # Handle path - can be array or string
                        path = url_data.get('path', [])
                        if isinstance(path, list):
                            path_str = '/'.join([str(p) for p in path if p])
                        elif isinstance(path, str):
                            path_str = path
                        else:
                            path_str = ''

                        # Construct the full URL
                        endpoint = f"{protocol}://{host_str}"
                        if port:
                            endpoint += f":{port}"
                        if path_str:
                            endpoint += f"/{path_str}"

                        # Handle query parameters
                        query = url_data.get('query', [])
                        if query and isinstance(query, list):
                            query_params = []
                            for q in query:
                                if isinstance(q, dict) and q.get('key'):
                                    key = q.get('key', '')
                                    value = q.get('value', '')
                                    query_params.append(f"{key}={value}")
                            if query_params:
                                endpoint += f"?{'&'.join(query_params)}"

                    # Skip if we couldn't extract a valid endpoint
                    if not endpoint:
                        continue

                    # Extract HTTP method
                    method = request.get('method', 'GET')

                    # Advanced authentication parsing
                    auth_config = parse_authentication(request)

                    # Extract request body for POST/PUT requests
                    body_data = None
                    if method in ['POST', 'PUT', 'PATCH'] and 'body' in request:
                        body = request['body']
                        if body and body.get('mode') == 'raw':
                            raw_data = body.get('raw', '')
                            if raw_data:
                                body_data = raw_data
                        elif body and body.get('mode') == 'formdata':
                            # Handle form data
                            form_data = body.get('formdata', [])
                            if form_data:
                                body_dict = {}
                                for item in form_data:
                                    if isinstance(item, dict) and item.get('type') != 'file':
                                        key = item.get('key', '')
                                        value = item.get('value', '')
                                        if key:
                                            body_dict[key] = value
                                if body_dict:
                                    body_data = json.dumps(body_dict)

                    # Determine API type (internal vs external)
                    api_type = determine_api_type(endpoint)

                    # Calculate expected response time threshold based on API type
                    threshold = calculate_threshold(api_type, endpoint)

                    # Build API configuration
                    api_config = {
                        'name': api_name,
                        'api_type': api_type,
                        'primary_endpoint': endpoint,
                        'backup_endpoint': '',
                        'method': method,
                        'auth': auth_config,
                        'expected_status_code': 200,
                        'performance_threshold_ms': threshold
                    }

                    # Add request body if present
                    if body_data:
                        api_config['data'] = body_data

                    apis.append(api_config)

                elif 'item' in item and item['item']:
                    # Recursive processing for folders
                    folder_name_new = f"{folder_name}/{item['name']}" if folder_name else item['name']
                    extract_requests(item['item'], folder_name_new)

        # Start extraction from root items
        if 'item' in collection_data and collection_data['item']:
            extract_requests(collection_data['item'])

        return apis

    except Exception as e:
        st.error(f"Failed to parse Postman collection: {str(e)}")
        print(f"DEBUG - Parse error: {e}")  # For debugging
        return []

def parse_authentication(request):
    """
    Advanced authentication parser for multiple auth types
    """
    auth_config = {'method': 'none'}

    # Check auth object first
    auth = request.get('auth', {})
    if not auth:
        return auth_config

    auth_type = auth.get('type', 'none')

    if auth_type == 'bearer':
        bearer_data = auth.get('bearer', [])
        token = None
        for bearer_item in bearer_data:
            if isinstance(bearer_item, dict) and bearer_item.get('key') == 'token':
                token = bearer_item.get('value', '')
                if token:
                    break
        if token:
            auth_config = {
                'method': 'bearer',
                'key_value': token
            }

    elif auth_type == 'apikey':
        apikey_data = auth.get('apikey', [])
        key_name = None
        key_value = None
        for apikey_item in apikey_data:
            if isinstance(apikey_item, dict):
                if apikey_item.get('key') == 'key':
                    key_name = apikey_item.get('value', '')
                elif apikey_item.get('key') == 'value':
                    key_value = apikey_item.get('value', '')

        if key_name and key_value:
            auth_config = {
                'method': 'api_key',
                'key_name': key_name,
                'key_value': key_value
            }

    # Fallback: check headers for auth information
    if auth_config['method'] == 'none':
        headers = request.get('header', [])
        for header in headers:
            if isinstance(header, dict) and not header.get('disabled', False):
                key = header.get('key', '').lower()
                value = header.get('value', '')

                if key == 'authorization':
                    if value.lower().startswith('bearer '):
                        auth_config = {
                            'method': 'bearer',
                            'key_value': value[7:].strip()
                        }
                        break
                    elif 'api' in value.lower() or 'token' in value.lower():
                        auth_config = {
                            'method': 'api_key',
                            'key_name': 'Authorization',
                            'key_value': value
                        }
                        break
                elif any(auth_keyword in key for auth_keyword in ['api-key', 'x-api-key', 'token', 'auth']):
                    auth_config = {
                        'method': 'api_key',
                        'key_name': header.get('key', ''),
                        'key_value': value
                    }
                    break

    return auth_config

def determine_api_type(endpoint):
    """
    Determine if API is internal or external based on endpoint
    """
    if any(indicator in endpoint.lower() for indicator in ['localhost', '127.0.0.1', '0.0.0.0', ':8000', ':3000', ':5000']):
        return 'internal'
    else:
        return 'external'

def calculate_threshold(api_type, endpoint):
    """
    Calculate performance threshold based on API characteristics
    """
    base_threshold = 1000  # 1 second default

    # Adjust based on API type
    if api_type == 'internal':
        base_threshold = 500  # 500ms for internal APIs
    else:
        base_threshold = 2000  # 2 seconds for external APIs

    # Adjust based on endpoint characteristics
    if any(keyword in endpoint.lower() for keyword in ['upload', 'download', 'file', 'large']):
        base_threshold *= 5  # 5x longer for file operations
    elif any(keyword in endpoint.lower() for keyword in ['search', 'query', 'report']):
        base_threshold *= 2  # 2x longer for search operations
    elif any(keyword in endpoint.lower() for keyword in ['status', 'health', 'ping']):
        base_threshold = min(base_threshold, 200)  # Very fast for health checks

    return base_threshold

def import_from_postman(collection_data, import_all=True, selected_indices=None):
    """
    Import APIs from parsed Postman collection
    """
    parsed_apis = parse_postman_collection(collection_data)

    if not parsed_apis:
        return []

    if import_all:
        return parsed_apis
    elif selected_indices:
        return [parsed_apis[i] for i in selected_indices if i < len(parsed_apis)]
    else:
        return []