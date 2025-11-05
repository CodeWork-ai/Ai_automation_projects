import streamlit as st
import pandas as pd
from datetime import datetime
import base64

def get_status_color(status):
    """Returns color for status display"""
    if status == "OK":
        return "green"
    elif status == "Warning":
        return "orange"
    elif status == "Down":
        return "red"
    elif status == "Calibrating":
        return "blue"
    else:
        return "grey"

def generate_report_data(monitors):
    """Generate comprehensive report data from monitors"""
    report_data = []
    
    for monitor in monitors:
        total_checks = len(monitor.logs)
        error_logs = [log for log in monitor.logs if log['level'] == 'ERROR']
        # A successful check is one that results in an "OK" or "Warning" status log
        success_logs = [log for log in monitor.logs if log['level'] == 'INFO' and ('successful' in log.get('message', '').lower() or 'switched to backup' in log.get('message', '').lower())]
        success_rate = (len(success_logs) / total_checks * 100) if total_checks > 0 else 0
        
        report_data.append({
            'API_Name': monitor.name,
            'API_Type': monitor.api_type,
            'Current_Status': monitor.status,
            'Primary_Endpoint': monitor.primary_endpoint,
            'Backup_Endpoint': monitor.backup_endpoint or 'None',
            'Current_Endpoint': monitor.current_endpoint,
            'Last_Response_Time_ms': monitor.response_time,
            'Success_Rate_Percent': round(success_rate, 2),
            'Total_Checks': total_checks,
            'Error_Count': len(error_logs),
            'Last_Check': monitor.last_check_timestamp.strftime('%Y-%m-%d %H:%M:%S') if monitor.last_check_timestamp else 'Never',
            'Performance_Threshold': monitor.config.get('performance_threshold_ms', 'Not Set'),
            'Authentication_Method': monitor.config.get('auth', {}).get('method', 'none')
        })
    
    return report_data

def create_download_link(df, filename, file_format='csv'):
    """Create download link for reports"""
    if file_format == 'csv':
        data = df.to_csv(index=False)
        mime = 'text/csv'
        ext = 'csv'
    elif file_format == 'json':
        data = df.to_json(orient='records', indent=2)
        mime = 'application/json'
        ext = 'json'
    
    b64 = base64.b64encode(data.encode()).decode()
    return f'<a href="data:{mime};base64,{b64}" download="{filename}.{ext}">📥 Download {ext.upper()} Report</a>'

def show_enhanced_dashboard(monitors):
    st.title("🔍 API Monitor Agent Dashboard")
    
    if not monitors:
        st.info("No APIs configured yet. Use the sidebar to add APIs to monitor.")
        return
    
    # Summary Metrics
    total_apis = len(monitors)
    healthy_apis = len([m for m in monitors if m.status == 'OK'])
    warning_apis = len([m for m in monitors if m.status == 'Warning'])
    down_apis = len([m for m in monitors if m.status == 'Down'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌐 Total APIs", total_apis)
    col2.metric("✅ Healthy", healthy_apis)
    col3.metric("⚠️ Warning", warning_apis)
    col4.metric("❌ Down", down_apis)
    
    st.markdown("---")
    
    # Reports Section
    with st.expander("📊 Reports & Analytics", expanded=False):
        report_data = generate_report_data(monitors)
        df = pd.DataFrame(report_data)
        
        st.dataframe(df, use_container_width=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_monitor_report_{timestamp}"
        
        r_col1, r_col2 = st.columns(2)
        r_col1.markdown(create_download_link(df, filename, 'csv'), unsafe_allow_html=True)
        r_col2.markdown(create_download_link(df, filename, 'json'), unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.subheader("API Status Details")
    
    for i, monitor in enumerate(monitors):
        with st.container(border=True):
            color = get_status_color(monitor.status)
            
            # Header and Main Info
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                st.markdown(f"**{monitor.name}**")
                st.markdown(f"Status: <span style='color:{color}; font-weight:bold'>{monitor.status}</span>", unsafe_allow_html=True)
            with c2:
                st.text(f"Endpoint: {monitor.current_endpoint}")
                st.text(f"Type: {monitor.api_type}")
            with c3:
                st.text(f"Response: {monitor.response_time} ms")
                last_check = monitor.last_check_timestamp.strftime('%H:%M:%S') if monitor.last_check_timestamp else "N/A"
                st.text(f"Last Check: {last_check}")
            
            # Status explanation
            if monitor.status == "OK":
                st.success(f"API responded as expected within the {monitor.config.get('performance_threshold_ms')}ms threshold.")
            elif monitor.status == "Warning":
                st.warning("Performance issues or rate limits detected. Check logs for details.")
            elif monitor.status == "Down":
                latest_error = next((log['message'] for log in monitor.logs if log['level'] == 'ERROR'), "API is not responding correctly.")
                st.error(f"**Failure**: {latest_error}")
            elif monitor.status == "Calibrating":
                st.info("⌛ Calibrating performance threshold... The first check is in progress.")

            # Expander for Logs and cURL command
            with st.expander("Show Details"):
                # cURL Command
                st.markdown("##### cURL Command for Debugging")
                st.code(monitor.generate_curl_command(), language='bash')

                # Logs
                st.markdown("##### Recent Activity")
                if monitor.logs:
                    for log in monitor.logs[:5]:
                        if log['level'] == 'ERROR':
                            st.error(f"🔴 [{log['timestamp']}] {log['message']}")
                        elif log['level'] == 'WARN':
                            st.warning(f"🟡 [{log['timestamp']}] {log['message']}")
                        else:
                            st.info(f"🟢 [{log['timestamp']}] {log['message']}")
                else:
                    st.info("No logs available yet.")