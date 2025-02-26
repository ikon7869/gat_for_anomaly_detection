import streamlit as st
import requests

st.title("Manufacturing Anomaly Detection")

st.write("Enter machine details for a specific timestamp:")

col1, col2, col3 = st.columns(3)
with col1:
    m1_status = st.selectbox("Machine 1 Status", ["Running", "Stopped"])
    m1_worker = st.number_input("Machine 1 Worker Count", min_value=0, value=0)
with col2:
    m2_status = st.selectbox("Machine 2 Status", ["Running", "Stopped"])
    m2_worker = st.number_input("Machine 2 Worker Count", min_value=0, value=0)
with col3:
    m3_status = st.selectbox("Machine 3 Status", ["Running", "Stopped"])
    m3_worker = st.number_input("Machine 3 Worker Count", min_value=0, value=0)

if st.button("Check for Anomaly"):
    status_map = {"Running": 1, "Stopped": 0}
    payload = {
        "M1_Status": status_map[m1_status],
        "M1_Worker_Count": m1_worker,
        "M2_Status": status_map[m2_status],
        "M2_Worker_Count": m2_worker,
        "M3_Status": status_map[m3_status],
        "M3_Worker_Count": m3_worker,
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.write("Reconstruction Error:", result["reconstruction_error"])
            if result["anomaly"]:
                st.error("Anomaly detected!")
            else:
                st.success("No anomaly detected.")
        else:
            st.error("Error in prediction API: " + str(response.status_code))
    except Exception as e:
        st.error(f"Error connecting to the API: {e}")
