import streamlit as st
import requests
import os

# Grab the internal network routing variable mapped by Docker Compose
# Fall back to standard localhost if running outside a container network loop
API_BASE = os.getenv("API_URL", "http://localhost:8000")
API_URL = f"{API_BASE.rstrip('/')}/predict"

# Configure basic browser webpage metadata layout
st.set_page_config(page_title="Clinical ADR Analyzer", page_icon="🩺", layout="centered")

st.title("🩺 Clinical ADR Classification Dashboard")
st.markdown("---")
st.write("Submit a patient medical review text block below to screen for Adverse Drug Reactions (ADRs) using our fine-tuned Clinical BioBERT engine.")

# Render user interface text box
user_review = st.text_area(
    label="Patient Review Text Input:",
    placeholder="Type or paste medical documentation fields here (Max 1000 characters)...",
    height=150
)

# Render submission action trigger button
if st.button("Run BioBERT Inference Analysis", use_container_width=True):
    if not user_review.strip():
        st.warning("Please input a valid textual evaluation phrase before clicking analysis options.")
    else:
        with st.spinner("Streaming data across network container ports to Inference Engine..."):
            try:
                # Wrap text inside a clean JSON schema bundle and shoot it over the network
                payload = {"review": user_review}
                response = requests.post(API_URL, json=payload, timeout=10)
                
                # Handshake Handling 1: Success Gate
                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis Complete! Output Matrices Synced Successfully.")
                    
                    # Display raw results cleanly
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Predicted Class ID", value=result["prediction_class_id"])
                    with col2:
                        st.write("**Class Distribution Probabilities:**")
                        st.json(result["class_probabilities"])
                        
                # Handshake Handling 2: Input Guardrail Gate Blocked (Character Limits/Empty Strings)
                elif response.status_code == 422:
                    st.error("❌ Data Validation Rejected: Input string violates length limits (5-1000 characters).")
                    st.json(response.json()["detail"])
                    
                # Handshake Handling 3: Rate Limiter Guardrail Gate Blocked
                elif response.status_code == 429:
                    st.error("⚠️ Security Gate Triggered: Too many requests submitted. Rate limiter active.")
                    
                else:
                    st.error(f"Backend Server Error. HTTP Status Received: {response.status_code}")
                    st.write(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Network Connection Error: Could not resolve connection to FastAPI container backend engine.")
                st.write(f"Attempted Target Address Endpoint: `{API_URL}`")
