import streamlit as st
import numpy as np
import helper  # your helper.py with query_point_creator and preprocess functions
import pickle

# Load your trained model once
@st.cache_resource(show_spinner=False)
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🔍 Duplicate Question Pairs Detector")
st.markdown(
    """
    Enter two questions below and find out if they are duplicates or not.
    This model uses advanced NLP features and embeddings under the hood.
    """
)

# Text inputs with placeholders for better UX
q1 = st.text_area("Enter Question 1:", placeholder="Type your first question here...", height=100)
q2 = st.text_area("Enter Question 2:", placeholder="Type your second question here...", height=100)

if st.button("Check Duplicate"):
    if not q1.strip() or not q2.strip():
        st.warning("Please enter both questions before checking.")
    else:
        try:
            # Prepare features for the model using your helper function
            query_features = helper.query_point_creator(q1, q2)
            
            # Predict (model expects 2D array)
            prediction = model.predict(query_features)[0]
            
            if prediction == 1:
                st.success("✅ These questions are duplicates!")
            else:
                st.info("❌ These questions are NOT duplicates.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("Developed by Dev Saxena | Powered by XGBoost & NLP")

