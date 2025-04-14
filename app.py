import streamlit as st
import fitz  # PyMuPDF
import xgboost as xgb
import numpy as np
from sentence_transformers import SentenceTransformer

# Load BERT model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample data to simulate training
training_data = [
    ("Built 2 AI products, led a team of 4 in a healthtech startup", "Love working in small teams, dream of building a company", 92),
    ("Worked in MNC, entry-level Python skills, learning AI", "Curious about joining early-stage startups", 45),
    ("Designed apps and branding for 3 early-stage companies", "Want to join a product-led startup", 74),
    ("Built EdTech MVP, won hackathons, launched on Play Store", "Startup culture excites me", 89),
    ("Frontend intern, worked on React and UI/UX", "Interested in rapid learning, building impactful solutions", 71),
]

texts = [r + " " + b for r, b, _ in training_data]
scores = [s for _, _, s in training_data]
embeddings = bert_model.encode(texts)

# Train XGBoost regressor
xgb_model = xgb.XGBRegressor()
xgb_model.fit(embeddings, scores)


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="Startup Fit Score Predictor", layout="centered")

st.title("🚀 AI-Powered Startup Fit Score")
st.subheader("Upload your CV + Add your bio to get your personalized startup readiness score!")

uploaded_file = st.file_uploader("Upload your CV (PDF or TXT)", type=['pdf', 'txt'])

bio_text = st.text_area("Tell us about yourself (your startup interest, goals, mindset)", height=150)

if st.button("🔍 Predict Startup Fit Score"):

    if not uploaded_file or not bio_text.strip():
        st.warning("Please upload a resume and enter your bio.")
    else:
        # Extract resume text
        resume_text = ""
        if uploaded_file.name.endswith('.pdf'):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                resume_text += page.get_text()
        elif uploaded_file.name.endswith('.txt'):
            resume_text = uploaded_file.read().decode('utf-8')
        else:
            st.error("Unsupported file type.")
            st.stop()

        # Combine resume + bio
        full_text = resume_text + " " + bio_text
        embedding = bert_model.encode([full_text])
        predicted_score = xgb_model.predict(embedding)[0]

        st.success(f"🎯 Your Predicted Startup Fit Score: **{round(predicted_score, 2)} / 100**")
        st.caption("This score is an estimate of how well your profile fits early-stage startups.")

