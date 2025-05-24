# assignment/api/index.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch
import re

# --- Global Model Loading ---
# This ensures the SBERT model is loaded only once when the serverless function spins up.
# It might cause a 'cold start' delay on the very first request after deployment/inactivity.
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
try:
    model = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"Vercel API: Successfully loaded SentenceTransformer model: {SBERT_MODEL_NAME}")
except Exception as e:
    print(f"Vercel API Error: Failed to load models: {e}")
    raise e # Re-raise to ensure Vercel knows something went wrong on startup

# --- Text Cleaning Function ---
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z0-9\s.,;!?\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Semantic Textual Similarity API (Vercel)",
    description="API to calculate semantic similarity between two paragraphs using Sentence-BERT.",
    version="1.0.0"
)

# --- Define Request and Response Body Models ---
class TextPairRequest(BaseModel):
    text1: str
    text2: str

class SimilarityScoreResponse(Base2Model):
    similarity_score: float

# --- API Endpoint for Similarity Prediction ---
@app.post("/predict_similarity", response_model=SimilarityScoreResponse)
async def predict_similarity(request: TextPairRequest):
    """
    Calculates the semantic similarity score between text1 and text2.
    Returns a score between 0 and 1, where 1 means highly similar and 0 means highly dissimilar.
    """
    cleaned_text1 = clean_text(request.text1)
    cleaned_text2 = clean_text(request.text2)

    embeddings1 = model.encode([cleaned_text1], convert_to_tensor=True)
    embeddings2 = model.encode([cleaned_text2], convert_to_tensor=True)

    # Calculate cosine similarity using the base SBERT approach
    cosine_score = util.cos_sim(embeddings1[0], embeddings2[0]).item()
    final_similarity_score = (cosine_score + 1) / 2

    # --- If you implemented the optional fine-tuned regression model, UNCOMMENT this section: ---
    # IMPORTANT: Ensure your model_assets/ folder is NOT ignored in .gitignore
    #            and that the .joblib files are actually present in that folder.
    # import joblib
    # import numpy as np
    # REGRESSION_MODEL_PATH = './model_assets/sts_regression_model.joblib'
    # SCALER_PATH = './model_assets/scaler.joblib'
    # try:
    #     fine_tuned_regression_model = joblib.load(REGRESSION_MODEL_PATH)
    #     feature_scaler = joblib.load(SCALER_PATH)
    #     print(f"Vercel API: Successfully loaded regression model and scaler.")
    # except Exception as e:
    #     print(f"Vercel API Error: Failed to load regression models: {e}")
    #     raise e

    # X_inference = torch.cat((embeddings1, embeddings2), dim=1).cpu().numpy()
    # X_inference_scaled = feature_scaler.transform(X_inference)
    # predicted_score = fine_tuned_regression_model.predict(X_inference_scaled)[0]
    # final_similarity_score = max(0.0, min(1.0, predicted_score))
    # --- End of regression model logic ---

    return {"similarity_score": final_similarity_score}

# --- Optional: Root endpoint for health check or info ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Semantic Textual Similarity API</title>
        </head>
        <body>
            <h1>Semantic Textual Similarity API</h1>
            <p>Visit /docs for the FastAPI Swagger UI to test the API.</p>
            <p>Endpoint: <code>/predict_similarity</code> (POST)</p>
        </body>
    </html>
    """