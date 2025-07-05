import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import os
import nltk
import pandas as pd
import plotly.express as px
from gtts import gTTS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import aiohttp
import asyncio

# DeepL API configuration
#DEEPL_API_KEY = os.getenv("DEEPL_API_KEY") # Provided DeepL API key
DEEPL_API_KEY = st.secrets["DEEPL_API_KEY"]

DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

# Translation function
async def translate_text_async(text: str, target_lang: str) -> str:
    """
    Translate text using the DeepL API asynchronously.
    """
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "text": [text],
            "target_lang": target_lang,
            "source_lang": "EN"  # Assuming input is English
        }
        try:
            async with session.post(DEEPL_API_URL, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["translations"][0]["text"]
                elif response.status == 429:
                    raise Exception("Rate limit exceeded. Please try again later.")
                elif response.status == 403:
                    raise Exception("Invalid API key. Please check your DeepL API key.")
                else:
                    raise Exception(f"DeepL API error: {response.status} - {await response.text()}")
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
            return text  # Fallback to original text

# Synchronous wrapper for the async translation function
def translate_text(text, target_lang):
    """
    Wrapper for the async translation function that works in Streamlit.
    """
    # Map language codes to DeepL format
    deepl_lang_map = {
        "en": "EN",
        "ar": "AR",
        "ur": None,  # Urdu not supported by DeepL
        "ko": "KO",
        "fr": "FR",
        "es": "ES"
    }
    
    # Get the DeepL language code
    deepl_lang = deepl_lang_map.get(target_lang, "EN")
    
    # Check if language is supported
    if deepl_lang is None:
        raise Exception(f"Translation to {target_lang} is not supported by DeepL API")
    
    try:
        # Run the async function in the current event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        translated_text = loop.run_until_complete(translate_text_async(text, deepl_lang))
        loop.close()
        return translated_text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Define model architecture
class GenreClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GenreClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Load resources
@st.cache_resource
def load_model():
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    
    input_dim = tfidf.max_features
    output_dim = len(mlb.classes_)
    model = GenreClassifier(input_dim, output_dim)
    state_dict = torch.load('model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model, tfidf, mlb

# Load evaluation metrics
def load_metrics():
    metrics = {
        'overall': {
            'Hamming Loss': 0.1103,
            'Subset Accuracy': 0.1399,
            'F1 Score (Micro)': 0.6077,
            'F1 Score (Macro)': 0.5551,
            'Precision (Micro)': 0.5400,
            'Precision (Macro)': 0.5195,
            'Recall (Micro)': 0.6948,
            'Recall (Macro)': 0.6097
        },
        'by_genre': {
            'Action': {'precision': 0.49, 'recall': 0.77, 'f1-score': 0.60, 'support': 1388},
            'Adventure': {'precision': 0.48, 'recall': 0.65, 'f1-score': 0.55, 'support': 1268},
            'Animation': {'precision': 0.62, 'recall': 0.64, 'f1-score': 0.63, 'support': 504},
            'Comedy': {'precision': 0.53, 'recall': 0.75, 'f1-score': 0.63, 'support': 2495},
            'Crime': {'precision': 0.51, 'recall': 0.58, 'f1-score': 0.55, 'support': 1060},
            'Documentary': {'precision': 0.62, 'recall': 0.65, 'f1-score': 0.64, 'support': 254},
            'Drama': {'precision': 0.66, 'recall': 0.86, 'f1-score': 0.75, 'support': 4438},
            'Family': {'precision': 0.50, 'recall': 0.51, 'f1-score': 0.50, 'support': 750},
            'Fantasy': {'precision': 0.36, 'recall': 0.53, 'f1-score': 0.43, 'support': 433},
            'Historical': {'precision': 0.25, 'recall': 0.36, 'f1-score': 0.30, 'support': 168},
            'Horror': {'precision': 0.71, 'recall': 0.70, 'f1-score': 0.71, 'support': 815},
            'Musical': {'precision': 0.31, 'recall': 0.34, 'f1-score': 0.33, 'support': 488},
            'Mystery': {'precision': 0.32, 'recall': 0.45, 'f1-score': 0.37, 'support': 428},
            'Romance': {'precision': 0.49, 'recall': 0.62, 'f1-score': 0.55, 'support': 1506},
            'Science Fiction': {'precision': 0.59, 'recall': 0.68, 'f1-score': 0.63, 'support': 467},
            'Short Film': {'precision': 0.72, 'recall': 0.57, 'f1-score': 0.63, 'support': 635},
            'Thriller': {'precision': 0.44, 'recall': 0.68, 'f1-score': 0.53, 'support': 1372},
            'Western': {'precision': 0.76, 'recall': 0.62, 'f1-score': 0.68, 'support': 213}
        },
        'averages': {
            'micro avg': {'precision': 0.54, 'recall': 0.69, 'f1-score': 0.61, 'support': 18682},
            'macro avg': {'precision': 0.52, 'recall': 0.61, 'f1-score': 0.56, 'support': 18682},
            'weighted avg': {'precision': 0.55, 'recall': 0.69, 'f1-score': 0.61, 'support': 18682},
            'samples avg': {'precision': 0.56, 'recall': 0.70, 'f1-score': 0.58, 'support': 18682}
        }
    }
    return metrics

# Load confusion matrices
def get_confusion_matrix_images():
    genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Family', 'Fantasy', 'Historical', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Science_Fiction', 'Short_Film', 'Thriller', 'Western'
    ]
    return {genre: f'confusion_matrix/confusion_matrix_{genre}.png' for genre in genres}

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""
if 'plot_input' not in st.session_state:
    st.session_state.plot_input = ""

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Genre Predictor", page_icon="🎬", layout="wide")

# Custom CSS with orange-black gradient theme
st.markdown("""
    <style>
    /* Main background with gradient */
    .main { 
        background: linear-gradient(135deg, #0F0F0F 0%, #1A1A1A 100%);
        color: #F0F0F0; 
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(90deg, #FF8C00, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-shadow: 0 2px 10px rgba(255, 140, 0, 0.2);
        margin-bottom: 30px;
    }
    
    h2, h3, .subheader {
        color: #FF8C00;
        font-weight: 600;
    }
    
    /* Card container for sections */
    .card {
        background: linear-gradient(135deg, rgba(40, 40, 40, 0.9) 0%, rgba(28, 28, 28, 0.8) 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 140, 0, 0.1);
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #FF8C00, #FF4500);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255, 140, 0, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #FF4500, #FF8C00);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(255, 140, 0, 0.4);
    }
    .stButton>button:active {
        transform: translateY(1px);
    }
    
    /* Form input styling */
    .stTextArea textarea {
        background-color: rgba(30, 30, 30, 0.8);
        color: #F0F0F0;
        border: 1px solid rgba(255, 140, 0, 0.3);
        border-radius: 12px;
        padding: 15px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #FF8C00;
        box-shadow: 0 0 0 2px rgba(255, 140, 0, 0.2);
    }
    
    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(30, 30, 30, 0.8);
        border: 1px solid rgba(255, 140, 0, 0.3);
        border-radius: 12px;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover {
        border-color: #FF8C00;
    }
    .stSelectbox div[data-baseweb="select"] > div > div {
        color: #F0F0F0;
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        background: linear-gradient(90deg, #FF8C00, #FF4500);
        border: none;
    }
    .stSlider [data-baseweb="slider"] [data-testid="stTrack"] > div {
        background-image: linear-gradient(90deg, #FF8C00, #FF4500);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1A1A1A 0%, #0F0F0F 100%);
        border-right: 1px solid rgba(255, 140, 0, 0.2);
    }
    
    /* Radio buttons in sidebar */
    .sidebar .sidebar-content .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    .sidebar .sidebar-content .stRadio label {
        background: rgba(40, 40, 40, 0.6);
        border-left: 4px solid transparent;
        border-radius: 8px;
        padding: 12px 15px;
        transition: all 0.2s ease;
        color: #F0F0F0;
        margin-bottom: 8px;
    }
    .sidebar .sidebar-content .stRadio label:hover {
        background: rgba(50, 50, 50, 0.8);
        border-left: 4px solid rgba(255, 140, 0, 0.5);
        color: #FF8C00;
    }
    .sidebar .sidebar-content .stRadio input[type='radio']:checked + div {
        background: rgba(50, 50, 50, 0.8);
        border-left: 4px solid #FF8C00;
        color: #FF8C00;
        font-weight: bold;
    }
    
    /* Tables */
    .stTable table {
        background: rgba(30, 30, 30, 0.7);
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 140, 0, 0.2);
    }
    .stTable thead tr th {
        background-color: rgba(255, 140, 0, 0.15);
        color: #FF8C00;
        text-transform: uppercase;
        font-size: 14px;
        font-weight: 600;
        padding: 12px 15px;
    }
    .stTable tbody tr:nth-child(odd) {
        background-color: rgba(40, 40, 40, 0.5);
    }
    .stTable tbody tr:nth-child(even) {
        background-color: rgba(30, 30, 30, 0.5);
    }
    .stTable tbody tr:hover {
        background-color: rgba(255, 140, 0, 0.1);
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, rgba(40, 40, 40, 0.7) 0%, rgba(28, 28, 28, 0.6) 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 140, 0, 0.15);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: #F0F0F0;
        font-size: 16px;
        font-weight: 500;
    }
    .stMetric .stMetricValue {
        background: linear-gradient(90deg, #FF8C00, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 32px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(40, 40, 40, 0.7);
        border-radius: 8px;
        border: 1px solid rgba(255, 140, 0, 0.2);
        color: #F0F0F0;
        font-weight: 500;
        padding: 10px 15px;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(50, 50, 50, 0.8);
        border-color: rgba(255, 140, 0, 0.4);
        color: #FF8C00;
    }
    .streamlit-expanderContent {
        background: rgba(25, 25, 25, 0.7);
        border: 1px solid rgba(255, 140, 0, 0.1);
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 15px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-image: linear-gradient(90deg, #FF8C00, #FF4500);
    }
    
    /* Spinner */
    .stSpinner > div > div {
        border-top-color: #FF8C00 !important;
    }
    
    /* Alerts and messages */
    .stAlert {
        background: rgba(30, 30, 30, 0.7);
        border-radius: 10px;
        border-left: 5px solid;
    }
    .stAlert.success {
        border-left-color: #FF8C00;
    }
    .stAlert.info {
        border-left-color: #1E90FF;
    }
    .stAlert.warning {
        border-left-color: #FFD700;
    }
    .stAlert.error {
        border-left-color: #FF4500;
    }
    
    /* Custom classes for sections */
    .section-title {
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid rgba(255, 140, 0, 0.3);
        padding-bottom: 10px;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #1E90FF, #00BFFF);
        color: white;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #00BFFF, #1E90FF);
    }
    
    /* Audio player */
    audio {
        width: 100%;
        border-radius: 30px;
        background: rgba(30, 30, 30, 0.8);
    }
    </style>
""", unsafe_allow_html=True)

# Custom container function
def custom_container(title="", content_function=None):
    st.markdown(f"<div class='section-title'><h3>{title}</h3></div>", unsafe_allow_html=True)
    
    # Begin card container
    
    
    # Execute content function if provided
    if content_function:
        result = content_function()
    
    # End card container
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Return result from content function if it returned something
    if content_function and 'result' in locals():
        return result

# Sidebar navigation
with st.sidebar:
    st.markdown("<h2 style='text-align:center; margin-bottom:30px;'>Navigation</h2>", unsafe_allow_html=True)
    option = st.radio(
        label="Choose a section:",  # Added proper label
        options=["Input Summary", "Predict Genre", "Convert Summary to Audio", "Model Evaluation", "Confusion Matrices", "View History", "Provide Feedback"],
        format_func=lambda x: x.replace("_", " ").title(),
        label_visibility="collapsed"  # Hide the label since we have the h2 above
    )
    
    # Logo or branding
    st.markdown("<div style='text-align:center; margin-top:50px;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#FF8C00;'>🎬 Movie Genius</h4>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px;'>Powered by AI</p>", unsafe_allow_html=True)
    
    
    if option == "View History" and st.button("Clear History"):
        st.session_state.prediction_history = []
        st.success("History cleared!")

# App header
st.markdown("<h1 style='text-align:center;'>🎬 Movie Genre Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; margin-bottom:30px;'>Discover the genres of any movie based on its plot summary. Get predictions, listen to audio summaries, and more with our AI-powered tool.</p>", unsafe_allow_html=True)

# Model Highlights Card
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#FF8C00; margin-top:0;'>Model Highlights</h3><div style='display:flex; flex-wrap:wrap; gap:20px;'><div style='flex:1; min-width:200px; background:rgba(255, 140, 0, 0.1); padding:15px; border-radius:12px; border-left:4px solid #FF8C00;'><h4 style='margin:0; color:#FF8C00;'>High Accuracy</h4><p style='margin:5px 0 0 0; font-size:14px;'>Achieves 60.77% F1 Score (Micro) across 18 genres.</p></div><div style='flex:1; min-width:200px; background:rgba(255, 140, 0, 0.1); padding:15px; border-radius:12px; border-left:4px solid #FF8C00;'><h4 style='margin:0; color:#FF8C00;'>Multi-Genre Support</h4><p style='margin:5px 0 0 0; font-size:14px;'>Predicts multiple genres simultaneously with confidence scores.</p></div><div style='flex:1; min-width:200px; background:rgba(255, 140, 0, 0.1); padding:15px; border-radius:12px; border-left:4px solid #FF8C00;'><h4 style='margin:0; color:#FF8C00;'>Robust NLP</h4><p style='margin:5px 0 0 0; font-size:14px;'>Uses TF-IDF and neural networks for precise text analysis.</p></div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Main content area
def main_content():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Example summaries dropdown
        examples = {
            "": "",
            "The Shawshank Redemption": "Over the course of several years, two convicts form a friendship, seeking consolation and, eventually, redemption through basic compassion.",
            "The Dark Knight": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
            "Titanic": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
            "Toy Story": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
            "The Godfather": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."
        }
        
        selected_example = st.selectbox("Select an example plot summary:", list(examples.keys()))
        
        # Movie plot summary input - use the example text as default if selected
        default_text = examples[selected_example] if selected_example else ""
        user_input = st.text_area("Enter the Movie Plot Summary", 
                                value=default_text,
                                height=200, 
                                placeholder="Type or paste your movie plot here...",
                                key="plot_input")
        
        if st.button("✨ Process Summary"):
            if user_input.strip():
                if 'original_text' not in st.session_state:
                    st.session_state.original_text = ""
                st.session_state.original_text = user_input
                st.success("Summary processed successfully! Navigate to other sections to analyze the plot.")
            else:
                st.error("Please enter a plot summary first.")
    
    with col2:
        # Quick stats in cards
        st.markdown("<div style='padding:20px 0;'></div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Predictions", len(st.session_state.prediction_history))
        with col_b:
            st.metric("Supported Languages", 6)
            
        # Additional fun stat or feature highlight
        st.markdown("<div style='padding:10px 0;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='background: rgba(255, 140, 0, 0.1); padding:15px; border-radius:12px; border-left:4px solid #FF8C00;'><h4 style='margin:0; color:#FF8C00;'>Did you know?</h4><p style='margin:5px 0 0 0; font-size:14px;'>Our AI can identify over 20 different movie genres!</p></div>", unsafe_allow_html=True)
    
    return user_input

# Functionality based on user choice
if option == "Input Summary":
    custom_container("Enter Plot Summary", main_content)

elif option == "Predict Genre":
    custom_container("Predict Movie Genres", lambda: None)
    
    if st.session_state.original_text:
        user_input = st.session_state.original_text
    else:
        st.warning("No plot summary found. Please enter one below or go to the Input Summary section.")
        user_input = main_content()
    
    st.write("Enter a plot summary to see predicted genres and their confidence scores.")
    
    if st.button("🔮 Predict Genres") and user_input.strip():
        progress_bar = st.progress(0)
        progress_bar.progress(10)
        
        with st.spinner("Loading model..."):
            model, tfidf, mlb = load_model()
            progress_bar.progress(50)

        with st.spinner("Analyzing plot and predicting genres..."):
            cleaned_input = preprocess_text(user_input)
            X_input = tfidf.transform([cleaned_input]).toarray()
            X_tensor = torch.FloatTensor(X_input)
            
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.sigmoid(logits).numpy()[0]
                predicted_indices = (probs > 0.5).nonzero()[0]
                predicted_genres = mlb.classes_[predicted_indices]
                confidence_scores = probs[predicted_indices]
            
            progress_bar.progress(100)

        if len(predicted_genres) == 0:
            st.warning("⚠️ No genres predicted with high confidence. Showing top 3 possibilities:")
            top_indices = probs.argsort()[-3:][::-1]
            predicted_genres = mlb.classes_[top_indices]
            confidence_scores = probs[top_indices]
        
        st.success("✅ Predicted Genres:")
        
        # Genre badges
        genres_html = "".join([
            f"""<span style="display:inline-block; background:linear-gradient(90deg, #FF8C00, #FF4500);
            padding:8px 15px; margin:5px; border-radius:20px; color:white; font-weight:600;">
            {genre}</span>"""
            for genre in predicted_genres
        ])
        st.markdown(f"<div style='margin:15px 0;'>{genres_html}</div>", unsafe_allow_html=True)
        
        # Display confidence scores
        st.subheader("Confidence Scores")
        score_data = pd.DataFrame({
            "Genre": predicted_genres,
            "Confidence": confidence_scores
        })
        score_data["Confidence"] = score_data["Confidence"].apply(lambda x: f"{x:.2%}")
        st.table(score_data)

        # Visualize confidence scores with custom colors
        fig = px.bar(
            score_data, 
            x="Genre", 
            y=[float(x.strip('%'))/100 for x in score_data["Confidence"]], 
            title="Genre Prediction Confidence",
            labels={"y": "Confidence"},
            color_discrete_sequence=px.colors.sequential.Oranges_r
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F0F0F0'),
            title_font=dict(color='#FF8C00', size=20),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)', 
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)', 
                zerolinecolor='rgba(255,255,255,0.1)',
                tickformat='.0%'
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Save to history
        st.session_state.prediction_history.append({
            "summary": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "genres": predicted_genres.tolist(),
            "confidence": confidence_scores.tolist()
        })

elif option == "Convert Summary to Audio":
    custom_container("Convert Summary to Audio", lambda: None)
    
    if st.session_state.original_text:
        user_input = st.session_state.original_text
    else:
        st.warning("No plot summary found. Please enter one below or go to the Input Summary section.")
        user_input = main_content()
    
    st.write("Translate and convert your plot summary to audio in your preferred language.")
    
    # Language options with flags and DeepL support
    language_options = {
        "English": {"code": "en", "deepl_code": "EN", "note": "US English", "flag": "🇺🇸"},
        "Arabic": {"code": "ar", "deepl_code": "AR", "note": "Modern Standard Arabic", "flag": "🇦🇪"},
        "Urdu": {"code": "ur", "deepl_code": None, "note": "Audio only (translation not available)", "flag": "🇵🇰"},
        "Korean": {"code": "ko", "deepl_code": "KO", "note": "Clear Hangul pronunciation", "flag": "🇰🇷"},
        "French": {"code": "fr", "deepl_code": "FR", "note": "Parisian French accent", "flag": "🇫🇷"},
        "Spanish": {"code": "es", "deepl_code": "ES", "note": "Neutral Latin American Spanish", "flag": "🇪🇸"}
    }

    # Format function for selectbox
    def format_language(lang):
        return f"{language_options[lang]['flag']} {lang} ({language_options[lang]['note']})"

    selected_lang = st.selectbox(
        "Choose Language",
        options=list(language_options.keys()),
        format_func=format_language
    )

    # Translation toggle
    enable_translation = st.checkbox("🌐 Enable Translation", value=True, 
                                   help="Translate the text before converting to audio")

    # Audio controls with better layout
    col1, col2 = st.columns(2)
    with col1:
        speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
    with col2:
        volume = st.slider("Volume", 0.1, 1.0, 0.5, 0.1)

    # Display original text
    if user_input.strip():
        st.markdown("<h4>Original Text (English):</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:rgba(30, 30, 30, 0.8); padding:15px; border-radius:12px; border:1px solid rgba(255, 140, 0, 0.3);'>{user_input}</div>", unsafe_allow_html=True)

    # Translation and audio generation
    if st.button("🔊 Generate Audio") and user_input.strip():
        lang_info = language_options[selected_lang]
        
        # Step 1: Translation (if enabled)
        text_for_audio = user_input
        if enable_translation and selected_lang != "English":
            # Check if translation is supported for this language
            if lang_info["deepl_code"] is None:
                st.warning(f"⚠️ Translation to {selected_lang} is not available. Audio will be generated from the original English text with {selected_lang} pronunciation.")
                text_for_audio = user_input
            else:
                with st.spinner(f"Translating to {selected_lang}..."):
                    try:
                        translated_text = translate_text(user_input, lang_info["code"])
                        text_for_audio = translated_text
                        
                        # Display translated text
                        st.markdown(f"<h4>Translated Text ({selected_lang}):</h4>", unsafe_allow_html=True)
                        st.markdown(f"<div style='background:rgba(40, 40, 40, 0.9); padding:15px; border-radius:12px; border:1px solid rgba(0, 255, 0, 0.3);'>{translated_text}</div>", unsafe_allow_html=True)
                        st.success("✅ Translation completed!")
                    except Exception as e:
                        st.error(f"Translation failed: {str(e)}")
                        st.info("Using original English text for audio generation.")
                        text_for_audio = user_input
        
        # Step 2: Audio generation
        with st.spinner(f"Generating {selected_lang} audio..."):
            try:
                audio_file = "summary_audio.mp3"
                tts = gTTS(text=text_for_audio, lang=lang_info["code"], slow=(speed < 1.0))
                tts.save(audio_file)

                # Audio player with custom styling
                st.markdown("<h4>Preview Audio</h4>", unsafe_allow_html=True)
                st.audio(audio_file, format="audio/mp3")
                
                # Download option
                with open(audio_file, "rb") as f:
                    st.download_button(
                        label="💾 Download Audio",
                        data=f,
                        file_name=f"movie_summary_{lang_info['code']}.mp3",
                        mime="audio/mpeg"
                    )
                
                # Show what was actually converted
                if enable_translation and selected_lang != "English" and text_for_audio != user_input:
                    st.info(f"🎯 Audio generated from translated {selected_lang} text")
                elif selected_lang == "Urdu":
                    st.info(f"🎯 Audio generated with Urdu pronunciation (translation not available)")
                else:
                    st.info(f"🎯 Audio generated from original English text")
                    
            except Exception as e:
                st.error(f"Audio generation failed: {str(e)}")
                st.info("Try shortening the text or selecting another language.")
            finally:
                # Clean up temporary file
                if 'audio_file' in locals() and os.path.exists(audio_file):
                    os.remove(audio_file)

elif option == "Model Evaluation":
    custom_container("Model Evaluation", lambda: None)
    
    metrics = load_metrics()
    
    st.write("Overall Model Performance Metrics:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("F1 Score (Micro)", f"{metrics['overall']['F1 Score (Micro)']:.4f}")
    with col2:
        st.metric("Precision (Micro)", f"{metrics['overall']['Precision (Micro)']:.4f}")
    with col3:
        st.metric("Recall (Micro)", f"{metrics['overall']['Recall (Micro)']:.4f}")
    
    st.write("Detailed Metrics by Genre:")
    metrics_df = pd.DataFrame(metrics['by_genre']).T
    metrics_df = metrics_df.style.format({
        'precision': '{:.2f}',
        'recall': '{:.2f}',
        'f1-score': '{:.2f}',
        'support': '{:.0f}'
    })
    st.table(metrics_df)
    
    # Visualization with Plotly
    st.write("Genre Performance Visualization:")
    metrics_df = pd.DataFrame(metrics['by_genre']).T.reset_index()
    metrics_df.columns = ['Genre', 'Precision', 'Recall', 'F1-Score', 'Support']
    
    fig = px.bar(
        metrics_df, 
        x='Genre', 
        y=['Precision', 'Recall', 'F1-Score'],
        title='Model Performance by Genre',
        color_discrete_sequence=['#FF8C00', '#FF4500', '#FFA500'],
        barmode='group'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0',
        legend_title_font_color='#FF8C00',
        title_font_color='#FF8C00'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif option == "Confusion Matrices":
    custom_container("Confusion Matrices", lambda: None)
    
    confusion_matrices = get_confusion_matrix_images()
    
    st.write("Explore Confusion Matrices for Each Genre:")
    selected_genre = st.selectbox("Select Genre", list(confusion_matrices.keys()))
    
    if selected_genre:
        image_path = confusion_matrices[selected_genre]
        st.markdown(f"<h4>{selected_genre} Confusion Matrix</h4>", unsafe_allow_html=True)
        try:
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
        except:
            st.error(f"Could not load confusion matrix for {selected_genre}. Please ensure the image exists in the 'confusion_matrix' directory.")

elif option == "View History":
    custom_container("Prediction History", lambda: None)
    
    if st.session_state.prediction_history:
        for i, entry in enumerate(st.session_state.prediction_history):
            with st.expander(f"Prediction {i+1}: {entry['summary']}"):
                # Generate genre badges with proper styling
                genres_html = "".join([
                    f"<span style='display:inline-block; background:linear-gradient(90deg, #FF8C00, #FF4500); "
                    f"padding:5px 10px; margin:3px; border-radius:15px; color:white; font-size:14px;'>"
                    f"{genre}</span>"
                    for genre in entry['genres']
                ])
                st.markdown(f"<div style='margin:10px 0;'>{genres_html}</div>", unsafe_allow_html=True)
                
                # Confidence scores as horizontal bars
                st.write("**Confidence Scores**:")
                for genre, conf in zip(entry['genres'], entry['confidence']):
                    percentage = conf * 100
                    st.markdown(f"""
                    <div style="margin-bottom:8px;">
                        <div style="display:flex; align-items:center;">
                            <div style="width:100px;">{genre}</div>
                            <div style="flex-grow:1; background:rgba(50,50,50,0.5); border-radius:10px; height:15px; position:relative;">
                                <div style="position:absolute; top:0; left:0; width:{percentage}%; 
                                height:100%; background:linear-gradient(90deg, #FF8C00, #FF4500); border-radius:10px;"></div>
                            </div>
                            <div style="width:50px; text-align:right; margin-left:10px;">{percentage:.1f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("📋 No predictions yet. Start by making a prediction!")
        st.markdown("""
        <div style="text-align:center; padding:40px 0;">
            <div style="font-size:60px; margin-bottom:20px;">🎬</div>
            <p>Your prediction history will appear here</p>
        </div>
        """, unsafe_allow_html=True)

elif option == "Provide Feedback":
    custom_container("Provide Feedback", lambda: None)
    
    st.write("Help us improve by sharing your thoughts!")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        with st.form("feedback_form"):
            feedback = st.text_area("Your Feedback", 
                                    placeholder="E.g., 'The model missed Romance genre' or 'The UI is great!'")
            
            rating = st.slider("Rate Your Experience", 1, 5, 4)
            rating_emojis = {1: "😞", 2: "😐", 3: "🙂", 4: "😊", 5: "🤩"}
            
            st.markdown(f"<div style='text-align:center; font-size:30px;'>{rating_emojis[rating]}</div>", 
                      unsafe_allow_html=True)
            
            submitted = st.form_submit_button("📤 Submit Feedback")
            if submitted and feedback.strip():
                # Save feedback to a local file (for simplicity)
                with open("feedback.txt", "a") as f:
                    f.write(f"Rating: {rating}, Feedback: {feedback}\n")
                st.success("Thank you for your feedback! We appreciate your input.")
    
    with col2:
        st.markdown("""
        <div style="background:linear-gradient(135deg, rgba(40, 40, 40, 0.7) 0%, rgba(28, 28, 28, 0.6) 100%); 
             padding:20px; border-radius:12px; border:1px solid rgba(255, 140, 0, 0.2);">
            <h4 style="color:#FF8C00; margin-top:0;">Why Feedback Matters</h4>
            <p style="font-size:14px;">Your feedback helps us improve our genre prediction model and user experience. 
            We review all comments to make our tool better for movie enthusiasts like you!</p>
            <div style="text-align:center; font-size:24px; margin:15px 0;">
                🎯 👍 🚀
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top:60px; text-align:center; padding:20px; font-size:14px; opacity:0.7;">
    <p>🎬 Movie Genre Predictor | Powered by AI Technology</p>
    <p style="font-size:12px;">© 2025 Movie Genius | All rights reserved</p>
</div>
""", unsafe_allow_html=True)
