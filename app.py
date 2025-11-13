import streamlit as st
import json
from datetime import datetime
from io import BytesIO
import re
from transformers import pipeline
import torch
from typing import Dict, List, Optional
import pandas as pd
from langdetect import detect, LangDetectException
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Police Recognition Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DARK THEME CSS with Perfect Text Visibility
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Root and body styling */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    /* Info boxes - Dark theme */
    .info-box {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border-left: 6px solid #3b82f6;
        margin: 15px 0;
        color: #e0e7ff !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }
    
    .info-box h4 {
        color: #93c5fd !important;
        margin-bottom: 10px;
    }
    
    .success-box {
        padding: 18px;
        border-radius: 12px;
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border-left: 6px solid #10b981;
        margin: 15px 0;
        color: #d1fae5 !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    
    .success-box h4 {
        color: #6ee7b7 !important;
    }
    
    .warning-box {
        padding: 18px;
        border-radius: 12px;
        background: linear-gradient(135deg, #92400e 0%, #b45309 100%);
        border-left: 6px solid #f59e0b;
        margin: 15px 0;
        color: #fef3c7 !important;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: #ffffff !important;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h2, .metric-card p {
        color: #ffffff !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #1f2937;
        padding: 10px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 0 25px;
        background-color: #374151;
        border-radius: 10px;
        color: #d1d5db !important;
        font-weight: 600;
        border: 2px solid #4b5563;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff !important;
        border-color: #667eea;
    }
    
    /* TEXT INPUT - CRITICAL FOR VISIBILITY */
    .stTextArea textarea {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
        border: 2px solid #4b5563 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        border-radius: 8px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3) !important;
        background-color: #111827 !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
        opacity: 1 !important;
    }
    
    .stTextInput input {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
        border: 2px solid #4b5563 !important;
        font-size: 16px !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stTextInput input::placeholder {
        color: #9ca3af !important;
    }
    
    /* Labels and headers */
    label, .stMarkdown, h1, h2, h3, h4, h5, h6, p, span, div {
        color: #f3f4f6 !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #e0e7ff !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #f3f4f6 !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        color: #d1d5db !important;
    }
    
    .stRadio [role="radiogroup"] label {
        color: #e5e7eb !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #f3f4f6 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #e0e7ff !important;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #374151 !important;
        color: #f3f4f6 !important;
        border: 2px solid #4b5563 !important;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #1f2937;
        border: 2px dashed #4b5563;
        border-radius: 10px;
        padding: 20px;
    }
    
    [data-testid="stFileUploader"] label {
        color: #d1d5db !important;
    }
    
    /* Buttons */
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        color: #ffffff !important;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #4b5563;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #374151 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #111827;
        border: 1px solid #4b5563;
        border-top: none;
    }
    
    /* Dataframe */
    .dataframe {
        color: #f3f4f6 !important;
        background-color: #1f2937;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #e0e7ff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background-color: #065f46 !important;
        color: #d1fae5 !important;
        border-left: 5px solid #10b981;
    }
    
    .stError {
        background-color: #991b1b !important;
        color: #fecaca !important;
        border-left: 5px solid #ef4444;
    }
    
    .stWarning {
        background-color: #92400e !important;
        color: #fef3c7 !important;
        border-left: 5px solid #f59e0b;
    }
    
    .stInfo {
        background-color: #1e3a8a !important;
        color: #dbeafe !important;
        border-left: 5px solid #3b82f6;
    }
    
    /* Download buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: #ffffff !important;
    }
    
    /* Selectbox and multiselect */
    .stSelectbox, .stMultiSelect {
        color: #f3f4f6 !important;
    }
    
    .stSelectbox > div > div {
        background-color: #1f2937 !important;
        color: #f3f4f6 !important;
        border: 2px solid #4b5563 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Links */
    a {
        color: #93c5fd !important;
    }
    
    a:hover {
        color: #bfdbfe !important;
    }
    
    /* HR divider */
    hr {
        border-color: #4b5563 !important;
    }
    
    /* Container backgrounds */
    .element-container {
        color: #f3f4f6 !important;
    }
    
    /* Chart labels */
    .vega-embed text {
        fill: #d1d5db !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# HARDCODED DATABASE - Police Officers and Departments
OFFICER_DATABASE = {
    "names": [
        "Officer John Smith", "Sergeant Mary Johnson", "Inspector David Brown",
        "Captain Sarah Williams", "Detective Michael Jones", "Officer Emily Davis",
        "Lieutenant Robert Miller", "Chief Patricia Wilson", "Officer James Moore",
        "Constable Jennifer Taylor", "Officer Christopher Anderson", "SI Rajesh Kumar",
        "ASI Priya Sharma", "Inspector Amit Patel", "Constable Sunita Verma",
        "Officer Rahul Singh", "Detective Anita Desai", "Captain Vijay Reddy",
        "SI Lakshmi Iyer", "Officer Arjun Nair", "Inspector Kavita Menon",
        "Constable Ravi Krishnan", "Officer Deepak Gupta", "Sergeant Pooja Rao"
    ],
    "departments": [
        "Central Police Station", "North District Police", "South Precinct",
        "East Division Police", "West Police Department", "Metropolitan Police",
        "City Police Commissionerate", "Traffic Police Department",
        "Crime Investigation Department", "Special Task Force",
        "Cyber Crime Unit", "14th Precinct", "5th District Station",
        "Bangalore City Police", "Mumbai Police", "Delhi Police",
        "Kolkata Police", "Chennai Police", "Bhubaneswar Police",
        "Cuttack Police", "Puri Police Station", "Kendrapara Police"
    ],
    "locations": [
        "Downtown", "Riverside", "Central District", "North Zone",
        "South Sector", "East Block", "West Avenue", "Main Street",
        "Park Avenue", "City Center", "Bangalore", "Mumbai", "Delhi",
        "Kolkata", "Chennai", "Bhubaneswar", "Cuttack", "Puri"
    ]
}

# Language mapping
LANG_CODE_MAP = {
    'hi': 'Hindi', 'bn': 'Bengali', 'te': 'Telugu', 'ta': 'Tamil',
    'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam',
    'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'en': 'English',
    'ur': 'Urdu', 'ne': 'Nepali', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'zh-cn': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'
}

# Cache models
@st.cache_resource(show_spinner=False)
def load_models():
    """Load ML models"""
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=-1
        )
        
        qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1
        )
        
        translation_cache = {}
        
        return sentiment_analyzer, summarizer, translation_cache, qa_model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None, None, None

def detect_language(text: str) -> str:
    """Detect language"""
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text: str, translation_cache: dict) -> tuple:
    """Translate using Helsinki models"""
    try:
        detected_lang = detect_language(text)
        
        if detected_lang == 'en':
            return text, 'en'
        
        helsinki_map = {
            'hi': 'hi', 'bn': 'bn', 'gu': 'gu', 'mr': 'mr',
            'ta': 'ta', 'te': 'te', 'ur': 'ur', 'or': 'or',
            'es': 'es', 'fr': 'fr', 'de': 'de'
        }
        
        src_lang = helsinki_map.get(detected_lang)
        
        if not src_lang:
            return text, detected_lang
        
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
        
        if model_name not in translation_cache:
            try:
                translator = pipeline("translation", model=model_name, device=-1)
                translation_cache[model_name] = translator
            except:
                return text, detected_lang
        else:
            translator = translation_cache[model_name]
        
        result = translator(text[:512], max_length=512)
        translated = result[0]['translation_text']
        
        return translated, detected_lang
        
    except:
        return text, detect_language(text)

def extract_officer_info_from_database(text: str) -> Dict:
    """Extract officers and departments using hardcoded database"""
    text_lower = text.lower()
    
    found_officers = []
    found_departments = []
    found_locations = []
    
    for officer in OFFICER_DATABASE["names"]:
        if officer.lower() in text_lower:
            found_officers.append(officer)
        else:
            last_name = officer.split()[-1]
            if last_name.lower() in text_lower:
                found_officers.append(officer)
    
    for dept in OFFICER_DATABASE["departments"]:
        if dept.lower() in text_lower:
            found_departments.append(dept)
    
    for loc in OFFICER_DATABASE["locations"]:
        if loc.lower() in text_lower:
            found_locations.append(loc)
    
    officer_patterns = [
        r'(?:Officer|Constable|Inspector|Sergeant|Detective|Chief|Captain|Lieutenant|SI|ASI|CI|PSI)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    
    for pattern in officer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            found_officers.append(f"Officer {match}")
    
    if not found_officers:
        found_officers = ["(Officer name not found - please specify)"]
    
    if not found_departments:
        found_departments = ["(Department not specified)"]
    
    return {
        "officers": list(set(found_officers)),
        "departments": list(set(found_departments)),
        "locations": list(set(found_locations))
    }

def analyze_sentiment(text: str, sentiment_analyzer) -> Dict:
    """Sentiment analysis"""
    try:
        result = sentiment_analyzer(text[:512])[0]
        
        if result['label'] == 'POSITIVE':
            sentiment_score = result['score']
        else:
            sentiment_score = -result['score']
        
        return {
            "label": result['label'],
            "score": result['score'],
            "normalized_score": sentiment_score
        }
    except:
        return {"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0}

def extract_competency_tags(text: str) -> List[str]:
    """Extract competency tags"""
    competencies = {
        "community_engagement": ["community", "engagement", "outreach", "relationship", "trust", "friendly", "public"],
        "de-escalation": ["de-escalate", "calm", "peaceful", "resolved", "mediation", "defused", "negotiation"],
        "rapid_response": ["quick", "fast", "immediate", "prompt", "timely", "rapid", "swift", "emergency"],
        "professionalism": ["professional", "courteous", "respectful", "polite", "dignified", "conduct"],
        "life_saving": ["saved", "rescue", "life-saving", "emergency", "critical", "revived", "medical"],
        "investigation": ["investigation", "solved", "detective", "evidence", "arrest", "caught", "crime"],
        "compassion": ["compassion", "care", "kindness", "empathy", "understanding", "helped", "sympathetic"],
        "bravery": ["brave", "courage", "heroic", "danger", "risk", "fearless", "valor", "heroism"]
    }
    
    text_lower = text.lower()
    found_tags = []
    
    for tag, keywords in competencies.items():
        if any(keyword in text_lower for keyword in keywords):
            found_tags.append(tag)
    
    return found_tags if found_tags else ["general_commendation"]

def generate_summary(text: str, summarizer) -> str:
    """Generate comprehensive summary"""
    try:
        if len(text) < 100:
            return text
        
        text_to_summarize = text[:1024]
        summary = summarizer(text_to_summarize, max_length=150, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except:
        sentences = text.split('.')[:3]
        return '. '.join(sentences) + '.'

def calculate_recognition_score(sentiment_score: float, tags: List[str], text_length: int) -> float:
    """Calculate score"""
    base_score = (sentiment_score + 1) / 2
    
    high_value_tags = ["life_saving", "bravery", "de-escalation"]
    tag_boost = sum(0.15 for tag in tags if tag in high_value_tags)
    
    length_boost = min(0.1, text_length / 1000 * 0.1)
    
    final_score = min(1.0, base_score + tag_boost + length_boost)
    return round(final_score, 3)

def create_pdf_summary(result: Dict) -> BytesIO:
    """Create PDF summary report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    story.append(Paragraph("🚔 Police Recognition Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(result['summary'], styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Recognition Score', f"{result['recognition_score']}/1.0"],
        ['Sentiment', result['sentiment_label']],
        ['Language', result['language_name']],
        ['Text Length', f"{result['text_length']} characters"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Identified Officers", heading_style))
    for officer in result['extracted_officers']:
        story.append(Paragraph(f"• {officer}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Departments", heading_style))
    for dept in result['extracted_departments']:
        story.append(Paragraph(f"• {dept}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Competency Tags", heading_style))
    for tag in result['suggested_tags']:
        story.append(Paragraph(f"• {tag.replace('_', ' ').title()}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def process_text(text: str, models_tuple) -> Dict:
    """Main processing pipeline"""
    sentiment_analyzer, summarizer, translator_cache, qa_model = models_tuple
    
    original_text = text
    translated_text, detected_lang = translate_to_english(text, translator_cache)
    processing_text = translated_text if detected_lang != 'en' else original_text
    
    entities = extract_officer_info_from_database(processing_text)
    sentiment = analyze_sentiment(processing_text, sentiment_analyzer)
    tags = extract_competency_tags(processing_text)
    summary = generate_summary(processing_text, summarizer)
    
    score = calculate_recognition_score(
        sentiment['normalized_score'],
        tags,
        len(processing_text)
    )
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "original_text": original_text,
        "detected_language": detected_lang,
        "language_name": LANG_CODE_MAP.get(detected_lang, detected_lang.upper()),
        "translated_text": translated_text if detected_lang != 'en' else None,
        "summary": summary,
        "extracted_officers": entities['officers'],
        "extracted_departments": entities['departments'],
        "extracted_locations": entities['locations'],
        "sentiment_label": sentiment['label'],
        "sentiment_score": sentiment['normalized_score'],
        "suggested_tags": tags,
        "recognition_score": score,
        "text_length": len(processing_text)
    }
    
    return result

def answer_question(question: str, context: str, qa_model) -> str:
    """Q&A"""
    try:
        result = qa_model(question=question, context=context[:2000])
        return result['answer']
    except Exception as e:
        return f"Unable to answer: {str(e)}"

# Main App
def main():
    st.markdown('<h1 class="main-header">🚔 Police Recognition Analytics Platform</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Welcome to the AI-Powered Police Recognition System</h4>
        Analyze public feedback, news articles, and documents to identify and recognize outstanding police work. 
        Features include automatic translation, entity extraction, sentiment analysis, and comprehensive PDF reports.
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading AI models... Please wait..."):
        models = load_models()
    
    if models[0] is None:
        st.error("❌ Failed to load models. Please refresh the page.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/police-badge.png", width=100)
        st.title("📍 Navigation")
        
        st.markdown("---")
        st.subheader("📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 Total", len(st.session_state.processed_data))
        with col2:
            if st.session_state.processed_data:
                avg_score = sum(d['recognition_score'] for d in st.session_state.processed_data) / len(st.session_state.processed_data)
                st.metric("⭐ Avg", f"{avg_score:.2f}")
        
        st.markdown("---")
        st.subheader("🌐 Supported Languages")
        
        st.markdown("""
        **✅ Indic Languages:**
        - Hindi • Odia • Bengali
        - Tamil • Telugu • Marathi
        - Gujarati • Kannada
        
        **✅ European:**
        - Spanish • French • German
        """)
        
        st.markdown("---")
        st.subheader("👮 Officer Database")
        st.info(f"**{len(OFFICER_DATABASE['names'])}** officers  \n**{len(OFFICER_DATABASE['departments'])}** departments")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.processed_data = []
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Process Feedback", "📊 Dashboard", "💬 Q&A Chat", "📈 Export Data"])
    
    with tab1:
        st.header("📝 Process New Feedback")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_method = st.radio(
                "📥 Select Input Method:",
                ["✍️ Text Input", "📄 File Upload"],
                horizontal=True
            )
            
            text_to_process = ""
            
            if input_method == "✍️ Text Input":
                text_to_process = st.text_area(
                    "Enter feedback, article, or document (any language):",
                    height=250,
                    placeholder="Example:\nOfficer Smith from Central Police Station showed exceptional bravery...\n\nया Odia में: ଅଫିସର ସ୍ମିଥ ଅସାଧାରଣ ସାହସ ଦେଖାଇଥିଲେ...",
                    key="main_text_input"
                )
            else:
                uploaded_file = st.file_uploader(
                    "📤 Upload Document (TXT or PDF)",
                    type=['txt', 'pdf']
                )
                
                if uploaded_file:
                    try:
                        if uploaded_file.type == "text/plain":
                            text_to_process = uploaded_file.getvalue().decode("utf-8")
                            st.success(f"✅ Loaded {len(text_to_process)} characters")
                        elif uploaded_file.type == "application/pdf":
                            import pdfplumber
                            with pdfplumber.open(uploaded_file) as pdf:
                                text_to_process = ""
                                for page in pdf.pages:
                                    text_to_process += page.extract_text() or ""
                            st.success(f"✅ Extracted {len(text_to_process)} characters from PDF")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                    
                    if text_to_process:
                        with st.expander("📄 Preview"):
                            st.text_area("Content:", text_to_process[:500] + "...", height=150, key="preview", disabled=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🌍 Multi-Language</h4>
                • Auto-detect language<br>
                • Translate to English<br>
                • 15+ languages
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="success-box">
                <h4>✨ AI Features</h4>
                ✅ Sentiment Analysis<br>
                ✅ Entity Extraction<br>
                ✅ Auto-Summary<br>
                ✅ Competency Tags<br>
                ✅ PDF Reports
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚀 Analyze Feedback", type="primary", use_container_width=True):
            if text_to_process and text_to_process.strip():
                with st.spinner("🔍 Analyzing..."):
                    try:
                        result = process_text(text_to_process, models)
                        st.session_state.processed_data.append(result)
                        
                        st.success("✅ Analysis Complete!")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h2>{result['recognition_score']}</h2>
                                <p>Recognition Score</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            emoji = "😊" if result['sentiment_label'] == 'POSITIVE' else "😐"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h2>{emoji}</h2>
                                <p>{result['sentiment_label']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h2>{len(result['extracted_officers'])}</h2>
                                <p>Officers</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h2>{result['language_name']}</h2>
                                <p>Language</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Details
                        with st.expander("📋 View Details", expanded=True):
                            if result['translated_text']:
                                st.markdown(f"""
                                <div class="warning-box">
                                    <b>🌐 Translation:</b> Original in <b>{result['language_name']}</b>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("📄 Original")
                                    st.text_area("", result['original_text'], height=150, key="orig", disabled=True)
                                with col2:
                                    st.subheader("🔄 English")
                                    st.text_area("", result['translated_text'], height=150, key="trans", disabled=True)
                            
                            st.subheader("📝 Summary")
                            st.markdown(f"""
                            <div class="info-box">
                                {result['summary']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("👮 Officers")
                                for o in result['extracted_officers']:
                                    st.markdown(f"• **{o}**")
                                
                                st.subheader("🏢 Departments")
                                for d in result['extracted_departments']:
                                    st.markdown(f"• **{d}**")
                            
                            with col2:
                                st.subheader("🏷️ Tags")
                                for t in result['suggested_tags']:
                                    st.markdown(f"• **{t.replace('_', ' ').title()}**")
                                
                                if result['extracted_locations']:
                                    st.subheader("📍 Locations")
                                    for l in result['extracted_locations']:
                                        st.markdown(f"• **{l}**")
                        
                        # Export
                        st.markdown("---")
                        st.subheader("📥 Export")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            pdf_buffer = create_pdf_summary(result)
                            st.download_button(
                                "📄 PDF Summary",
                                data=pdf_buffer,
                                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                        with col2:
                            st.download_button(
                                "📋 JSON Data",
                                data=json.dumps(result, indent=2, ensure_ascii=False),
                                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter text")
    
    with tab2:
        st.header("📊 Dashboard")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total", len(df))
            with col2:
                st.metric("⭐ Avg Score", f"{df['recognition_score'].mean():.2f}")
            with col3:
                pos_pct = (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100
                st.metric("😊 Positive", f"{pos_pct:.0f}%")
            with col4:
                total = sum(len(o) for o in df['extracted_officers'])
                st.metric("👮 Officers", total)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Officers")
                all_officers = [o for officers in df['extracted_officers'] for o in officers if not o.startswith("(")]
                
                if all_officers:
                    st.bar_chart(pd.Series(all_officers).value_counts().head(10))
                else:
                    st.info("No officers yet")
            
            with col2:
                st.subheader("📊 Tags")
                all_tags = [t for tags in df['suggested_tags'] for t in tags]
                
                if all_tags:
                    st.bar_chart(pd.Series(all_tags).value_counts())
            
            st.markdown("---")
            st.subheader("📜 Recent")
            
            for idx in range(min(5, len(df))):
                row = df.iloc[-(idx+1)]
                with st.expander(f"#{len(df)-idx} | Score: {row['recognition_score']} | {row['language_name']}"):
                    st.markdown(f"**📝** {row['summary']}")
                    st.markdown(f"**👮** {', '.join(row['extracted_officers'])}")
                    st.markdown(f"**🏷️** {', '.join([t.replace('_', ' ').title() for t in row['suggested_tags']])}")
        else:
            st.info("No data yet")
    
    with tab3:
        st.header("💬 Q&A Chat")
        
        if st.session_state.processed_data:
            st.markdown("""
            <div class="info-box">
                Ask questions about analyzed feedback
            </div>
            """, unsafe_allow_html=True)
            
            all_texts = " ".join([
                d['translated_text'] if d['translated_text'] else d['original_text'] 
                for d in st.session_state.processed_data
            ])
            
            question = st.text_input(
                "💭 Ask:",
                placeholder="What acts of bravery were mentioned?",
                key="qa_q"
            )
            
            if st.button("🔍 Get Answer", type="primary"):
                if question:
                    with st.spinner("🤔 Searching..."):
                        answer = answer_question(question, all_texts[:2000], models[3])
                        st.session_state.chat_history.append({"q": question, "a": answer})
                        st.rerun()
            
            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("💬 History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                    st.markdown(f"""
                    <div class="info-box">
                        <b>❓ Q{len(st.session_state.chat_history) - i + 1}:</b> {chat['q']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <b>✅ A:</b> {chat['a']}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("---")
        else:
            st.warning("Process feedback first")
    
    with tab4:
        st.header("📈 Export Data")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            st.subheader("📊 Data Table")
            
            all_cols = df.columns.tolist()
            default_cols = [c for c in ['timestamp', 'recognition_score', 'sentiment_label', 'language_name', 'extracted_officers'] if c in all_cols]
            
            selected = st.multiselect("Columns:", all_cols, default=default_cols)
            
            if selected:
                st.dataframe(df[selected], use_container_width=True, height=400)
            
            st.markdown("---")
            st.subheader("📥 Bulk Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📄 CSV",
                    data=csv_data,
                    file_name=f"data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_data = df.to_json(orient='records', indent=2, force_ascii=False)
                st.download_button(
                    "📋 JSON",
                    data=json_data,
                    file_name=f"data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("No data available")

if __name__ == "__main__":
    main()
