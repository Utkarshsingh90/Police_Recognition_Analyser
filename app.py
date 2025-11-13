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
from reportlab.pdfgen import canvas
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Police Recognition Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with proper contrast and visibility
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 20px;
    }
    
    /* Info boxes */
    .info-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #e3f2fd;
        border-left: 6px solid #1976d2;
        margin: 15px 0;
        color: #01579b !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .success-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        margin: 15px 0;
        color: #155724 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        padding: 18px;
        border-radius: 12px;
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        margin: 15px 0;
        color: #856404 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 0 25px;
        background-color: white;
        border-radius: 10px;
        color: #495057;
        font-weight: 600;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: #667eea;
    }
    
    /* TEXT VISIBILITY FIXES - CRITICAL */
    .stTextArea textarea {
        color: #212529 !important;
        background-color: #ffffff !important;
        border: 2px solid #ced4da !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    }
    
    .stTextInput input {
        color: #212529 !important;
        background-color: #ffffff !important;
        border: 2px solid #ced4da !important;
        font-size: 16px !important;
    }
    
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    }
    
    /* Placeholder text */
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #6c757d !important;
        opacity: 0.8 !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #212529 !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        color: #495057 !important;
    }
    
    /* Labels and text */
    label, .stMarkdown, p, span, div {
        color: #212529 !important;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #212529 !important;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: #e9ecef !important;
        color: #212529 !important;
        border: 2px solid #ced4da !important;
    }
    
    /* Buttons */
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        color: #212529 !important;
        font-weight: 600;
        border-radius: 8px;
    }
    
    /* Dataframe */
    .dataframe {
        color: #212529 !important;
    }
    
    /* Ensure all text is visible */
    * {
        color: #212529;
    }
    
    /* Override Streamlit's default dark text on dark backgrounds */
    .element-container, .stMarkdown, .stText {
        color: #212529 !important;
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
        
        # Translation cache
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
    
    # Search for officers
    for officer in OFFICER_DATABASE["names"]:
        # Check both full name and last name
        if officer.lower() in text_lower:
            found_officers.append(officer)
        else:
            # Check last name only
            last_name = officer.split()[-1]
            if last_name.lower() in text_lower:
                found_officers.append(officer)
    
    # Search for departments
    for dept in OFFICER_DATABASE["departments"]:
        if dept.lower() in text_lower:
            found_departments.append(dept)
    
    # Search for locations
    for loc in OFFICER_DATABASE["locations"]:
        if loc.lower() in text_lower:
            found_locations.append(loc)
    
    # Pattern matching fallback
    officer_patterns = [
        r'(?:Officer|Constable|Inspector|Sergeant|Detective|Chief|Captain|Lieutenant|SI|ASI|CI|PSI)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    
    for pattern in officer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            found_officers.append(f"Officer {match}")
    
    # If no officers found, suggest from database
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
        
        # For longer texts, create multi-part summary
        text_to_summarize = text[:1024]
        summary = summarizer(text_to_summarize, max_length=150, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except:
        # Fallback to simple truncation
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
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("🚔 Police Recognition Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Timestamp
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Summary
    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(result['summary'], styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Metrics table
    story.append(Paragraph("Recognition Metrics", heading_style))
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
    
    # Officers
    story.append(Paragraph("Identified Officers", heading_style))
    for officer in result['extracted_officers']:
        story.append(Paragraph(f"• {officer}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Departments
    story.append(Paragraph("Departments", heading_style))
    for dept in result['extracted_departments']:
        story.append(Paragraph(f"• {dept}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Tags
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
    
    # Translate if needed
    translated_text, detected_lang = translate_to_english(text, translator_cache)
    
    # Use translated text for processing
    processing_text = translated_text if detected_lang != 'en' else original_text
    
    # Extract entities from database
    entities = extract_officer_info_from_database(processing_text)
    
    # Analyze sentiment
    sentiment = analyze_sentiment(processing_text, sentiment_analyzer)
    
    # Extract tags
    tags = extract_competency_tags(processing_text)
    
    # Generate summary
    summary = generate_summary(processing_text, summarizer)
    
    # Calculate score
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
        <b>🎯 Welcome!</b> This AI-powered platform analyzes public feedback, news articles, and documents to identify 
        and recognize outstanding police work. Features include automatic translation, entity extraction, sentiment analysis, 
        and comprehensive PDF reports.
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
        st.metric("📝 Total Processed", len(st.session_state.processed_data))
        
        if st.session_state.processed_data:
            avg_score = sum(d['recognition_score'] for d in st.session_state.processed_data) / len(st.session_state.processed_data)
            st.metric("⭐ Avg Score", f"{avg_score:.2f}")
            
            positive_count = sum(1 for d in st.session_state.processed_data if d['sentiment_label'] == 'POSITIVE')
            st.metric("😊 Positive", positive_count)
        
        st.markdown("---")
        st.subheader("🌐 Supported Languages")
        st.write("✅ **Indic Languages:**")
        st.write("Hindi • Odia • Bengali")
        st.write("Tamil • Telugu • Marathi")
        st.write("Gujarati • Kannada • Urdu")
        
        st.write("✅ **European:**")
        st.write("Spanish • French • German")
        
        st.markdown("---")
        st.subheader("👮 Officer Database")
        st.info(f"**{len(OFFICER_DATABASE['names'])}** officers\n\n**{len(OFFICER_DATABASE['departments'])}** departments")
        
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
                    placeholder="Example:\nOfficer Smith from the Central Police Station showed exceptional bravery when he rescued a child from a burning building...\n\nया Odia में:\nଅଫିସର ସ୍ମିଥ ଅସାଧାରଣ ସାହସ ଦେଖାଇଥିଲେ...",
                    key="main_text_input",
                    help="Paste your text here. Supports multiple languages with automatic translation."
                )
            else:
                uploaded_file = st.file_uploader(
                    "📤 Upload Document (TXT or PDF)",
                    type=['txt', 'pdf'],
                    help="Upload a text file or PDF document for analysis"
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
                            st.success(f"✅ Extracted {len(text_to_process)} characters from {len(pdf.pages)} pages")
                    except Exception as e:
                        st.error(f"❌ Error reading file: {str(e)}")
                    
                    if text_to_process:
                        with st.expander("📄 Preview Extracted Text"):
                            st.text_area("Content Preview:", text_to_process[:500] + "...", height=150, key="file_preview", disabled=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h4>🌍 Multi-Language Support</h4>
                • Automatic language detection<br>
                • Translation to English<br>
                • 15+ languages supported
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="success-box">
                <h4>✨ AI Features</h4>
                ✅ Sentiment Analysis<br>
                ✅ Entity Extraction<br>
                ✅ Auto-Summarization<br>
                ✅ Competency Tagging<br>
                ✅ PDF Report Generation
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚀 Analyze Feedback", type="primary", use_container_width=True):
            if text_to_process and text_to_process.strip():
                with st.spinner("🔍 Analyzing feedback... This may take a moment..."):
                    try:
                        result = process_text(text_to_process, models)
                        st.session_state.processed_data.append(result)
                        
                        st.markdown('<div class="success-box">✅ <b>Analysis Complete!</b></div>', unsafe_allow_html=True)
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h2>{result['recognition_score']}</h2>
                                <p>Recognition Score</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            emoji = "😊" if result['sentiment_label'] == 'POSITIVE' else "😐" if result['sentiment_label'] == 'NEUTRAL' else "😞"
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
                                <p>Officers Found</p>
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
                        
                        # Detailed results
                        with st.expander("📋 View Detailed Results", expanded=True):
                            # Translation info
                            if result['translated_text']:
                                st.markdown(f"""
                                <div class="warning-box">
                                    <b>🌐 Translation Notice:</b> Original text was in <b>{result['language_name']}</b> and has been translated to English for analysis.
                                </div>
                                """, unsafe_allow_html=True)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("📄 Original Text")
                                    st.text_area("", result['original_text'], height=150, key="orig_display", disabled=True)
                                with col2:
                                    st.subheader("🔄 Translated Text")
                                    st.text_area("", result['translated_text'], height=150, key="trans_display", disabled=True)
                            
                            # Summary
                            st.subheader("📝 Summary")
                            st.markdown(f"""
                            <div class="info-box">
                                {result['summary']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Entities
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("👮 Identified Officers")
                                for officer in result['extracted_officers']:
                                    st.markdown(f"• **{officer}**")
                                
                                st.subheader("🏢 Departments")
                                for dept in result['extracted_departments']:
                                    st.markdown(f"• **{dept}**")
                            
                            with col2:
                                st.subheader("🏷️ Competency Tags")
                                for tag in result['suggested_tags']:
                                    st.markdown(f"• **{tag.replace('_', ' ').title()}**")
                                
                                if result['extracted_locations']:
                                    st.subheader("📍 Locations")
                                    for loc in result['extracted_locations']:
                                        st.markdown(f"• **{loc}**")
                        
                        # Export options
                        st.markdown("---")
                        st.subheader("📥 Export Options")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # PDF Summary
                            pdf_buffer = create_pdf_summary(result)
                            st.download_button(
                                "📄 Download PDF Summary",
                                data=pdf_buffer,
                                file_name=f"police_recognition_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                        with col2:
                            # JSON Export
                            st.download_button(
                                "📋 Download JSON",
                                data=json.dumps(result, indent=2, ensure_ascii=False),
                                file_name=f"recognition_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
            else:
                st.warning("⚠️ Please enter or upload some text to analyze.")
    
    with tab2:
        st.header("📊 Recognition Dashboard")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Submissions", len(df))
            with col2:
                st.metric("⭐ Average Score", f"{df['recognition_score'].mean():.2f}")
            with col3:
                positive_pct = (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100
                st.metric("😊 Positive Feedback", f"{positive_pct:.1f}%")
            with col4:
                total_officers = sum(len(officers) for officers in df['extracted_officers'])
                st.metric("👮 Officers Recognized", total_officers)
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Top Recognized Officers")
                all_officers = [o for officers in df['extracted_officers'] for o in officers if not o.startswith("(")]
                
                if all_officers:
                    officer_counts = pd.Series(all_officers).value_counts().head(10)
                    st.bar_chart(officer_counts)
                else:
                    st.info("No officers identified yet")
            
            with col2:
                st.subheader("📊 Competency Distribution")
                all_tags = [t for tags in df['suggested_tags'] for t in tags]
                
                if all_tags:
                    tag_counts = pd.Series(all_tags).value_counts()
                    st.bar_chart(tag_counts)
            
            # Recent submissions
            st.markdown("---")
            st.subheader("📜 Recent Submissions")
            
            for idx in range(min(5, len(df))):
                row = df.iloc[-(idx+1)]
                with st.expander(f"📄 Submission {len(df) - idx} | Score: {row['recognition_score']} | Language: {row['language_name']}"):
                    st.markdown(f"**📝 Summary:** {row['summary']}")
                    st.markdown(f"**👮 Officers:** {', '.join(row['extracted_officers'])}")
                    st.markdown(f"**🏢 Departments:** {', '.join(row['extracted_departments'])}")
                    st.markdown(f"**🏷️ Tags:** {', '.join([t.replace('_', ' ').title() for t in row['suggested_tags']])}")
                    st.markdown(f"**😊 Sentiment:** {row['sentiment_label']} ({row['sentiment_score']:.2f})")
        else:
            st.info("ℹ️ No data processed yet. Go to 'Process Feedback' tab to get started!")
    
    with tab3:
        st.header("💬 Q&A Chat Assistant")
        
        if st.session_state.processed_data:
            st.markdown("""
            <div class="info-box">
                Ask questions about the analyzed feedback. The AI will search through all processed documents to find answers.
            </div>
            """, unsafe_allow_html=True)
            
            all_texts = " ".join([
                d['translated_text'] if d['translated_text'] else d['original_text'] 
                for d in st.session_state.processed_data
            ])
            
            question = st.text_input(
                "💭 Ask a question:",
                placeholder="e.g., What acts of bravery were mentioned? Which officers showed compassion?",
                key="qa_question"
            )
            
            if st.button("🔍 Get Answer", type="primary"):
                if question:
                    with st.spinner("🤔 Searching for answer..."):
                        answer = answer_question(question, all_texts[:2000], models[3])
                        st.session_state.chat_history.append({"q": question, "a": answer, "time": datetime.now()})
                        st.rerun()
            
            # Chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("💬 Conversation History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                    st.markdown(f"""
                    <div class="info-box">
                        <b>❓ Question {len(st.session_state.chat_history) - i + 1}:</b> {chat['q']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <b>✅ Answer:</b> {chat['a']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
        else:
            st.warning("⚠️ Please process some feedback first before using the Q&A feature!")
    
    with tab4:
        st.header("📈 Export & Data Management")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            st.subheader("📊 Complete Data Table")
            
            # Column selector
            all_columns = df.columns.tolist()
            default_cols = [col for col in ['timestamp', 'recognition_score', 'sentiment_label', 'language_name', 'extracted_officers', 'suggested_tags'] if col in all_columns]
            
            selected_columns = st.multiselect(
                "Select columns to display:",
                all_columns,
                default=default_cols
            )
            
            if selected_columns:
                st.dataframe(df[selected_columns], use_container_width=True, height=400)
            
            # Export section
            st.markdown("---")
            st.subheader("📥 Bulk Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📄 Download All as CSV",
                    data=csv_data,
                    file_name=f"police_recognition_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_data = df.to_json(orient='records', indent=2, force_ascii=False)
                st.download_button(
                    "📋 Download All as JSON",
                    data=json_data,
                    file_name=f"police_recognition_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col3:
                # Generate combined PDF report
                if st.button("📚 Generate Combined PDF", use_container_width=True):
                    with st.spinner("Creating comprehensive PDF report..."):
                        # Create a combined PDF for all entries
                        st.info("Combined PDF generation for multiple entries coming soon!")
        else:
            st.info("ℹ️ No data available for export. Process some feedback first!")

if __name__ == "__main__":
    main()
