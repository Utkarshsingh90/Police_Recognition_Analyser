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
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Police Recognition Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for appealing UI with fixed text visibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e3f2fd;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
        color: #1e3a8a;
    }
    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 10px 0;
        color: #155724;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        color: #262730;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white !important;
    }
    
    /* Fix for text area visibility */
    .stTextArea textarea {
        color: #262730 !important;
        background-color: #ffffff !important;
        border: 2px solid #cbd5e0 !important;
    }
    
    /* Fix for text input visibility */
    .stTextInput input {
        color: #262730 !important;
        background-color: #ffffff !important;
        border: 2px solid #cbd5e0 !important;
    }
    
    /* Ensure placeholder text is visible */
    .stTextArea textarea::placeholder {
        color: #718096 !important;
    }
    
    .stTextInput input::placeholder {
        color: #718096 !important;
    }
    
    /* Fix radio buttons */
    .stRadio > label {
        color: #262730 !important;
    }
    
    /* Fix general text color */
    .element-container {
        color: #262730 !important;
    }
    
    /* Fix file uploader text */
    .uploadedFile {
        color: #262730 !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda !important;
        color: #155724 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Language code mapping
LANG_CODE_MAP = {
    'hi': 'Hindi', 'bn': 'Bengali', 'te': 'Telugu', 'ta': 'Tamil',
    'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam',
    'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'en': 'English',
    'ur': 'Urdu', 'ne': 'Nepali', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'zh-cn': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'
}

# Cache models for performance
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all required ML models"""
    try:
        # Sentiment analysis model
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        
        # NER model - using multilingual
        ner_model = pipeline(
            "ner", 
            model="Davlan/xlm-roberta-base-ner-hrl",
            aggregation_strategy="simple",
            device=-1
        )
        
        # Summarization model
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=-1
        )
        
        # Lightweight translation pipeline - using smaller Helsinki-NLP models
        # These are loaded on-demand per language pair
        translation_cache = {}
        
        # Q&A model
        qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1
        )
        
        return sentiment_analyzer, ner_model, summarizer, translation_cache, qa_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def detect_language(text: str) -> str:
    """Detect language of text"""
    try:
        lang = detect(text)
        return lang
    except:
        return "en"

def translate_to_english(text: str, translation_cache: dict) -> tuple:
    """Translate using lightweight Helsinki-NLP models"""
    try:
        detected_lang = detect_language(text)
        
        if detected_lang == 'en':
            return text, 'en'
        
        # Map to Helsinki model language codes
        helsinki_map = {
            'hi': 'hi', 'bn': 'bn', 'gu': 'gu', 'mr': 'mr',
            'ta': 'ta', 'te': 'te', 'ur': 'ur', 'ne': 'ne',
            'es': 'es', 'fr': 'fr', 'de': 'de', 'zh-cn': 'zh'
        }
        
        src_lang = helsinki_map.get(detected_lang)
        
        # For unsupported languages, return original
        if not src_lang:
            return text, detected_lang
        
        # Load translation model on demand (cached)
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
        
        if model_name not in translation_cache:
            try:
                translator = pipeline("translation", model=model_name, device=-1)
                translation_cache[model_name] = translator
            except:
                # Fallback for unsupported language pairs
                return text, detected_lang
        else:
            translator = translation_cache[model_name]
        
        # Translate
        result = translator(text[:512], max_length=512)
        translated = result[0]['translation_text']
        
        return translated, detected_lang
        
    except Exception as e:
        return text, detect_language(text)

def extract_officer_info(text: str, ner_model) -> Dict:
    """Extract officer names and departments using NER"""
    try:
        entities = ner_model(text)
        
        officers = []
        departments = []
        locations = []
        
        for ent in entities:
            entity_text = ent['word'].replace('▁', ' ').strip()
            entity_type = ent['entity_group']
            
            if entity_type == 'PER':
                context_words = ['officer', 'constable', 'inspector', 'sergeant', 'detective', 
                                'chief', 'captain', 'lieutenant', 'cop', 'police', 'asi', 'si', 'ci']
                if any(word in text.lower() for word in context_words):
                    officers.append(entity_text)
            elif entity_type == 'ORG':
                departments.append(entity_text)
            elif entity_type == 'LOC':
                locations.append(entity_text)
        
        # Pattern matching for better extraction
        officer_patterns = [
            r'(?:Officer|Constable|Inspector|Sergeant|Detective|Chief|Captain|Lt\.|Sgt\.|ASI|SI|CI|PSI)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]
        
        for pattern in officer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            officers.extend(matches)
        
        dept_patterns = [
            r'(\d+(?:st|nd|rd|th)\s+(?:Precinct|District|Division|Station|Police\s+Station|Thana))',
            r'([A-Z][a-z]+\s+(?:Police|Department|Precinct|Station|Thana))',
        ]
        
        for pattern in dept_patterns:
            matches = re.findall(pattern, text)
            departments.extend(matches)
        
        return {
            "officers": list(set([o.strip() for o in officers if o.strip()])),
            "departments": list(set([d.strip() for d in departments if d.strip()])),
            "locations": list(set([l.strip() for l in locations if l.strip()]))
        }
    except Exception as e:
        return {"officers": [], "departments": [], "locations": []}

def analyze_sentiment_detailed(text: str, sentiment_analyzer) -> Dict:
    """Perform sentiment analysis"""
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
        "community_engagement": ["community", "engagement", "outreach", "relationship", "trust", "friendly"],
        "de-escalation": ["de-escalate", "calm", "peaceful", "resolved", "mediation", "defused"],
        "rapid_response": ["quick", "fast", "immediate", "prompt", "timely", "rapid", "swift"],
        "professionalism": ["professional", "courteous", "respectful", "polite", "dignified"],
        "life_saving": ["saved", "rescue", "life-saving", "emergency", "critical", "revived"],
        "investigation": ["investigation", "solved", "detective", "evidence", "arrest", "caught"],
        "compassion": ["compassion", "care", "kindness", "empathy", "understanding", "helped"],
        "bravery": ["brave", "courage", "heroic", "danger", "risk", "fearless", "valor"]
    }
    
    text_lower = text.lower()
    found_tags = []
    
    for tag, keywords in competencies.items():
        if any(keyword in text_lower for keyword in keywords):
            found_tags.append(tag)
    
    return found_tags if found_tags else ["general_commendation"]

def generate_summary(text: str, summarizer) -> str:
    """Generate summary"""
    try:
        if len(text) < 100:
            return text
        
        summary = summarizer(text[:1024], max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except:
        return text[:200] + "..."

def calculate_recognition_score(sentiment_score: float, tags: List[str], text_length: int) -> float:
    """Calculate recognition score"""
    base_score = (sentiment_score + 1) / 2
    
    high_value_tags = ["life_saving", "bravery", "de-escalation"]
    tag_boost = sum(0.15 for tag in tags if tag in high_value_tags)
    
    length_boost = min(0.1, text_length / 1000 * 0.1)
    
    final_score = min(1.0, base_score + tag_boost + length_boost)
    return round(final_score, 3)

def process_text(text: str, models_tuple) -> Dict:
    """Process text through pipeline"""
    sentiment_analyzer, ner_model, summarizer, translator_cache, qa_model = models_tuple
    
    original_text = text
    translated_text, detected_lang = translate_to_english(text, translator_cache)
    
    processing_text = translated_text if detected_lang != 'en' else original_text
    
    entities = extract_officer_info(processing_text, ner_model)
    sentiment = analyze_sentiment_detailed(processing_text, sentiment_analyzer)
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
    """Answer questions"""
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
        <b>Welcome!</b> This platform uses AI to analyze public feedback, news articles, and social media posts 
        to identify and recognize outstanding police work. Supports multiple languages including Odia, Hindi, Bengali, and more!
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading AI models... Please wait..."):
        models = load_models()
    
    if models[0] is None:
        st.error("❌ Failed to load models. Please refresh the page or check logs.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/police-badge.png", width=100)
        st.title("📍 Navigation")
        
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.metric("Total Processed", len(st.session_state.processed_data))
        
        if st.session_state.processed_data:
            avg_score = sum(d['recognition_score'] for d in st.session_state.processed_data) / len(st.session_state.processed_data)
            st.metric("Avg Score", f"{avg_score:.2f}")
        
        st.markdown("---")
        st.subheader("🌐 Languages")
        st.write("✅ Hindi • Odia • Bengali")
        st.write("✅ Tamil • Telugu • Marathi")
        st.write("✅ Gujarati • Kannada")
        st.write("✅ Spanish • French • German")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Data"):
            st.session_state.processed_data = []
            st.session_state.chat_history = []
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Process Feedback", "📊 Dashboard", "💬 Q&A", "📈 Data"])
    
    with tab1:
        st.header("Process New Feedback")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_method = st.radio("Input Method:", ["Text Input", "File Upload"], horizontal=True)
            
            text_to_process = ""
            
            if input_method == "Text Input":
                text_to_process = st.text_area(
                    "Enter feedback (any language):",
                    height=200,
                    placeholder="Example: Officer Smith showed great compassion...\n\nOdia: ଅଫିସର ସ୍ମିଥ ବହୁତ ଦୟାଳୁ ଥିଲେ...",
                    key="main_input"
                )
            else:
                uploaded_file = st.file_uploader("Upload TXT or PDF", type=['txt', 'pdf'])
                
                if uploaded_file:
                    if uploaded_file.type == "text/plain":
                        text_to_process = uploaded_file.getvalue().decode("utf-8")
                    elif uploaded_file.type == "application/pdf":
                        try:
                            import pdfplumber
                            with pdfplumber.open(uploaded_file) as pdf:
                                text_to_process = ""
                                for page in pdf.pages:
                                    text_to_process += page.extract_text() or ""
                        except Exception as e:
                            st.error(f"PDF error: {str(e)}")
                    
                    if text_to_process:
                        st.text_area("Preview:", text_to_process[:300] + "...", height=100, key="preview")
        
        with col2:
            st.info("**🌍 Multi-language**\n- Auto-detect\n- Auto-translate\n- 15+ languages")
            st.success("**✨ AI Features**\n✅ Sentiment\n✅ Entity extraction\n✅ Summarization\n✅ Tagging")
        
        if st.button("🚀 Analyze", type="primary", use_container_width=True):
            if text_to_process and text_to_process.strip():
                with st.spinner("🔍 Processing..."):
                    result = process_text(text_to_process, models)
                    st.session_state.processed_data.append(result)
                
                st.success("✅ Complete!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Score", result['recognition_score'])
                with col2:
                    st.metric("Sentiment", result['sentiment_label'])
                with col3:
                    st.metric("Officers", len(result['extracted_officers']))
                with col4:
                    st.metric("Language", result['language_name'])
                
                # Details
                with st.expander("📋 Details", expanded=True):
                    if result['translated_text']:
                        st.warning(f"Translated from {result['language_name']}")
                        st.text_area("Original:", result['original_text'], height=80, key="orig")
                        st.text_area("English:", result['translated_text'], height=80, key="trans")
                    
                    st.info(f"**Summary:** {result['summary']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**👮 Officers:**")
                        for o in result['extracted_officers'] or ["None"]:
                            st.write(f"- {o}")
                        
                        st.write("**🏢 Departments:**")
                        for d in result['extracted_departments'] or ["None"]:
                            st.write(f"- {d}")
                    
                    with col2:
                        st.write("**🏷️ Tags:**")
                        for t in result['suggested_tags']:
                            st.write(f"- {t.replace('_', ' ').title()}")
                
                st.download_button(
                    "📥 Export JSON",
                    json.dumps(result, indent=2, ensure_ascii=False),
                    f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            else:
                st.warning("⚠️ Please enter text")
    
    with tab2:
        st.header("Dashboard")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(df))
            with col2:
                st.metric("Avg Score", f"{df['recognition_score'].mean():.2f}")
            with col3:
                pos_pct = (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100
                st.metric("Positive", f"{pos_pct:.0f}%")
            with col4:
                total_officers = sum(len(o) for o in df['extracted_officers'])
                st.metric("Officers", total_officers)
            
            st.subheader("🏆 Top Officers")
            all_officers = [o for officers in df['extracted_officers'] for o in officers]
            if all_officers:
                st.bar_chart(pd.Series(all_officers).value_counts().head(10))
            else:
                st.info("No officers identified yet")
            
            st.subheader("📊 Tags")
            all_tags = [t for tags in df['suggested_tags'] for t in tags]
            if all_tags:
                st.bar_chart(pd.Series(all_tags).value_counts())
        else:
            st.info("No data yet. Process some feedback first!")
    
    with tab3:
        st.header("💬 Q&A")
        
        if st.session_state.processed_data:
            all_texts = " ".join([
                d['translated_text'] if d['translated_text'] else d['original_text'] 
                for d in st.session_state.processed_data
            ])
            
            question = st.text_input("Ask a question:", placeholder="What bravery was shown?")
            
            if st.button("Ask", type="primary"):
                if question:
                    with st.spinner("Thinking..."):
                        answer = answer_question(question, all_texts[:2000], models[4])
                        st.session_state.chat_history.append({"q": question, "a": answer})
                        st.rerun()
            
            if st.session_state.chat_history:
                for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                    st.markdown(f"**Q:** {chat['q']}")
                    st.info(f"**A:** {chat['a']}")
                    st.markdown("---")
        else:
            st.warning("Process feedback first!")
    
    with tab4:
        st.header("📈 Data Export")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            st.dataframe(df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button("📄 CSV", csv, "data.csv", "text/csv", use_container_width=True)
            with col2:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button("📋 JSON", json_data, "data.json", "application/json", use_container_width=True)
        else:
            st.info("No data available")

if __name__ == "__main__":
    main()
