import streamlit as st
import json
from datetime import datetime
from io import BytesIO
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
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
        border: 1px solid #cbd5e0 !important;
    }
    
    /* Fix for text input visibility */
    .stTextInput input {
        color: #262730 !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e0 !important;
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

# Cache models for performance
@st.cache_resource
def load_models():
    """Load all required ML models"""
    try:
        # Sentiment analysis model
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Use CPU
        )
        
        # NER model for entity extraction - using multilingual model
        ner_model = pipeline(
            "ner", 
            model="Davlan/xlm-roberta-base-ner-hrl",  # Multilingual NER
            aggregation_strategy="simple",
            device=-1
        )
        
        # Summarization model
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=-1
        )
        
        # Translation model for Indic languages
        indic_translator = pipeline(
            "translation",
            model="ai4bharat/indictrans2-indic-en-1B",
            device=-1
        )
        
        # Q&A model
        qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=-1
        )
        
        return sentiment_analyzer, ner_model, summarizer, indic_translator, qa_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

def detect_language(text: str) -> str:
    """Detect language of text"""
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"

def translate_to_english(text: str, translator) -> tuple:
    """Translate non-English text to English"""
    try:
        detected_lang = detect_language(text)
        
        # List of Indic languages supported
        indic_langs = ['hi', 'bn', 'te', 'ta', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'as']
        
        if detected_lang in indic_langs:
            # For Indic languages, use specialized translator
            result = translator(text, max_length=512)
            if isinstance(result, list) and len(result) > 0:
                translated = result[0].get('translation_text', text)
            else:
                translated = text
            return translated, detected_lang
        elif detected_lang != 'en':
            # For other languages, try generic translation
            try:
                from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
                # Fallback for non-Indic languages
                return text, detected_lang
            except:
                return text, detected_lang
        else:
            return text, 'en'
    except Exception as e:
        return text, 'unknown'

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
                # Check if it's likely an officer name
                context_words = ['officer', 'constable', 'inspector', 'sergeant', 'detective', 
                                'chief', 'captain', 'lieutenant', 'cop', 'police']
                if any(word in text.lower() for word in context_words):
                    officers.append(entity_text)
            elif entity_type == 'ORG':
                departments.append(entity_text)
            elif entity_type == 'LOC':
                locations.append(entity_text)
        
        # Enhanced pattern matching for officer names
        officer_patterns = [
            r'(?:Officer|Constable|Inspector|Sergeant|Detective|Chief|Captain|Lt\.|Sgt\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'(?:PC|DC|DI|DS)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]
        
        for pattern in officer_patterns:
            matches = re.findall(pattern, text)
            officers.extend(matches)
        
        # Department pattern matching
        dept_patterns = [
            r'(\d+(?:st|nd|rd|th)\s+(?:Precinct|District|Division|Station))',
            r'([A-Z][a-z]+\s+(?:Police|Department|Precinct|Station))',
        ]
        
        for pattern in dept_patterns:
            matches = re.findall(pattern, text)
            departments.extend(matches)
        
        return {
            "officers": list(set(officers)),
            "departments": list(set(departments)),
            "locations": list(set(locations))
        }
    except Exception as e:
        st.warning(f"Entity extraction issue: {str(e)}")
        return {"officers": [], "departments": [], "locations": []}

def analyze_sentiment_detailed(text: str, sentiment_analyzer) -> Dict:
    """Perform sentiment analysis with detailed scores"""
    try:
        result = sentiment_analyzer(text[:512])[0]
        
        # Convert to normalized score between -1 and 1
        if result['label'] == 'POSITIVE':
            sentiment_score = result['score']
        else:
            sentiment_score = -result['score']
        
        return {
            "label": result['label'],
            "score": result['score'],
            "normalized_score": sentiment_score
        }
    except Exception as e:
        return {"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0}

def extract_competency_tags(text: str) -> List[str]:
    """Extract competency tags from text using keyword matching"""
    competencies = {
        "community_engagement": ["community", "engagement", "outreach", "relationship", "trust", "friendly", "neighbor"],
        "de-escalation": ["de-escalate", "calm", "peaceful", "resolved", "mediation", "conflict resolution", "defused"],
        "rapid_response": ["quick", "fast", "immediate", "prompt", "timely", "rapid", "swift", "rushed"],
        "professionalism": ["professional", "courteous", "respectful", "polite", "manner", "dignified"],
        "life_saving": ["saved", "rescue", "life-saving", "emergency", "critical", "revived", "resuscitated"],
        "investigation": ["investigation", "solved", "detective", "evidence", "case", "arrest", "caught"],
        "compassion": ["compassion", "care", "kindness", "empathy", "understanding", "helped", "caring", "gentle"],
        "bravery": ["brave", "courage", "heroic", "danger", "risk", "fearless", "valor"]
    }
    
    text_lower = text.lower()
    found_tags = []
    
    for tag, keywords in competencies.items():
        if any(keyword in text_lower for keyword in keywords):
            found_tags.append(tag)
    
    return found_tags if found_tags else ["general_commendation"]

def generate_summary(text: str, summarizer) -> str:
    """Generate summary of the text"""
    try:
        if len(text) < 100:
            return text
        
        # Limit input length for summarization
        text_to_summarize = text[:1024]
        summary = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        # Fallback to simple truncation
        return text[:200] + "..."

def calculate_recognition_score(sentiment_score: float, tags: List[str], text_length: int) -> float:
    """Calculate a recognition score based on multiple factors"""
    base_score = (sentiment_score + 1) / 2  # Normalize to 0-1
    
    # Boost for specific high-value tags
    high_value_tags = ["life_saving", "bravery", "de-escalation"]
    tag_boost = sum(0.15 for tag in tags if tag in high_value_tags)
    
    # Boost for detailed feedback
    length_boost = min(0.1, text_length / 1000 * 0.1)
    
    final_score = min(1.0, base_score + tag_boost + length_boost)
    return round(final_score, 3)

def process_text(text: str, models_tuple) -> Dict:
    """Process text through the entire pipeline"""
    sentiment_analyzer, ner_model, summarizer, translator, qa_model = models_tuple
    
    # Translate if needed
    original_text = text
    translated_text, detected_lang = translate_to_english(text, translator)
    
    # Use translated text for processing
    processing_text = translated_text if detected_lang != 'en' else original_text
    
    # Extract entities
    entities = extract_officer_info(processing_text, ner_model)
    
    # Analyze sentiment
    sentiment = analyze_sentiment_detailed(processing_text, sentiment_analyzer)
    
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
    """Answer questions about the processed text"""
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
    
    # Load models with progress
    with st.spinner("🔄 Loading AI models... This may take a moment on first run."):
        models = load_models()
    
    if models[0] is None:
        st.error("Failed to load models. Please check your internet connection and try again.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/police-badge.png", width=100)
        st.title("Navigation")
        
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.metric("Total Processed", len(st.session_state.processed_data))
        
        if st.session_state.processed_data:
            avg_score = sum(d['recognition_score'] for d in st.session_state.processed_data) / len(st.session_state.processed_data)
            st.metric("Avg Recognition Score", f"{avg_score:.2f}")
        
        st.markdown("---")
        
        st.subheader("🌐 Supported Languages")
        st.write("• English")
        st.write("• Hindi (हिंदी)")
        st.write("• Odia (ଓଡ଼ିଆ)")
        st.write("• Bengali (বাংলা)")
        st.write("• Telugu (తెలుగు)")
        st.write("• Tamil (தமிழ்)")
        st.write("• Marathi (मराठी)")
        st.write("• And 5+ more!")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.processed_data = []
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("💾 Export Data", type="primary"):
            if st.session_state.processed_data:
                df = pd.DataFrame(st.session_state.processed_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "police_recognition_data.csv",
                    "text/csv"
                )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Process Feedback", "📊 Dashboard", "💬 Q&A Chat", "📈 Detailed View"])
    
    with tab1:
        st.header("Process New Feedback")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_method = st.radio("Input Method:", ["Text Input", "File Upload"], horizontal=True)
            
            text_to_process = ""
            
            if input_method == "Text Input":
                text_to_process = st.text_area(
                    "Paste feedback, news article, or social media post (any language):",
                    height=200,
                    placeholder="Example: Officer Smith from the 14th Precinct showed incredible compassion when helping a lost child find their parents...\n\nया: ଅଫିସର ସ୍ମିଥ ଏକ ହଜିଯାଇଥିବା ପିଲାକୁ ସାହାଯ୍ୟ କରିଥିଲେ..."
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload file (TXT, PDF)",
                    type=['txt', 'pdf']
                )
                
                if uploaded_file:
                    if uploaded_file.type == "text/plain":
                        text_to_process = uploaded_file.getvalue().decode("utf-8")
                    elif uploaded_file.type == "application/pdf":
                        try:
                            import pdfplumber
                            with pdfplumber.open(uploaded_file) as pdf:
                                text_to_process = ""
                                for page in pdf.pages:
                                    text_to_process += page.extract_text()
                        except Exception as e:
                            st.error(f"Error reading PDF: {str(e)}")
                    
                    if text_to_process:
                        st.text_area("Extracted Text Preview:", text_to_process[:500] + "...", height=150)
        
        with col2:
            st.info("**🌍 Language Support:**\n- Automatic detection\n- Auto-translation to English\n- Supports 12+ Indian languages")
            
            st.success("**✨ Features:**\n✅ Sentiment Analysis\n✅ Entity Extraction\n✅ Auto-Summarization\n✅ Competency Tagging\n✅ Multi-language")
        
        if st.button("🚀 Process Feedback", type="primary", use_container_width=True):
            if text_to_process:
                with st.spinner("🔍 Analyzing feedback..."):
                    result = process_text(text_to_process, models)
                    st.session_state.processed_data.append(result)
                
                st.markdown('<div class="success-box">✅ <b>Processing Complete!</b></div>', unsafe_allow_html=True)
                
                # Display results in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{result['recognition_score']}</h3>
                        <p>Recognition Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    sentiment_emoji = "😊" if result['sentiment_label'] == 'POSITIVE' else "😐"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{sentiment_emoji}</h3>
                        <p>{result['sentiment_label']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(result['extracted_officers'])}</h3>
                        <p>Officers Found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    lang_name = result.get('detected_language', 'unknown').upper()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{lang_name}</h3>
                        <p>Language</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed results
                with st.expander("📋 View Detailed Results", expanded=True):
                    if result['translated_text']:
                        st.warning(f"**Original text was in {result['detected_language'].upper()} - Translated to English**")
                        st.text_area("Original:", result['original_text'], height=100, key="orig")
                        st.text_area("Translated:", result['translated_text'], height=100, key="trans")
                    
                    st.subheader("📝 Summary")
                    st.info(result['summary'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("👮 Extracted Officers")
                        if result['extracted_officers']:
                            for officer in result['extracted_officers']:
                                st.write(f"- {officer}")
                        else:
                            st.write("No specific officers mentioned")
                        
                        st.subheader("🏢 Departments")
                        if result['extracted_departments']:
                            for dept in result['extracted_departments']:
                                st.write(f"- {dept}")
                        else:
                            st.write("No departments identified")
                    
                    with col2:
                        st.subheader("🏷️ Competency Tags")
                        for tag in result['suggested_tags']:
                            st.write(f"- {tag.replace('_', ' ').title()}")
                        
                        st.subheader("📍 Locations")
                        if result['extracted_locations']:
                            for loc in result['extracted_locations']:
                                st.write(f"- {loc}")
                        else:
                            st.write("No locations identified")
                
                # Export single result
                st.download_button(
                    "📥 Export This Result (JSON)",
                    json.dumps(result, indent=2),
                    f"recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            else:
                st.warning("⚠️ Please enter or upload some text to process.")
    
    with tab2:
        st.header("Recognition Dashboard")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Submissions", len(df))
            with col2:
                st.metric("Avg Score", f"{df['recognition_score'].mean():.2f}")
            with col3:
                positive_pct = (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100
                st.metric("Positive Feedback", f"{positive_pct:.1f}%")
            with col4:
                total_officers = sum(len(officers) for officers in df['extracted_officers'])
                st.metric("Officers Recognized", total_officers)
            
            st.markdown("---")
            
            # Top officers
            st.subheader("🏆 Top Recognized Officers")
            all_officers = []
            for officers in df['extracted_officers']:
                all_officers.extend(officers)
            
            if all_officers:
                officer_counts = pd.Series(all_officers).value_counts().head(10)
                st.bar_chart(officer_counts)
            else:
                st.info("No officers identified yet in processed feedback.")
            
            # Tag distribution
            st.subheader("📊 Competency Tag Distribution")
            all_tags = []
            for tags in df['suggested_tags']:
                all_tags.extend(tags)
            
            if all_tags:
                tag_counts = pd.Series(all_tags).value_counts()
                st.bar_chart(tag_counts)
            
            # Recent submissions
            st.subheader("📜 Recent Submissions")
            for idx, row in df.tail(5).iterrows():
                with st.expander(f"Submission {idx + 1} - Score: {row['recognition_score']}"):
                    st.write(f"**Summary:** {row['summary']}")
                    st.write(f"**Officers:** {', '.join(row['extracted_officers']) if row['extracted_officers'] else 'None identified'}")
                    st.write(f"**Tags:** {', '.join(row['suggested_tags'])}")
                    st.write(f"**Sentiment:** {row['sentiment_label']} ({row['sentiment_score']:.2f})")
        else:
            st.info("ℹ️ No data processed yet. Go to 'Process Feedback' tab to get started!")
    
    with tab3:
        st.header("💬 Q&A Chat")
        st.write("Ask questions about the processed feedback")
        
        if st.session_state.processed_data:
            # Combine all processed texts for context
            all_texts = " ".join([
                d['translated_text'] if d['translated_text'] else d['original_text'] 
                for d in st.session_state.processed_data
            ])
            
            question = st.text_input(
                "Ask a question:", 
                placeholder="e.g., What acts of bravery were mentioned?",
                key="qa_input"
            )
            
            if st.button("Get Answer", type="primary"):
                if question:
                    with st.spinner("🤔 Thinking..."):
                        answer = answer_question(question, all_texts[:2000], models[4])
                        st.session_state.chat_history.append({"question": question, "answer": answer})
                        st.rerun()
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("---")
                st.subheader("Chat History")
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")
                    st.info(f"**A:** {chat['answer']}")
                    st.markdown("---")
        else:
            st.warning("⚠️ Please process some feedback first before using Q&A!")
    
    with tab4:
        st.header("📈 Detailed Data View")
        
        if st.session_state.processed_data:
            df = pd.DataFrame(st.session_state.processed_data)
            
            # Data table with selection
            st.subheader("Complete Data Table")
            
            # Select columns to display
            available_columns = df.columns.tolist()
            default_cols = [col for col in ['timestamp', 'recognition_score', 'sentiment_label', 'extracted_officers', 'suggested_tags', 'detected_language'] if col in available_columns]
            
            columns_to_show = st.multiselect(
                "Select columns to display:",
                available_columns,
                default=default_cols
            )
            
            if columns_to_show:
                st.dataframe(df[columns_to_show], use_container_width=True)
            
            # Export options
            st.markdown("---")
            st.subheader("📥 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    "📄 Download CSV",
                    csv,
                    "full_data.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    "📋 Download JSON",
                    json_data,
                    "full_data.json",
                    "application/json",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ No data available yet.")

if __name__ == "__main__":
    main()
