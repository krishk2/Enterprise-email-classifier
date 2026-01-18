import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
import os
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PARTICLE BACKGROUND ---
import random
particles_html = ""
for i in range(50):
    left = random.randint(0, 100)
    size = random.randint(2, 6)
    duration = random.randint(10, 20)
    delay = random.randint(0, 10)
    opacity = random.uniform(0.1, 0.5)
    particles_html += f'<div class="circle" style="left: {left}%; width: {size}px; height: {size}px; animation-duration: {duration}s; animation-delay: {delay}s; opacity: {opacity};"></div>'

st.markdown(f"""
<style>
/* Base Background */
.stApp {{
background-color: #1e1e2f;
}}

/* Particles */
.circle {{
position: fixed;
bottom: -10px;
background: white;
border-radius: 50%;
z-index: 0;
animation: float linear infinite;
pointer-events: none;
}}

@keyframes float {{
0% {{ transform: translateY(0) translateX(0); opacity: 0; }}
20% {{ opacity: 0.5; }}
100% {{ transform: translateY(-110vh) translateX(20px); opacity: 0; }}
}}

/* Ensure content is above particles */
.main .block-container {{
z-index: 1;
position: relative;
}}

/* Email Card Wrapper (Glassmorphism) */
.email-card {{
background: rgba(255, 255, 255, 0.05);
backdrop-filter: blur(10px);
-webkit-backdrop-filter: blur(10px);
border: 1px solid rgba(255, 255, 255, 0.1);
border-radius: 10px;
margin-bottom: 10px;
padding: 10px 15px;
transition: transform 0.2s, box-shadow 0.2s;
cursor: pointer;
display: flex;
align-items: center;
gap: 15px;
box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}}
.email-card:hover {{
transform: translateY(-2px);
box-shadow: 0 10px 15px rgba(0, 0, 0, 0.3);
background: rgba(255, 255, 255, 0.1);
}}

/* Text Visibility */
h1, h2, h3, h4, h5, h6, p, div, label, span, .stMarkdown {{
color: #e0e0e0;
}}
.subject-text {{
font-weight: 700;
font-size: 1.05rem;
color: #ffffff;
}}
.body-snippet {{
font-size: 0.9rem;
color: #b0b0b0;
white-space: nowrap;
overflow: hidden;
text-overflow: ellipsis;
max-width: 600px;
}}

/* Priority Indicators */
.prio-high {{ color: #ff5252; font-weight: bold; font-size: 1.2rem; }}
.prio-medium {{ color: #ffca28; font-size: 1.2rem; }}
.prio-low {{ color: #42a5f5; font-size: 1.2rem; }}

/* Badges */
.badge {{
padding: 4px 8px;
border-radius: 4px;
font-size: 0.75rem;
font-weight: 600;
text-transform: uppercase;
margin-left: auto;
}}
.badge-complaint {{ background-color: #3e2723; color: #ffccbc; border: 1px solid #ffccbc; }}
.badge-request {{ background-color: #0d47a1; color: #bbdefb; border: 1px solid #bbdefb; }}
.badge-feedback {{ background-color: #1b5e20; color: #c8e6c9; border: 1px solid #c8e6c9; }}
.badge-spam {{ background-color: #212121; color: #9e9e9e; border: 1px solid #9e9e9e; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
background-color: rgba(30, 30, 47, 0.95);
border-right: 1px solid rgba(255,255,255,0.1);
}}
div[data-testid="metric-container"] {{
background-color: rgba(255,255,255,0.05);
padding: 10px;
border-radius: 10px;
}}
</style>

<div class="particles">
{particles_html}
</div>
""", unsafe_allow_html=True)


# --- NLTK SETUP ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'system', 'problem', 'issue', 'failure', 'bug', 'urgent', 'critical', 'thank', 'you', 'please'}
PRIORITY_TERMS = r'\b(dear|support|team|ni|hello|hope|well|customer|data|please|message|team|nan|could|would|assistance|update)\b'
HTML_ARTIFACTS = r'\b(href|src|width|height|font|size|face|arial|sans|serif|color|border|style|nbsp|img|align|center|br|div|table|tr|td|span|strong|em)\b'

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(HTML_ARTIFACTS, '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Re-init lemmatizer here for Streamlit thread safety
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    base_dir = "models"
    try:
        return {
            'u_body': joblib.load(os.path.join(base_dir, 'urgency_body_model.pkl')),
            'u_body_vec': joblib.load(os.path.join(base_dir, 'urgency_body_tfidf.pkl')),
            'u_subj': joblib.load(os.path.join(base_dir, 'urgency_subject_model.pkl')),
            'u_subj_vec': joblib.load(os.path.join(base_dir, 'urgency_subject_tfidf.pkl')),
            'c_body': joblib.load(os.path.join(base_dir, 'class_body_model.pkl')),
            'c_body_vec': joblib.load(os.path.join(base_dir, 'class_body_tfidf.pkl')),
            'c_subj': joblib.load(os.path.join(base_dir, 'class_subject_model.pkl')),
            'c_subj_vec': joblib.load(os.path.join(base_dir, 'class_subject_tfidf.pkl'))
        }
    except FileNotFoundError:
        return None

models = load_models()

# --- PREDICTION LOGIC ---
def get_prediction(subject, body, models):
    # Rule Based
    text_combined = (str(subject) + " " + str(body)).lower()
    high_triggers = ['deadline', 'asap', 'immediate', 'urgency', 'critical', 'breach', 'emergency', 'shuts down', 'exploded', 'security alert', 'system down', 'outage', 'unacceptable']
    
    rule_urgency = None
    rule_conf = 0.0
    
    for word in high_triggers:
        if word in text_combined:
            rule_urgency = 'high'
            rule_conf = 0.99
            break
            
    # ML Prediction (Urgency)
    u_b_vec = models['u_body_vec'].transform([clean_text(body)])
    u_s_vec = models['u_subj_vec'].transform([clean_text(subject)])
    p_b = models['u_body'].predict_proba(u_b_vec)[0]
    p_s = models['u_subj'].predict_proba(u_s_vec)[0]
    avg_u = (p_b + p_s) / 2
    ml_urgency = models['u_body'].classes_[np.argmax(avg_u)]
    ml_u_conf = np.max(avg_u)
    
    final_u = rule_urgency if rule_urgency else ml_urgency
    final_u_conf = 0.99 if rule_urgency else ml_u_conf
        
    # ML Prediction (Type)
    cleaned_body = clean_text(body)
    cleaned_subj = clean_text(subject)
    c_b_vec = models['c_body_vec'].transform([cleaned_body])
    c_s_vec = models['c_subj_vec'].transform([cleaned_subj])
    p_b_c = models['c_body'].predict_proba(c_b_vec)[0]
    p_s_c = models['c_subj'].predict_proba(c_s_vec)[0]
    avg_c = (p_b_c + p_s_c) / 2
    final_type = models['c_body'].classes_[np.argmax(avg_c)]
    final_type_conf = np.max(avg_c)
    
    # Rule Based Type Correction
    critical_type_triggers = ['bug', 'failure', 'issue', 'crash', 'error', 'outage', 'down', 'breach']
    spam_indicators = ['buy', 'cheap', 'click here', 'prize', 'winner', 'casino', 'investment', 'bitcoin', 'offer', 'deal', 'limited', 'gift']
    
    is_spaminess_detected = any(s in text_combined for s in spam_indicators)
    
    for word in critical_type_triggers:
        if word in text_combined and final_type.lower() == 'spam' and not is_spaminess_detected:
            final_type = 'Complaint'
            final_type_conf = 0.95
            break
            
    return final_u, final_u_conf, final_type, final_type_conf

# --- BATCH PREDICTION ---
def batch_predict(df, models):
    results = []
    progress_bar = st.progress(0)
    for i, row in df.iterrows():
        s = row.get('subject', '')
        b = row.get('body', '')
        u, uc, t, tc = get_prediction(s, b, models)
        results.append({
            'Subject': s, 
            'Body': b, 
            'Urgency': u, 
            'Category': t,
            'Conf': f"{tc:.2f}"
        })
        progress_bar.progress((i + 1) / len(df))
    progress_bar.empty()
    return pd.DataFrame(results)

# --- MOCK DATA ---
def generate_mock_data():
    return pd.DataFrame([
        {'subject': 'System Outage - Critical', 'body': 'The entire production database is down. We need immediate assistance.'},
        {'subject': 'Feature Request: Dark Mode', 'body': 'Can you please add dark mode to the dashboard? It would be nice.'},
        {'subject': 'Thank you!', 'body': 'Thanks for the quick help yesterday. Appreciate it.'},
        {'subject': 'Bug in login page', 'body': 'I cannot login when I use Firefox. It just spins forever.'},
        {'subject': 'Cheap Rolex', 'body': 'Buy cheap watches now! 50% discount.'},
        {'subject': 'Meeting notes', 'body': 'Here are the minutes from the sprint planning.'},
    ])

# --- RENDER HELPER ---
def render_email_row(idx, row):
    # Determine Style
    u_icon = "🔴" if row['Urgency'] == 'high' else "🟡" if row['Urgency'] == 'medium' else "🔵"
    badge_class = f"badge-{row['Category'].lower()}"
    
    # Snippet
    snippet = str(row['Body'])[:80] + "..." if len(str(row['Body'])) > 80 else str(row['Body'])
    
    # HTML Card
    card_html = f"""
    <div class="email-card">
        <div class="prio-{row['Urgency']}">{u_icon}</div>
        <div style="flex-grow: 1;">
            <div class="subject-text">{row['Subject']}</div>
            <div class="body-snippet">{snippet}</div>
        </div>
        <div class="badge {badge_class}">{row['Category']}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# --- MAIN APP ---

# --- IMPORT DB MODULE ---
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from db import create_user, verify_user, save_classification_event, get_user_history

# --- MAIN APP ---
def main():
    if models is None:
        st.error("❌ Models not found! Run `src/train_models.py` first.")
        return

    # --- SESSION STATE AUTH ---
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['username'] = None

    # --- LOGIN PAGE ---
    if not st.session_state['authenticated']:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.title("🔒 Login")
            auth_mode = st.radio("Mode", ["Login", "Register"], horizontal=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if auth_mode == "Login":
                if st.button("Login", use_container_width=True):
                    if verify_user(username, password):
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
            else:
                if st.button("Register", use_container_width=True):
                    success, msg = create_user(username, password)
                    if success:
                        st.success("Account created! User can now login.")
                    else:
                        st.error(msg)
        return # Stop execution here if not logged in

    # --- DASHBOARD (LOGGED IN) ---
    st.sidebar.title(f"👤 {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        # Clear entire session state to prevent data leaks between users
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        
    st.sidebar.title("📨 Mailbox")
    page = st.sidebar.radio("Navigate", ["Inbox", "Compose", "History & Analytics"], index=0)
    st.sidebar.markdown("---")
    
    # State Management for Inbox
    if 'inbox_data' not in st.session_state:
        st.session_state['inbox_data'] = pd.DataFrame()
    if 'selected_email_idx' not in st.session_state:
        st.session_state['selected_email_idx'] = None
        
    if page == "Inbox":
        
        # --- DETAIL VIEW ---
        if st.session_state['selected_email_idx'] is not None:
            # Get Data
            try:
                row = st.session_state['inbox_data'].iloc[st.session_state['selected_email_idx']]
                
                # Header / Nav
                col_nav, col_title = st.columns([1, 5])
                with col_nav:
                     if st.button("← Back", use_container_width=True):
                         st.session_state['selected_email_idx'] = None
                         st.rerun()
                
                # Main Content
                with st.container():
                    st.markdown(f'<div style="background-color: rgba(30, 30, 40, 0.8); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);"><h2>{row["Subject"]}</h2>', unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                        <span class="badge badge-{row['Category'].lower()}">{row['Category']}</span>
                        <span class="badge" style="background-color: #eee; color: #333;">Urgency: {row['Urgency'].upper()}</span>
                        <span class="badge" style="background-color: #eee; color: #333;">Conf: {row['Conf']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### Message Body")
                    st.write(row['Body'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error loading email: {e}")
                st.session_state['selected_email_idx'] = None
                
        # --- LIST VIEW ---
        else:
            st.markdown('<h1 style="color: white; text-shadow: 2px 2px 4px #000000;">Inbox</h1>', unsafe_allow_html=True)
            
            c1, c2 = st.columns([3, 1])
            with c1:
                uploaded_file = st.file_uploader("Import Emails (CSV)", type=['csv'])
            with c2:
                st.write("")
                st.write("")
                if st.button("🔄 Demo Data"):
                    mock_df = generate_mock_data()
                    st.session_state['inbox_data'] = batch_predict(mock_df, models)
                    # Save to DB
                    for _, r in st.session_state['inbox_data'].iterrows():
                         save_classification_event(st.session_state['username'], r['Subject'], r['Body'], r['Category'], r['Urgency'], float(r['Conf']))
                    st.toast("Demo data loaded & saved to history!")
                    st.rerun()

            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'subject' in df.columns and 'body' in df.columns:
                        st.session_state['inbox_data'] = batch_predict(df, models)
                         # Save to DB
                        for _, r in st.session_state['inbox_data'].iterrows():
                             save_classification_event(st.session_state['username'], r['Subject'], r['Body'], r['Category'], r['Urgency'], float(r['Conf']))
                        st.toast("Emails classified & saved to DB!")
                    else:
                        st.error("CSV must have 'subject' and 'body' columns.")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

            # INBOX DISPLAY
            if not st.session_state['inbox_data'].empty:
                df_display = st.session_state['inbox_data'].copy()
                
                filter_cat = st.selectbox("Filter", ["All", "High Priority", "Complaint", "Request", "Feedback", "Spam"])
                if filter_cat == "High Priority":
                    df_display = df_display[df_display['Urgency'] == 'high']
                elif filter_cat != "All":
                    df_display = df_display[df_display['Category'] == filter_cat]
                    
                prio_map = {'high': 0, 'medium': 1, 'low': 2}
                df_display['_prio_rank'] = df_display['Urgency'].map(prio_map)
                df_display = df_display.sort_values('_prio_rank')
                
                st.caption(f"Showing {len(df_display)} emails")
                
                for idx, row in df_display.iterrows():
                    with st.container():
                        col_card, col_btn = st.columns([6, 1])
                        with col_card:
                            render_email_row(idx, row)
                        with col_btn:
                            st.write("")
                            if st.button("Open", key=f"open_{idx}"):
                                st.session_state['selected_email_idx'] = idx 
                                st.rerun()
            else:
                st.info("Inbox is empty. Upload CSV to classify & save to DB.")

    elif page == "Compose":
        st.header("Compose / Quick Check")
        s = st.text_input("Subject")
        b = st.text_area("Body", height=150)
        if st.button("Send (Classify)"):
            if b:
                u, uc, t, tc = get_prediction(s, b, models)
                save_classification_event(st.session_state['username'], s, b, t, u, float(tc))
                st.markdown(f"### Result")
                col1, col2 = st.columns(2)
                col1.metric("Urgency", u.upper(), f"{uc:.1%}")
                col2.metric("Category", t, f"{tc:.1%}")
                st.success("Result saved to History.")
            else:
                st.warning("Body is empty.")

    elif page == "History & Analytics":
        st.header("History & Analytics")
        history_data = get_user_history(st.session_state['username'])
        
        if history_data:
            df = pd.DataFrame(history_data)
            
            # Metrics
            total = len(df)
            high_prio = len(df[df['urgency'] == 'high'])
            st.markdown(f"**Total Processed for {st.session_state['username']}:** {total} | **High Urgency:** {high_prio}")
            st.markdown("---")
            
            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.pie(df, names='category', title='Historical Category Distribution')
                st.plotly_chart(fig1, use_container_width=True)
            with c2:
                fig2 = px.bar(df, x='urgency', title='Historical Urgency Distribution', color='urgency', 
                              color_discrete_map={'high':'red', 'medium':'orange', 'low':'blue'})
                st.plotly_chart(fig2, use_container_width=True)
                
            st.subheader("Recent Activity Log")
            st.subheader("Recent Activity Log")
            # st.dataframe(df[['timestamp', 'subject', 'category', 'urgency', 'confidence']].head(50), use_container_width=True)
            
            # Interactive History
            for i, row in df.head(50).iterrows():
                # Format timestamp
                ts = row['timestamp'].strftime("%Y-%m-%d %H:%M")
                label = f"{ts} | {row['urgency'].upper()} | {row['subject']}"
                
                with st.expander(label):
                    st.markdown(f"**Category:** {row['category']} | **Confidence:** {row['confidence']}")
                    st.markdown("---")
                    st.write(row['body'])
            
        else:
            st.warning("No history found. Go to Inbox/Compose to classify some emails.")

if __name__ == "__main__":
    main()

