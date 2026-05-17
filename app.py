# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import datetime
import plotly.graph_objects as go
from gtts import gTTS
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LANGUAGE CONFIG ---
LANG_CODE = {"English": "en", "Spanish": "es", "Hindi": "hi", "Telugu": "te"}

LANGUAGES = {
    "English": {
        "title": "Diabetes Risk Predictor",
        "subtitle": "Clinical AI Assessment System",
        "personal": "👤 Personal Details", "health": "🏥 Health Metrics", "predict_btn": "Predict Diabetes Risk",
        "gender": "Gender", "age": "Age", "smoking": "Smoking History", "bp": "Hypertension",
        "bmi": "BMI (Body Mass Index)", "heart": "Heart Disease", "hba1c": "HbA1c Level (%)", "glucose": "Blood Glucose Level (mg/dL)",
        "sidebar_goals": "🎯 My Health Goals", "sidebar_history": "📜 Prediction History", "target_bmi": "Target BMI", "target_glucose": "Target Glucose (mg/dL)", "target_hba1c": "Target HbA1c (%)",
        "voice_result_high": "Warning! Your diabetes risk is {prob} percent. You are in the {cat} category. Please consult a doctor immediately.",
        "voice_result_low": "Good news! Your diabetes risk is {prob} percent. You are in the {cat} category. Keep maintaining a healthy lifestyle.",
        "voice_advice": "Health advice: {advice}"
    },
    "Spanish": {
        "title": "Predictor de Riesgo de Diabetes",
        "subtitle": "Sistema de Evaluación Clínica de IA",
        "personal": "👤 Detalles Personales", "health": "🏥 Métricas de Salud", "predict_btn": "Predecir Riesgo de Diabetes",
        "gender": "Género", "age": "Edad", "smoking": "Historial de Tabaquismo", "bp": "Hipertensión",
        "bmi": "IMC", "heart": "Enf. del Corazón", "hba1c": "Nivel de HbA1c (%)", "glucose": "Glucosa en Sangre (mg/dL)",
        "sidebar_goals": "🎯 Mis Metas de Salud", "sidebar_history": "📜 Historial", "target_bmi": "IMC Objetivo", "target_glucose": "Glucosa Objetivo", "target_hba1c": "HbA1c Objetivo (%)",
        "voice_result_high": "Advertencia! Su riesgo de diabetes es del {prob} por ciento. Está en la categoría {cat}. Por favor consulte a un médico.",
        "voice_result_low": "Buenas noticias! Su riesgo de diabetes es del {prob} por ciento. Categoría: {cat}. Siga manteniendo un estilo de vida saludable.",
        "voice_advice": "Consejo de salud: {advice}"
    },
    "Hindi": {
        "title": "मधुमेह जोखिम भविष्यवक्ता",
        "subtitle": "क्लिनिकल एआई सिस्टम",
        "personal": "👤 व्यक्तिगत विवरण", "health": "🏥 स्वास्थ्य मीट्रिक", "predict_btn": "मधुमेह जोखिम का आकलन करें",
        "gender": "लिंग", "age": "उम्र", "smoking": "धूम्रपान का इतिहास", "bp": "उच्च रक्तचाप",
        "bmi": "बीएमआई", "heart": "हृदय रोग", "hba1c": "एचबीए1सी स्तर (%)", "glucose": "रक्त शर्करा (mg/dL)",
        "sidebar_goals": "🎯 मेरे लक्ष्य", "sidebar_history": "📜 इतिहास", "target_bmi": "लक्षित बीएमआई", "target_glucose": "लक्षित शर्करा", "target_hba1c": "लक्षित एचबीए1सी (%)",
        "voice_result_high": "चेतावनी! आपका मधुमेह जोखिम {prob} प्रतिशत है। आप {cat} श्रेणी में हैं। कृपया तुरंत डॉक्टर से परामर्श करें।",
        "voice_result_low": "शुभ समाचार! आपका मधुमेह जोखिम {prob} प्रतिशत है। आप {cat} श्रेणी में हैं। स्वस्थ जीवनशैली बनाए रखें।",
        "voice_advice": "स्वास्थ्य सलाह: {advice}"
    },
    "Telugu": {
        "title": "మధుమేహ ప్రమాద సూచిక",
        "subtitle": "క్లినికల్ ఏఐ అంచనా వ్యవస్థ",
        "personal": "👤 వ్యక్తిగత వివరాలు", "health": "🏥 ఆరోగ్య కొలమానాలు", "predict_btn": "మధుమేహ ప్రమాదాన్ని అంచనా వేయండి",
        "gender": "లింగం", "age": "వయస్సు", "smoking": "ధూమపాన చరిత్ర", "bp": "అధిక రక్తపోటు",
        "bmi": "శరీర ద్రవ్యరాశి సూచిక (BMI)", "heart": "గుండె జబ్బు", "hba1c": "HbA1c స్థాయి (%)", "glucose": "రక్తంలో చక్కెర (mg/dL)",
        "sidebar_goals": "🎯 నా లక్ష్యాలు", "sidebar_history": "📜 అంచనా చరిత్ర", "target_bmi": "లక్ష్య BMI", "target_glucose": "లక్ష్య గ్లూకోజ్", "target_hba1c": "లక్ష్య HbA1c (%)",
        "voice_result_high": "హెచ్చరిక! మీ మధుమేహ ప్రమాదం {prob} శాతం ఉంది. మీరు {cat} వర్గంలో ఉన్నారు. వెంటనే డాక్టర్‌ని సంప్రదించండి.",
        "voice_result_low": "శుభవార్త! మీ మధుమేహ ప్రమాదం {prob} శాతం ఉంది. మీరు Telugu వర్గంలో ఉన్నారు. ఆరోగ్యకరమైన జీవనశైలిని కొనసాగించండి.",
        "voice_advice": "ఆరోగ్య సలహా: {advice}"
    }
}

# --- STATE INIT ---
if 'page' not in st.session_state: st.session_state.page = 'form'
if 'history' not in st.session_state: st.session_state.history = []
if 'chat_log' not in st.session_state: st.session_state.chat_log = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🌐 Settings")
    selected_lang = st.selectbox("Language / Idioma / भाषा / తెలుగు", list(LANGUAGES.keys()))
    t, lc = LANGUAGES[selected_lang], LANG_CODE[selected_lang]

    st.markdown(f"## {t['sidebar_goals']}")
    goal_bmi = st.number_input(t['target_bmi'], min_value=15.0, max_value=40.0, value=22.0)
    goal_glucose = st.number_input(t['target_glucose'], min_value=70, max_value=150, value=90)
    goal_hba1c = st.number_input(t['target_hba1c'], min_value=4.0, max_value=10.0, value=5.5)

    st.markdown(f"## {t['sidebar_history']}")
    if len(st.session_state.history) > 0:
        for idx, item in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"**{item['Time']}** | `Risk: {item['Risk %']}%`  \n`BMI: {item['BMI']}` | `Gluc: {item['Glucose']}`")
            if idx < 4 and idx < len(st.session_state.history)-1: st.divider()
        if st.button("🗑️ Clear History", key="clear_hist"):
            st.session_state.history = []; st.rerun()
    else:
        st.info("No predictions yet.")

    st.markdown("---")
    st.markdown("### 🛠️ AI Sensitivity")
    threshold_input = st.slider("Step: Sensitivity (%)", min_value=10, max_value=90, value=50, step=5)
    clinical_threshold = threshold_input / 100.0




# --- PREVIOUS PREMIUM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #eef2ff, #c7d2fe);
    }

    [data-testid="stAppViewContainer"]::before {
        content: ''; position: absolute; top: -10%; left: -10%; width: 500px; height: 500px;
        background: #a5b4fc; border-radius: 50%; filter: blur(80px); z-index: -1; pointer-events: none;
    }
    [data-testid="stAppViewContainer"]::after {
        content: ''; position: absolute; bottom: -10%; right: -10%; width: 600px; height: 600px;
        background: #e0e7ff; border-radius: 50%; filter: blur(80px); z-index: -1; pointer-events: none;
    }

    [data-testid="stMainBlockContainer"] {
        background: rgba(255,255,255,0.85); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.5); border-radius: 24px; padding: 3rem;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.08); max-width: 1000px; margin-top: 30px; margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #059669; color: white; font-weight: bold; border-radius: 12px;
        padding: 1rem; transition: all 0.3s ease; width: 100%; border: none; font-size: 1.1rem;
    }
    .stButton>button:hover { background-color: #047857; transform: translateY(-2px); box-shadow: 0 10px 20px -10px #059669; }

    h1, h2, h3, p, label, .stSelectbox label, .stNumberInput label {
        color: #1e293b !important; font-family: 'Inter', sans-serif !important;
    }
    h1, h2, h3, p { text-align: center; font-weight: bold !important; }

    .result-box-low { background-color: #dcfce7; color: #166534; padding: 40px 20px; border-radius: 16px; text-align: center; font-size: 26px; font-weight: bold; border: 1px solid #bbf7d0; margin: 20px 0; }
    .result-box-high { background-color: #fee2e2; color: #991b1b; padding: 40px 20px; border-radius: 16px; text-align: center; font-size: 26px; font-weight: bold; border: 1px solid #fecaca; margin: 20px 0; }
    .conf-score { font-size: 18px; font-weight: normal; margin-top: 10px; display: block; }
    .voice-box { background: linear-gradient(135deg, #6366f1, #8b5cf6); border-radius: 16px; padding: 20px; color: white; text-align: center; margin: 15px 0; box-shadow: 0 8px 20px rgba(99,102,241,0.3); }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
@st.cache_data(show_spinner=False)
def synthesize_speech(text, lang):
    try:
        tts = gTTS(text=text, lang=lang); buf = BytesIO(); tts.write_to_fp(buf); buf.seek(0); return buf.read()
    except: return None

@st.cache_resource
def load_model():
    try: return joblib.load('diabetes_model.pkl')
    except: return None

model = load_model()
if model is None: st.error("Model missing!"); st.stop()

# --- FORM PAGE ---
if st.session_state.page == 'form':
    st.markdown(f"<h1><span style='background-color:#ffffff;color:black;border-radius:50%;padding:2px 10px;font-size:0.9em;box-shadow:0 0 15px rgba(0,0,0,0.1);vertical-align:middle;'>🩺</span> {t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>{t['subtitle']}</p><br>", unsafe_allow_html=True)

    with st.container():
        st.markdown(f"<h3>{t['personal']}</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: gender = st.selectbox(t['gender'], ["Female", "Male", "Other"])
        with c2: age = st.number_input(t['age'], min_value=0.0, max_value=120.0, value=30.0)
        with c3: smoking = st.selectbox(t['smoking'], ["never", "No Info", "current", "former", "ever", "not current"])

    with st.container():
        st.markdown(f"<h3>{t['health']}</h3>", unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4:
            ht = st.selectbox(t['bp'], ["No", "Yes"])
            bmi = st.number_input(t['bmi'], min_value=10.0, max_value=60.0, value=25.0)
        with c5:
            hd = st.selectbox(t['heart'], ["No", "Yes"])
            hba1c = st.number_input(t['hba1c'], min_value=3.0, max_value=15.0, value=5.5)
        glucose = st.number_input(t['glucose'], min_value=50, max_value=400, value=100)

    if st.button(t['predict_btn']):
        input_df = pd.DataFrame([{
            'gender': gender, 'age': age, 'hypertension': 1 if ht=="Yes" else 0,
            'heart_disease': 1 if hd=="Yes" else 0, 'smoking_history': smoking,
            'bmi': bmi, 'HbA1c_level': hba1c, 'blood_glucose_level': glucose
        }])
        prob = model.predict_proba(input_df)[0][1] * 100
        st.session_state.prediction = 1 if prob >= (clinical_threshold * 100) else 0
        st.session_state.probability = prob
        st.session_state.bmi, st.session_state.hba1c, st.session_state.glucose = bmi, hba1c, glucose
        st.session_state.raw_df, st.session_state.lang = input_df, selected_lang
        st.session_state.history.append({"Time": datetime.datetime.now().strftime("%I:%M %p"), "Risk %": round(prob, 1), "BMI": bmi, "Glucose": glucose})
        st.session_state.page = 'result'; st.rerun()

# --- RESULT PAGE ---
elif st.session_state.page == 'result':
    prob, pred = st.session_state.probability, st.session_state.prediction
    t_res = LANGUAGES[st.session_state.lang]; lc_res = LANG_CODE[st.session_state.lang]

    if prob < 30: risk_cat = "Optimal (Safe)"
    elif prob < 60: risk_cat = "Borderline (Monitor)"
    else: risk_cat = "Elevated (Warning)"

    tabs = st.tabs(["📊 Overview", "📉 Trend Simulator", "⚖️ Averages", "🧠 AI Logic", "⚕️ Doctors", "💬 Virtual Bot"])

    with tabs[0]:
        cls = 'result-box-high' if pred == 1 else 'result-box-low'
        lbl = '⚠️ HIGH' if pred == 1 else '✅ LOW'
        st.markdown(f"<div class='{cls}'>{lbl} RISK DETECTED<br><span class='conf-score'>Category: {risk_cat} | Prob: {prob:.1f}%</span></div>", unsafe_allow_html=True)
        
        # Plotly Gauge
        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob, title={'text': "Probability of Diabetes (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': 'black'},
                   'steps': [{'range': [0, 30], 'color': "#10b981"}, {'range': [30, 60], 'color': "#f59e0b"}, 
                            {'range': [60, 85], 'color': "#f97316"}, {'range': [85, 100], 'color': "#ef4444"}]}))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

        st.markdown("<div class='voice-box'>🔊 AI Voice Summary — Click to hear your result</div>", unsafe_allow_html=True)
        if st.button("🔊 Play Audio"):
            text = f"Your diabetes risk is {round(prob, 1)} percent. You are in the {risk_cat} category. Please consult a doctor."
            audio = synthesize_speech(text, lc_res)
            if audio: st.audio(audio, format="audio/mp3", autoplay=True)

    with tabs[1]:
        st.markdown("### 📉 Risk Reduction Simulator")
        sim_df = st.session_state.raw_df.copy(); sim_probs = [prob]
        for _ in range(3):
            sim_df['bmi'] -= 1.0; sim_df['blood_glucose_level'] -= 5
            sim_probs.append(model.predict_proba(sim_df)[0][1] * 100)
        fig_trend = go.Figure(go.Scatter(x=["Now", "M1", "M2", "M3"], y=sim_probs, mode='lines+markers', line=dict(color='#6366f1')))
        st.plotly_chart(fig_trend)

    with tabs[2]:
        st.markdown("### ⚖️ You vs Health Goals")
        fig_bar = go.Figure(data=[
            go.Bar(name='You', x=['BMI', 'Glucose'], y=[st.session_state.bmi, st.session_state.glucose], marker_color='#4338ca'),
            go.Bar(name='Goal', x=['BMI', 'Glucose'], y=[goal_bmi, goal_glucose], marker_color='#10b981')
        ])
        st.plotly_chart(fig_bar)

    with tabs[3]:
        st.markdown("### 🧠 AI Logic & Indicators")
        if st.session_state.hba1c > 5.7: st.error(f"🔴 HbA1c ({st.session_state.hba1c}%): Primary indicator strongly triggered.")
        else: st.success(f"🟢 HbA1c ({st.session_state.hba1c}%): Optimal range.")
        if st.session_state.glucose > 125: st.error(f"🔴 Glucose ({st.session_state.glucose}): Exceeds diabetic threshold.")
        elif st.session_state.glucose > 99: st.warning(f"🟡 Glucose ({st.session_state.glucose}): Prediabetic range.")

    with tabs[4]:
        st.markdown("### ⚕️ Specialist Referral System")
        if st.session_state.hba1c > 6.0: st.error("🧬 **Endocrinologist**: Elevated HbA1c requires specialist review.")
        if st.session_state.bmi > 25: st.info("🥗 **Nutritionist**: A weight-loss plan will improve your health score.")
        st.warning("🔬 **General Practitioner**: We recommend a professional fasting glucose test.")

    with tabs[5]:
        st.markdown("### 💬 Virtual Assistant")
        user_q = st.text_input("Ask about diet, exercise, or diabetes:", placeholder="e.g. How to reduce blood sugar?")
        def bot_ans(q):
            q = q.lower()
            if "sugar" in q or "glucose" in q: return "To reduce sugar: avoid refined carbs, drink water, and walk after meals."
            if "weight" in q or "bmi" in q: return "For weight loss: Maintain a 300-500 calorie deficit and do cardio."
            return "I am an AI assistant. Consult a doctor for medical emergencies."
        if user_q:
            ans = bot_ans(user_q)
            st.session_state.chat_log.append({"q": user_q, "a": ans})
        for log in reversed(st.session_state.chat_log[-4:]):
            st.markdown(f"**👤 You:** {log['q']}\n\n**🤖 Bot:** {log['a']}\n---")

    if st.button("⬅️ Retake Assessment"): st.session_state.page = 'form'; st.rerun()

