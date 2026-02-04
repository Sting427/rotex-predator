import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk
import feedparser
import requests
import time
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import random
import sqlite3
from datetime import datetime, timedelta
import qrcode
from io import BytesIO
from scipy import signal

# --- 🛡️ SAFE IMPORT FOR GRAPHVIZ ---
try:
    import graphviz
    graphviz_installed = True
except ImportError:
    graphviz_installed = False

# --- 🌑 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ROTex // PREDATOR ELITE",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# --- 🎨 THE "YOUTUBE-STYLE" UI ENGINE (v42.0) ---
st.markdown("""
    <style>
    /* 1. IMPORT FUTURISTIC FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Rajdhani:wght@500;700;800&display=swap');

    /* 2. ANIMATED MESH GRADIENT BACKGROUND */
    .stApp {
        background-color: #000000;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        background-size: 200% 200%;
        animation: gradient-animation 15s ease infinite;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 3. GLASSMORPHISM CARDS */
    div[data-testid="metric-container"], .info-card, .job-card, .skunk-card, .target-card, .target-safe, .guide-card, .stDataFrame, .stPlotlyChart, .video-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }

    div[data-testid="metric-container"]:hover, .info-card:hover, .guide-card:hover, .video-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 210, 255, 0.2);
        border-color: rgba(0, 210, 255, 0.5);
    }
    
    /* 4. HEADERS */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #fff, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(0, 210, 255, 0.3);
    }

    /* 5. SIDEBAR & MENU OPTIMIZATION */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* MENU TILES */
    section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.03);
        padding: 12px 15px;
        margin-bottom: 8px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
    }

    section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(0, 210, 255, 0.1);
        border-color: rgba(0, 210, 255, 0.5);
        padding-left: 25px; /* Slide animation */
        color: #ffffff !important;
        cursor: pointer;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.1);
    }

    /* 6. LOGO ANIMATION */
    .rotex-text {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 800;
        font-size: 50px;
        background: linear-gradient(90deg, #00d2ff, #ffffff, #00d2ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 5s linear infinite;
    }
    @keyframes shine { 0% { background-position: 200%; } 100% { background-position: 0%; } }
    
    .rotex-tagline { font-family: 'Rajdhani'; letter-spacing: 4px; color: #666; font-size: 12px; text-transform: uppercase; }

    /* 7. ALERTS */
    .chaos-alert { border-left: 4px solid #ff0000 !important; background: rgba(255, 0, 0, 0.1) !important; animation: pulse-red 2s infinite; }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }

    .target-card { border-left: 4px solid #ff4b4b !important; background: linear-gradient(90deg, rgba(255,0,0,0.1), transparent); }
    .target-safe { border-left: 4px solid #00ff88 !important; background: linear-gradient(90deg, rgba(0,255,136,0.1), transparent); }
    
    .login-box {
        background: rgba(0,0,0,0.8);
        border: 1px solid #333;
        padding: 50px;
        border-radius: 20px;
        text-align: center;
        max-width: 450px;
        margin: auto;
        box-shadow: 0 0 100px rgba(0, 210, 255, 0.1);
    }
    
    /* 8. NEON BUTTONS */
    .stButton > button {
        background: linear-gradient(45deg, #0b1c2c, #1a1a2e);
        border: 1px solid #00d2ff;
        color: #00d2ff;
        font-family: 'Rajdhani';
        font-weight: 700;
        letter-spacing: 2px;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #00d2ff;
        color: #000;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.6);
    }
    
    /* 9. CHAT BUBBLES & NOTES */
    .chat-bubble-me, .note-bubble {
        background: rgba(0, 210, 255, 0.1);
        border: 1px solid #00d2ff;
        border-radius: 15px 15px 0px 15px;
        padding: 10px;
        margin: 5px;
        text-align: right;
        float: right;
        clear: both;
        max-width: 70%;
    }
    .chat-bubble-other {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid #666;
        border-radius: 15px 15px 15px 0px;
        padding: 10px;
        margin: 5px;
        text-align: left;
        float: left;
        clear: both;
        max-width: 70%;
    }
    .note-bubble {
        text-align: left; float: left; border-radius: 0px 15px 15px 15px; width: 100%; max-width: 100%;
        border-left: 3px solid #00d2ff;
    }

    /* 10. YOUTUBE CHIPS (For Video Vault) */
    div[data-testid="stHorizontalBlock"] > div[class*="stRadio"] > div[role="radiogroup"] {
        background: transparent !important;
        border: none !important;
    }

    /* 11. RECOMMENDATION CARD (NEW) */
    .rec-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #333;
        transition: all 0.2s;
    }
    .rec-card:hover {
        border-color: #00d2ff;
        background: rgba(0, 210, 255, 0.1);
        cursor: pointer;
    }
    .rec-title {
        font-size: 13px;
        font-weight: bold;
        color: #eee;
        margin-bottom: 4px;
        line-height: 1.3;
    }
    .rec-meta {
        font-size: 10px;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 🟢 SYSTEM STATUS MARKER ---
st.success("SYSTEM STATUS: v42.0 (YOUTUBE ENGINE + FULL RESTORE)")

# --- 🗄️ DATABASE ---
DB_FILE = "rotex_core.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # CORE TABLES
    c.execute('''CREATE TABLE IF NOT EXISTS deals (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, buyer TEXT, qty REAL, price REAL, cost REAL, margin REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, defects INTEGER, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, role TEXT, salary REAL, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, user TEXT, message TEXT)''')
    
    # VIDEO LIBRARY (Updated with RATING column)
    c.execute('''CREATE TABLE IF NOT EXISTS video_library (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, type TEXT, url TEXT, category TEXT, rating TEXT DEFAULT 'None')''')
    
    # VIDEO NOTES
    c.execute('''CREATE TABLE IF NOT EXISTS video_notes (id INTEGER PRIMARY KEY AUTOINCREMENT, video_id INTEGER, user TEXT, note TEXT, timestamp TEXT)''')

    # --- MIGRATION CHECK (Add 'rating' column if it doesn't exist for older v38 users) ---
    try:
        c.execute("ALTER TABLE video_library ADD COLUMN rating TEXT DEFAULT 'None'")
    except sqlite3.OperationalError:
        pass # Column already exists, all good.

    # --- AUTO-SEED DATA ---
    c.execute("SELECT count(*) FROM employees")
    if c.fetchone()[0] == 0:
        fake_employees = [
            ("Abdul Rahim", "Knitting Operator", 12500, "Active"),
            ("Fatima Begum", "Sewing Operator", 13000, "Active"),
            ("Kamrul Hasan", "Shift Supervisor", 28000, "Active"),
            ("Suma Akter", "Quality Inspector", 18500, "Active"),
        ]
        for _ in range(5): 
            for emp in fake_employees:
                c.execute("INSERT INTO employees (name, role, salary, status) VALUES (?, ?, ?, ?)", 
                          (emp[0] + f" {random.randint(1,99)}", emp[1], emp[2] + random.randint(-500, 500), emp[3]))
        
        # Seed Videos
        videos = [
            ("Textile Factory Safety", "youtube", "https://www.youtube.com/watch?v=1p55GjA1jCQ", "Training", "None"),
            ("Advanced Knitting Tech", "youtube", "https://www.youtube.com/watch?v=F07gB5lH6qE", "R&D", "None"),
        ]
        c.executemany("INSERT INTO video_library (title, type, url, category, rating) VALUES (?, ?, ?, ?, ?)", videos)
        
        conn.commit()
    conn.commit(); conn.close()

def db_log_deal(buyer, qty, price, cost, margin):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO deals (timestamp, buyer, qty, price, cost, margin) VALUES (?, ?, ?, ?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M"), buyer, qty, price, cost, margin))
    conn.commit(); conn.close()

def db_add_employee(name, role, salary):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO employees (name, role, salary, status) VALUES (?, ?, ?, ?)", (name, role, salary, "Active"))
    conn.commit(); conn.close()

def db_post_chat(user, message):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO chat_logs (timestamp, user, message) VALUES (?, ?, ?)", (datetime.now().strftime("%H:%M"), user, message))
    conn.commit(); conn.close()

def db_add_video(title, url, category):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO video_library (title, type, url, category, rating) VALUES (?, ?, ?, ?, 'None')", (title, "youtube", url, category))
    conn.commit(); conn.close()

def db_delete_video(vid_id):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("DELETE FROM video_library WHERE id=?", (vid_id,))
    c.execute("DELETE FROM video_notes WHERE video_id=?", (vid_id,))
    conn.commit(); conn.close()

def db_add_note(vid_id, note):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO video_notes (video_id, user, note, timestamp) VALUES (?, ?, ?, ?)", (vid_id, "CEO", note, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit(); conn.close()

# NEW: RATE VIDEO
def db_rate_video(vid_id, rating):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("UPDATE video_library SET rating=? WHERE id=?", (rating, vid_id))
    conn.commit(); conn.close()

def db_fetch_table(table_name):
    conn = sqlite3.connect(DB_FILE); df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY id DESC", conn); conn.close()
    return df
init_db()

# --- 🔒 SECURITY ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "TEXTILE_KING":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else: st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown('<div class="login-box"><div class="rotex-logo-container"><div class="rotex-text">ROTex</div><div class="rotex-tagline">System v33.0</div></div></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2: st.text_input("IDENTITY VERIFICATION", type="password", on_change=password_entered, key="password", label_visibility="collapsed", placeholder="Enter Key...")
        return False
    return st.session_state["password_correct"]

# --- 🧠 LOGIC & UTILS ---
@st.cache_data(ttl=3600)
def load_market_data():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    df_safe = pd.DataFrame(index=dates)
    df_safe['Cotton_USD'] = np.random.normal(85, 2, 100)
    df_safe['Gas_USD'] = np.random.normal(3.0, 0.1, 100)
    df_safe['Yarn_Fair_Value'] = ((df_safe['Cotton_USD']/100) * 1.6) + (df_safe['Gas_USD'] * 0.15) + 0.40
    try:
        data = yf.download(['CT=F', 'NG=F'], period="1y", interval="1d", progress=False)
        if data is None or data.empty: return df_safe
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0): data = data.xs('Close', level=0, axis=1)
            else: return df_safe
        if 'Close' in data: data = data['Close']
        if data.empty: return df_safe
        data.columns = ['Cotton_USD', 'Gas_USD']
        data['Yarn_Fair_Value'] = ((data['Cotton_USD']/100) * 1.6) + (data['Gas_USD'] * 0.15) + 0.40
        return data.dropna()
    except Exception: return df_safe

def get_news_stealth():
    try: return feedparser.parse(requests.get("https://news.google.com/rss/search?q=Bangladesh+Textile+Industry+when:3d&hl=en-BD&gl=BD&ceid=BD:en").content).entries[:4]
    except: return []

def process_fabric_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = img.copy()
    count = 0
    for c in contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            count += 1
    return output_img, count

def sanitize_text(text):
    if not text: return ""
    return text.encode('latin-1', 'replace').decode('latin-1')

def create_pdf_report(yarn, cotton, gas, news, df_hist):
    plt.figure(figsize=(10, 4)); plt.plot(df_hist.index, df_hist['Yarn_Fair_Value'], color='#00d2ff'); plt.savefig('temp.png'); plt.close()
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 20, "ROTex EXECUTIVE REPORT", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(0, 10, f"Yarn Index: ${yarn:.2f} | Cotton: ${cotton:.2f} | Gas: ${gas:.2f}", ln=True)
    pdf.image('temp.png', x=10, w=190)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14); pdf.cell(0, 10, "Market Intel:", ln=True)
    pdf.set_font("Arial", "", 10)
    for item in news:
        safe_title = sanitize_text(item.title)
        pdf.multi_cell(0, 10, f"- {safe_title}")
    return pdf.output(dest='S').encode('latin-1')

def generate_noise_pattern(freq, chaos):
    w, h = 300, 300
    x = np.linspace(0, freq, w)
    y = np.linspace(0, freq, h)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X + random.random()*chaos) * np.cos(Y + random.random()*chaos)
    Z_norm = cv2.normalize(Z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    Z_color = cv2.applyColorMap(Z_norm, cv2.COLORMAP_JET)
    return Z_color

# --- 🚀 LAUNCH ---
if check_password():
    with st.sidebar:
        st.markdown('<div class="rotex-logo-container"><div class="rotex-text">ROTex</div><div class="rotex-tagline">System Online</div></div>', unsafe_allow_html=True)
        
        # ELITE MENU
        menu = st.radio("MAIN MENU", [
            "MARKET INTELLIGENCE", "COMPETITOR PRICING", "CHAOS THEORY", 
            "ESG PULSE 🌿", "NEURAL SCHEDULER 🧠", "SMART GRID ⚡", 
            "VIDEO VAULT 🎥", "LIVE SUPPORT 💬", 
            "HR COMMAND", "R&D INNOVATION", "QUALITY LAB", 
            "FACTORY STATUS", "FABRIC SCANNER", "LOGISTICS", "COSTING", 
            "DATABASE", "SYSTEM GUIDE"
        ])
        st.divider()
        if st.button("LOGOUT"): st.session_state["password_correct"] = False; st.rerun()

    df = load_market_data()
    if not df.empty: yarn_cost = df['Yarn_Fair_Value'].iloc[-1]
    else: yarn_cost = 4.50

    # 1. MARKET INTELLIGENCE
    if menu == "MARKET INTELLIGENCE":
        st.markdown("## 📡 MARKET INTELLIGENCE")
        st.markdown(f"<div style='background:rgba(0,0,0,0.5); padding:10px; border-radius:5px; white-space:nowrap; overflow:hidden; color:#00ff88; font-family:monospace;'>LIVE FEED: COTTON: ${df['Cotton_USD'].iloc[-1]:.2f} ▲ | GAS: ${df['Gas_USD'].iloc[-1]:.2f} ▼ | YARN FAIR VALUE: ${yarn_cost:.2f} ▲</div>", unsafe_allow_html=True)
        st.write("")
        news_items = get_news_stealth()
        col_metrics, col_btn = st.columns([3, 1])
        with col_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Yarn Index", f"${yarn_cost:.2f}", "+1.2%")
            c2.metric("Cotton Futures", f"${df['Cotton_USD'].iloc[-1]:.2f}", "-0.5%")
            c3.metric("Energy Index", f"${df['Gas_USD'].iloc[-1]:.2f}", "+0.1%")
        with col_btn:
            pdf = create_pdf_report(yarn_cost, df['Cotton_USD'].iloc[-1], df['Gas_USD'].iloc[-1], news_items, df)
            st.download_button("📄 DOWNLOAD RESEARCH PDF", pdf, "ROTex_Executive_Brief.pdf", "application/pdf", use_container_width=True)

        st.markdown("### 📈 Market Trend Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], line=dict(color='#00d2ff', width=3), name='Yarn Index'))
        fig.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

        col_map, col_intel = st.columns([2, 1])
        with col_map:
            st.markdown("### 🗺️ Geopolitical Risk Tracker")
            map_data = pd.DataFrame({'lat': [23.8, 31.2, 21.0, 39.9, 25.2], 'lon': [90.4, 121.4, 105.8, 116.4, 55.3], 'name': ["DHAKA", "SHANGHAI", "HANOI", "BEIJING", "DUBAI"], 'risk': [10, 50, 30, 80, 20], 'color': [[0, 255, 136], [255, 0, 0], [255, 165, 0], [255, 0, 0], [0, 100, 255]]})
            layer = pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_fill_color='color', get_radius=200000, pickable=True)
            view_state = pdk.ViewState(latitude=25, longitude=90, zoom=2, pitch=45)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10", tooltip={"text": "{name}\nRisk Level: {risk}%"}))
        with col_intel:
            st.markdown("### 🧠 Global Feed")
            for item in news_items: st.markdown(f'<div class="info-card" style="font-size:12px; padding:10px;"><a href="{item.link}" target="_blank" style="color:#00d2ff; text-decoration:none;">➤ {item.title[:60]}...</a></div>', unsafe_allow_html=True)

    # 2. COMPETITOR PRICING
    elif menu == "COMPETITOR PRICING":
        st.markdown("## ⚔️ COMPETITOR PRICING SIMULATOR")
        col_ctrl, col_sim = st.columns([1, 2])
        with col_ctrl:
            st.markdown("### 🎛️ Controls")
            fabric = st.selectbox("Fabric Class", ["Cotton Single Jersey", "CVC Fleece", "Poly Mesh"])
            my_quote = st.number_input("Your Quote ($/kg)", 4.50)
            shock = st.slider("Global Price Shock (%)", -20, 20, 0)
        with col_sim:
            base = yarn_cost * (1 + shock/100)
            china = base * 0.94; india = base * 0.96; vietnam = base * 0.98
            diff = my_quote - min(china, india, vietnam)
            prob = max(0, min(100, 100 - (diff * 200))) 
            fig = go.Figure(go.Indicator(mode = "gauge+number", value = prob, title = {'text': "Win Probability"}, gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00d2ff"}}))
            fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("🇨🇳 China", f"${china:.2f}")
            c2.metric("🇮🇳 India", f"${india:.2f}")
            c3.metric("🇻🇳 Vietnam", f"${vietnam:.2f}")

    # 3. CHAOS THEORY
    elif menu == "CHAOS THEORY":
        st.markdown("## ☣️ DOOMSDAY SIMULATOR")
        col_doom1, col_doom2 = st.columns([1, 3])
        with col_doom1:
            st.markdown("### 🌪️ SELECT DISASTER")
            scenario = st.radio("Scenario Trigger", ["None (Business as Usual)", "Suez Canal Blockage (14 Days)", "Cotton Crop Failure (India)", "Cyber Attack (Port System)"])
        with col_doom2:
            if scenario == "None (Business as Usual)":
                st.success("✅ STATUS: OPTIMAL")
                data = [{"source": [90.4, 23.8], "target": [-74.0, 40.7], "color": [0, 255, 136]}] 
                impact_cost = 0; days_left = 45
            elif scenario == "Suez Canal Blockage (14 Days)":
                st.markdown('<div class="chaos-alert"><h3>🚨 ALERT: CANAL BLOCKED</h3><p>Rerouting via Cape of Good Hope (+14 Days)</p></div>', unsafe_allow_html=True)
                data = [{"source": [90.4, 23.8], "target": [18.4, -33.9], "color": [255, 0, 0]}, {"source": [18.4, -33.9], "target": [-74.0, 40.7], "color": [255, 0, 0]}] 
                impact_cost = 25000; days_left = 12
            elif scenario == "Cotton Crop Failure (India)":
                st.markdown('<div class="chaos-alert"><h3>🚨 ALERT: RAW MATERIAL SHORTAGE</h3><p>Price Surge +30% Imminent</p></div>', unsafe_allow_html=True)
                data = [{"source": [77.0, 20.0], "target": [90.4, 23.8], "color": [255, 165, 0]}] 
                impact_cost = 50000; days_left = 20
            elif scenario == "Cyber Attack (Port System)":
                st.markdown('<div class="chaos-alert"><h3>🚨 ALERT: PORT BLACKOUT</h3><p>Zero Movement.</p></div>', unsafe_allow_html=True)
                data = []; impact_cost = 100000; days_left = 3
            st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ArcLayer", data=data, get_width=8, get_source_position="source", get_target_position="target", get_source_color="color", get_target_color="color")], initial_view_state=pdk.ViewState(latitude=20, longitude=10, zoom=1, pitch=40), map_style="mapbox://styles/mapbox/dark-v10"))
            c1, c2, c3 = st.columns(3)
            c1.metric("Financial Impact", f"-${impact_cost:,}", delta_color="inverse")
            c2.metric("Days to Shutdown", f"{days_left} Days", delta_color="inverse" if days_left < 15 else "normal")

    # ESG PULSE
    elif menu == "ESG PULSE 🌿":
        st.markdown("## 🌿 ESG CARBON COMMAND")
        col_esg1, col_esg2 = st.columns([1, 2])
        with col_esg1:
            st.markdown("### 🏭 Production Input")
            daily_prod = st.slider("Daily Production (kg)", 1000, 20000, 5000)
            energy_mix = st.radio("Energy Source", ["National Grid (Heavy Gas)", "Solar Hybrid (30%)", "Coal (Legacy)"])
            co2_factor = 0.6 if energy_mix == "Solar Hybrid (30%)" else (0.9 if "Grid" in energy_mix else 1.2)
            total_co2 = daily_prod * co2_factor
            if co2_factor < 0.8: st.success("✅ EU EXPORT READY")
            else: st.warning("⚠️ SURCHARGE RISK (CBAM)")
        with col_esg2:
            st.markdown("### 💨 Real-Time Emissions Tracker")
            dates = pd.date_range(end=datetime.today(), periods=30)
            df_esg = pd.DataFrame({"Date": dates, "CO2 (Tons)": np.random.normal(total_co2/1000, 0.5, 30)})
            fig = px.area(df_esg, x="Date", y="CO2 (Tons)", color_discrete_sequence=["#00ff88"])
            fig.add_hline(y=4.0, line_dash="dot", line_color="red", annotation_text="EU Limit")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    # NEURAL SCHEDULER
    elif menu == "NEURAL SCHEDULER 🧠":
        st.markdown("## 🧠 NEURAL PRODUCTION SCHEDULER")
        if st.button("🤖 GENERATE OPTIMAL SCHEDULE"):
            df_gantt = pd.DataFrame([
                dict(Task="Order #901 (H&M)", Start='2026-02-01', Finish='2026-02-05', Machine="Knitting-A", Status="Running"),
                dict(Task="Order #902 (Zara)", Start='2026-02-02', Finish='2026-02-06', Machine="Dyeing-1", Status="Scheduled"),
                dict(Task="Order #903 (Uniqlo)", Start='2026-02-06', Finish='2026-02-10', Machine="Knitting-A", Status="Pending"),
                dict(Task="Maintenance", Start='2026-02-05', Finish='2026-02-06', Machine="Knitting-B", Status="Critical"),
            ])
            fig = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Machine", color="Status", 
                              color_discrete_map={"Running": "#00d2ff", "Critical": "#ff0055", "Scheduled": "#00ff88"})
            fig.update_yaxes(categoryorder="total ascending")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', title="AI Generated Timeline (Efficiency: 98%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="skunk-card" style="text-align:center;"><h3>AWAITING INPUT</h3><p>Click Generate to run Neural Planning Algorithm</p></div>', unsafe_allow_html=True)

    # SMART GRID
    elif menu == "SMART GRID ⚡":
        st.markdown("## ⚡ SMART ENERGY GRID")
        c1, c2, c3 = st.columns(3)
        c1.metric("Live Load", "450 kW", "Peak Zone")
        c2.metric("Hourly Cost", "BDT 4,500", "+15% (Peak Rate)")
        c3.metric("Power Factor", "0.98", "Optimal")
        
        x = list(range(24))
        y = [random.randint(300, 500) if i > 10 and i < 18 else random.randint(100, 200) for i in x]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', line=dict(color='#ffaa00', width=2), name="Power Usage (kW)"))
        fig.add_vrect(x0=17, x1=23, annotation_text="PEAK HOURS (Avoid)", annotation_position="top left", fillcolor="red", opacity=0.1, line_width=0)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Hour of Day", yaxis_title="Kilowatts (kW)")
        st.plotly_chart(fig, use_container_width=True)

    # --- 🆕 NEW FEATURE: VIDEO VAULT (YOUTUBE REDESIGN + RECS) ---
    elif menu == "VIDEO VAULT 🎥":
        st.markdown("## 🎥 ROTex STREAM")
        
        # 1. YOUTUBE-STYLE HEADER (Search + Upload)
        c_search, c_add = st.columns([4, 1])
        with c_search:
            search_query = st.text_input("Search", placeholder="Search your library...", label_visibility="collapsed")
        with c_add:
            show_upload = st.checkbox("➕ ADD VIDEO", help="Open Upload Studio")

        # 2. UPLOAD STUDIO (Hidden by default)
        if show_upload:
            with st.container():
                st.markdown("<div style='background: #111; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #333;'>", unsafe_allow_html=True)
                u_c1, u_c2 = st.columns([1, 1])
                with u_c1:
                    st.markdown("#### 🔗 YouTube Link")
                    yt_url = st.text_input("URL", key="yt_url_in")
                    yt_title = st.text_input("Title", key="yt_title_in")
                    yt_cat = st.selectbox("Category", ["Training", "R&D", "Market", "Safety", "General"], key="yt_cat_in")
                    if st.button("PUBLISH TO VAULT", use_container_width=True):
                        if yt_url and yt_title:
                            db_add_video(yt_title, yt_url, yt_cat)
                            st.toast("✅ Video Published!"); time.sleep(1); st.rerun()
                with u_c2:
                    st.markdown("#### 📂 Local File")
                    up_file = st.file_uploader("Drag & Drop MP4", type=['mp4'])
                    if up_file: st.video(up_file)
                st.markdown("</div>", unsafe_allow_html=True)

        # 3. FILTER CHIPS
        df_videos = db_fetch_table("video_library")
        categories = ["All"] + list(df_videos['category'].unique()) if not df_videos.empty else ["All"]
        selected_cat = st.radio("Filters", categories, horizontal=True, label_visibility="collapsed")

        # 4. CONTENT GRID + RECOMMENDATION ENGINE
        if not df_videos.empty:
            filtered_df = df_videos.copy()
            if search_query: filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False)]
            if selected_cat != "All": filtered_df = filtered_df[filtered_df['category'] == selected_cat]

            if not filtered_df.empty:
                for index, row in filtered_df.iterrows():
                    with st.container():
                        c_video, c_recs = st.columns([2.5, 1])
                        
                        # --- LEFT: VIDEO PLAYER ---
                        with c_video:
                            st.markdown(f"<div class='video-card'>", unsafe_allow_html=True)
                            clean_url = row['url'].replace("/live/", "/watch?v=")
                            st.video(clean_url)
                            
                            # Meta Data & Rating Badge
                            rating_icon = "🤍"
                            if row['rating'] == 'Like': rating_icon = "▲"
                            elif row['rating'] == 'Dislike': rating_icon = "▼"
                            
                            m1, m2 = st.columns([3, 1])
                            with m1:
                                st.markdown(f"<div style='font-size:18px; font-weight:700; color:white;'>{row['title']}</div>", unsafe_allow_html=True)
                                st.caption(f"{row['category']} • Rating: {rating_icon}")
                            with m2:
                                c_a, c_b, c_c = st.columns(3)
                                # CYBER ICONS
                                with c_a: 
                                    if st.button("▲", key=f"up_{row['id']}", help="Like"): db_rate_video(row['id'], "Like"); st.rerun()
                                with c_b: 
                                    if st.button("▼", key=f"dn_{row['id']}", help="Dislike"): db_rate_video(row['id'], "Dislike"); st.rerun()
                                with c_c: 
                                    if st.button("🗑️", key=f"dl_{row['id']}", help="Delete"): db_delete_video(row['id']); st.rerun()
                            
                            with st.expander("📝 Notes"):
                                n_in = st.text_input("Add Note", key=f"n_{row['id']}")
                                if st.button("Save", key=f"ns_{row['id']}") and n_in: db_add_note(row['id'], n_in); st.rerun()
                                # Notes would display here
                            st.markdown("</div>", unsafe_allow_html=True)

                        # --- RIGHT: RECOMMENDATION ENGINE ---
                        with c_recs:
                            st.markdown(f"**Recommended**")
                            rec_topics = [
                                f"Latest {row['category']} Trends",
                                f"Advanced {row['category']} Techniques",
                                f"Global {row['category']} News"
                            ]
                            for topic in rec_topics:
                                search_url = f"https://www.youtube.com/results?search_query={topic.replace(' ', '+')}"
                                st.markdown(f"""
                                <a href="{search_url}" target="_blank" style="text-decoration:none;">
                                    <div class="rec-card">
                                        <div class="rec-title">▶ {topic}</div>
                                        <div class="rec-meta">YouTube Search • {row['category']}</div>
                                    </div>
                                </a>
                                """, unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.info("No videos found matching criteria.")
        else:
            st.warning("Vault Empty. Click '➕ ADD VIDEO' to begin.")

    # LIVE SUPPORT
    elif menu == "LIVE SUPPORT 💬":
        st.markdown("## 💬 TACTICAL COMMS")
        chat_container = st.container()
        with st.form("chat_form", clear_on_submit=True):
            user_msg = st.text_input("Enter Message:", placeholder="Report status...")
            submitted = st.form_submit_button("SEND")
            if submitted and user_msg: db_post_chat("CEO (You)", user_msg)
        with chat_container:
            df_chat = db_fetch_table("chat_logs")
            if not df_chat.empty:
                for index, row in df_chat.head(10).iterrows():
                    if row['user'] == "CEO (You)":
                        st.markdown(f"<div class='chat-bubble-me'><b>{row['user']}</b> [{row['timestamp']}]<br>{row['message']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chat-bubble-other'><b>{row['user']}</b> [{row['timestamp']}]<br>{row['message']}</div>", unsafe_allow_html=True)

    # HR COMMAND
    elif menu == "HR COMMAND":
        st.markdown("## 👥 HUMAN RESOURCES COMMAND")
        hr_tabs = st.tabs(["📋 Staff Directory", "💰 Payroll Engine", "⏱️ Attendance Log"])
        with hr_tabs[0]:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("### Add New Hire")
                name = st.text_input("Full Name"); role = st.selectbox("Designation", ["Operator", "Supervisor", "Manager", "QC Inspector"]); salary = st.number_input("Base Salary (BDT)", 12000)
                if st.button("Onboard Employee"):
                    db_add_employee(name, role, salary)
                    st.success(f"Welcome, {name}!")
            with c2:
                st.markdown("### Active Roster")
                st.dataframe(db_fetch_table("employees"), use_container_width=True)
        with hr_tabs[1]:
            st.markdown("### 💸 Batch Payroll Processor")
            if st.button("RUN MONTHLY PAYROLL"):
                progress = st.progress(0)
                for i in range(100): time.sleep(0.01); progress.progress(i+1)
                st.success("✅ Payroll Generated")
        with hr_tabs[2]:
            st.markdown("### ⏱️ Live Attendance")
            att_data = pd.DataFrame({"Employee": ["Rahim", "Karim", "Fatima", "Suma"], "Time In": ["08:01 AM", "08:05 AM", "07:55 AM", "08:10 AM"], "Status": ["On Time", "On Time", "Early", "Late"]})
            st.table(att_data)

    # R&D INNOVATION
    elif menu == "R&D INNOVATION":
        st.markdown("## 🔬 R&D INNOVATION LAB")
        tab1, tab2, tab3 = st.tabs(["🔊 Loom Whisperer", "🧬 Algo-Weaver", "⛓️ Digital Passport"])
        with tab1:
            if st.button("SCAN FREQUENCIES"):
                x = np.linspace(-5, 5, 100); y = np.linspace(-5, 5, 100); X, Y = np.meshgrid(x, y); R = np.sqrt(X**2 + Y**2); Z = np.sin(R)
                fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')])
                fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            c1, c2 = st.columns(2); freq = c1.slider("Pattern Frequency", 1, 20, 10); chaos = c2.slider("Chaos Factor", 1, 10, 5)
            if st.button("GENERATE"):
                st.image(generate_noise_pattern(freq, chaos), use_column_width=True, channels="BGR")
        with tab3: st.info("System Operational. Minting active.")

    # QUALITY LAB
    elif menu == "QUALITY LAB":
        st.markdown("## 🧪 QUALITY CONTROL LAB")
        test = st.selectbox("Select Protocol", ["GSM Calc", "Shrinkage Sim", "AQL Inspector"])
        if test == "GSM Calc":
            c1, c2 = st.columns(2); w = c1.number_input("Sample Weight (g)", 2.5); a = c2.selectbox("Cut Size", ["100 cm²", "A4"])
            if st.button("CALCULATE GSM"):
                res = w * 100 if a == "100 cm²" else w * 16
                st.metric("RESULT", f"{res:.1f} g/m²")
        elif test == "Shrinkage Sim":
            st.write("### 📏 Dimensional Stability")
            c1, c2 = st.columns(2); l_b = c1.number_input("Length Before (cm)", 50.0); l_a = c2.number_input("Length After (cm)", 48.0)
            c3, c4 = st.columns(2); w_b = c3.number_input("Width Before (cm)", 50.0); w_a = c4.number_input("Width After (cm)", 49.0)
            if st.button("CALCULATE SHRINKAGE"):
                shrink_l = ((l_b - l_a) / l_b) * 100; shrink_w = ((w_b - w_a) / w_b) * 100
                col_res1, col_res2 = st.columns(2); col_res1.metric("Length Shrinkage", f"-{shrink_l:.1f}%"); col_res2.metric("Width Shrinkage", f"-{shrink_w:.1f}%")
        elif test == "AQL Inspector":
            qty = st.number_input("Lot Qty", 5000); st.info("Inspect 200 pcs. Reject if > 10 defects (AQL 2.5).")

    # FACTORY STATUS
    elif menu == "FACTORY STATUS":
        st.markdown("## 🏭 FACTORY STATUS")
        c1, c2, c3 = st.columns(3)
        fig_speed = go.Figure(go.Indicator(mode="gauge+number", value=random.randint(750, 850), title={'text': "Loom RPM"}, gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#00ff88"}}))
        fig_speed.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        c1.plotly_chart(fig_speed, use_container_width=True)
        fig_temp = go.Figure(go.Indicator(mode="gauge+number", value=random.randint(28, 40), title={'text': "Temp (°C)"}, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#ff0055"}}))
        fig_temp.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        c2.plotly_chart(fig_temp, use_container_width=True)
        c3.info("Loom #4: Bearing Failure predicted in 48 hours.")

    # FABRIC SCANNER
    elif menu == "FABRIC SCANNER":
        st.markdown("## 👁️ FABRIC DEFECT SCANNER")
        up = st.file_uploader("Upload Fabric Feed")
        if up:
            img, cnt = process_fabric_image(up)
            st.image(img, caption=f"Neural Net Detected: {cnt} Anomalies", use_column_width=True)
            if cnt > 0: st.error("⚠️ QUALITY THRESHOLD BREACHED")
            else: st.success("✅ GRADE A CERTIFIED")

    # LOGISTICS
    elif menu == "LOGISTICS":
        st.markdown("## 🌍 GLOBAL LOGISTICS")
        data = [{"source": [90.4, 23.8], "target": [-74.0, 40.7], "color": [0, 255, 136]}] 
        layer = pdk.Layer("ArcLayer", data=data, get_width=5, get_source_position="source", get_target_position="target", get_source_color="color", get_target_color="color")
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1, pitch=40), map_style="mapbox://styles/mapbox/dark-v10"))
        st.dataframe(pd.DataFrame({"Vessel": ["Ever Given", "Maersk Alabama"], "Dest": ["NYC", "Hamburg"], "ETA": ["4 Days", "12 Days"], "Status": ["On Time", "Delayed"]}), use_container_width=True)

    # COSTING
    elif menu == "COSTING":
        st.markdown("## 💰 COSTING CALCULATOR")
        p = st.number_input("Price", 4.50)
        margin = p - (yarn_cost+0.75)
        st.metric("Margin", f"${margin:.2f}/kg")
        if st.button("Save"): db_log_deal("Test", 0, p, 0, 0); st.success("Saved")

    elif menu == "DATABASE":
        st.markdown("## 🗄️ ORDER HISTORY")
        st.dataframe(db_fetch_table("deals"), use_container_width=True)

    # SYSTEM GUIDE
    elif menu == "SYSTEM GUIDE":
        st.markdown("## 🎓 ROTex SYSTEM GUIDE")
        tab_guide1, tab_guide2, tab_guide3, tab_guide4 = st.tabs(["Market Logic", "Quality Standards", "R&D Blueprints", "Training Video"])
        with tab_guide1:
            st.markdown('<div class="guide-card"><h3>📈 HOW PRICING WORKS</h3><p>Reverse-Costing Algorithm.</p></div>', unsafe_allow_html=True)
            if graphviz_installed: st.graphviz_chart('''digraph G { rankdir=LR; node [shape=box, style=filled, fillcolor="#222", fontcolor="white"]; A [label="NYMEX Cotton"]; B [label="Henry Hub Gas"]; C [label="Processing Cost"]; D [label="FINAL YARN COST"]; A -> D; B -> D; C -> D; }''')
            else: st.warning("Schematic unavailable.")
        with tab_guide2:
            st.markdown('<div class="guide-card"><h3>🧪 QUALITY PROTOCOLS (ISO 6330)</h3></div>', unsafe_allow_html=True)
            col_g1, col_g2 = st.columns(2)
            col_g1.info("**GSM Tolerance:** ±5%\n- Under 130: Reject (Sheer)\n- 140-160: Standard\n- 180+: Heavy")
            col_g2.info("**Shrinkage Tolerance:** ±5%\n- Length: Max -5%\n- Width: Max -5%\n- Spirality: Max 4%")
        with tab_guide3:
            st.markdown('<div class="guide-card"><h3>👽 ALIEN TECH BLUEPRINTS</h3></div>', unsafe_allow_html=True)
            if graphviz_installed: st.graphviz_chart('''digraph G { rankdir=TD; node [shape=box, style=filled, fillcolor="#222", fontcolor="white"]; Mic -> FFT -> Freq -> AI -> Alert; }''')
        with tab_guide4:
             st.markdown('<div class="guide-card"><h3>🎥 OPERATOR TRAINING</h3></div>', unsafe_allow_html=True)
             st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
             st.caption("Module 1: System Calibration & Maintenance")
