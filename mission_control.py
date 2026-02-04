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

    /* 2. BACKGROUND */
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

    /* 3. GLASS CARDS */
    div[data-testid="metric-container"], .info-card, .job-card, .skunk-card, .target-card, .target-safe, .guide-card, .stDataFrame, .stPlotlyChart, .video-card, .rec-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }

    div[data-testid="metric-container"]:hover, .info-card:hover, .guide-card:hover, .video-card:hover, .rec-card:hover {
        transform: translateY(-5px);
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

    /* 5. SIDEBAR */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
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
        padding-left: 25px;
        color: #ffffff !important;
        cursor: pointer;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.1);
    }

    /* 6. LOGO */
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
    
    /* 8. BUTTONS & CHIPS */
    .stButton > button {
        background: rgba(255,255,255,0.05);
        border: 1px solid #333;
        color: #aaa;
        font-family: 'Inter';
        font-size: 12px;
        border-radius: 20px; /* Pill shape */
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        border-color: #00d2ff;
        color: #00d2ff;
        background: rgba(0, 210, 255, 0.1);
    }

    div[data-testid="stHorizontalBlock"] > div[class*="stRadio"] > div[role="radiogroup"] {
        background: transparent !important;
        border: none !important;
    }
    
    /* 9. CHAT & REC CARD */
    .chat-bubble-me, .note-bubble { background: rgba(0, 210, 255, 0.1); border: 1px solid #00d2ff; border-radius: 15px 15px 0px 15px; padding: 10px; margin: 5px; text-align: right; float: right; clear: both; max-width: 70%; }
    .chat-bubble-other { background: rgba(255, 255, 255, 0.1); border: 1px solid #666; border-radius: 15px 15px 15px 0px; padding: 10px; margin: 5px; text-align: left; float: left; clear: both; max-width: 70%; }
    .note-bubble { text-align: left; float: left; border-radius: 0px 15px 15px 15px; width: 100%; max-width: 100%; border-left: 3px solid #00d2ff; }
    
    /* 10. RECOMMENDATION CARD STYLE */
    .rec-card {
        cursor: pointer;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
    }
    .rec-card:hover {
        border-color: #00d2ff;
        background: rgba(0, 210, 255, 0.05);
    }
    .rec-title { font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 4px; line-height: 1.4; }
    .rec-meta { font-size: 10px; color: #888; font-family: 'Rajdhani'; letter-spacing: 1px; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

# --- 🟢 SYSTEM STATUS MARKER ---
st.success("SYSTEM STATUS: v42.0 (FULL FEATURES RESTORED + REC ENGINE)")

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

    try: c.execute("ALTER TABLE video_library ADD COLUMN rating TEXT DEFAULT 'None'")
    except: pass

    # Seed Data
    c.execute("SELECT count(*) FROM employees")
    if c.fetchone()[0] == 0:
        fake_employees = [("Abdul Rahim", "Operator", 12500, "Active"), ("Fatima Begum", "Operator", 13000, "Active"), ("Kamrul Hasan", "Supervisor", 28000, "Active")]
        for _ in range(5): 
            for emp in fake_employees: c.execute("INSERT INTO employees (name, role, salary, status) VALUES (?, ?, ?, ?)", (emp[0], emp[1], emp[2], emp[3]))
        c.executemany("INSERT INTO video_library (title, type, url, category, rating) VALUES (?, ?, ?, ?, ?)", 
                      [("Textile Factory Safety", "youtube", "https://www.youtube.com/watch?v=1p55GjA1jCQ", "Training", "None"),
                       ("Advanced Knitting Tech", "youtube", "https://www.youtube.com/watch?v=F07gB5lH6qE", "R&D", "None")])
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
    c.execute("DELETE FROM video_library WHERE id=?", (vid_id,)); c.execute("DELETE FROM video_notes WHERE video_id=?", (vid_id,))
    conn.commit(); conn.close()

def db_add_note(vid_id, note):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO video_notes (video_id, user, note, timestamp) VALUES (?, ?, ?, ?)", (vid_id, "CEO", note, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit(); conn.close()

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
        if st.session_state["password"] == "TEXTILE_KING": st.session_state["password_correct"] = True; del st.session_state["password"]
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
    output_img = img.copy(); count = 0
    for c in contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3); count += 1
    return output_img, count

def create_pdf_report(yarn, cotton, gas, news, df_hist):
    plt.figure(figsize=(10, 4)); plt.plot(df_hist.index, df_hist['Yarn_Fair_Value'], color='#00d2ff'); plt.savefig('temp.png'); plt.close()
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 20, "ROTex EXECUTIVE REPORT", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align="C")
    pdf.image('temp.png', x=10, w=190); pdf.ln(10)
    for item in news: pdf.multi_cell(0, 10, f"- {item.title.encode('latin-1', 'replace').decode('latin-1')}")
    return pdf.output(dest='S').encode('latin-1')

def generate_noise_pattern(freq, chaos):
    w, h = 300, 300
    x = np.linspace(0, freq, w); y = np.linspace(0, freq, h); X, Y = np.meshgrid(x, y)
    Z = np.sin(X + random.random()*chaos) * np.cos(Y + random.random()*chaos)
    Z_norm = cv2.normalize(Z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(Z_norm, cv2.COLORMAP_JET)

# --- 🚀 LAUNCH ---
if check_password():
    with st.sidebar:
        st.markdown('<div class="rotex-logo-container"><div class="rotex-text">ROTex</div><div class="rotex-tagline">System Online</div></div>', unsafe_allow_html=True)
        menu = st.radio("MAIN MENU", ["MARKET INTELLIGENCE", "COMPETITOR PRICING", "CHAOS THEORY", "ESG PULSE 🌿", "NEURAL SCHEDULER 🧠", "SMART GRID ⚡", "VIDEO VAULT 🎥", "LIVE SUPPORT 💬", "HR COMMAND", "R&D INNOVATION", "QUALITY LAB", "FACTORY STATUS", "FABRIC SCANNER", "LOGISTICS", "COSTING", "DATABASE", "SYSTEM GUIDE"])
        st.divider(); 
        if st.button("LOGOUT"): st.session_state["password_correct"] = False; st.rerun()

    df = load_market_data()
    yarn_cost = df['Yarn_Fair_Value'].iloc[-1] if not df.empty else 4.50

    if menu == "MARKET INTELLIGENCE":
        st.markdown("## 📡 MARKET INTELLIGENCE")
        st.markdown(f"<div style='background:rgba(0,0,0,0.5); padding:10px; border-radius:5px; white-space:nowrap; overflow:hidden; color:#00ff88; font-family:monospace;'>LIVE FEED: COTTON: ${df['Cotton_USD'].iloc[-1]:.2f} ▲ | GAS: ${df['Gas_USD'].iloc[-1]:.2f} ▼ | YARN FAIR VALUE: ${yarn_cost:.2f} ▲</div>", unsafe_allow_html=True)
        news_items = get_news_stealth()
        col_metrics, col_btn = st.columns([3, 1])
        with col_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Yarn Index", f"${yarn_cost:.2f}", "+1.2%"); c2.metric("Cotton", f"${df['Cotton_USD'].iloc[-1]:.2f}", "-0.5%"); c3.metric("Energy", f"${df['Gas_USD'].iloc[-1]:.2f}", "+0.1%")
        with col_btn:
            pdf = create_pdf_report(yarn_cost, df['Cotton_USD'].iloc[-1], df['Gas_USD'].iloc[-1], news_items, df)
            st.download_button("📄 DOWNLOAD REPORT", pdf, "ROTex_Report.pdf", "application/pdf", use_container_width=True)
        fig = go.Figure(); fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], line=dict(color='#00d2ff', width=3)))
        fig.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        col_map, col_intel = st.columns([2, 1])
        with col_map:
            st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ScatterplotLayer", data=pd.DataFrame({'lat': [23.8, 31.2, 21.0, 39.9, 25.2], 'lon': [90.4, 121.4, 105.8, 116.4, 55.3], 'name': ["DHAKA", "SHANGHAI", "HANOI", "BEIJING", "DUBAI"], 'risk': [10, 50, 30, 80, 20], 'color': [[0, 255, 136], [255, 0, 0], [255, 165, 0], [255, 0, 0], [0, 100, 255]]}), get_position='[lon, lat]', get_fill_color='color', get_radius=200000, pickable=True)], initial_view_state=pdk.ViewState(latitude=25, longitude=90, zoom=2), map_style="mapbox://styles/mapbox/dark-v10"))
        with col_intel:
            for item in news_items: st.markdown(f'<div class="info-card" style="font-size:12px; padding:10px;"><a href="{item.link}" target="_blank" style="color:#00d2ff;">➤ {item.title[:60]}...</a></div>', unsafe_allow_html=True)

    elif menu == "COMPETITOR PRICING":
        st.markdown("## ⚔️ COMPETITOR PRICING")
        c1, c2 = st.columns([1, 2]); 
        with c1: 
            st.selectbox("Fabric", ["Single Jersey", "Fleece"]); shock = st.slider("Shock %", -20, 20, 0)
            my_quote = st.number_input("Your Quote", 4.50)
        base = yarn_cost * (1 + shock/100); 
        fig = go.Figure(go.Indicator(mode = "gauge+number", value = 75, title = {'text': "Win Prob"}, gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00d2ff"}}))
        fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}); c2.plotly_chart(fig, use_container_width=True)
        st.info("Simulation running on active Live Market Data.")

    elif menu == "CHAOS THEORY":
        st.markdown("## ☣️ DOOMSDAY SIMULATOR")
        c1, c2 = st.columns([1, 3]); 
        with c1: scen = st.radio("Scenario", ["None", "Suez Blockage", "Cotton Failure"])
        data = [{"source": [90.4, 23.8], "target": [-74.0, 40.7], "color": [0, 255, 136]}] if scen == "None" else []
        if scen != "None": st.markdown(f'<div class="chaos-alert"><h3>🚨 ALERT: {scen.upper()}</h3></div>', unsafe_allow_html=True)
        st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ArcLayer", data=data, get_width=5, get_source_position="source", get_target_position="target", get_source_color="color", get_target_color="color")], initial_view_state=pdk.ViewState(latitude=20, longitude=10, zoom=1), map_style="mapbox://styles/mapbox/dark-v10"))

    elif menu == "ESG PULSE 🌿":
        st.markdown("## 🌿 ESG PULSE")
        c1, c2 = st.columns(2); c1.metric("Carbon", "0.6 kg/kg", "EU Compliant"); c2.metric("Credits", "$450", "+12%")
        st.plotly_chart(px.area(x=pd.date_range(end=datetime.today(), periods=30), y=np.random.normal(5, 0.5, 30), template="plotly_dark"), use_container_width=True)

    elif menu == "NEURAL SCHEDULER 🧠":
        st.markdown("## 🧠 NEURAL SCHEDULER"); st.button("GENERATE SCHEDULE")
        st.plotly_chart(px.timeline(pd.DataFrame([dict(Task="Job A", Start='2026-02-01', Finish='2026-02-05', Resource="Loom 1")]), x_start="Start", x_end="Finish", y="Resource", template="plotly_dark"), use_container_width=True)

    elif menu == "SMART GRID ⚡":
        st.markdown("## ⚡ SMART GRID"); c1, c2, c3 = st.columns(3); c1.metric("Load", "450 kW"); c2.metric("Cost", "BDT 4500"); c3.metric("PF", "0.98")
        st.plotly_chart(go.Figure(go.Scatter(y=[random.randint(100,500) for _ in range(24)], fill='tozeroy', line=dict(color='#ffaa00'))).update_layout(template="plotly_dark", height=300), use_container_width=True)

    # --- 🎥 VIDEO VAULT (v42.0 YOUTUBE RECS) ---
    elif menu == "VIDEO VAULT 🎥":
        st.markdown("## 🎥 ROTex STREAM")
        
        # 1. HEADER
        c_search, c_add = st.columns([4, 1])
        with c_search:
            search_query = st.text_input("Search", placeholder="Search library...", label_visibility="collapsed")
        with c_add:
            show_upload = st.checkbox("➕ ADD VIDEO")

        if show_upload:
            with st.container():
                st.markdown("<div style='background: #111; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #333;'>", unsafe_allow_html=True)
                u1, u2 = st.columns(2)
                with u1:
                    yt_url = st.text_input("Link"); yt_title = st.text_input("Title"); yt_cat = st.selectbox("Category", ["Training", "R&D", "Market", "Safety"])
                    if st.button("PUBLISH") and yt_url: db_add_video(yt_title, yt_url, yt_cat); st.rerun()
                with u2: st.file_uploader("Upload File")
                st.markdown("</div>", unsafe_allow_html=True)

        # 2. FILTERS
        df_videos = db_fetch_table("video_library")
        cats = ["All"] + list(df_videos['category'].unique()) if not df_videos.empty else ["All"]
        sel_cat = st.radio("Filters", cats, horizontal=True, label_visibility="collapsed")

        # 3. CONTENT GRID + RECOMMENDATION ENGINE
        if not df_videos.empty:
            if search_query: df_videos = df_videos[df_videos['title'].str.contains(search_query, case=False)]
            if sel_cat != "All": df_videos = df_videos[df_videos['category'] == sel_cat]

            for index, row in df_videos.iterrows():
                with st.container():
                    c_video, c_recs = st.columns([2, 1])
                    
                    # --- LEFT: VIDEO PLAYER ---
                    with c_video:
                        st.markdown(f"<div class='video-card'>", unsafe_allow_html=True)
                        st.video(row['url'].replace("/live/", "/watch?v="))
                        
                        m1, m2 = st.columns([3, 1])
                        with m1:
                            st.markdown(f"<div style='font-size:18px; font-weight:700; color:white;'>{row['title']}</div>", unsafe_allow_html=True)
                            st.caption(f"{row['category']} • Rating: {row['rating']}")
                        with m2:
                            c_a, c_b, c_c = st.columns(3)
                            # CYBER ICONS: ▲ (Like), ▼ (Dislike)
                            with c_a: 
                                if st.button("▲", key=f"up_{row['id']}", help="Like"): db_rate_video(row['id'], "Like"); st.rerun()
                            with c_b: 
                                if st.button("▼", key=f"dn_{row['id']}", help="Dislike"): db_rate_video(row['id'], "Dislike"); st.rerun()
                            with c_c: 
                                if st.button("🗑️", key=f"dl_{row['id']}", help="Delete"): db_delete_video(row['id']); st.rerun()
                        st.markdown("</div>", unsafe_allow_html=True)

                    # --- RIGHT: RECOMMENDATION ENGINE ---
                    with c_recs:
                        st.markdown(f"**Recommended**")
                        # The "Ghost" Algorithm: Suggests based on Category
                        rec_topics = [
                            f"Latest {row['category']} Trends",
                            f"Advanced {row['category']} Techniques",
                            f"Global {row['category']} News"
                        ]
                        for topic in rec_topics:
                            search_url = f"https://www.youtube.com/results?search_query={topic.replace(' ', '+')}"
                            st.markdown(f"""
                            <a href="{search_url}" target="_blank" style="text-decoration:none;">
                                <div class="rec-card" style="margin-bottom:10px; cursor:pointer;">
                                    <div class="rec-title">▶ {topic}</div>
                                    <div class="rec-meta">YouTube Search • {row['category']}</div>
                                </div>
                            </a>
                            """, unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("Library Empty.")

    elif menu == "LIVE SUPPORT 💬":
        st.markdown("## 💬 TACTICAL COMMS")
        with st.form("c"): 
            if st.form_submit_button("SEND") and (msg:=st.text_input("Msg")): db_post_chat("CEO", msg)
        df_c = db_fetch_table("chat_logs")
        for _, r in df_c.head(5).iterrows(): st.info(f"{r['user']}: {r['message']}")

    elif menu == "HR COMMAND":
        st.markdown("## 👥 HUMAN RESOURCES COMMAND")
        hr_tabs = st.tabs(["📋 Staff Directory", "💰 Payroll Engine", "⏱️ Attendance Log"])
        with hr_tabs[0]:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("### Add New Hire")
                name = st.text_input("Full Name"); role = st.selectbox("Designation", ["Operator", "Supervisor"])
                if st.button("Onboard Employee"): db_add_employee(name, role, 12000); st.success("Added")
            with c2: st.dataframe(db_fetch_table("employees"), use_container_width=True)
        with hr_tabs[1]: st.markdown("### 💸 Payroll"); st.button("RUN PAYROLL"); st.success("Payroll Active")
        with hr_tabs[2]: st.table(pd.DataFrame({"Staff": ["Rahim", "Karim"], "Status": ["On Time", "Late"]}))

    elif menu == "R&D INNOVATION":
        st.markdown("## 🔬 R&D INNOVATION LAB")
        tab1, tab2, tab3 = st.tabs(["🔊 Loom Whisperer", "🧬 Algo-Weaver", "⛓️ Digital Passport"])
        with tab1: st.button("SCAN FREQUENCIES"); st.success("System Nominal")
        with tab2: st.image(generate_noise_pattern(10, 5))
        with tab3: st.info("Active")

    elif menu == "QUALITY LAB":
        st.markdown("## 🧪 QUALITY LAB"); st.metric("GSM", "150", "OK")

    elif menu == "FACTORY STATUS":
        st.markdown("## 🏭 FACTORY STATUS")
        c1, c2 = st.columns(2); c1.metric("RPM", "850"); c2.metric("Temp", "35°C")
        st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=850)), use_container_width=True)

    elif menu == "FABRIC SCANNER":
        st.markdown("## 👁️ SCANNER"); st.file_uploader("Upload Fabric")

    elif menu == "LOGISTICS":
        st.markdown("## 🌍 LOGISTICS"); st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ArcLayer", data=[{"source": [90.4, 23.8], "target": [-74.0, 40.7], "color": [0, 255, 136]}])], initial_view_state=pdk.ViewState(latitude=20, zoom=1)))

    elif menu == "COSTING":
        st.markdown("## 💰 COSTING"); st.metric("Margin", "$0.85/kg")

    elif menu == "DATABASE":
        st.markdown("## 🗄️ ARCHIVES"); st.dataframe(db_fetch_table("deals"))

    elif menu == "SYSTEM GUIDE":
        st.markdown("## 🎓 GUIDE"); st.info("v42.0 Manual")
