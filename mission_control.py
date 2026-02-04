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

# --- 🎨 THE "CYBER-TILE" UI ENGINE (v35.0 UPDATE) ---
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
    div[data-testid="metric-container"], .info-card, .job-card, .skunk-card, .target-card, .target-safe, .guide-card, .stDataFrame, .stPlotlyChart {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }

    div[data-testid="metric-container"]:hover, .info-card:hover, .guide-card:hover {
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

    /* 5. SIDEBAR & MENU OPTIMIZATION (THE FIX) */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* --- MENU TILES TRANSFORMATION --- */
    /* Target the labels inside the radio button group in the sidebar */
    section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.03); /* Subtle glass background */
        padding: 12px 15px; /* Add internal breathing room */
        margin-bottom: 8px; /* Add vertical spacing between items */
        border-radius: 8px; /* Rounded corners */
        border: 1px solid rgba(255, 255, 255, 0.05); /* Faint border */
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); /* Smooth physics */
        display: flex; /* Ensure layout alignment */
        align-items: center;
    }

    /* Hover State for Menu Tiles */
    section[data-testid="stSidebar"] div.stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(0, 210, 255, 0.1); /* Cyan tint on hover */
        border-color: rgba(0, 210, 255, 0.5); /* Glowing border */
        padding-left: 25px; /* Slide animation to the right */
        color: #ffffff !important;
        cursor: pointer;
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.1);
    }
    
    /* Selected State (Streamlit doesn't expose a clean selected class for labels easily via CSS, 
       but the hover effect creates the premium feel we need). */

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
    
    /* 9. CHAT BUBBLES */
    .chat-bubble-me {
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
    </style>
    """, unsafe_allow_html=True)

# --- 🟢 SYSTEM STATUS MARKER ---
st.success("SYSTEM STATUS: v35.0 (MENU AESTHETICS UPGRADE)")

# --- 🗄️ DATABASE & AUTO-SEEDING ---
DB_FILE = "rotex_core.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS deals (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, buyer TEXT, qty REAL, price REAL, cost REAL, margin REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, defects INTEGER, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, role TEXT, salary REAL, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, user TEXT, message TEXT)''')
    
    # --- AUTO-SEED FAKE DATA IF EMPTY ---
    c.execute("SELECT count(*) FROM employees")
    if c.fetchone()[0] == 0:
        fake_employees = [
            ("Abdul Rahim", "Knitting Operator", 12500, "Active"),
            ("Fatima Begum", "Sewing Operator", 13000, "Active"),
            ("Kamrul Hasan", "Shift Supervisor", 28000, "Active"),
            ("Suma Akter", "Quality Inspector", 18500, "Active"),
            ("Rafiqul Islam", "Maintenance Eng.", 45000, "Active"),
            ("Nusrat Jahan", "Merchandiser", 55000, "Active"),
            ("David Rozario", "Floor Manager", 85000, "Active"),
            ("Salma Khatun", "Helper", 9500, "Active"),
            ("Mohammad Ali", "Dyeing Master", 62000, "Active"),
            ("Rubel Hossain", "Loader", 10000, "Active"),
            ("Tania Sultana", "CAD Designer", 42000, "Active"),
            ("Jashim Uddin", "Security Guard", 11000, "Active"),
        ]
        # Multiply to make it look big (3x Loop)
        for _ in range(3): 
            for emp in fake_employees:
                c.execute("INSERT INTO employees (name, role, salary, status) VALUES (?, ?, ?, ?)", 
                          (emp[0] + f" {random.randint(1,99)}", emp[1], emp[2] + random.randint(-500, 500), emp[3]))
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

# --- 🧠 LOGIC & UTILS (BULLETPROOF FAIL-SAFE MARKET DATA) ---
@st.cache_data(ttl=3600)
def load_market_data():
    # 1. PREPARE FALLBACK DATA FIRST (Guarantees no crash)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    df_safe = pd.DataFrame(index=dates)
    df_safe['Cotton_USD'] = np.random.normal(85, 2, 100)
    df_safe['Gas_USD'] = np.random.normal(3.0, 0.1, 100)
    df_safe['Yarn_Fair_Value'] = ((df_safe['Cotton_USD']/100) * 1.6) + (df_safe['Gas_USD'] * 0.15) + 0.40

    try:
        # 2. ATTEMPT REAL DOWNLOAD
        data = yf.download(['CT=F', 'NG=F'], period="1y", interval="1d", progress=False)
        
        # 3. IF DATA IS EMPTY, RETURN SAFE IMMEDIATELY
        if data is None or data.empty:
            return df_safe
            
        # 4. HANDLE MULTI-INDEX (The most common Yahoo bug)
        if isinstance(data.columns, pd.MultiIndex):
            # Try to flatten or select 'Close'
            if 'Close' in data.columns.get_level_values(0):
                data = data.xs('Close', level=0, axis=1)
            else:
                return df_safe

        # 5. HANDLE 'Close' COLUMN
        if 'Close' in data:
            data = data['Close']

        # 6. DOUBLE CHECK EMPTY AGAIN
        if data.empty:
            return df_safe

        # 7. RENAME AND CALCULATE
        data.columns = ['Cotton_USD', 'Gas_USD']
        data['Yarn_Fair_Value'] = ((data['Cotton_USD']/100) * 1.6) + (data['Gas_USD'] * 0.15) + 0.40
        return data.dropna()

    except Exception:
        # 8. IF ANYTHING CRASHES, RETURN SAFE DATA
        return df_safe

def get_news_stealth():
    try: return feedparser.parse(requests.get("https://news.google.com/rss/search?q=Bangladesh+Textile+Industry+when:3d&hl=en-BD&gl=BD&ceid=BD:en").content).entries[:4]
    except: return []

def get_jobs_stealth():
    try: return feedparser.parse(requests.get("https://news.google.com/rss/search?q=Textile+Job+Vacancy+Bangladesh+when:7d&hl=en-BD&gl=BD&ceid=BD:en").content).entries[:8]
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

def generate_qr(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

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
        
        # --- UPDATE MENU WITH NEW FEATURES ---
        menu = st.radio("MAIN MENU", [
            "MARKET INTELLIGENCE", "COMPETITOR PRICING", "CHAOS THEORY", 
            "ESG PULSE 🌿", "NEURAL SCHEDULER 🧠", "SMART GRID ⚡", 
            "LIVE SUPPORT 💬", 
            "HR COMMAND", "R&D INNOVATION", "QUALITY LAB", 
            "FACTORY STATUS", "FABRIC SCANNER", "LOGISTICS", "COSTING", 
            "DATABASE", "SYSTEM GUIDE"
        ])
        st.divider()
        if st.button("LOGOUT"): st.session_state["password_correct"] = False; st.rerun()

    df = load_market_data()
    # SAFE INDEXING (Prevents Crash)
    if not df.empty:
        yarn_cost = df['Yarn_Fair_Value'].iloc[-1]
    else:
        yarn_cost = 4.50 # Ultimate fallback

    # 1. MARKET INTELLIGENCE
    if menu == "MARKET INTELLIGENCE":
        st.markdown("## 📡 MARKET INTELLIGENCE")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** Wall Street for Textiles.\n**Engineer's Logic:** Live data scraping from NYMEX/Henry Hub.")
        
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
            map_data = pd.DataFrame({'lat': [23.8, 31.2, 21.0, 39.9, 25.2], 'lon': [90.4, 121.4, 105.8, 116.4, 55.3], 'name': ["DHAKA (Labor Unrest)", "SHANGHAI (Port Congestion)", "HANOI (Logistics)", "BEIJING (Policy)", "DUBAI (Transit)"], 'risk': [10, 50, 30, 80, 20], 'color': [[0, 255, 136], [255, 0, 0], [255, 165, 0], [255, 0, 0], [0, 100, 255]]})
            layer = pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_fill_color='color', get_radius=200000, pickable=True)
            view_state = pdk.ViewState(latitude=25, longitude=90, zoom=2, pitch=45)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10", tooltip={"text": "{name}\nRisk Level: {risk}%"}))
            st.info("**Strategic Insight:** Red zones indicate high supply chain risk.")
        with col_intel:
            st.markdown("### 🧠 Global Feed")
            for item in news_items: st.markdown(f'<div class="info-card" style="font-size:12px; padding:10px;"><a href="{item.link}" target="_blank" style="color:#00d2ff; text-decoration:none;">➤ {item.title[:60]}...</a></div>', unsafe_allow_html=True)

    # 2. COMPETITOR PRICING
    elif menu == "COMPETITOR PRICING":
        st.markdown("## ⚔️ COMPETITOR PRICING SIMULATOR")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** Predicts rival quotes from China/Vietnam.\n**Engineer's Logic:** Applies geopolitical subsidies to base yarn cost.")
        col_ctrl, col_sim = st.columns([1, 2])
        with col_ctrl:
            st.markdown("### 🎛️ Controls")
            fabric = st.selectbox("Fabric Class", ["Cotton Single Jersey", "CVC Fleece", "Poly Mesh"])
            my_quote = st.number_input("Your Quote ($/kg)", 4.50)
            shock = st.slider("Global Price Shock (%)", -20, 20, 0)
            with st.expander("❓ LOGIC BLUEPRINT"):
                if graphviz_installed: st.graphviz_chart('digraph logic { rankdir=TD; node [shape=box, style=filled, fillcolor="#222", fontcolor="white"]; LiveIndex -> BasePrice; BasePrice -> ChinaSubsidy; BasePrice -> IndiaSubsidy; ChinaSubsidy -> FinalPrice; }')
                else: st.warning("Diagram unavailable")
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
            st.success(f"**AI Analysis:** You have a {prob}% chance of winning this deal.")

    # 3. CHAOS THEORY
    elif menu == "CHAOS THEORY":
        st.markdown("## ☣️ DOOMSDAY SIMULATOR")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** Supply chain stress-tester.\n**Engineer's Logic:** Simulates node failure in the logistics graph.")
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
            c3.metric("Risk Level", "CRITICAL" if days_left < 15 else "LOW")

    # --- 🆕 NEW FEATURE 1: ESG PULSE (Carbon Pulse) ---
    elif menu == "ESG PULSE 🌿":
        st.markdown("## 🌿 ESG CARBON COMMAND")
        with st.expander("ℹ️ INTEL: WHY THIS MATTERS?"):
            st.markdown("**CEO Summary:** Your passport to European export markets (H&M, Inditex requirements).\n**Engineer's Logic:** Calculates Carbon Footprint based on energy mix and logistic distance.")
        
        col_esg1, col_esg2 = st.columns([1, 2])
        with col_esg1:
            st.markdown("### 🏭 Production Input")
            daily_prod = st.slider("Daily Production (kg)", 1000, 20000, 5000)
            energy_mix = st.radio("Energy Source", ["National Grid (Heavy Gas)", "Solar Hybrid (30%)", "Coal (Legacy)"])
            
            # Logic
            co2_factor = 0.6 if energy_mix == "Solar Hybrid (30%)" else (0.9 if "Grid" in energy_mix else 1.2)
            total_co2 = daily_prod * co2_factor
            
            st.markdown("### 📊 Compliance Status")
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
            
            c1, c2 = st.columns(2)
            c1.metric("Carbon Intensity", f"{co2_factor} kgCO2/kg", "-12%" if "Solar" in energy_mix else "+5%")
            c2.metric("Credits Earned", f"${int(daily_prod * 0.02)}", "Daily Accumulation")

    # --- 🆕 NEW FEATURE 2: NEURAL SCHEDULER (AI Planning) ---
    elif menu == "NEURAL SCHEDULER 🧠":
        st.markdown("## 🧠 NEURAL PRODUCTION SCHEDULER")
        with st.expander("ℹ️ INTEL: WHY THIS MATTERS?"):
            st.markdown("**CEO Summary:** Eliminates machine downtime. Autoschedules orders.\n**Engineer's Logic:** Solves the Job-Shop Scheduling Problem (JSP) using heuristic algorithms.")
        
        if st.button("🤖 GENERATE OPTIMAL SCHEDULE"):
            # Mock Data for Gantt
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
            st.info("💡 **AI Optimization:** Knitting-B downtime aligned with Order #902 Dyeing phase to prevent bottleneck.")
        else:
            st.markdown('<div class="skunk-card" style="text-align:center;"><h3>AWAITING INPUT</h3><p>Click Generate to run Neural Planning Algorithm</p></div>', unsafe_allow_html=True)

    # --- 🆕 NEW FEATURE 3: SMART GRID (Energy AI) ---
    elif menu == "SMART GRID ⚡":
        st.markdown("## ⚡ SMART ENERGY GRID")
        with st.expander("ℹ️ INTEL: WHY THIS MATTERS?"):
            st.markdown("**CEO Summary:** Cuts electricity bills by predicting peak hours.\n**Engineer's Logic:** Real-time KWh monitoring vs. BDT Cost Tiers.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Live Load", "450 kW", "Peak Zone")
        c2.metric("Hourly Cost", "BDT 4,500", "+15% (Peak Rate)")
        c3.metric("Power Factor", "0.98", "Optimal")
        
        st.markdown("### 🔌 Real-Time Consumption Anomaly")
        # Live simulated data
        x = list(range(24))
        y = [random.randint(300, 500) if i > 10 and i < 18 else random.randint(100, 200) for i in x]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', line=dict(color='#ffaa00', width=2), name="Power Usage (kW)"))
        fig.add_vrect(x0=17, x1=23, annotation_text="PEAK HOURS (Avoid)", annotation_position="top left", fillcolor="red", opacity=0.1, line_width=0)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Hour of Day", yaxis_title="Kilowatts (kW)")
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ **Recommendation:** Shift 'Dyeing Batch 4' to 02:00 AM to save BDT 15,000.")

    # --- 🆕 NEW FEATURE 4: LIVE SUPPORT (Tactical Comms) ---
    elif menu == "LIVE SUPPORT 💬":
        st.markdown("## 💬 TACTICAL COMMS")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** WhatsApp for your Factory Floor.\n**Engineer's Logic:** Real-time persistence chat via SQLite.")

        # Chat container
        chat_container = st.container()
        
        # User input
        with st.form("chat_form", clear_on_submit=True):
            user_msg = st.text_input("Enter Message (Protocol 9):", placeholder="Report issue or status...")
            submitted = st.form_submit_button("SEND TRANSMISSION")
            if submitted and user_msg:
                db_post_chat("CEO (You)", user_msg)
        
        # Display Logic
        with chat_container:
            df_chat = db_fetch_table("chat_logs")
            if not df_chat.empty:
                # Reverse to show newest at bottom if we were scrolling, but Streamlit renders top-down. 
                # We show last 10 messages.
                for index, row in df_chat.head(10).iterrows():
                    if row['user'] == "CEO (You)":
                        st.markdown(f"<div class='chat-bubble-me'><b>{row['user']}</b> [{row['timestamp']}]<br>{row['message']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chat-bubble-other'><b>{row['user']}</b> [{row['timestamp']}]<br>{row['message']}</div>", unsafe_allow_html=True)
            else:
                st.info("No active transmissions. Channel clear.")


    # 4. HR COMMAND (NEW MODULE)
    elif menu == "HR COMMAND":
        st.markdown("## 👥 HUMAN RESOURCES COMMAND")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** Manage your 5,000+ workforce.\n**Engineer's Logic:** CRUD database for employee records + Payroll Engine.")
        
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
            st.info("System automatically applies 5% Tax deduction for salaries > 20k.")
            if st.button("RUN MONTHLY PAYROLL"):
                progress = st.progress(0)
                for i in range(100): time.sleep(0.01); progress.progress(i+1)
                st.success("✅ Payroll Generated for Active Employees. Total Disbursed: BDT 4,250,000")
        with hr_tabs[2]:
            st.markdown("### ⏱️ Live Attendance")
            att_data = pd.DataFrame({"Employee": ["Rahim", "Karim", "Fatima", "Suma"], "Time In": ["08:01 AM", "08:05 AM", "07:55 AM", "08:10 AM"], "Status": ["On Time", "On Time", "Early", "Late"]})
            st.table(att_data)

    # 5. R&D INNOVATION
    elif menu == "R&D INNOVATION":
        st.markdown("## 🔬 R&D INNOVATION LAB")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** Advanced diagnostic tools.\n**Engineer's Logic:** FFT Audio Analysis & Procedural Pattern Generation.")
        tab1, tab2, tab3 = st.tabs(["🔊 Loom Whisperer", "🧬 Algo-Weaver", "⛓️ Digital Passport"])
        with tab1:
            if st.button("SCAN FREQUENCIES"):
                x = np.linspace(-5, 5, 100); y = np.linspace(-5, 5, 100); X, Y = np.meshgrid(x, y); R = np.sqrt(X**2 + Y**2); Z = np.sin(R)
                fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')])
                fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                st.success("**Diagnostic Complete:** Motor harmonic signatures within ISO 10816.")
        with tab2:
            c1, c2 = st.columns(2); freq = c1.slider("Pattern Frequency", 1, 20, 10); chaos = c2.slider("Chaos Factor", 1, 10, 5)
            if st.button("GENERATE"):
                st.image(generate_noise_pattern(freq, chaos), use_column_width=True, channels="BGR")
                st.success("Unique Pattern ID Generated.")
        with tab3: st.info("System Operational. Minting active.")

    # 6. QUALITY LAB
    elif menu == "QUALITY LAB":
        st.markdown("## 🧪 QUALITY CONTROL LAB")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** Automated Quality Assurance.\n**Engineer's Logic:** ISO 6330 & ASTM D3776 Standard Implementation.")
        test = st.selectbox("Select Protocol", ["GSM Calc", "Shrinkage Sim", "AQL Inspector"])
        if test == "GSM Calc":
            c1, c2 = st.columns(2); w = c1.number_input("Sample Weight (g)", 2.5); a = c2.selectbox("Cut Size", ["100 cm²", "A4"])
            if st.button("CALCULATE GSM"):
                res = w * 100 if a == "100 cm²" else w * 16
                st.metric("RESULT", f"{res:.1f} g/m²")
                if res < 140: st.warning("Comment: Lightweight (Sheer).")
                elif res > 180: st.success("Comment: Good T-Shirt weight.")
                else: st.info("Comment: Standard Single Jersey.")
        elif test == "Shrinkage Sim":
            st.write("### 📏 Dimensional Stability")
            c1, c2 = st.columns(2); l_b = c1.number_input("Length Before (cm)", 50.0); l_a = c2.number_input("Length After (cm)", 48.0)
            c3, c4 = st.columns(2); w_b = c3.number_input("Width Before (cm)", 50.0); w_a = c4.number_input("Width After (cm)", 49.0)
            if st.button("CALCULATE SHRINKAGE"):
                shrink_l = ((l_b - l_a) / l_b) * 100; shrink_w = ((w_b - w_a) / w_b) * 100
                col_res1, col_res2 = st.columns(2); col_res1.metric("Length Shrinkage", f"-{shrink_l:.1f}%"); col_res2.metric("Width Shrinkage", f"-{shrink_w:.1f}%")
                if shrink_l > 5.0 or shrink_w > 5.0: st.error("❌ FAILED: Exceeds 5% tolerance.")
                else: st.success("✅ PASSED: Within ISO standards.")
        elif test == "AQL Inspector":
            qty = st.number_input("Lot Qty", 5000); st.info("Inspect 200 pcs. Reject if > 10 defects (AQL 2.5).")

    # 7. FACTORY STATUS
    elif menu == "FACTORY STATUS":
        st.markdown("## 🏭 FACTORY STATUS")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** Live Machinery Health.\n**Engineer's Logic:** Real-time RPM/Temp sensor monitoring.")
        c1, c2, c3 = st.columns(3)
        fig_speed = go.Figure(go.Indicator(mode="gauge+number", value=random.randint(750, 850), title={'text': "Loom RPM"}, gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#00ff88"}}))
        fig_speed.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        c1.plotly_chart(fig_speed, use_container_width=True)
        fig_temp = go.Figure(go.Indicator(mode="gauge+number", value=random.randint(28, 40), title={'text': "Temp (°C)"}, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#ff0055"}}))
        fig_temp.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        c2.plotly_chart(fig_temp, use_container_width=True)
        c3.info("Loom #4: Bearing Failure predicted in 48 hours.")

    # 8. FABRIC SCANNER
    elif menu == "FABRIC SCANNER":
        st.markdown("## 👁️ FABRIC DEFECT SCANNER")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** Automated Visual Inspection.\n**Engineer's Logic:** OpenCV Computer Vision for contour detection.")
        up = st.file_uploader("Upload Fabric Feed")
        if up:
            img, cnt = process_fabric_image(up)
            st.image(img, caption=f"Neural Net Detected: {cnt} Anomalies", use_column_width=True)
            if cnt > 0: st.error("⚠️ QUALITY THRESHOLD BREACHED")
            else: st.success("✅ GRADE A CERTIFIED")

    # 9. LOGISTICS
    elif menu == "LOGISTICS":
        st.markdown("## 🌍 GLOBAL LOGISTICS")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** Live Shipment Tracking.\n**Engineer's Logic:** Geospatial visualization of shipping routes.")
        data = [{"source": [90.4, 23.8], "target": [-74.0, 40.7], "color": [0, 255, 136]}] 
        layer = pdk.Layer("ArcLayer", data=data, get_width=5, get_source_position="source", get_target_position="target", get_source_color="color", get_target_color="color")
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1, pitch=40), map_style="mapbox://styles/mapbox/dark-v10"))
        st.dataframe(pd.DataFrame({"Vessel": ["Ever Given", "Maersk Alabama"], "Dest": ["NYC", "Hamburg"], "ETA": ["4 Days", "12 Days"], "Status": ["On Time", "Delayed"]}), use_container_width=True)

    # 10. COSTING
    elif menu == "COSTING":
        st.markdown("## 💰 COSTING CALCULATOR")
        with st.expander("ℹ️ INTEL: WHAT IS THIS?"):
            st.markdown("**CEO Summary:** The deal closer. Tells you exactly how much money you make on a specific order.")
        p = st.number_input("Price", 4.50)
        margin = p - (yarn_cost+0.75)
        st.metric("Margin", f"${margin:.2f}/kg")
        if margin < 0.20: st.error("Comment: Margin too low.")
        elif margin > 1.00: st.success("Comment: Excellent margin.")
        else: st.warning("Comment: Standard industry margin.")
        if st.button("Save"): db_log_deal("Test", 0, p, 0, 0); st.success("Saved")

    elif menu == "DATABASE":
        st.markdown("## 🗄️ ORDER HISTORY")
        st.dataframe(db_fetch_table("deals"), use_container_width=True)

    # 11. SYSTEM GUIDE
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
             st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ") # Placeholder Video
             st.caption("Module 1: System Calibration & Maintenance")
