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

# --- 🌑 PAGE CONFIGURATION (ELITE MODE) ---
st.set_page_config(
    page_title="ROTex // PREDATOR ELITE",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded" # Expanded to show off the new menu
)

# --- 🎨 THE "ELITE" CSS INJECTION ---
# This is where the magic happens. 
st.markdown("""
    <style>
    /* 1. IMPORT FUTURISTIC FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Rajdhani:wght@500;700;800&family=Syncopate:wght@700&display=swap');

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

    /* 4. TYPOGRAPHY OVERHAUL */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #fff, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(0, 210, 255, 0.3);
    }
    
    .rotex-logo {
        font-family: 'Syncopate', sans-serif;
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff 0%, #ffffff 50%, #00d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 10px #00d2ff, 0 0 20px #00d2ff; }
        to { text-shadow: 0 0 20px #00d2ff, 0 0 30px #00d2ff; }
    }

    /* 5. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    div[class*="stRadio"] > label {
        color: #888 !important;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        transition: color 0.3s;
    }
    div[class*="stRadio"] > label:hover {
        color: #00d2ff !important;
        cursor: pointer;
    }

    /* 6. NEON BUTTONS */
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

    /* 7. SCROLLBAR CUSTOMIZATION */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: #00d2ff; }
    
    /* 8. ALERTS & ANIMATIONS */
    .chaos-alert { 
        border-left: 4px solid #ff0000 !important; 
        background: rgba(255, 0, 0, 0.1) !important; 
        animation: pulse-red 1.5s infinite; 
    }
    @keyframes pulse-red { 
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4); } 
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 0, 0); } 
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } 
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .status-live { background: rgba(0, 255, 136, 0.2); color: #00ff88; border: 1px solid #00ff88; }
    
    </style>
    """, unsafe_allow_html=True)

# --- 🗄️ DATABASE & AUTO-SEEDING (UNCHANGED LOGIC) ---
DB_FILE = "rotex_core.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS deals (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, buyer TEXT, qty REAL, price REAL, cost REAL, margin REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, defects INTEGER, status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, role TEXT, salary REAL, status TEXT)''')
    
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
        # ELITE LOGIN SCREEN
        st.markdown("""
        <div style='background: rgba(0,0,0,0.8); border: 1px solid #00d2ff; padding: 50px; border-radius: 20px; text-align: center; max-width: 450px; margin: auto; box-shadow: 0 0 50px rgba(0, 210, 255, 0.2);'>
            <div class="rotex-logo">ROTex</div>
            <div style='color: #888; letter-spacing: 4px; font-size: 12px; margin-bottom: 20px;'>PREDATOR SYSTEM v33.0</div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2: 
            st.text_input("IDENTITY VERIFICATION", type="password", on_change=password_entered, key="password", label_visibility="collapsed", placeholder="Enter Access Key...")
        return False
    return st.session_state["password_correct"]

# --- 🧠 LOGIC & UTILS (UNCHANGED) ---
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
        st.markdown('<div class="rotex-logo">ROTex</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom: 20px; color: #666;">PREDATOR OS // v33.0</div>', unsafe_allow_html=True)
        
        # ELITE MENU
        menu = st.radio("COMMAND PROTOCOLS", [
            "MARKET INTELLIGENCE", "COMPETITOR PRICING", "CHAOS THEORY", 
            "ESG PULSE 🌿", "NEURAL SCHEDULER 🧠", "SMART GRID ⚡",
            "HR COMMAND", "R&D INNOVATION", "QUALITY LAB", 
            "FACTORY STATUS", "FABRIC SCANNER", "LOGISTICS", "COSTING", 
            "DATABASE", "SYSTEM GUIDE"
        ])
        st.divider()
        if st.button("TERMINATE SESSION"): st.session_state["password_correct"] = False; st.rerun()

    df = load_market_data()
    if not df.empty: yarn_cost = df['Yarn_Fair_Value'].iloc[-1]
    else: yarn_cost = 4.50 

    # 1. MARKET INTELLIGENCE
    if menu == "MARKET INTELLIGENCE":
        st.markdown("## 📡 MARKET INTELLIGENCE")
        
        # STATUS TICKER
        st.markdown(f"""
        <div style='background: rgba(0, 210, 255, 0.1); border: 1px solid #00d2ff; padding: 10px; border-radius: 8px; display: flex; align-items: center; justify-content: space-between;'>
            <span style='color: #00d2ff; font-weight: bold;'>⚡ LIVE FEED</span>
            <span style='color: white;'>COTTON: ${df['Cotton_USD'].iloc[-1]:.2f}</span>
            <span style='color: white;'>GAS: ${df['Gas_USD'].iloc[-1]:.2f}</span>
            <span style='color: #00ff88;'>YARN: ${yarn_cost:.2f}</span>
        </div>
        <br>
        """, unsafe_allow_html=True)

        news_items = get_news_stealth()
        col_metrics, col_btn = st.columns([3, 1])
        with col_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Yarn Index", f"${yarn_cost:.2f}", "+1.2%")
            c2.metric("Cotton Futures", f"${df['Cotton_USD'].iloc[-1]:.2f}", "-0.5%")
            c3.metric("Energy Index", f"${df['Gas_USD'].iloc[-1]:.2f}", "+0.1%")
        with col_btn:
            pdf = create_pdf_report(yarn_cost, df['Cotton_USD'].iloc[-1], df['Gas_USD'].iloc[-1], news_items, df)
            st.download_button("📄 DOWNLOAD REPORT", pdf, "ROTex_Executive.pdf", "application/pdf", use_container_width=True)

        st.markdown("### 📈 Market Trend Analysis")
        # ELITE PLOTLY CONFIG
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], fill='tozeroy', line=dict(color='#00d2ff', width=2), name='Yarn Index'))
        fig.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

        col_map, col_intel = st.columns([2, 1])
        with col_map:
            st.markdown("### 🗺️ Geopolitical Risk")
            map_data = pd.DataFrame({'lat': [23.8, 31.2, 21.0, 39.9, 25.2], 'lon': [90.4, 121.4, 105.8, 116.4, 55.3], 'name': ["DHAKA", "SHANGHAI", "HANOI", "BEIJING", "DUBAI"], 'risk': [10, 50, 30, 80, 20], 'color': [[0, 255, 136], [255, 0, 0], [255, 165, 0], [255, 0, 0], [0, 100, 255]]})
            layer = pdk.Layer("ScatterplotLayer", data=map_data, get_position='[lon, lat]', get_fill_color='color', get_radius=200000, pickable=True, stroked=True, filled=True, radius_min_pixels=10, radius_max_pixels=100)
            view_state = pdk.ViewState(latitude=25, longitude=90, zoom=2, pitch=45)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10", tooltip={"text": "{name}\nRisk: {risk}%"}))
        with col_intel:
            st.markdown("### 🧠 Global Intel")
            for item in news_items: st.markdown(f'<div class="info-card" style="font-size:12px; padding:10px; margin-bottom: 10px;"><a href="{item.link}" target="_blank" style="color:#00d2ff; text-decoration:none;">➤ {item.title[:60]}...</a></div>', unsafe_allow_html=True)

    # 2. COMPETITOR PRICING
    elif menu == "COMPETITOR PRICING":
        st.markdown("## ⚔️ COMPETITOR SIMULATOR")
        col_ctrl, col_sim = st.columns([1, 2])
        with col_ctrl:
            st.markdown("### 🎛️ CONTROLS")
            fabric = st.selectbox("Fabric Class", ["Cotton Single Jersey", "CVC Fleece", "Poly Mesh"])
            my_quote = st.number_input("Your Quote ($/kg)", 4.50)
            shock = st.slider("Market Shock (%)", -20, 20, 0)
        with col_sim:
            base = yarn_cost * (1 + shock/100)
            china = base * 0.94; india = base * 0.96; vietnam = base * 0.98
            diff = my_quote - min(china, india, vietnam)
            prob = max(0, min(100, 100 - (diff * 200))) 
            
            fig = go.Figure(go.Indicator(mode = "gauge+number", value = prob, title = {'text': "WIN PROBABILITY"}, gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00d2ff"}, 'bgcolor': "rgba(0,0,0,0)"}))
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white", 'family': "Rajdhani"})
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
            st.markdown("### 🌪️ TRIGGER")
            scenario = st.radio("Select Disaster", ["None", "Suez Blockage", "Cotton Failure", "Cyber Attack"])
        with col_doom2:
            if scenario == "None":
                st.success("✅ STATUS: OPTIMAL")
                data = [{"source": [90.4, 23.8], "target": [-74.0, 40.7], "color": [0, 255, 136]}] 
                impact_cost = 0; days_left = 45
            elif scenario == "Suez Blockage":
                st.markdown('<div class="chaos-alert"><h3>🚨 ALERT: CANAL BLOCKED</h3><p>Rerouting via Cape of Good Hope (+14 Days)</p></div>', unsafe_allow_html=True)
                data = [{"source": [90.4, 23.8], "target": [18.4, -33.9], "color": [255, 0, 0]}, {"source": [18.4, -33.9], "target": [-74.0, 40.7], "color": [255, 0, 0]}] 
                impact_cost = 25000; days_left = 12
            elif scenario == "Cotton Failure":
                st.markdown('<div class="chaos-alert"><h3>🚨 ALERT: RAW MATERIAL SHORTAGE</h3><p>Price Surge +30% Imminent</p></div>', unsafe_allow_html=True)
                data = [{"source": [77.0, 20.0], "target": [90.4, 23.8], "color": [255, 165, 0]}] 
                impact_cost = 50000; days_left = 20
            elif scenario == "Cyber Attack":
                st.markdown('<div class="chaos-alert"><h3>🚨 ALERT: PORT BLACKOUT</h3><p>Zero Movement.</p></div>', unsafe_allow_html=True)
                data = []; impact_cost = 100000; days_left = 3
            
            st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ArcLayer", data=data, get_width=8, get_source_position="source", get_target_position="target", get_source_color="color", get_target_color="color")], initial_view_state=pdk.ViewState(latitude=20, longitude=10, zoom=1, pitch=40), map_style="mapbox://styles/mapbox/dark-v10"))

    # --- 🆕 ESG PULSE ---
    elif menu == "ESG PULSE 🌿":
        st.markdown("## 🌿 ESG CARBON COMMAND")
        col_esg1, col_esg2 = st.columns([1, 2])
        with col_esg1:
            st.markdown("### 🏭 Production Input")
            daily_prod = st.slider("Daily Production (kg)", 1000, 20000, 5000)
            energy_mix = st.radio("Energy Source", ["National Grid", "Solar Hybrid (30%)", "Coal (Legacy)"])
            co2_factor = 0.6 if energy_mix == "Solar Hybrid (30%)" else (0.9 if "Grid" in energy_mix else 1.2)
            total_co2 = daily_prod * co2_factor
            if co2_factor < 0.8: st.markdown('<div class="target-safe">✅ EU EXPORT READY</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="target-card">⚠️ SURCHARGE RISK (CBAM)</div>', unsafe_allow_html=True)
            
        with col_esg2:
            st.markdown("### 💨 Real-Time Emissions")
            dates = pd.date_range(end=datetime.today(), periods=30)
            df_esg = pd.DataFrame({"Date": dates, "CO2 (Tons)": np.random.normal(total_co2/1000, 0.5, 30)})
            fig = px.area(df_esg, x="Date", y="CO2 (Tons)", color_discrete_sequence=["#00ff88"])
            fig.add_hline(y=4.0, line_dash="dot", line_color="red", annotation_text="EU Limit")
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    # --- 🆕 NEURAL SCHEDULER ---
    elif menu == "NEURAL SCHEDULER 🧠":
        st.markdown("## 🧠 NEURAL SCHEDULER")
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
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', title="AI Timeline (98% Efficiency)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Awaiting Command to Run AI Algorithm...")

    # --- 🆕 SMART GRID ---
    elif menu == "SMART GRID ⚡":
        st.markdown("## ⚡ SMART ENERGY GRID")
        c1, c2, c3 = st.columns(3)
        c1.metric("Live Load", "450 kW", "Peak Zone")
        c2.metric("Hourly Cost", "BDT 4,500", "+15%")
        c3.metric("Power Factor", "0.98", "Optimal")
        
        x = list(range(24))
        y = [random.randint(300, 500) if i > 10 and i < 18 else random.randint(100, 200) for i in x]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', line=dict(color='#ffaa00', width=2), name="Usage"))
        fig.add_vrect(x0=17, x1=23, annotation_text="PEAK HOURS", fillcolor="red", opacity=0.1, line_width=0)
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Hour", yaxis_title="kW")
        st.plotly_chart(fig, use_container_width=True)

    # 4. HR COMMAND
    elif menu == "HR COMMAND":
        st.markdown("## 👥 HR COMMAND")
        hr_tabs = st.tabs(["Directory", "Payroll", "Attendance"])
        with hr_tabs[0]:
            st.dataframe(db_fetch_table("employees"), use_container_width=True)
        with hr_tabs[1]:
            if st.button("RUN PAYROLL"):
                progress = st.progress(0)
                for i in range(100): time.sleep(0.01); progress.progress(i+1)
                st.success("✅ Payroll Disbursed")

    # 5. R&D INNOVATION
    elif menu == "R&D INNOVATION":
        st.markdown("## 🔬 R&D LAB")
        tab1, tab2 = st.tabs(["Loom Whisperer", "Algo-Weaver"])
        with tab1:
            if st.button("SCAN HARMONICS"):
                x = np.linspace(-5, 5, 100); y = np.linspace(-5, 5, 100); X, Y = np.meshgrid(x, y); R = np.sqrt(X**2 + Y**2); Z = np.sin(R)
                fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')])
                fig.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            c1, c2 = st.columns(2); freq = c1.slider("Frequency", 1, 20, 10); chaos = c2.slider("Chaos", 1, 10, 5)
            if st.button("GENERATE"):
                st.image(generate_noise_pattern(freq, chaos), use_column_width=True, channels="BGR")

    # 6. FACTORY STATUS
    elif menu == "FACTORY STATUS":
        st.markdown("## 🏭 FACTORY HEALTH")
        c1, c2 = st.columns(2)
        fig_speed = go.Figure(go.Indicator(mode="gauge+number", value=random.randint(750, 850), title={'text': "RPM"}, gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#00ff88"}, 'bgcolor': "rgba(0,0,0,0)"}))
        fig_speed.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        c1.plotly_chart(fig_speed, use_container_width=True)
        
        fig_temp = go.Figure(go.Indicator(mode="gauge+number", value=random.randint(28, 40), title={'text': "TEMP °C"}, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#ff0055"}, 'bgcolor': "rgba(0,0,0,0)"}))
        fig_temp.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        c2.plotly_chart(fig_temp, use_container_width=True)

    # 7. COSTING (UNCHANGED)
    elif menu == "COSTING":
        st.markdown("## 💰 MARGIN CALCULATOR")
        p = st.number_input("Price", 4.50)
        margin = p - (yarn_cost+0.75)
        st.metric("Margin", f"${margin:.2f}/kg")
        if st.button("Save Deal"): db_log_deal("Test", 0, p, 0, 0); st.success("Saved")

    # 8. OTHER MODULES (Simplified for Elite Display)
    elif menu == "DATABASE":
        st.markdown("## 🗄️ ARCHIVES")
        st.dataframe(db_fetch_table("deals"), use_container_width=True)

    elif menu == "SYSTEM GUIDE":
        st.markdown("## 🎓 SYSTEM MANUAL")
        st.info("Refer to Engineering Documentation v33.0")
