import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PAGE CONFIG
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="AI-Based IDS | Biniyam Ademasu",
    page_icon="ðŸ›¡ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CUSTOM CSS â€” CYBER DARK THEME
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');

  /* Base */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background: #070a12; color: #e2e8f0; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2a3a;
  }

  /* Cards */
  .metric-card {
    background: linear-gradient(135deg, #0d1117, #0f1923);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,212,255,0.08);
    transition: all 0.3s ease;
  }
  .metric-card:hover { border-color: #00d4ff; box-shadow: 0 0 30px rgba(0,212,255,0.2); }
  .metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4ff; font-family: 'Fira Code'; }
  .metric-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

  /* Header banner */
  .header-banner {
    background: linear-gradient(135deg, #0d1117 0%, #0a1628 50%, #07101e 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .header-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,212,255,0.015) 2px, rgba(0,212,255,0.015) 4px);
    pointer-events: none;
  }
  .header-title { font-size: 2rem; font-weight: 800; color: #fff; margin: 0; }
  .header-subtitle { color: #00d4ff; font-family: 'Fira Code'; font-size: 0.95rem; margin: 6px 0 0; }
  .header-badge {
    display: inline-block;
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.4);
    color: #00ff88;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'Fira Code';
    margin-top: 10px;
  }

  /* Alert boxes */
  .alert-normal {
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.4);
    border-radius: 10px; padding: 16px 20px; margin: 10px 0;
  }
  .alert-attack {
    background: rgba(255,59,59,0.1);
    border: 1px solid rgba(255,59,59,0.5);
    border-radius: 10px; padding: 16px 20px; margin: 10px 0;
  }
  .alert-title { font-size: 1.3rem; font-weight: 700; margin: 0 0 4px; }
  .alert-normal .alert-title { color: #00ff88; }
  .alert-attack .alert-title { color: #ff3b3b; }

  /* Prediction result */
  .result-normal { color: #00ff88; font-size: 2.5rem; font-weight: 800; font-family: 'Fira Code'; }
  .result-attack { color: #ff3b3b; font-size: 2.5rem; font-weight: 800; font-family: 'Fira Code'; }

  /* Section headers */
  .section-title {
    color: #00d4ff;
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin: 24px 0 16px;
  }

  /* Stmetric override */
  [data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 12px 16px;
  }
  [data-testid="metric-container"] label { color: #64748b !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'Fira Code'; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0066ff);
    color: #000;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    padding: 10px 24px;
    width: 100%;
    transition: all 0.3s;
  }
  .stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #0d1117; border-bottom: 1px solid #1e3a5f; }
  .stTabs [data-baseweb="tab"] { color: #64748b; }
  .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff; }

  /* Tables */
  .dataframe { background: #0d1117 !important; color: #e2e8f0 !important; }
  .dataframe th { background: #0f1923 !important; color: #00d4ff !important; }
  .dataframe tr:hover td { background: #1e2a3a !important; }

  /* File uploader */
  [data-testid="stFileUploadDropzone"] {
    background: #0d1117 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 10px !important;
  }
  [data-testid="stFileUploadDropzone"]:hover { border-color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FEATURES DEFINITION (from dataset structure)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FEATURES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

PROTOCOL_TYPES = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "ssh", "dns", "ftp_data", "other"]
FLAGS = ["SF", "S0", "REJ", "RSTO", "RSTR", "SH", "OTH", "S1", "S2", "S3"]

ATTACK_CATEGORIES = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit"],
    "Normal": ["normal"]
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# DATA & MODEL LOADING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data
def generate_demo_data(n=5000, seed=42):
    """Generate realistic synthetic IDS data for demo purposes."""
    rng = np.random.default_rng(seed)
    normal = int(n * 0.6)
    attack = n - normal

    def make_records(n, is_attack):
        label = 1 if is_attack else 0
        return {
            "duration":               rng.integers(0, 2000, n) if is_attack else rng.integers(0, 100, n),
            "protocol_type":          rng.choice([0,1,2], n),
            "service":                rng.choice(range(7), n),
            "flag":                   rng.choice(range(10), n),
            "src_bytes":              rng.integers(1000, 500000, n) if is_attack else rng.integers(0, 5000, n),
            "dst_bytes":              rng.integers(0, 1000, n) if is_attack else rng.integers(0, 50000, n),
            "land":                   rng.choice([0,1], n, p=[0.95,0.05]),
            "wrong_fragment":         rng.integers(0, 5, n) if is_attack else rng.integers(0, 1, n),
            "urgent":                 rng.integers(0, 3, n) if is_attack else rng.integers(0, 1, n),
            "hot":                    rng.integers(0, 30, n) if is_attack else rng.integers(0, 5, n),
            "num_failed_logins":      rng.integers(0, 5, n) if is_attack else rng.integers(0, 1, n),
            "logged_in":              rng.choice([0,1], n, p=[0.3,0.7] if is_attack else [0.1,0.9]),
            "num_compromised":        rng.integers(0, 20, n) if is_attack else rng.integers(0, 1, n),
            "root_shell":             rng.choice([0,1], n, p=[0.7,0.3] if is_attack else [0.99,0.01]),
            "su_attempted":           rng.choice([0,1], n, p=[0.8,0.2] if is_attack else [0.99,0.01]),
            "num_root":               rng.integers(0, 10, n) if is_attack else rng.integers(0, 1, n),
            "num_file_creations":     rng.integers(0, 10, n),
            "num_shells":             rng.integers(0, 5, n) if is_attack else rng.integers(0, 1, n),
            "num_access_files":       rng.integers(0, 10, n),
            "num_outbound_cmds":      rng.integers(0, 2, n),
            "is_host_login":          rng.choice([0,1], n, p=[0.9,0.1]),
            "is_guest_login":         rng.choice([0,1], n, p=[0.9,0.1]),
            "count":                  rng.integers(1, 512, n),
            "srv_count":              rng.integers(1, 512, n),
            "serror_rate":            rng.uniform(0.5, 1.0, n) if is_attack else rng.uniform(0, 0.1, n),
            "srv_serror_rate":        rng.uniform(0.5, 1.0, n) if is_attack else rng.uniform(0, 0.1, n),
            "rerror_rate":            rng.uniform(0, 1, n),
            "srv_rerror_rate":        rng.uniform(0, 1, n),
            "same_srv_rate":          rng.uniform(0, 0.5, n) if is_attack else rng.uniform(0.5, 1.0, n),
            "diff_srv_rate":          rng.uniform(0, 1, n),
            "srv_diff_host_rate":     rng.uniform(0, 1, n),
            "dst_host_count":         rng.integers(1, 255, n),
            "dst_host_srv_count":     rng.integers(1, 255, n),
            "dst_host_same_srv_rate": rng.uniform(0, 1, n),
            "dst_host_diff_srv_rate": rng.uniform(0, 1, n),
            "dst_host_same_src_port_rate": rng.uniform(0, 1, n),
            "dst_host_srv_diff_host_rate": rng.uniform(0, 1, n),
            "dst_host_serror_rate":   rng.uniform(0.5, 1.0, n) if is_attack else rng.uniform(0, 0.1, n),
            "dst_host_srv_serror_rate": rng.uniform(0.5, 1.0, n) if is_attack else rng.uniform(0, 0.1, n),
            "dst_host_rerror_rate":   rng.uniform(0, 1, n),
            "dst_host_srv_rerror_rate": rng.uniform(0, 1, n),
            "label": np.full(n, label),
        }

    normal_df = pd.DataFrame(make_records(normal, False))
    attack_df = pd.DataFrame(make_records(attack, True))
    df = pd.concat([normal_df, attack_df]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

@st.cache_resource
def train_models(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=80, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        results[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "cm":        confusion_matrix(y_test, y_pred).tolist(),
        }
        trained[name] = model

    return trained, scaler, X_test, y_test, results

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SIDEBAR
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
      <div style='font-size:3rem;'>ðŸ›¡ï¸</div>
      <div style='color:#00d4ff; font-weight:700; font-size:1.1rem;'>AI-Based IDS</div>
      <div style='color:#64748b; font-size:0.8rem; font-family: Fira Code;'>v2.0 â€” Biniyam Ademasu</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**ðŸ¤– Model Selection**")
    selected_model_name = st.selectbox(
        "Choose classifier",
        ["Random Forest", "Gradient Boosting", "Decision Tree", "K-Nearest Neighbors"],
        help="Random Forest usually achieves the best accuracy"
    )

    st.markdown("---")
    st.markdown("**ðŸ“‚ Data Source**")
    data_source = st.radio("Choose data source", ["Demo Dataset", "Upload CSV"])

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your network traffic CSV",
            type=["csv"],
            help="CSV should contain the 41 KDD Cup features + a 'label' column (0=normal, 1=attack)"
        )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; text-align:center;'>
      Built by <strong style='color:#00d4ff'>Biniyam Ademasu Afework</strong><br>
      B.Sc. Cyber Security â€” Bahir Dar University<br><br>
      <a href='https://github.com/bini576' style='color:#00d4ff; text-decoration:none;'>github.com/bini576</a>
    </div>
    """, unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LOAD DATA & TRAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.spinner("ðŸ”„ Loading data and training models..."):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "label" not in df.columns:
                st.error("âŒ CSV must contain a 'label' column (0 = Normal, 1 = Attack)")
                st.stop()
            # Encode any string columns
            for col in df.select_dtypes(include="object").columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        df = generate_demo_data()

    trained_models, scaler, X_test, y_test, results = train_models(df)
    best_model = trained_models[selected_model_name]
    best_results = results[selected_model_name]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HEADER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown(f"""
<div class="header-banner">
  <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
    <div>
      <div class="header-title">ðŸ›¡ï¸ AI-Based Intrusion Detection System</div>
      <div class="header-subtitle">$ network_monitor --model {selected_model_name.lower().replace(" ","_")} --mode live_analysis</div>
      <span class="header-badge">â— SYSTEM ACTIVE</span>
    </div>
    <div style='text-align:right;'>
      <div style='color:#64748b; font-size:0.8rem; font-family: Fira Code;'>Dataset: {len(df):,} records</div>
      <div style='color:#64748b; font-size:0.8rem; font-family: Fira Code;'>Normal: {int((df["label"]==0).sum()):,} | Attack: {int((df["label"]==1).sum()):,}</div>
      <div style='color:#00ff88; font-size:0.8rem; font-family: Fira Code; margin-top:4px;'>Model Accuracy: {best_results["accuracy"]}%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TOP METRIC CARDS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value">{best_results["accuracy"]}%</div>
      <div class="metric-label">ðŸŽ¯ Accuracy</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value">{best_results["precision"]}%</div>
      <div class="metric-label">ðŸ” Precision</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value">{best_results["recall"]}%</div>
      <div class="metric-label">ðŸ“¡ Recall</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
      <div class="metric-value">{best_results["f1"]}%</div>
      <div class="metric-label">âš¡ F1-Score</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TABS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
tab1, tab2, tab3, tab4 = st.tabs(["ðŸ” Live Classifier", "ðŸ“Š Model Analytics", "ðŸ“ˆ Dataset Explorer", "ðŸ“š About"])

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 1 â€” LIVE CLASSIFIER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab1:
    st.markdown('<div class="section-title">ðŸ” Real-Time Traffic Classification</div>', unsafe_allow_html=True)
    st.markdown("Input network connection features below and the model will classify it instantly.")

    input_mode = st.radio("Input mode", ["ðŸŽ›ï¸ Manual Feature Input", "ðŸ“‹ Paste Single Record", "ðŸŽ² Generate Random Sample"], horizontal=True)

    if input_mode == "ðŸŽ² Generate Random Sample":
        sample_type = st.selectbox("Sample type", ["Random Normal Traffic", "Random Attack Traffic", "Surprise Me!"])
        if st.button("âš¡ Generate & Classify"):
            rng = np.random.default_rng()
            is_attack = sample_type == "Random Attack Traffic" or (sample_type == "Surprise Me!" and rng.random() > 0.5)
            temp_df = generate_demo_data(n=2, seed=int(rng.integers(0, 10000)))
            sample = temp_df.drop("label", axis=1).iloc[0:1]
            actual = temp_df["label"].iloc[0]

            scaled = scaler.transform(sample)
            pred = best_model.predict(scaled)[0]
            proba = best_model.predict_proba(scaled)[0]

            col_a, col_b = st.columns([1, 1])
            with col_a:
                if pred == 0:
                    st.markdown(f"""<div class="alert-normal">
                      <div class="alert-title">âœ… NORMAL TRAFFIC</div>
                      <div style='color:#94a3b8; font-size:0.9rem;'>Connection classified as benign</div>
                      <div style='margin-top:10px;'>
                        <span style='color:#00ff88; font-weight:700; font-size:1.5rem;'>{proba[0]*100:.1f}%</span>
                        <span style='color:#64748b; font-size:0.85rem;'> confidence</span>
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="alert-attack">
                      <div class="alert-title">ðŸš¨ ATTACK DETECTED</div>
                      <div style='color:#94a3b8; font-size:0.9rem;'>Malicious connection detected!</div>
                      <div style='margin-top:10px;'>
                        <span style='color:#ff3b3b; font-weight:700; font-size:1.5rem;'>{proba[1]*100:.1f}%</span>
                        <span style='color:#64748b; font-size:0.85rem;'> confidence</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with col_b:
                fig = go.Figure(go.Bar(
                    x=["Normal", "Attack"],
                    y=[proba[0]*100, proba[1]*100],
                    marker_color=["#00ff88", "#ff3b3b"],
                    text=[f"{v:.1f}%" for v in [proba[0]*100, proba[1]*100]],
                    textposition="auto"
                ))
                fig.update_layout(
                    title="Prediction Confidence",
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    font_color="#e2e8f0",
                    height=200,
                    margin=dict(t=40, b=0, l=0, r=0),
                    showlegend=False,
                    yaxis=dict(range=[0, 110], gridcolor="#1e2a3a"),
                    xaxis=dict(gridcolor="#1e2a3a"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("ðŸ”Ž Sample Features"):
                st.dataframe(sample.T.rename(columns={0: "Value"}), use_container_width=True)

    elif input_mode == "ðŸŽ›ï¸ Manual Feature Input":
        st.markdown("**Key network features (simplified):**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            duration = st.number_input("Duration (sec)", 0, 60000, 0)
            protocol = st.selectbox("Protocol", PROTOCOL_TYPES)
            service = st.selectbox("Service", SERVICES)
            flag = st.selectbox("Flag", FLAGS)
            src_bytes = st.number_input("Source Bytes", 0, 1000000, 0)
        with col2:
            dst_bytes = st.number_input("Dest Bytes", 0, 1000000, 0)
            logged_in = st.selectbox("Logged In", [0, 1])
            count = st.slider("Count", 1, 512, 1)
            srv_count = st.slider("Srv Count", 1, 512, 1)
            serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0, 0.01)
        with col3:
            rerror_rate = st.slider("REJ Error Rate", 0.0, 1.0, 0.0, 0.01)
            same_srv_rate = st.slider("Same Srv Rate", 0.0, 1.0, 1.0, 0.01)
            diff_srv_rate = st.slider("Diff Srv Rate", 0.0, 1.0, 0.0, 0.01)
            hot = st.number_input("Hot Indicators", 0, 100, 0)
            num_failed_logins = st.number_input("Failed Logins", 0, 10, 0)
        with col4:
            root_shell = st.selectbox("Root Shell", [0, 1])
            num_compromised = st.number_input("Compromised", 0, 100, 0)
            dst_host_count = st.slider("Dst Host Count", 1, 255, 100)
            dst_host_serror_rate = st.slider("Dst Host SErr Rate", 0.0, 1.0, 0.0, 0.01)
            wrong_fragment = st.number_input("Wrong Fragments", 0, 10, 0)

        if st.button("ðŸ›¡ï¸ Classify Traffic"):
            proto_map = {p: i for i, p in enumerate(PROTOCOL_TYPES)}
            svc_map = {s: i for i, s in enumerate(SERVICES)}
            flag_map = {f: i for i, f in enumerate(FLAGS)}

            input_data = np.zeros(41)
            input_data[0]  = duration
            input_data[1]  = proto_map.get(protocol, 0)
            input_data[2]  = svc_map.get(service, 0)
            input_data[3]  = flag_map.get(flag, 0)
            input_data[4]  = src_bytes
            input_data[5]  = dst_bytes
            input_data[7]  = wrong_fragment
            input_data[9]  = hot
            input_data[10] = num_failed_logins
            input_data[11] = logged_in
            input_data[12] = num_compromised
            input_data[13] = root_shell
            input_data[22] = count
            input_data[23] = srv_count
            input_data[24] = serror_rate
            input_data[25] = serror_rate
            input_data[26] = rerror_rate
            input_data[27] = rerror_rate
            input_data[28] = same_srv_rate
            input_data[29] = diff_srv_rate
            input_data[31] = dst_host_count
            input_data[36] = dst_host_serror_rate
            input_data[37] = dst_host_serror_rate

            scaled = scaler.transform(input_data.reshape(1, -1))
            pred = best_model.predict(scaled)[0]
            proba = best_model.predict_proba(scaled)[0]

            if pred == 0:
                st.markdown(f"""<div class="alert-normal">
                  <div class="alert-title">âœ… NORMAL TRAFFIC â€” {proba[0]*100:.1f}% Confidence</div>
                  <div>This connection matches normal network behavior patterns.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="alert-attack">
                  <div class="alert-title">ðŸš¨ ATTACK DETECTED â€” {proba[1]*100:.1f}% Confidence</div>
                  <div>This connection exhibits malicious patterns. Take immediate action!</div>
                </div>""", unsafe_allow_html=True)

    else:  # Paste
        st.markdown("Paste a comma-separated record (41 numeric features):")
        pasted = st.text_area("Feature values", placeholder="0,0,1,0,491,0,0,0,0,0,1,0,0,0,0,0,0,0,2,2,0.0,0.0,0.0,0.0,1.0,0.0,0.0,150,25,0.17,0.03,0.17,0.0,0.0,0.0,0.05,0.0,0.0,0.0,0.0,0.0", height=100)
        if st.button("ðŸ›¡ï¸ Classify Pasted Record"):
            try:
                vals = [float(x.strip()) for x in pasted.split(",")]
                if len(vals) != 41:
                    st.error(f"Expected 41 features, got {len(vals)}")
                else:
                    scaled = scaler.transform(np.array(vals).reshape(1, -1))
                    pred = best_model.predict(scaled)[0]
                    proba = best_model.predict_proba(scaled)[0]
                    label = "âœ… NORMAL" if pred == 0 else "ðŸš¨ ATTACK"
                    color = "#00ff88" if pred == 0 else "#ff3b3b"
                    st.markdown(f"""<div style='text-align:center; padding:30px;'>
                      <div style='color:{color}; font-size:2.5rem; font-weight:800; font-family:Fira Code;'>{label}</div>
                      <div style='color:#64748b; margin-top:8px;'>Confidence: {max(proba)*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Parse error: {e}")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 2 â€” MODEL ANALYTICS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab2:
    st.markdown('<div class="section-title">ðŸ“Š Model Performance Comparison</div>', unsafe_allow_html=True)

    # Comparison bar chart
    metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    fig = go.Figure()
    for metric, color in [("accuracy","#00d4ff"),("precision","#00ff88"),("recall","#a855f7"),("f1","#f59e0b")]:
        fig.add_trace(go.Bar(name=metric.capitalize(), x=metrics_df["Model"], y=metrics_df[metric], marker_color=color))
    fig.update_layout(
        barmode="group", title="All Models â€” All Metrics (%)",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e2e8f0", height=380,
        legend=dict(bgcolor="#0d1117"),
        yaxis=dict(range=[0,110], gridcolor="#1e2a3a"),
        xaxis=dict(gridcolor="#1e2a3a"),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f'<div class="section-title">ðŸ”² Confusion Matrix â€” {selected_model_name}</div>', unsafe_allow_html=True)
        cm = best_results["cm"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted Normal", "Predicted Attack"],
            y=["Actual Normal", "Actual Attack"],
            colorscale=[[0,"#0d1117"],[0.5,"#1e3a5f"],[1,"#00d4ff"]],
            text=cm, texttemplate="%{text}",
            showscale=False,
        ))
        fig_cm.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e2e8f0", height=300,
            margin=dict(t=10, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-title">ðŸ† Model Ranking</div>', unsafe_allow_html=True)
        ranking = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)
        for rank, (name, r) in enumerate(ranking, 1):
            emoji = ["ðŸ¥‡","ðŸ¥ˆ","ðŸ¥‰","4ï¸âƒ£"][rank-1]
            highlight = "border: 1px solid #00d4ff;" if name == selected_model_name else ""
            st.markdown(f"""
            <div style='background:#0d1117; border-radius:8px; padding:12px 16px; margin:6px 0; {highlight}'>
              <span style='font-weight:700; color:#e2e8f0;'>{emoji} {name}</span>
              <span style='float:right; color:#00d4ff; font-family:Fira Code; font-weight:700;'>{r["f1"]}% F1</span>
              <div style='color:#64748b; font-size:0.8rem; margin-top:2px;'>
                Acc: {r["accuracy"]}% | Prec: {r["precision"]}% | Rec: {r["recall"]}%
              </div>
            </div>""", unsafe_allow_html=True)

    # Feature importance (Random Forest)
    if selected_model_name == "Random Forest":
        st.markdown('<div class="section-title">ðŸ”‘ Top Feature Importances</div>', unsafe_allow_html=True)
        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(15)
        fig_fi = px.bar(importance, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=["#1e3a5f","#00d4ff"])
        fig_fi.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e2e8f0", height=400, coloraxis_showscale=False,
            margin=dict(t=10), yaxis=dict(autorange="reversed"),
            xaxis=dict(gridcolor="#1e2a3a"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 3 â€” DATASET EXPLORER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab3:
    st.markdown('<div class="section-title">ðŸ“ˆ Dataset Statistics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        label_counts = df["label"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["Normal", "Attack"],
            values=label_counts.values,
            marker_colors=["#00ff88", "#ff3b3b"],
            hole=0.5,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            title="Traffic Distribution",
            paper_bgcolor="#0d1117", font_color="#e2e8f0", height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_src = px.histogram(df, x="src_bytes", color=df["label"].map({0:"Normal",1:"Attack"}),
                               nbins=50, title="Source Bytes Distribution",
                               color_discrete_map={"Normal":"#00ff88","Attack":"#ff3b3b"})
        fig_src.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e2e8f0", height=300, legend_title="Traffic Type",
            xaxis=dict(gridcolor="#1e2a3a"), yaxis=dict(gridcolor="#1e2a3a"),
        )
        st.plotly_chart(fig_src, use_container_width=True)

    st.markdown('<div class="section-title">ðŸ“‹ Sample Records</div>', unsafe_allow_html=True)
    sample_size = st.slider("Records to display", 5, 50, 10)
    display_df = df.head(sample_size).copy()
    display_df["label"] = display_df["label"].map({0: "âœ… Normal", 1: "ðŸš¨ Attack"})
    st.dataframe(display_df[["duration","src_bytes","dst_bytes","count","serror_rate","label"]], use_container_width=True)

    st.markdown('<div class="section-title">ðŸ“Š Feature Correlations (Numeric)</div>', unsafe_allow_html=True)
    top_cols = ["duration","src_bytes","dst_bytes","count","serror_rate","rerror_rate","same_srv_rate","dst_host_count","label"]
    corr = df[top_cols].corr()
    fig_corr = px.imshow(corr, color_continuous_scale=[[0,"#ff3b3b"],[0.5,"#0d1117"],[1,"#00d4ff"]],
                         title="Correlation Heatmap", aspect="auto")
    fig_corr.update_layout(paper_bgcolor="#0d1117", font_color="#e2e8f0", height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TAB 4 â€” ABOUT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
with tab4:
    st.markdown("""
    <div style='background:#0d1117; border:1px solid #1e3a5f; border-radius:12px; padding:28px;'>
      <h2 style='color:#00d4ff; margin-top:0;'>ðŸ›¡ï¸ AI-Based Intrusion Detection System</h2>

      <p style='color:#94a3b8;'>
        This project implements a machine learning-based Intrusion Detection System (IDS) capable of
        analyzing network connection records and classifying them as <strong style='color:#00ff88'>normal</strong>
        or <strong style='color:#ff3b3b'>malicious/attack</strong> in real-time.
      </p>

      <h3 style='color:#e2e8f0;'>ðŸŽ¯ How It Works</h3>
      <div style='color:#94a3b8; line-height:1.8;'>
        <strong style='color:#00d4ff;'>1. Data Collection</strong> â€” Network traffic features are captured (41 KDD Cup features)<br>
        <strong style='color:#00d4ff;'>2. Preprocessing</strong> â€” Feature scaling and encoding applied<br>
        <strong style='color:#00d4ff;'>3. Training</strong> â€” ML classifiers trained on labeled traffic data<br>
        <strong style='color:#00d4ff;'>4. Prediction</strong> â€” New traffic classified as Normal or Attack in real-time<br>
        <strong style='color:#00d4ff;'>5. Alerting</strong> â€” Results displayed with confidence scores
      </div>

      <h3 style='color:#e2e8f0;'>âš”ï¸ Attack Categories Detected</h3>
      <div style='display:grid; grid-template-columns: repeat(2, 1fr); gap:12px; margin-top:10px;'>
        <div style='background:#0a1628; border:1px solid #1e3a5f; border-radius:8px; padding:12px;'>
          <strong style='color:#ff3b3b;'>DoS Attacks</strong>
          <div style='color:#64748b; font-size:0.85rem;'>back, land, neptune, pod, smurf, teardrop</div>
        </div>
        <div style='background:#0a1628; border:1px solid #1e3a5f; border-radius:8px; padding:12px;'>
          <strong style='color:#f59e0b;'>Probe Attacks</strong>
          <div style='color:#64748b; font-size:0.85rem;'>ipsweep, nmap, portsweep, satan</div>
        </div>
        <div style='background:#0a1628; border:1px solid #1e3a5f; border-radius:8px; padding:12px;'>
          <strong style='color:#a855f7;'>R2L Attacks</strong>
          <div style='color:#64748b; font-size:0.85rem;'>ftp_write, guess_passwd, imap, spy...</div>
        </div>
        <div style='background:#0a1628; border:1px solid #1e3a5f; border-radius:8px; padding:12px;'>
          <strong style='color:#00d4ff;'>U2R Attacks</strong>
          <div style='color:#64748b; font-size:0.85rem;'>buffer_overflow, loadmodule, rootkit...</div>
        </div>
      </div>

      <h3 style='color:#e2e8f0; margin-top:20px;'>ðŸ‘¨â€ðŸ’» About the Developer</h3>
      <p style='color:#94a3b8;'>
        <strong style='color:#00d4ff;'>Biniyam Ademasu Afework</strong> â€” Cybersecurity Analyst &amp; Security Engineer<br>
        B.Sc. Cyber Security, Bahir Dar University, Ethiopia (GPA: 3.43/4.00, 2026)<br>
        Skills: Python, Machine Learning, Network Security, Penetration Testing<br><br>
        ðŸ”— <a href='https://bini576.github.io' style='color:#00d4ff;'>Portfolio</a> &nbsp;|&nbsp;
        ðŸ™ <a href='https://github.com/bini576' style='color:#00d4ff;'>GitHub</a> &nbsp;|&nbsp;
        ðŸ’¼ <a href='https://linkedin.com/in/biniyam-ademasu' style='color:#00d4ff;'>LinkedIn</a>
      </p>
    </div>
    """, unsafe_allow_html=True)
