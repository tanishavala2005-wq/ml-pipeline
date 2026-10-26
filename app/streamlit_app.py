import json
import os
import sys

# Let Python find the src folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.preprocess import load_config, load_data
from src.predict import load_model, predict_churn
from src.train import main as train_model

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

REPO_URL = "https://github.com/tanishavala2005-wq/ml-pipeline"

YES_NO = ["Yes", "No"]
YES_NO_NO_INTERNET = ["Yes", "No", "No internet service"]
MAP_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Background colours the visitor can pick in the sidebar (bg = page, panel = sidebar and inputs)
THEMES = {
    "Lavender": {"bg": "#f5f3ff", "panel": "#ede9fe", "line": "#c4b5fd"},
    "Sky": {"bg": "#f0f9ff", "panel": "#e0f2fe", "line": "#7dd3fc"},
    "Mint": {"bg": "#f0fdf4", "panel": "#dcfce7", "line": "#86efac"},
    "Peach": {"bg": "#fff7ed", "panel": "#ffedd5", "line": "#fdba74"},
    "Rose": {"bg": "#fff1f2", "panel": "#ffe4e6", "line": "#fda4af"},
    "White (default)": {"bg": "#ffffff", "panel": "#f0f2f6", "line": "#d1d5db"},
    # dark palettes also set a light text colour
    "Dark navy": {"bg": "#0b1020", "panel": "#151c33", "line": "#334155", "text": "#e2e8f0"},
    "Dark purple": {"bg": "#14102b", "panel": "#1f1a3d", "line": "#4c3f8f", "text": "#ede9fe"},
    "Charcoal": {"bg": "#121212", "panel": "#1e1e1e", "line": "#3f3f46", "text": "#e5e5e5"},
}

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0, "TotalCharges": 170.0,
}

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
}

for key, value in HIGH_RISK.items():
    st.session_state.setdefault(key, value)


def load_example(example):
    for key, value in example.items():
        st.session_state[key] = value
    st.session_state.pop("result", None)  # an old result would no longer match the form


# ---------------------------------------------------------------- model + data
@st.cache_resource(show_spinner="Loading model (the first run trains it, about 30 seconds)...")
def get_model():
    config = load_config()
    # The saved model is not stored on GitHub, so train it once if it is missing
    if not os.path.exists(config["model_path"]):
        os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)
        train_model()
    return load_model()


@st.cache_data
def load_customers():
    """All customers with a clean numeric Churn column (1 = churned)."""
    config = load_config()
    df = load_data(config["data_path"]).copy()
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna(subset=MAP_COLS + ["Churn"]).reset_index(drop=True)


def similar_customers(df, customer, k=100):
    """The k customers closest to this one in (tenure, monthly, total) space."""
    X = df[MAP_COLS].to_numpy(dtype=float)
    spread = X.std(axis=0)
    spread[spread == 0] = 1.0
    point = np.array([customer[c] for c in MAP_COLS], dtype=float)
    distance = (((X - point) / spread) ** 2).sum(axis=1)
    return df.iloc[np.argsort(distance)[:k]]


# ---------------------------------------------------------------- business logic
def risk_level(probability):
    # 0.5 is the model's own decision threshold (churn = 1 above it)
    if probability >= 0.5:
        return "High", "Offer a retention discount or move the customer to a longer contract."
    if probability >= 0.35:
        return "Medium", "Keep an eye on this customer and check in on their experience."
    return "Low", "No action needed right now."


def risk_signals(customer):
    """Plain, data-backed signals. The churn rates come from the EDA on the full dataset."""
    risky, protective = [], []

    if customer["Contract"] == "Month-to-month":
        risky.append("Month-to-month contract (42.7% of these customers churned)")
    if customer["tenure"] <= 12:
        risky.append("Short tenure, 12 months or less (47.4% churned)")
    if customer["PaymentMethod"] == "Electronic check":
        risky.append("Pays by electronic check (45.3% churned)")
    if customer["InternetService"] == "Fiber optic":
        risky.append("Fiber optic internet (41.9% churned)")
    if customer["TechSupport"] == "No":
        risky.append("No tech support (41.6% churned)")
    if customer["OnlineSecurity"] == "No":
        risky.append("No online security (41.8% churned)")

    if customer["Contract"] == "Two year":
        protective.append("Two-year contract (only 2.8% churned)")
    if customer["tenure"] > 12:
        protective.append("Tenure above 12 months (17.1% churned)")

    return risky, protective


# ---------------------------------------------------------------- animated gauge (SVG + JS, no plotly)
GAUGE_HTML = """
<div style="font-family:system-ui,sans-serif;text-align:center">
<svg viewBox="0 0 200 150" style="width:100%;max-width:330px">
  <g fill="none" stroke-width="16">
    <path d="M20 100 A80 80 0 0 1 180 100" pathLength="100" stroke="#22c55e" stroke-opacity="0.3" stroke-dasharray="35 100"/>
    <path d="M20 100 A80 80 0 0 1 180 100" pathLength="100" stroke="#f59e0b" stroke-opacity="0.3" stroke-dasharray="15 100" stroke-dashoffset="-35"/>
    <path d="M20 100 A80 80 0 0 1 180 100" pathLength="100" stroke="#ef4444" stroke-opacity="0.3" stroke-dasharray="50 100" stroke-dashoffset="-50"/>
    <path id="val" d="M20 100 A80 80 0 0 1 180 100" pathLength="100" stroke="#22c55e" stroke-width="6" stroke-linecap="round" stroke-dasharray="0 100"/>
  </g>
  <g id="needle"><line x1="100" y1="100" x2="34" y2="100" stroke="#64748b" stroke-width="3" stroke-linecap="round"/></g>
  <circle cx="100" cy="100" r="7" fill="#64748b"/>
  <text id="num" x="100" y="132" text-anchor="middle" font-size="26" font-weight="800" fill="#22c55e">0.0%</text>
  <text x="100" y="147" text-anchor="middle" font-size="9" fill="#64748b">Churn probability</text>
</svg>
</div>
<script>
  var P = __P__;
  var val = document.getElementById('val'), needle = document.getElementById('needle'),
      num = document.getElementById('num');
  var start = null, DUR = 1200;
  function color(v) { return v < 35 ? '#22c55e' : v < 50 ? '#f59e0b' : '#ef4444'; }
  function frame(ts) {
    if (start === null) start = ts;
    var k = Math.min((ts - start) / DUR, 1), e = 1 - Math.pow(1 - k, 3), v = P * e;
    val.setAttribute('stroke-dasharray', v + ' 100');
    val.setAttribute('stroke', color(v));
    needle.setAttribute('transform', 'rotate(' + (v * 1.8) + ' 100 100)');
    num.textContent = v.toFixed(1) + '%';
    num.setAttribute('fill', color(v));
    if (k < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
</script>
"""


def gauge_html(probability):
    return GAUGE_HTML.replace("__P__", f"{probability * 100:.2f}")


# ---------------------------------------------------------------- 3D map (Three.js, no plotly)
MAP_HTML = """
<div id="wrap" style="position:relative;width:100%;height:620px;background:#0b1020;border-radius:12px;overflow:hidden;font-family:system-ui,sans-serif;color:#e2e8f0">
  <div id="stage" style="position:absolute;inset:0"></div>
  <div id="legend" style="position:absolute;left:14px;top:12px;font-size:13px;line-height:1.7;background:rgba(11,16,32,.65);padding:8px 12px;border-radius:8px"></div>
  <div style="position:absolute;right:14px;top:12px;display:flex;gap:8px">
    <button id="rot" style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:6px 12px;cursor:pointer">Pause</button>
    <button id="reset" style="background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:6px 12px;cursor:pointer">Reset view</button>
  </div>
  <div id="tip" style="position:absolute;display:none;pointer-events:none;background:rgba(15,23,42,.92);border:1px solid #334155;border-radius:6px;padding:6px 10px;font-size:12px"></div>
  <div id="err" style="position:absolute;inset:0;display:none;align-items:center;justify-content:center;padding:24px;text-align:center"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function () {
  var D = __DATA__;
  var wrap = document.getElementById('wrap'), stage = document.getElementById('stage');
  var errBox = document.getElementById('err');
  function fail(msg) { errBox.style.display = 'flex'; errBox.textContent = msg; }
  if (typeof THREE === 'undefined' || !THREE.OrbitControls) {
    fail('The 3D library could not be loaded. Check your internet connection and reload.'); return;
  }
  var renderer;
  try { renderer = new THREE.WebGLRenderer({antialias: true}); }
  catch (e) { fail('Your browser could not start WebGL, so the 3D map cannot be shown.'); return; }

  var W = function () { return stage.clientWidth; }, H = function () { return stage.clientHeight; };
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(W(), H());
  stage.appendChild(renderer.domElement);

  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1020);
  var camera = new THREE.PerspectiveCamera(45, W() / H(), 0.1, 100);
  var HOME = new THREE.Vector3(4.2, 2.6, 4.2);
  camera.position.copy(HOME.clone().multiplyScalar(2));      // starts far away, flies in
  var controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.07;
  controls.autoRotate = D.rotate; controls.autoRotateSpeed = 1.4;
  controls.minDistance = 1.5; controls.maxDistance = 9;

  var M = D.max;
  function clamp(v) { return Math.max(0, Math.min(v, 1.05)); }
  // x = tenure, y (up) = total charges, z = monthly charges, all scaled into a 2x2x2 box
  function norm(p) {
    return new THREE.Vector3(clamp(p[0] / M[0]) * 2 - 1, clamp(p[2] / M[2]) * 2 - 1, clamp(p[1] / M[1]) * 2 - 1);
  }

  function tex(draw) {
    var c = document.createElement('canvas'); c.width = c.height = 64;
    draw(c.getContext('2d')); return new THREE.CanvasTexture(c);
  }
  var dotTex = tex(function (g) {
    var gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.6, 'rgba(255,255,255,.9)');
    gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.beginPath(); g.arc(32, 32, 32, 0, 6.2832); g.fill();
  });
  var ringTex = tex(function (g) {
    g.strokeStyle = '#fff'; g.lineWidth = 6; g.beginPath(); g.arc(32, 32, 24, 0, 6.2832); g.stroke();
  });

  function cloud(list, color, size, opacity, map) {
    var geo = new THREE.BufferGeometry().setFromPoints(list.map(norm));
    var mat = new THREE.PointsMaterial({color: color, size: size, map: map, transparent: true,
                                        opacity: opacity, depthWrite: false, alphaTest: 0.04});
    var pts = new THREE.Points(geo, mat); scene.add(pts); return pts;
  }
  var stayed = cloud(D.stayed, 0x38bdf8, 0.05, 0.75, dotTex);
  var churned = cloud(D.churned, 0xfb7185, 0.05, 0.85, dotTex);
  if (D.near.length) cloud(D.near, 0xf8fafc, 0.1, 1, ringTex);

  // frame, floor grid and axis labels
  scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
                                   new THREE.LineBasicMaterial({color: 0x334066})));
  var grid = new THREE.GridHelper(2, 10, 0x26304d, 0x1a2340); grid.position.y = -1; scene.add(grid);
  function label(text, x, y, z) {
    var c = document.createElement('canvas'); c.width = 256; c.height = 64;
    var g = c.getContext('2d'); g.fillStyle = '#cbd5e1'; g.font = '600 26px system-ui, sans-serif';
    g.textAlign = 'center'; g.fillText(text, 128, 40);
    var s = new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(c), transparent: true}));
    s.position.set(x, y, z); s.scale.set(1, 0.25, 1); scene.add(s);
  }
  label('Tenure (months)', 0, -1.12, 1.25);
  label('Monthly charges', 1.3, -1.12, 0);
  label('Total charges', -1.2, 1.15, 1.2);

  // the customer: pulsing yellow diamond, halo and a drop line to the floor
  var me = null, halo = null;
  if (D.me) {
    var p = norm(D.me);
    var geo = new THREE.OctahedronGeometry(0.085);
    me = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({color: 0xfacc15}));
    me.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({color: 0x111827})));
    me.scale.y = 1.4; me.position.copy(p); scene.add(me);
    halo = new THREE.Sprite(new THREE.SpriteMaterial({map: dotTex, color: 0xfacc15, transparent: true,
                                                       opacity: 0.35, depthWrite: false}));
    halo.position.copy(p); halo.scale.set(0.4, 0.4, 1); scene.add(halo);
    var drop = new THREE.BufferGeometry().setFromPoints([p, new THREE.Vector3(p.x, -1, p.z)]);
    scene.add(new THREE.Line(drop, new THREE.LineBasicMaterial({color: 0xfacc15, transparent: true, opacity: 0.5})));
  }

  // legend
  function row(color, text, ring) {
    var style = ring ? 'border:2px solid ' + color : 'background:' + color;
    return '<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;' + style + '"></span>' + text + '</div>';
  }
  document.getElementById('legend').innerHTML =
    row('#38bdf8', 'Stayed') + row('#fb7185', 'Churned') +
    (D.me ? row('#facc15', 'Your customer') + row('#f8fafc', '100 most similar', true) : '');

  // buttons
  var rotBtn = document.getElementById('rot');
  function syncBtn() { rotBtn.textContent = controls.autoRotate ? 'Pause' : 'Rotate'; }
  rotBtn.onclick = function () { controls.autoRotate = !controls.autoRotate; syncBtn(); };
  document.getElementById('reset').onclick = function () { camera.position.copy(HOME); controls.target.set(0, 0, 0); };
  syncBtn();

  // hover tooltip
  var ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.035;
  var mouse = new THREE.Vector2(), tip = document.getElementById('tip');
  renderer.domElement.addEventListener('mousemove', function (ev) {
    var r = renderer.domElement.getBoundingClientRect();
    mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(mouse, camera);
    var hit = ray.intersectObjects([stayed, churned])[0];
    if (!hit) { tip.style.display = 'none'; return; }
    var d = (hit.object === stayed ? D.stayed : D.churned)[hit.index];
    tip.innerHTML = '<b>' + (hit.object === stayed ? 'Stayed' : 'Churned') + '</b><br>Tenure: ' + d[0] +
                    ' months<br>Monthly: ' + d[1].toFixed(0) + '<br>Total: ' + d[2].toFixed(0);
    tip.style.left = (ev.clientX - r.left + 14) + 'px'; tip.style.top = (ev.clientY - r.top + 14) + 'px';
    tip.style.display = 'block';
  });

  window.addEventListener('resize', function () {
    camera.aspect = W() / H(); camera.updateProjectionMatrix(); renderer.setSize(W(), H());
  });

  var t0 = null, FLY = 1700;
  function tick(now) {
    requestAnimationFrame(tick);
    if (t0 === null) t0 = now;
    var k = Math.min((now - t0) / FLY, 1), e = 1 - Math.pow(1 - k, 3);
    if (k < 1) camera.position.copy(HOME.clone().multiplyScalar(2 - e));   // fly-in
    var t = now / 1000;
    if (me) { me.rotation.y = t * 1.6; me.scale.setScalar(1 + 0.12 * Math.sin(t * 4)); me.scale.y *= 1.4;
              halo.scale.setScalar(0.4 + 0.1 * Math.sin(t * 4)); }
    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(tick);
})();
</script>
"""


def map_html(sample, customer, near, max_values, rotate):
    def points(frame):
        return frame[MAP_COLS].astype(float).round(1).values.tolist()

    data = {
        "stayed": points(sample[sample["Churn"] == 0]),
        "churned": points(sample[sample["Churn"] == 1]),
        "near": points(near) if near is not None else [],
        "me": [float(customer[c]) for c in MAP_COLS] if customer else None,
        "max": max_values,
        "rotate": bool(rotate),
    }
    return MAP_HTML.replace("__DATA__", json.dumps(data))


# ---------------------------------------------------------------- page
model = get_model()

st.markdown(
    """
    <style>
    @keyframes shift { 0% {background-position: 0% 50%;} 100% {background-position: 200% 50%;} }
    @keyframes rise  { from {opacity: 0; transform: translateY(12px);} to {opacity: 1; transform: none;} }
    .hero { display: flex; align-items: center; gap: 14px; animation: rise 0.8s ease-out; }
    .hero-icon { font-size: 2.6rem; line-height: 1; }
    .hero-title {
        font-size: 2.6rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.02em;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777, #2563eb);
        background-size: 200% auto;
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        animation: shift 6s linear infinite;
    }
    .hero-sub { opacity: 0.7; margin: 2px 0 1.2rem 0; }
    .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 1.2rem; }
    .kpi {
        padding: 14px 16px; border-radius: 12px; border-top: 3px solid var(--c);
        border-left: 1px solid rgba(127,127,127,.25); border-right: 1px solid rgba(127,127,127,.25);
        border-bottom: 1px solid rgba(127,127,127,.25); background: rgba(127,127,127,.07);
    }
    .kpi b { display: block; font-size: 1.7rem; line-height: 1.25; }
    .kpi span { opacity: 0.7; font-size: 0.85rem; }
    .risk {
        padding: 16px 18px; border-radius: 12px; border-left: 6px solid var(--c);
        border-top: 1px solid rgba(127,127,127,.25); border-right: 1px solid rgba(127,127,127,.25);
        border-bottom: 1px solid rgba(127,127,127,.25); background: rgba(127,127,127,.07);
    }
    .risk .lvl { font-size: 1.5rem; font-weight: 800; color: var(--c); margin-bottom: 2px; }
    [data-testid="stForm"] { border-radius: 14px; padding: 1.2rem 1.4rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; border-radius: 8px 8px 0 0; }
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #2563eb, #7c3aed); color: #fff; border: 0;
        font-weight: 700; padding: 0.6rem 1.6rem;
    }
    [data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.12); color: #fff; }
    @media (max-width: 760px) {
        .kpis { grid-template-columns: repeat(2, 1fr); }
        .hero-title, .hero-icon { font-size: 1.8rem; }
    }
    @media (prefers-reduced-motion: reduce) { .hero, .hero-title { animation: none; } }
    </style>
    <div class="hero"><span class="hero-icon">📉</span><span class="hero-title">Customer Churn Predictor</span></div>
    <div class="hero-sub">Predict who is about to leave, then see where they sit among real customers.</div>
    <div class="kpis">
      <div class="kpi" style="--c:#2563eb"><b>7,043</b><span>Customers in the data</span></div>
      <div class="kpi" style="--c:#db2777"><b>26.5%</b><span>Churned overall</span></div>
      <div class="kpi" style="--c:#7c3aed"><b>0.841</b><span>Test ROC-AUC</span></div>
      <div class="kpi" style="--c:#16a34a"><b>0.783</b><span>Test recall</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About this project")
    st.write(
        "An end-to-end machine learning pipeline that predicts whether a telecom "
        "customer is likely to leave."
    )
    st.markdown(
        "- **Data:** 7,043 customers, 26.5% churned\n"
        "- **Model:** Logistic Regression, tuned with 5-fold CV\n"
        "- **Test ROC-AUC:** 0.841\n"
        "- **Test Recall:** 0.783"
    )
    st.markdown(f"[View the code on GitHub]({REPO_URL})")
    st.caption("Built with scikit-learn, MLflow, FastAPI and Streamlit.")
    st.divider()
    theme_name = st.selectbox("Background colour", list(THEMES), index=list(THEMES).index("Dark navy"))

t = THEMES[theme_name]

BASE_CSS = """
<style>
.stApp { background: __BG__; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: __PANEL__; }
[data-baseweb="select"] div, [data-baseweb="input"], [data-baseweb="input"] > div,
[data-baseweb="base-input"], [data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] button { background: __PANEL__ !important; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {
    border-color: __LINE__ !important; }
/* newer Streamlit builds the dropdown with React Aria (data-rac) instead of BaseWeb */
[data-testid="stSelectbox"] [data-rac] { background: __PANEL__ !important; border-color: __LINE__ !important; }
/* the list that opens when a dropdown is clicked */
[data-testid="stSelectboxVirtualDropdown"] { background: __PANEL__ !important; border: 1px solid __LINE__; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] { background: __LINE__ !important; }
.stButton button { background: __PANEL__; border-color: __LINE__; }
[data-testid="stForm"] { background: __FORM__; border-color: __LINE__; }
</style>
"""

# Extra rules for dark palettes: light text everywhere, and dark-looking tables
DARK_CSS = """
<style>
.stApp, [data-testid="stSidebar"] { color: __TEXT__; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
[data-testid="stMarkdownContainer"] *, [data-testid="stWidgetLabel"] *,
[data-testid="stMetric"] *, [data-testid="stCaptionContainer"] *,
[data-testid="stSlider"] *, .stTabs [data-baseweb="tab"] * { color: __TEXT__; }
input, textarea, [data-baseweb="select"] *, [data-testid="stNumberInput"] button {
    color: __TEXT__ !important; -webkit-text-fill-color: __TEXT__ !important; }
[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg { fill: __TEXT__ !important; }
[data-testid="stSelectbox"] [data-rac], [data-testid="stSelectbox"] button, [data-testid="stSelectbox"] svg {
    color: __TEXT__ !important; -webkit-text-fill-color: __TEXT__ !important; }
[data-testid="stFormSubmitButton"] button * { color: #fff; }
[data-testid="stSelectboxVirtualDropdown"] * { color: __TEXT__ !important; }
[data-testid="stDataFrame"] { filter: invert(0.92) hue-rotate(180deg); }
</style>
"""

dark = "text" in t
css = BASE_CSS + (DARK_CSS if dark else "")
for key, value in {
    "__BG__": t["bg"], "__PANEL__": t["panel"], "__LINE__": t["line"],
    "__FORM__": "rgba(255,255,255,.05)" if dark else "rgba(255,255,255,.55)",
    "__TEXT__": t.get("text", ""),
}.items():
    css = css.replace(key, value)
st.markdown(css, unsafe_allow_html=True)

tab_predict, tab_map, tab_perf = st.tabs(["🔮 Predict", "🧊 3D customer map", "📊 Model performance"])

# ---------- Predict ----------
with tab_predict:
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Load high-risk example", on_click=load_example, args=(HIGH_RISK,))
    b2.button("Load low-risk example", on_click=load_example, args=(LOW_RISK,))

    with st.form("customer_form"):
        st.subheader("Customer profile")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", ["Female", "Male"], key="gender")
        senior = c2.selectbox("Senior citizen (1 = yes)", [0, 1], key="SeniorCitizen")
        partner = c3.selectbox("Partner", YES_NO, key="Partner")
        dependents = c4.selectbox("Dependents", YES_NO, key="Dependents")

        st.subheader("Services")
        s1, s2, s3 = st.columns(3)
        phone = s1.selectbox("Phone service", YES_NO, key="PhoneService")
        multiple = s1.selectbox("Multiple lines", ["Yes", "No", "No phone service"], key="MultipleLines")
        internet = s1.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="InternetService")
        security = s2.selectbox("Online security", YES_NO_NO_INTERNET, key="OnlineSecurity")
        backup = s2.selectbox("Online backup", YES_NO_NO_INTERNET, key="OnlineBackup")
        protection = s2.selectbox("Device protection", YES_NO_NO_INTERNET, key="DeviceProtection")
        support = s3.selectbox("Tech support", YES_NO_NO_INTERNET, key="TechSupport")
        tv = s3.selectbox("Streaming TV", YES_NO_NO_INTERNET, key="StreamingTV")
        movies = s3.selectbox("Streaming movies", YES_NO_NO_INTERNET, key="StreamingMovies")

        st.subheader("Account and billing")
        a1, a2, a3 = st.columns(3)
        tenure = a1.slider("Tenure (months)", 0, 72, key="tenure")
        contract = a1.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="Contract")
        paperless = a2.selectbox("Paperless billing", YES_NO, key="PaperlessBilling")
        payment = a2.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="PaymentMethod",
        )
        monthly = a3.number_input("Monthly charges", 0.0, 200.0, step=1.0, key="MonthlyCharges")
        total = a3.number_input("Total charges", 0.0, 10000.0, step=10.0, key="TotalCharges")

        submitted = st.form_submit_button("Predict churn risk")

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        probability = predict_churn(customer, model)["probability"]
        st.session_state["result"] = {"customer": customer, "probability": probability}

    result = st.session_state.get("result")
    if result:
        customer, probability = result["customer"], result["probability"]
        level, advice = risk_level(probability)

        st.divider()
        st.subheader("Result")
        r1, r2 = st.columns([1, 2])
        with r1:
            components.html(gauge_html(probability), height=250)

        colors = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
        r2.markdown(
            f'<div class="risk" style="--c:{colors[level]}"><div class="lvl">{level} risk</div>'
            f'<div>{advice}</div></div>',
            unsafe_allow_html=True,
        )

        try:
            near = similar_customers(load_customers(), customer)
            r2.metric(
                "Churn rate among the 100 most similar customers",
                f"{near['Churn'].mean() * 100:.0f}%",
                help="Similar = closest in tenure, monthly charges and total charges. "
                     "This is a lookup in the data, not the model's prediction.",
            )
        except Exception:
            r2.caption("Similar-customer lookup is unavailable (dataset not found).")

        risky, protective = risk_signals(customer)
        e1, e2 = st.columns(2)
        e1.markdown("**Risk signals in this profile**")
        e1.markdown("\n".join(f"- {item}" for item in risky) or "- none found")
        e2.markdown("**Protective signals in this profile**")
        e2.markdown("\n".join(f"- {item}" for item in protective) or "- none found")
        st.caption("Churn rates in brackets come from the training data (overall churn rate: 26.5%).")
        st.info("Open the **3D customer map** tab to see this customer among real customers.")

# ---------- 3D map ----------
with tab_map:
    st.subheader("Where does this customer sit among real customers?")
    st.write(
        "Each dot is one customer. The map turns on its own. Drag to rotate, scroll to zoom, "
        "and hover over a dot to see its numbers."
    )
    rotate = st.toggle("Auto-rotate", value=True)

    try:
        customers = load_customers()
        sample = customers.sample(min(1500, len(customers)), random_state=42)
        result = st.session_state.get("result")
        customer = result["customer"] if result else None
        near = similar_customers(customers, customer) if customer else None
        max_values = [float(customers[c].max()) for c in MAP_COLS]

        components.html(map_html(sample, customer, near, max_values, rotate), height=640)

        if customer:
            st.caption(
                "The yellow diamond is your customer. White rings mark the 100 customers "
                "closest to it. This map is for exploring only. The prediction comes from "
                "the Logistic Regression model, not from this chart."
            )
        else:
            st.caption("Run a prediction in the Predict tab to add your customer as a yellow diamond.")
    except Exception as error:
        st.warning(f"The 3D map needs the dataset in the repo. Details: {error}")

# ---------- Model performance ----------
with tab_perf:
    st.subheader("Model comparison (test set, 1,409 customers)")
    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.738, 0.764, 0.752],
        "Precision": [0.504, 0.548, 0.528],
        "Recall": [0.783, 0.620, 0.599],
        "F1": [0.614, 0.582, 0.561],
        "ROC-AUC": [0.841, 0.822, 0.806],
    })
    st.dataframe(results, hide_index=True)
    st.bar_chart(results.set_index("Model")[["Precision", "Recall", "F1", "ROC-AUC"]], stack=False)

    st.write(
        "Logistic Regression was chosen because it had the best ROC-AUC and Recall. "
        "Missing a customer who is about to leave costs more than a false alarm, "
        "so Recall mattered more than accuracy."
    )

    st.subheader("Confusion matrix of the final model")
    matrix = pd.DataFrame(
        [[749, 286], [81, 293]],
        index=["Actually stayed", "Actually churned"],
        columns=["Predicted: stay", "Predicted: churn"],
    )
    st.dataframe(matrix)
    st.caption("The model caught 293 of the 374 customers who actually churned.")