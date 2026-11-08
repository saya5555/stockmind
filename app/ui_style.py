"""Visual theme for the StockMind dashboard: glass UI. CSS only, no data logic."""

import streamlit as st

_CSS = """
<style>
:root {
  --sm-accent: #4F46E5;
  --sm-accent-2: #14B8A6;
  --sm-ink: #0F172A;
  --sm-muted: #64748B;
  --sm-glass: linear-gradient(135deg, rgba(255, 255, 255, 0.74), rgba(255, 255, 255, 0.40));
  --sm-glass-edge: 1px solid rgba(255, 255, 255, 0.70);
  --sm-glass-blur: blur(18px) saturate(1.7);
  --sm-glass-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.95),
    inset 0 -1px 0 rgba(255, 255, 255, 0.30),
    0 1px 2px rgba(15, 23, 42, 0.05),
    0 22px 44px -22px rgba(49, 46, 129, 0.32);
  --sm-glass-shadow-hover:
    inset 0 1px 0 rgba(255, 255, 255, 1),
    inset 0 -1px 0 rgba(255, 255, 255, 0.35),
    0 2px 4px rgba(15, 23, 42, 0.06),
    0 28px 52px -20px rgba(79, 70, 229, 0.42);
}
.stApp {
  background: #EDEFFA !important;
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", Roboto, sans-serif;
}
.stApp::before {
  content: "";
  position: fixed;
  inset: -10%;
  z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(38% 42% at 12% 8%, rgba(99, 102, 241, 0.40), transparent 70%),
    radial-gradient(34% 40% at 88% 6%, rgba(20, 184, 166, 0.34), transparent 70%),
    radial-gradient(40% 46% at 72% 94%, rgba(244, 114, 182, 0.26), transparent 70%),
    radial-gradient(36% 40% at 6% 90%, rgba(56, 189, 248, 0.28), transparent 70%);
  will-change: transform;
  animation: sm-drift 26s ease-in-out infinite alternate;
}
@keyframes sm-drift {
  0% { transform: translate3d(-2%, -1%, 0) scale(1); }
  100% { transform: translate3d(3%, 2%, 0) scale(1.08); }
}
[data-testid="stAppViewContainer"] {
  position: relative;
  z-index: 1;
  background: transparent !important;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {
  visibility: hidden;
  height: 0;
}
.block-container {
  max-width: 1180px;
  padding-top: 2.4rem;
  padding-bottom: 4rem;
}
.stApp h1 {
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.1;
  background: linear-gradient(90deg, #312E81 0%, #4F46E5 45%, #14B8A6 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.stApp h2, .stApp h3 {
  color: var(--sm-ink);
  font-weight: 700;
  letter-spacing: -0.02em;
}
[data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stExpander"] {
  background: var(--sm-glass);
  border: var(--sm-glass-edge);
  border-radius: 20px;
  -webkit-backdrop-filter: var(--sm-glass-blur);
  backdrop-filter: var(--sm-glass-blur);
  box-shadow: var(--sm-glass-shadow);
}
[data-testid="stMetric"] {
  position: relative;
  overflow: hidden;
  padding: 18px 20px;
  transition: transform 0.3s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.3s ease;
}
[data-testid="stMetric"]::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(115deg, transparent 32%, rgba(255, 255, 255, 0.70) 48%, transparent 64%);
  transform: translateX(-130%);
  transition: transform 0.9s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-5px);
  box-shadow: var(--sm-glass-shadow-hover);
}
[data-testid="stMetric"]:hover::after { transform: translateX(130%); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
[data-testid="stMetricLabel"] {
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--sm-muted);
}
[data-testid="stMetricValue"] {
  font-weight: 800;
  letter-spacing: -0.02em;
}
[data-baseweb="tab-list"] {
  gap: 4px;
  max-width: 100%;
  padding: 6px;
  background: var(--sm-glass);
  border: var(--sm-glass-edge);
  border-radius: 999px;
  -webkit-backdrop-filter: var(--sm-glass-blur);
  backdrop-filter: var(--sm-glass-blur);
  box-shadow: var(--sm-glass-shadow);
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
button[data-baseweb="tab"] {
  height: auto;
  padding: 9px 18px;
  font-weight: 600;
  border-radius: 999px;
  transition: background 0.25s ease, color 0.25s ease, box-shadow 0.25s ease;
}
button[data-baseweb="tab"]:hover { background: rgba(255, 255, 255, 0.55); }
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--sm-accent);
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.72));
  box-shadow: inset 0 1px 0 #FFFFFF, 0 6px 16px -8px rgba(79, 70, 229, 0.55);
}
div[role="tabpanel"] { animation: sm-fade-up 0.5s cubic-bezier(0.2, 0.8, 0.2, 1) backwards; }
@keyframes sm-fade-up {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: var(--sm-glass-edge);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: var(--sm-glass-shadow);
}
[data-testid="stImage"] img {
  background: #FFFFFF;
  border: var(--sm-glass-edge);
  border-radius: 20px;
  box-shadow: var(--sm-glass-shadow);
}
[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.55);
  -webkit-backdrop-filter: blur(22px) saturate(1.6);
  backdrop-filter: blur(22px) saturate(1.6);
  border-right: var(--sm-glass-edge);
}
[data-baseweb="select"] > div, [data-baseweb="base-input"] {
  background: rgba(255, 255, 255, 0.62) !important;
  border: var(--sm-glass-edge) !important;
  border-radius: 14px !important;
  -webkit-backdrop-filter: blur(10px);
  backdrop-filter: blur(10px);
}
[data-testid^="stBaseButton"] {
  border-radius: 14px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid^="stBaseButton"]:hover { transform: translateY(-2px); }
::selection { background: rgba(79, 70, 229, 0.20); }
@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
  [data-testid="stMetric"], [data-testid="stAlert"], [data-testid="stExpander"],
  [data-baseweb="tab-list"] { background: rgba(255, 255, 255, 0.92); }
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>
"""


def inject_css() -> None:
    """Inject the dashboard theme. Call once, right after st.set_page_config()."""
    st.markdown(_CSS, unsafe_allow_html=True)
