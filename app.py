# app.py — DAR Global CEO Dashboard (with AI/ML Insights + Geo AI)
# Modified to use SQL Server via Streamlit connection instead of CSV files

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# Optional horizontal nav
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# Optional ML deps (fallback if not installed)
SKLEARN_OK = True
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score
except Exception:
    SKLEARN_OK = False

# -----------------------------------------------------------------------------
# Page config and theme
# -----------------------------------------------------------------------------
st.set_page_config(page_title="DAR Global - Executive Dashboard", layout="wide", initial_sidebar_state="expanded")

# Define theme variables BEFORE CSS
EXEC_PRIMARY = "#DAA520"
EXEC_BLUE = "#1E90FF"
EXEC_GREEN = "#32CD32"
EXEC_DANGER = "#DC143C"
EXEC_BG = "#1a1a1a"
EXEC_SURFACE = "#2d2d2d"

st.markdown(f"""
<style>
:root {{
    --exec-bg: {EXEC_BG};
    --exec-surface: {EXEC_SURFACE};
    --exec-primary: {EXEC_PRIMARY};
    --exec-blue: {EXEC_BLUE};
    --exec-green: {EXEC_GREEN};
}}

/* Full-width and trimmed paddings */
section.main div.block-container {{
    padding-left: 0.1rem !important;
    padding-right: 0.2rem !important;
    padding-top: 0.1rem !important;
    padding-bottom: 0.1rem !important;
    max-width: 100% !important;
}}

/* Bigger navigation bar */
div[role="tablist"] {{
    padding-top: 2px !important;
    padding-bottom: 2px !important;
    gap: 6px !important;
}}

div[role="tablist"] div, div[role="tablist"] button {{
    font-size: 30px !important;
    line-height: 36px !important;
    padding: 6px 14px !important;
}}

div[role="tablist"] button[aria-selected="true"], div[role="tab"][aria-selected="true"] {{
    border-bottom: 1px solid {EXEC_PRIMARY} !important;
}}

/* Remove Streamlit top headroom */
header[data-testid="stHeader"] {{
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
}}

div[role="tablist"] {{
    margin-bottom: 6px !important;
}}

/* Compact headings */
h1, .stMarkdown h1 {{
    margin-top: 2px !important;
    margin-bottom: 6px !important;
}}

h2, .stMarkdown h2 {{
    margin-top: 2px !important;
    margin-bottom: 6px !important;
}}

/* Metric cards */
div[data-testid="metric-container"] {{
    background: linear-gradient(135deg, var(--exec-surface) 0%, var(--exec-bg) 100%);
    border: 2px solid var(--exec-primary);
    padding: .65rem;
    border-radius: 10px;
    color: white;
}}

/* Insight callouts */
.insight-box {{
    background: linear-gradient(135deg, var(--exec-surface) 0%, var(--exec-bg) 100%);
    padding: 14px;
    border-radius: 10px;
    border-left: 5px solid var(--exec-green);
    color: white;
}}

/* Section wrapper */
.section {{
    background: linear-gradient(135deg, var(--exec-surface) 0%, var(--exec-bg) 100%);
    padding: 10px;
    border-radius: 6px;
    border: 1px solid #444;
}}

/* Dividers and element spacing */
hr {{
    margin: 6px 0 !important;
}}

[data-testid="stVerticalBlock"]:has(.main-header) + div {{
    margin-top: 6px !important;
}}

[data-testid="stDataFrame"] {{
    margin-top: 6px !important;
}}

.element-container:has(.plotly) {{
    margin-top: 6px !important;
}}

/* Legend nudge */
g.legend {{
    transform: translate(0, -10px);
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def format_currency(v):
    if pd.isna(v): return "$0"
    return f"${v/1e9:.1f}B" if v>=1e9 else (f"${v/1e6:.1f}M" if v>=1e6 else f"${v:,.0f}")

def format_number(v):
    if pd.isna(v): return "0"
    return f"{v/1e6:.1f}M" if v>=1e6 else (f"{v/1e3:.1f}K" if v>=1e3 else f"{v:,.0f}")

def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")

# -----------------------------------------------------------------------------
# Database loader (SQL Server via st.connection) — replaces CSV reading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    """
    Pulls all sheets from SQL using Streamlit's SQLConnection; override table
    names in secrets under [connections.sql.tables] without code changes.
    """
    from streamlit.connections import SQLConnection
    
    conn = st.connection("sql", type=SQLConnection)
    
    try:
        info = conn.query("SELECT @@SERVERNAME AS server, DB_NAME() AS db", ttl=60)
        st.caption(f"Connected to {info.iloc[0]['server']} / {info.iloc[0]['db']}")
    except Exception as e:
        st.error(f"Connectivity check failed: {e}")
        return {}

    # Optional table overrides from secrets
    tbl_cfg = st.secrets.get("connections", {}).get("sql", {}).get("tables", {})
    def T(key, default):
        return tbl_cfg.get(key, default)

    def fetch(table_fqn, label=None, limit=None):
        label = label or table_fqn
        top = f"TOP {int(limit)} " if limit else ""
        try:
            return conn.query(f"SELECT {top}* FROM {table_fqn}", ttl=600)
        except Exception as e:
            st.error(f"Query failed for {label}: {e}")
            return None

    # Pull raw tables
    ds = {}
    ds["leads"]        = fetch(T("leads","dbo.Lead"), "leads")
    ds["agents"]       = fetch(T("agents","dbo.Agents"), "agents")
    ds["calls"]        = fetch(T("calls","dbo.LeadCallRecord"), "calls")
    ds["schedules"]    = fetch(T("schedules","dbo.LeadSchedule"), "schedules")
    ds["transactions"] = fetch(T("transactions","dbo.LeadTransaction"), "transactions")
    ds["countries"] = fetch(T("countries","dbo.Country"), "countries")
    ds["lead_stages"] = fetch(T("lead_stages","dbo.LeadStage"), "lead_stages")  
    ds["lead_statuses"] = fetch(T("lead_statuses","dbo.LeadStatus"), "lead_statuses")
    ds["lead_sources"] = fetch(T("lead_sources","dbo.LeadSource"), "lead_sources")
    ds["lead_scoring"] = fetch(T("lead_scoring","dbo.LeadScoring"), "lead_scoring")
    ds["call_statuses"] = fetch(T("call_statuses","dbo.CallStatus"), "call_statuses")
    ds["sentiments"] = fetch(T("sentiments","dbo.CallSentiment"), "sentiments")
    ds["task_types"] = fetch(T("task_types","dbo.TaskType"), "task_types")
    ds["task_statuses"] = fetch(T("task_statuses","dbo.TaskStatus"), "task_statuses")
    ds["city_region"] = fetch(T("city_region","dbo.CityRegion"), "city_region")
    ds["timezone_info"] = fetch(T("timezone_info","dbo.TimezoneInfo"), "timezone_info")
    ds["priority"] = fetch(T("priority","dbo.Priority"), "priority")
    ds["meeting_status"] = fetch(T("meeting_status","dbo.MeetingStatus"), "meeting_status")
    ds["agent_meeting_assignment"] = fetch(T("agent_meeting_assignment","dbo.AgentMeetingAssignment"), "agent_meeting_assignment")

    # Normalization helpers (mirror your CSV logic)
    def norm(df):
        if df is None: return None
        out = df.copy()
        out.columns = out.columns.str.strip().str.replace(r"[^\w]+","_",regex=True).str.lower()
        return out

    def rename(df, mapping):
        # Only rename if source column exists
        cols = {src: dst for src, dst in mapping.items() if src in df.columns}
        return df.rename(columns=cols)

    # --- Normalize and align to canonical names used in pages ---
    if ds["leads"] is not None:
        df = norm(ds["leads"])
        df = rename(df, {
            "leadid":"LeadId","lead_id":"LeadId","leadcode":"LeadCode",
            "leadstageid":"LeadStageId","leadstatusid":"LeadStatusId","leadscoringid":"LeadScoringId",
            "assignedagentid":"AssignedAgentId","createdon":"CreatedOn","isactive":"IsActive",
            "countryid":"CountryId","cityregionid":"CityRegionId",
            "estimatedbudget":"EstimatedBudget","budget":"EstimatedBudget"
        })
        for col, default in [("EstimatedBudget",0.0),("LeadStageId",pd.NA),("LeadStatusId",pd.NA),
                             ("AssignedAgentId",pd.NA),("CreatedOn",pd.NaT),("IsActive",1)]:
            if col not in df.columns: df[col]=default
        
        df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], errors="coerce")
        df["EstimatedBudget"] = pd.to_numeric(df["EstimatedBudget"], errors="coerce").fillna(0.0)
        ds["leads"] = df

    if ds["agents"] is not None:
        df = norm(ds["agents"])
        df = rename(df,{"agentid":"AgentId","firstname":"FirstName","first_name":"FirstName","lastname":"LastName","last_name":"LastName","isactive":"IsActive"})
        for c, d in [("FirstName",""),("LastName",""),("Role",""),("IsActive",1)]:
            if c not in df.columns: df[c]=d
        ds["agents"] = df

    # Normalize and align to canonical names used in pages
    if ds["calls"] is not None:
        df = norm(ds["calls"])
        df = rename(df, {
            "leadcallid":"LeadCallId","lead_id":"LeadId","leadid":"LeadId",
            "callstatusid":"CallStatusId","calldatetime":"CallDateTime","call_datetime":"CallDateTime",
            "durationseconds":"DurationSeconds","sentimentid":"SentimentId",
            "assignedagentid":"AssignedAgentId","calldirection":"CallDirection","direction":"CallDirection"
        })
        if "calldatetime" in df.columns:
            df["CallDateTime"] = pd.to_datetime(df["calldatetime"], errors="coerce")
        if "CallDateTime" in df.columns:
            df["CallDateTime"] = pd.to_datetime(df["CallDateTime"], errors="coerce")
        ds["calls"] = df

    if ds["schedules"] is not None:
        df = norm(ds["schedules"])
        df = rename(df,{"scheduleid":"ScheduleId","leadid":"LeadId","tasktypeid":"TaskTypeId","scheduleddate":"ScheduledDate","taskstatusid":"TaskStatusId","assignedagentid":"AssignedAgentId","completeddate":"CompletedDate","isfollowup":"IsFollowUp"})
        if "ScheduledDate" in df.columns: df["ScheduledDate"] = pd.to_datetime(df["ScheduledDate"], errors="coerce")
        if "CompletedDate" in df.columns: df["CompletedDate"] = pd.to_datetime(df["CompletedDate"], errors="coerce")
        ds["schedules"] = df

    if ds["transactions"] is not None:
        df = norm(ds["transactions"])
        df = rename(df,{"transactionid":"TransactionId","leadid":"LeadId","tasktypeid":"TaskTypeId","transactiondate":"TransactionDate"})
        if "TransactionDate" in df.columns: df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
        ds["transactions"] = df

    # Normalize lookups
    for lk in ["countries","lead_stages","lead_statuses","lead_sources","lead_scoring","call_statuses","sentiments","task_types","task_statuses","city_region","timezone_info","priority","meeting_status","agent_meeting_assignment"]:
        if ds.get(lk) is not None: 
            ds[lk] = norm(ds[lk])

    return ds

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Filters")
    grain = st.radio("Time grain", ["Week","Month","Year"], index=1, horizontal=True)

data = load_data()

def filter_by_date(datasets, grain_sel: str):
    out = dict(datasets)
    cands=[]
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns:
        cands.append(pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce"))
    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns:
        cands.append(pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce"))
    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns:
        cands.append(pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce"))

    if cands:
        gmin=min([c.min() for c in cands if c is not None]).date()
        gmax=max([c.max() for c in cands if c is not None]).date()
    else:
        gmax=date.today(); gmin=gmax-timedelta(days=365)

    with st.sidebar:
        preset = st.select_slider("Quick range", ["Last 7 days","Last 30 days","Last 90 days","MTD","YTD","Custom"], value="Last 30 days")
        today=date.today()
        if preset=="Last 7 days": default_start, default_end = max(gmin, today-timedelta(days=6)), today
        elif preset=="Last 30 days": default_start, default_end = max(gmin, today-timedelta(days=29)), today
        elif preset=="Last 90 days": default_start, default_end = max(gmin, today-timedelta(days=89)), today
        elif preset=="MTD": default_start, default_end = max(gmin, today.replace(day=1)), today
        elif preset=="YTD": default_start, default_end = max(gmin, date(today.year,1,1)), today
        else: default_start, default_end = gmin, gmax
        step = timedelta(days=1 if grain_sel in ["Week","Month"] else 7)
        date_start, date_end = st.slider("Date range", min_value=gmin, max_value=gmax, value=(default_start, default_end), step=step)

    def add_period(dt):
        if grain_sel=="Week": return dt.dt.to_period("W").apply(lambda p: p.start_time.date())
        if grain_sel=="Month": return dt.dt.to_period("M").apply(lambda p: p.start_time.date())
        return dt.dt.to_period("Y").apply(lambda p: p.start_time.date())
    
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns:
        dt=pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce"); mask=dt.dt.date.between(date_start, date_end)
        out["leads"]=out["leads"].loc[mask].copy(); out["leads"]["period"]=add_period(dt.loc[mask])

    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns:
        dt=pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce"); mask=dt.dt.date.between(date_start, date_end)
        out["calls"]=out["calls"].loc[mask].copy(); out["calls"]["period"]=add_period(dt.loc[mask])

    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns:
        dt=pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce"); mask=dt.dt.date.between(date_start, date_end)
        out["schedules"]=out["schedules"].loc[mask].copy(); out["schedules"]["period"]=add_period(dt.loc[mask])

    if out.get("transactions") is not None and "TransactionDate" in out["transactions"].columns:
        dt=pd.to_datetime(out["transactions"]["TransactionDate"], errors="coerce"); mask=dt.dt.date.between(date_start, date_end)
        out["transactions"]=out["transactions"].loc[mask].copy(); out["transactions"]["period"]=add_period(dt.loc[mask])

    if out.get("agent_meeting_assignment") is not None:
        ama = out["agent_meeting_assignment"].copy()
        cols_lower = {c.lower(): c for c in ama.columns}
        if "startdatetime" in cols_lower:
            dtcol = cols_lower["startdatetime"]; dt = pd.to_datetime(ama[dtcol], errors="coerce"); mask = dt.dt.date.between(date_start, date_end)
            out["agent_meeting_assignment"] = ama.loc[mask].copy()

    return out

fdata = filter_by_date(data, grain)

# Navigation setup
NAV = [("Executive","speedometer2"," Executive Summary"),
       ("Lead Status","people"," Lead Status"),
       ("AI Calls","telephone"," AI Call Activity"),
       ("AI Insights","robot"," AI Insights"),
       ("Conversion","bar-chart-line"," Conversion"),
       ("Geo AI","globe"," Geo AI")]

# Initialize variables for both navigation types
selected = None
tabs = None

if HAS_OPTION_MENU:
    selected = option_menu(None, [n[0] for n in NAV], icons=[n[1] for n in NAV],
        orientation="horizontal", default_index=0,
        styles={"container":{"padding":"0!important","background-color":"#0f1116"},
                "icon":{"color":EXEC_PRIMARY,"font-size":"35px"},
                "nav-link":{"font-size":"35px","color":"#d0d0d0","--hover-color":"#21252b"},
                "nav-link-selected":{"background-color":EXEC_SURFACE}})
else:
    tabs = st.tabs([n[2] for n in NAV])

# -----------------------------------------------------------------------------
# Helper functions for pages
# -----------------------------------------------------------------------------
def recent_agg(df, when_col, cutoff, days=14):
    if df is None or len(df)==0 or when_col not in df:
        return pd.DataFrame({"LeadId":[], "n":[], "connected":[], "mean_dur":[], "last_days":[]})
    x = df.copy()
    x[when_col] = pd.to_datetime(x[when_col], errors="coerce")
    window = cutoff - pd.Timedelta(days=days)
    x = x[(x[when_col] >= window) & (x[when_col] <= cutoff)]
    g = x.groupby("LeadId").agg({
        "LeadId": "count",
        "CallStatusId": lambda s: (s==1).mean() if "CallStatusId" in x.columns else 0.0,
        "DurationSeconds": "mean" if "DurationSeconds" in x.columns else lambda s: 0.0
    }).reset_index()
    g.columns = ["LeadId","n","connected","mean_dur"]
    last = x.groupby("LeadId")[when_col].max().reset_index().rename(columns={when_col:"last_dt"})
    g = g.merge(last, on="LeadId", how="left")
    g["last_days"] = (cutoff - g["last_dt"]).dt.days.fillna(999)
    return g.drop(columns=["last_dt"], errors="ignore")

# -----------------------------------------------------------------------------
# Page functions
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    st.subheader("🎯 Executive Summary")
    leads = d.get("leads")
    if leads is None or len(leads)==0:
        st.info("No data available in the selected range.")
        return
    
    # Simple KPI display
    total_leads = len(leads)
    won_leads = int((leads.get("LeadStatusId", pd.Series()).eq(9)).sum())
    conv_rate = (won_leads/total_leads*100) if total_leads > 0 else 0.0
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Leads", f"{total_leads:,}")
    with c2: st.metric("Won Leads", f"{won_leads:,}")
    with c3: st.metric("Conversion Rate", f"{conv_rate:.1f}%")

def show_lead_status(d):
    st.subheader("📈 Lead Status")
    leads = d.get("leads")
    lead_statuses = d.get("lead_statuses")
    
    if leads is None or len(leads)==0:
        st.info("No lead data available.")
        return
    
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        # Simple status distribution
        status_counts = leads.get("LeadStatusId", pd.Series()).value_counts()
        status_map = dict(zip(lead_statuses.get("leadstatusid", []), lead_statuses.get("statusname_e", [])))
        
        df_status = pd.DataFrame({
            "Status": [status_map.get(k, f"Status {k}") for k in status_counts.index],
            "Count": status_counts.values
        })
        
        fig = px.pie(df_status, names="Status", values="Count", hole=0.4, title="Lead Status Distribution")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Lead status data not available for visualization.")

def show_calls(d):
    st.subheader("📞 AI Call Activity")
    calls = d.get("calls")
    
    if calls is None or len(calls)==0:
        st.info("No call data available.")
        return
    
    # Simple call metrics
    total_calls = len(calls)
    connected_calls = int((calls.get("CallStatusId", pd.Series()).eq(1)).sum())
    success_rate = (connected_calls/total_calls*100) if total_calls > 0 else 0.0
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Calls", f"{total_calls:,}")
    with c2: st.metric("Connected Calls", f"{connected_calls:,}")
    with c3: st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Show recent calls
    st.dataframe(calls.head(100), use_container_width=True)

def show_ai_insights(d):
    st.subheader("🤖 AI Insights")
    st.info("AI Insights page - Implementation in progress")

def show_conversions(d):
    st.subheader("📊 Conversion Analysis")
    st.info("Conversion page - Implementation in progress")

def show_geo_ai(d):
    st.subheader("🌍 Geo AI Analysis")
    st.info("Geo AI page - Implementation in progress")

# -----------------------------------------------------------------------------
# Navigation routing
# -----------------------------------------------------------------------------
if HAS_OPTION_MENU:
    if selected == "Executive":
        show_executive_summary(fdata)
    elif selected == "Lead Status":
        show_lead_status(fdata)
    elif selected == "AI Calls":
        show_calls(fdata)
    elif selected == "AI Insights":
        show_ai_insights(fdata)
    elif selected == "Conversion":
        show_conversions(fdata)
    elif selected == "Geo AI":
        show_geo_ai(fdata)
else:
    with tabs[0]: show_executive_summary(fdata)
    with tabs[1]: show_lead_status(fdata)
    with tabs[2]: show_calls(fdata)
    with tabs[3]: show_ai_insights(fdata)
    with tabs[4]: show_conversions(fdata)
    with tabs[5]: show_geo_ai(fdata)
