# app.py — DAR Global CEO Dashboard (connects to SQL Server)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os

# Optional horizontal nav
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# Optional ML deps
SKLEARN_OK = True
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score
except Exception:
    SKLEARN_OK = False

# -----------------------------------------------------------------------------
# Page config and theme
# -----------------------------------------------------------------------------
st.set_page_config(page_title="DAR Global - Executive Dashboard", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")

EXEC_PRIMARY="#DAA520"; EXEC_BLUE="#1E90FF"; EXEC_GREEN="#32CD32"; EXEC_DANGER="#DC143C"; EXEC_BG="#1a1a1a"; EXEC_SURFACE="#2d2d2d"

st.markdown(f"""
<style>
.main-header {{
    background: linear-gradient(135deg, {EXEC_BG} 0%, {EXEC_SURFACE} 100%);
    color: {EXEC_PRIMARY}; padding: 24px; border-radius: 12px; border: 2px solid {EXEC_PRIMARY};
    text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,.35);
}}
.main-header h1 {{ color: {EXEC_PRIMARY}; margin: 0 0 6px 0; }}
.main-header h3 {{ color: {EXEC_BLUE}; margin: 4px 0 0 0; }}
div[data-testid="metric-container"] {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    border: 2px solid {EXEC_PRIMARY}; padding: .75rem; border-radius: 10px; color: white;
}}
.insight-box {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    padding: 18px; border-radius: 10px; border-left: 5px solid {EXEC_GREEN}; color: white;
}}
.section {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    padding: 18px; border-radius: 10px; border: 1px solid #444;
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
# Database connection and data loader
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data_from_db():
    """Load data from SQL Server using st.connection"""
    try:
        # Create SQL connection using secrets
        conn = st.connection("sql")
        
        # Dictionary to hold all datasets
        ds = {}
        
        # Define table mappings (adjust table names as needed for your database)
        table_configs = {
            "leads": "Lead",
            "agents": "Agents", 
            "calls": "LeadCallRecord",
            "schedules": "LeadSchedule",
            "transactions": "LeadTransaction",
            "countries": "Country",
            "lead_stages": "LeadStage",
            "lead_statuses": "LeadStatus",
            "lead_sources": "LeadSource",
            "lead_scoring": "LeadScoring",
            "call_statuses": "CallStatus",
            "sentiments": "CallSentiment",
            "task_types": "TaskType",
            "task_statuses": "TaskStatus",
            "city_region": "CityRegion",
            "timezone_info": "TimezoneInfo",
            "priority": "Priority",
            "meeting_status": "MeetingStatus",
            "agent_meeting_assignment": "AgentMeetingAssignment"
        }
        
        # Load each table
        for key, table_name in table_configs.items():
            try:
                query = f"SELECT * FROM dbo.{table_name}"
                df = conn.query(query, ttl=600)  # Cache for 10 minutes
                ds[key] = df
                st.write(f"✅ Loaded {len(df)} records from {table_name}")
            except Exception as e:
                st.warning(f"⚠️ Could not load {table_name}: {str(e)}")
                ds[key] = None
        
        # Data normalization and column alignment
        def norm_columns(df):
            if df is None: return df
            df = df.copy()
            df.columns = df.columns.str.strip().str.replace(r"[^\w]+", "_", regex=True).str.lower()
            return df
        
        def rename_columns(df, mapping):
            if df is None: return df
            return df.rename(columns={c: mapping[c] for c in mapping if c in df.columns})
        
        # Normalize leads table
        if ds["leads"] is not None:
            df = norm_columns(ds["leads"])
            df = rename_columns(df, {
                "leadid": "LeadId", "lead_id": "LeadId", "leadcode": "LeadCode",
                "leadstageid": "LeadStageId", "leadstatusid": "LeadStatusId", "leadscoringid": "LeadScoringId",
                "assignedagentid": "AssignedAgentId", "createdon": "CreatedOn", "isactive": "IsActive",
                "countryid": "CountryId", "cityregionid": "CityRegionId",
                "estimatedbudget": "EstimatedBudget", "budget": "EstimatedBudget"
            })
            
            # Add missing columns with defaults
            for col, default in [("EstimatedBudget", 0.0), ("LeadStageId", pd.NA), ("LeadStatusId", pd.NA),
                                ("AssignedAgentId", pd.NA), ("CreatedOn", pd.NaT), ("IsActive", 1)]:
                if col not in df.columns:
                    df[col] = default
            
            # Convert data types
            df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], errors="coerce")
            df["EstimatedBudget"] = pd.to_numeric(df["EstimatedBudget"], errors="coerce").fillna(0.0)
            ds["leads"] = df
        
        # Normalize other tables as needed
        for key in ["agents", "calls", "schedules", "transactions"]:
            if ds[key] is not None:
                ds[key] = norm_columns(ds[key])
        
        # Normalize lookup tables
        for key in ["countries", "lead_stages", "lead_statuses", "lead_sources", "lead_scoring", 
                   "call_statuses", "sentiments", "task_types", "task_statuses", "city_region", 
                   "timezone_info", "priority", "meeting_status", "agent_meeting_assignment"]:
            if ds[key] is not None:
                ds[key] = norm_columns(ds[key])
        
        return ds
        
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
        st.info("💡 Make sure your secrets.toml is configured correctly and the database is accessible.")
        return None

# Fallback to CSV loader (for development/testing)
@st.cache_data(show_spinner=False) 
def load_data_from_csv(data_dir="data"):
    """Fallback CSV loader (same as original)"""
    # Your existing CSV loading logic here...
    # (keeping the original function for fallback)
    return {}

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
  <h1>🏗️ DAR Global — Executive Dashboard</h1>
  <h3>AI‑Powered Analytics</h3>
  <p style="margin: 6px 0 0 0; color: {EXEC_GREEN};">Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar filters and data loading
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Filters")
    grain = st.radio("Time grain", ["Week","Month","Year"], index=1, horizontal=True)
    
    # Data source toggle
    use_database = st.checkbox("Use Live Database", value=True, help="Uncheck to use local CSV files")

# Load data based on user preference
if use_database:
    data = load_data_from_db()
    if data is None:
        st.stop()  # Stop execution if database connection fails
else:
    data = load_data_from_csv("data")

def filter_by_date(datasets, grain_sel: str):
    out = dict(datasets)
    cands = []
    
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns: 
        cands.append(pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce"))
    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns: 
        cands.append(pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce"))
    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns: 
        cands.append(pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce"))

    if cands:
        gmin = min([c.min() for c in cands if c is not None]).date()
        gmax = max([c.max() for c in cands if c is not None]).date()
    else:
        gmax = date.today()
        gmin = gmax - timedelta(days=365)

    with st.sidebar:
        preset = st.select_slider("Quick range", ["Last 7 days","Last 30 days","Last 90 days","MTD","YTD","Custom"], value="Last 30 days")
        today = date.today()
        
        if preset == "Last 7 days": 
            default_start, default_end = max(gmin, today-timedelta(days=6)), today
        elif preset == "Last 30 days": 
            default_start, default_end = max(gmin, today-timedelta(days=29)), today
        elif preset == "Last 90 days": 
            default_start, default_end = max(gmin, today-timedelta(days=89)), today
        elif preset == "MTD": 
            default_start, default_end = max(gmin, today.replace(day=1)), today
        elif preset == "YTD": 
            default_start, default_end = max(gmin, date(today.year,1,1)), today
        else: 
            default_start, default_end = gmin, gmax
            
        step = timedelta(days=1 if grain_sel in ["Week","Month"] else 7)
        date_start, date_end = st.slider("Date range", min_value=gmin, max_value=gmax, 
                                        value=(default_start, default_end), step=step)

    def add_period(dt):
        if grain_sel == "Week": 
            return dt.dt.to_period("W").apply(lambda p: p.start_time.date())
        if grain_sel == "Month": 
            return dt.dt.to_period("M").apply(lambda p: p.start_time.date())
        return dt.dt.to_period("Y").apply(lambda p: p.start_time.date())

    # Filter datasets by date range
    for dataset_key, date_col in [("leads", "CreatedOn"), ("calls", "CallDateTime"), 
                                 ("schedules", "ScheduledDate"), ("transactions", "TransactionDate")]:
        if out.get(dataset_key) is not None and date_col in out[dataset_key].columns:
            dt = pd.to_datetime(out[dataset_key][date_col], errors="coerce")
            mask = dt.dt.date.between(date_start, date_end)
            out[dataset_key] = out[dataset_key].loc[mask].copy()
            out[dataset_key]["period"] = add_period(dt.loc[mask])

    return out

fdata = filter_by_date(data, grain)

# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
NAV = [
    ("Executive","speedometer2","🎯 Executive Summary"),
    ("Lead Status","people","📈 Lead Status"), 
    ("AI Calls","telephone","📞 AI Call Activity"),
    ("AI Insights","robot","🤖 AI Insights")
]

if HAS_OPTION_MENU:
    selected = option_menu(None, [n[0] for n in NAV], icons=[n[1] for n in NAV], orientation="horizontal", default_index=0,
                           styles={"container":{"padding":"0!important","background-color":"#0f1116"},
                                   "icon":{"color":EXEC_PRIMARY,"font-size":"16px"},
                                   "nav-link":{"font-size":"14px","color":"#d0d0d0","--hover-color":"#21252b"},
                                   "nav-link-selected":{"background-color":EXEC_SURFACE}})
else:
    tabs = st.tabs([n[2] for n in NAV])
    selected = None

# -----------------------------------------------------------------------------
# Executive Summary (Performance KPIs only)
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    leads = d.get("leads")
    lead_statuses = d.get("lead_statuses")
    meetings = d.get("agent_meeting_assignment")

    if leads is None or len(leads) == 0:
        st.info("No data available in the selected range.")
        return

    # Determine 'Won' status id
    won_status_id = 9
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        match = lead_statuses.loc[lead_statuses["statusname_e"].str.lower() == "won"]
        if not match.empty and "leadstatusid" in match.columns:
            won_status_id = int(match.iloc[0]["leadstatusid"])

    # Performance KPIs section
    st.subheader("Performance KPIs")
    today = pd.Timestamp.today().normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)
    date_ranges = {
        "Week to Date": (week_start, today),
        "Month to Date": (month_start, today), 
        "Year to Date": (year_start, today)
    }
    
    cols = st.columns(3)
    
    for (label, (start, end)), col in zip(date_ranges.items(), cols):
        # Filter leads for period
        leads_period = leads.loc[
            (pd.to_datetime(leads["CreatedOn"], errors="coerce") >= pd.Timestamp(start)) &
            (pd.to_datetime(leads["CreatedOn"], errors="coerce") <= pd.Timestamp(end))
        ] if "CreatedOn" in leads.columns else pd.DataFrame()
        
        # Filter meetings for period
        if meetings is not None and len(meetings) > 0:
            m = meetings.copy()
            m.columns = m.columns.str.lower()
            date_col = "startdatetime" if "startdatetime" in m.columns else None
            if date_col is not None:
                m["_dt"] = pd.to_datetime(m[date_col], errors="coerce")
                m = m[(m["_dt"] >= pd.Timestamp(start)) & (m["_dt"] <= pd.Timestamp(end))]
                if "meetingstatusid" in m.columns:
                    m = m[m["meetingstatusid"].isin({1, 6})]
                meetings_period = m
            else:
                meetings_period = pd.DataFrame()
        else:
            meetings_period = pd.DataFrame()

        # Calculate KPIs
        total_leads_p = int(len(leads_period))
        won_leads_p = int((leads_period["LeadStatusId"] == won_status_id).sum()) if "LeadStatusId" in leads_period.columns else 0
        conv_rate_p = (won_leads_p / total_leads_p * 100.0) if total_leads_p else 0.0
        meetings_scheduled = int(meetings_period["leadid"].nunique()) if "leadid" in meetings_period.columns else 0

        with col:
            st.markdown(f"#### {label}")
            st.markdown("**Total Leads**")
            st.markdown(f"<span style='font-size:2rem;'>{total_leads_p}</span>", unsafe_allow_html=True)
            st.markdown("**Conversion Rate**")
            st.markdown(f"<span style='font-size:2rem;'>{conv_rate_p:.1f}%</span>", unsafe_allow_html=True)
            st.markdown("**Meetings Scheduled**")
            st.markdown(f"<span style='font-size:2rem;'>{meetings_scheduled}</span>", unsafe_allow_html=True)

# Include the rest of your existing functions (show_lead_status, show_calls, show_ai_insights)
# ... (keeping them as they were in the original file)

# -----------------------------------------------------------------------------
# Router
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
else:
    with tabs[0]: 
        show_executive_summary(fdata)
    with tabs[1]: 
        show_lead_status(fdata)
    with tabs[2]: 
        show_calls(fdata)
    with tabs[3]: 
        show_ai_insights(fdata)
