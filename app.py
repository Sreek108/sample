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

    # -----------------------------------------------------------------------------
    def add_period(dt):
        if grain_sel=="Week": return dt.dt.to_period("W").apply(lambda p: p.start_time.date())
        if grain_sel=="Month": return dt.dt.to_period("M").apply(lambda p: p.start_time.date())
        return dt.dt.to_period("Y").apply(lambda p: p.start_time.date())
    
    # -----------------------------------------------------------------------------
    # Leads
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns:
        dt=pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce"); mask=dt.dt.date.between(date_start, date_end)
        out["leads"]=out["leads"].loc[mask].copy(); out["leads"]["period"]=add_period(dt.loc[mask])

    # Calls
    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns:
        dt=pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce"); mask=dt.dt.date.between(date_start, date_end)
        out["calls"]=out["calls"].loc[mask].copy(); out["calls"]["period"]=add_period(dt.loc[mask])

    # Schedules
    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns:
        dt=pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce"); mask=dt.dt.date.between(date_start, date_end)
        out["schedules"]=out["schedules"].loc[mask].copy(); out["schedules"]["period"]=add_period(dt.loc[mask])

    # Transactions
    if out.get("transactions") is not None and "TransactionDate" in out["transactions"].columns:
        dt=pd.to_datetime(out["transactions"]["TransactionDate"], errors="coerce"); mask=dt.dt.date.between(date_start, date_end)
        out["transactions"]=out["transactions"].loc[mask].copy(); out["transactions"]["period"]=add_period(dt.loc[mask])

    # AgentMeetingAssignment (align to window via StartDateTime)
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

if HAS_OPTION_MENU:
    selected = option_menu(None, [n[0] for n in NAV], icons=[n[1] for n in NAV],
        orientation="horizontal", default_index=0,
        styles={"container":{"padding":"0!important","background-color":"#0f1116"},
                "icon":{"color":EXEC_PRIMARY,"font-size":"35px"},
                "nav-link":{"font-size":"35px","color":"#d0d0d0","--hover-color":"#21252b"},
                "nav-link-selected":{"background-color":EXEC_SURFACE}})
else:
    tabs = st.tabs([n[2] for n in NAV])
    selected = None

# -----------------------------------------------------------------------------
# Executive Summary
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    leads = d.get("leads")
    agents = d.get("agents")
    calls = d.get("calls")
    lead_statuses = d.get("lead_statuses")
    countries = d.get("countries")
    
    if leads is None or len(leads) == 0:
        st.info("No data available in the selected range.")
        return

    # -----------------------------------------------------------------------------
    # Won status id
    won_status_id = 9
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        match = lead_statuses.loc[lead_statuses["statusname_e"].str.lower() == "won"]
        if not match.empty and "leadstatusid" in match.columns:
            won_status_id = int(match.iloc[0]["leadstatusid"])

    st.subheader("Performance KPIs")
    today = pd.Timestamp.today().normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)
    date_ranges = {"Week to Date":(week_start,today),"Month to Date":(month_start,today),"Year to Date":(year_start,today)}
    cols = st.columns(3)

    meetings = d.get("agent_meeting_assignment")
    for (label, (start, end)), col in zip(date_ranges.items(), cols):
        leads_period = leads.loc[(pd.to_datetime(leads["CreatedOn"], errors="coerce")>=pd.Timestamp(start)) & (pd.to_datetime(leads["CreatedOn"], errors="coerce")<=pd.Timestamp(end))] if "CreatedOn" in leads.columns else pd.DataFrame()
        
        if meetings is not None and len(meetings) > 0:
            m = meetings.copy()
            m.columns = m.columns.str.lower()
            date_col = "startdatetime" if "startdatetime" in m.columns else None
            if date_col is not None:
                m["dt"] = pd.to_datetime(m[date_col], errors="coerce")
                m = m[(m["dt"]>=pd.Timestamp(start)) & (m["dt"]<=pd.Timestamp(end))]
                if "meetingstatusid" in m.columns:
                    m = m[m["meetingstatusid"].isin([1,6])]
                meetings_period = m
            else:
                meetings_period = pd.DataFrame()
        else:
            meetings_period = pd.DataFrame()

        # Won status id
        total_leads_p = int(len(leads_period))
        won_leads_p = int((leads_period["LeadStatusId"]==won_status_id).sum()) if "LeadStatusId" in leads_period.columns else 0
        conv_rate_p = (won_leads_p/total_leads_p*100.0) if total_leads_p else 0.0
        meetings_scheduled = int(meetings_period["leadid"].nunique()) if "leadid" in meetings_period.columns else 0

        with col:
            st.markdown(f"#### {label}")
            st.markdown("**Total Leads**"); st.markdown(f"<span style='font-size:2rem;'>{total_leads_p}</span>", unsafe_allow_html=True)
            st.markdown("**Conversion Rate**"); st.markdown(f"<span style='font-size:2rem;'>{conv_rate_p:.1f}%</span>", unsafe_allow_html=True)
            st.markdown("**Meetings Scheduled**"); st.markdown(f"<span style='font-size:2rem;'>{meetings_scheduled}</span>", unsafe_allow_html=True)

    # Won status id
    st.markdown("---")
    st.subheader("Trend at a glance")
    trend_style = st.radio("Trend style", ["Line","Bars","Bullet"], index=0, horizontal=True, key="__trend_style_exec")

    leads_local = leads.copy()
    if "period" not in leads_local.columns:
        dt = pd.to_datetime(leads_local.get("CreatedOn"), errors="coerce")
        leads_local["period"] = dt.dt.to_period("M").apply(lambda p: p.start_time.date())

    leads_ts = leads_local.groupby("period").size().reset_index(name="value")
    
    if "LeadStatusId" in leads_local.columns:
        per_leads = leads_local.groupby("period").size().rename("total")
        per_won = leads_local.loc[leads_local["LeadStatusId"].eq(won_status_id)].groupby("period").size().rename("won")
        conv_ts = pd.concat([per_leads, per_won], axis=1).fillna(0.0).reset_index()
        conv_ts["value"] = (conv_ts["won"]/conv_ts["total"]*100).round(1)
    else:
        conv_ts = pd.DataFrame({"period":[], "value":[]})

    meetings = d.get("agent_meeting_assignment")
    if meetings is not None and len(meetings) > 0:
        m = meetings.copy()
        m.columns = m.columns.str.lower()
        date_col = "startdatetime" if "startdatetime" in m.columns else None
        if date_col is not None:
            m["period"] = pd.to_datetime(m[date_col], errors="coerce").dt.to_period("W").apply(lambda p: p.start_time.date())
            if "meetingstatusid" in m.columns:
                m = m[m["meetingstatusid"].isin([1,6])]
            meet_ts = m.groupby("period").size().reset_index(name="value").rename(columns={"period":"period"})
        else:
            meet_ts = pd.DataFrame({"period":[], "value":[]})
    else:
        meet_ts = pd.DataFrame({"period":[], "value":[]})

    def _index(df):
        df = df.copy()
        if df.empty: 
            df["idx"] = []
            return df
        base = df["value"].iloc[0] if df["value"].iloc[0] != 0 else 1.0
        df["idx"] = (df["value"]/base) * 100.0
        return df

    # Trend at a glance
    leads_ts = _index(leads_ts)
    conv_ts = _index(conv_ts)
    meet_ts = _index(meet_ts)

    def _apply_axes(fig, ys, title):
        ymin = float(pd.Series(ys).min()) if len(ys) else 0
        ymax = float(pd.Series(ys).max()) if len(ys) else 1
        pad = max(1.0, (ymax-ymin)*0.12)
        rng = [ymin-pad, ymax+pad]
        fig.update_layout(height=180, title=dict(text=title, x=0.01, font=dict(size=12, color="#cfcfcf")),
                          margin=dict(l=6,r=6,t=24,b=8), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=4, ticks="outside")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=3, ticks="outside", range=rng)
        return fig

    # Trend at a glance
    def tile_line(df,color,title):
        df = df.dropna().sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["period"], y=df["idx"], mode="lines+markers", line=dict(color=color, width=3, shape="spline"), marker=dict(size=5,color=color)))
        return _apply_axes(fig, df["idx"], title)

    def tile_bar(df,color,title):
        df = df.dropna().sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["period"], y=df["idx"], marker=dict(color=color, line=dict(color="rgba(255,255,255,0.15)", width=0.5)), opacity=0.9))
        return _apply_axes(fig, df["idx"], title)

    def tile_bullet(df,title,bar_color):
        if df.empty:
            fig = go.Figure()
            return _apply_axes(fig, [0,1], title)
        cur = float(df["idx"].iloc[-1])
        fig = go.Figure(go.Indicator(mode="number+gauge+delta", value=cur, number={'valueformat':".0f"}, delta={'reference':100},
                                     gauge={'shape':"bullet",'axis':{'range':[80,120]},
                                            'steps':[{'range':[80,95],'color':"rgba(220,20,60,0.35)"},{'range':[95,105],'color':"rgba(255,215,0,0.35)"},
                                                     {'range':[105,120],'color':"rgba(50,205,50,0.35)"}],
                                            'bar':{'color':bar_color},'threshold':{'line':{'color':'#fff','width':2},'value':100}}))
        fig.update_layout(height=120, margin=dict(l=8,r=8,t=26,b=8), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        return fig

    # Trend at a glance
    s1,s2,s3 = st.columns(3)
    if trend_style=="Line":
        with s1: st.plotly_chart(tile_line(leads_ts,EXEC_BLUE,"Leads trend (indexed)"), use_container_width=True)
        with s2: st.plotly_chart(tile_line(conv_ts,EXEC_GREEN,"Conversion rate (indexed)"), use_container_width=True)
        with s3: st.plotly_chart(tile_line(meet_ts,EXEC_PRIMARY,"Meeting scheduled (indexed)"), use_container_width=True)
    elif trend_style=="Bars":
        with s1: st.plotly_chart(tile_bar(leads_ts,EXEC_BLUE,"Leads trend (indexed)"), use_container_width=True)
        with s2: st.plotly_chart(tile_bar(conv_ts,EXEC_GREEN,"Conversion rate (indexed)"), use_container_width=True)
        with s3: st.plotly_chart(tile_bar(meet_ts,EXEC_PRIMARY,"Meeting scheduled (indexed)"), use_container_width=True)
    else:
        with s1: st.plotly_chart(tile_bullet(leads_ts,"Leads index",EXEC_BLUE), use_container_width=True)
        with s2: st.plotly_chart(tile_bullet(conv_ts,"Conversion index",EXEC_GREEN), use_container_width=True)
        with s3: st.plotly_chart(tile_bullet(meet_ts,"Meetings index",EXEC_PRIMARY), use_container_width=True)

    # Trend at a glance
    st.markdown("---")
    st.subheader("Lead conversion snapshot")
    # Lead conversion snapshot (funnel)
    render_funnel_and_markets(d)

# Helper function for funnel and markets
def render_funnel_and_markets(d):
    # 1. Define the helper FIRST (fully indented body, no imports here)
    def norm(df):
        if df is None: return None
        x = df.copy()
        x.columns = x.columns.str.strip().str.lower()
        return x

    # normalize helper
    leads = norm(d.get("leads"))
    statuses = norm(d.get("lead_statuses"))
    stages = norm(d.get("lead_stages"))
    ama = norm(d.get("agent_meeting_assignment"))
    trx = norm(d.get("transactions"))  # loader uses "transactions"
    task_types = norm(d.get("task_types"))
    meeting_status = norm(d.get("meeting_status"))  # loader uses "meeting_status"
    countries = d.get("countries")

    if leads is None or "leadid" not in leads.columns:
        st.info("Leads table not available")
        return

    # tables match loader keys
    snapshot_dt = st.session_state.get("snapshot_dt", pd.Timestamp.today().normalize())

    # snapshot
    if "createdon" in leads.columns:
        leads["createdon"] = pd.to_datetime(leads["createdon"], errors="coerce")
    if ama is not None and "startdatetime" in ama.columns:
        ama["startdatetime"] = pd.to_datetime(ama["startdatetime"], errors="coerce")
    if trx is not None and "transactiondate" in trx.columns:
        trx["transactiondate"] = pd.to_datetime(trx["transactiondate"], errors="coerce")

    # parse dates
    def ids_from_status_names(names):
        if statuses is None or {"leadstatusid","statusname_e"} - set(statuses.columns):
            return set()
        return set(statuses.loc[statuses["statusname_e"].str.lower().isin([n.lower() for n in names]), "leadstatusid"]
                  .dropna().astype(int))

    def ids_from_stage_names(stage_names):
        if statuses is None or stages is None:
            return set()
        needs = {"leadstatusid","leadstageid","statusname_e"} - set(statuses.columns)
        need_t = {"leadstageid","stagename_e"} - set(stages.columns)
        if needs or need_t:
            return set()
        j = statuses.merge(stages[["leadstageid","stagename_e"]], on="leadstageid", how="left")
        return set(j.loc[j["stagename_e"].str.lower().isin([n.lower() for n in stage_names]), "leadstatusid"]
                  .dropna().astype(int))

    def task_ids(names):
        if task_types is None or {"tasktypeid","typename_e"} - set(task_types.columns):
            return set()
        return set(task_types.loc[task_types["typename_e"].str.lower().isin([n.lower() for n in names]), "tasktypeid"]
                  .dropna().astype(int))

    def meeting_status_ids(names):
        if meeting_status is not None and {"meetingstatusid","statusname_e"}.issubset(meeting_status.columns):
            return set(meeting_status.loc[meeting_status["statusname_e"].str.lower().isin([n.lower() for n in names]), "meetingstatusid"].dropna().astype(int))
        return {1, 6}  # fallback

    # lookups
    qualified_sid = ids_from_stage_names(["Qualified"]) or ids_from_status_names(["Qualified"])
    won_sid = ids_from_status_names(["Won"])
    lost_sid = ids_from_status_names(["Lost"])
    negot_tid = task_ids(["Negotiation","Proposal"])
    signed_tid = task_ids(["Contract Signed","Signed"])
    lost_tid = task_ids(["Lost","Cancelled","Closed - Lost"])
    meeting_ok = meeting_status_ids(["Scheduled","Confirmed","Rescheduled"])

    # lookups
    cohort = leads.copy()
    if "createdon" in cohort.columns:
        cohort = cohort[cohort["createdon"].isna() | (cohort["createdon"] <= snapshot_dt)]

    keep_cols = [c for c in ["leadid","leadstatusid","leadsourceid","assignedagentid","leadstageid"] if c in cohort.columns]
    cohort = cohort[keep_cols].copy()

    # cohort
    meet_flag = pd.Series(False, index=cohort["leadid"])
    if ama is not None and {"leadid","meetingstatusid","startdatetime"}.issubset(ama.columns):
        m = ama.loc[
            (ama["leadid"].isin(cohort["leadid"])) &
            (ama["meetingstatusid"].isin(meeting_ok)) &
            (ama["startdatetime"] <= snapshot_dt), "leadid"]
        if not m.empty:
            meet_flag = m.groupby("leadid").size().ge(1)

    nego_flag = pd.Series(False, index=cohort["leadid"])
    signed_tx = pd.Series(False, index=cohort["leadid"])
    lost_tx = pd.Series(False, index=cohort["leadid"])
    if trx is not None and {"leadid","tasktypeid","transactiondate"}.issubset(trx.columns):
        base = trx.loc[(trx["leadid"].isin(cohort["leadid"])) & (trx["transactiondate"] <= snapshot_dt)]
        if not base.empty:
            if negot_tid:
                n = base.loc[base["tasktypeid"].isin(negot_tid), "leadid"]
                if not n.empty:
                    nego_flag = n.groupby("leadid").size().ge(1)
            if signed_tid:
                s = base.loc[base["tasktypeid"].isin(signed_tid), "leadid"]
                if not s.empty:
                    signed_tx = s.groupby("leadid").size().ge(1)
            if lost_tid:
                l = base.loc[base["tasktypeid"].isin(lost_tid), "leadid"]
                if not l.empty:
                    lost_tx = l.groupby("leadid").size().ge(1)

    cohort = cohort.set_index("leadid").join([
        meet_flag.rename("has_meeting"),
        nego_flag.rename("negotiation_tx"),
        signed_tx.rename("signed_tx"),
        lost_tx.rename("lost_tx")
    ], how="left")
    cohort[["has_meeting","negotiation_tx","signed_tx","lost_tx"]] = cohort[["has_meeting","negotiation_tx","signed_tx","lost_tx"]].fillna(False)

    # evidence flags
    cohort["qualified_now"] = cohort.get("leadstatusid", pd.Series(index=cohort.index)).isin(qualified_sid) if qualified_sid else False
    cohort["signed_now"] = cohort.get("leadstatusid", pd.Series(index=cohort.index)).isin(won_sid) if won_sid else False
    cohort["lost_now"] = cohort.get("leadstatusid", pd.Series(index=cohort.index)).isin(lost_sid) if lost_sid else False

    # evidence flags
    is_signed = cohort["signed_now"] | cohort["signed_tx"]
    is_lost = (~is_signed) & (cohort["lost_now"] | cohort["lost_tx"])
    is_nego = (~is_signed) & (~is_lost) & cohort["negotiation_tx"]
    is_meeting = (~is_signed) & (~is_lost) & (~is_nego) & cohort["has_meeting"]
    is_qualified = (~is_signed) & (~is_lost) & (~is_nego) & (~is_meeting) & cohort["qualified_now"]

    stage = np.select(
        [is_signed, is_lost, is_nego, is_meeting, is_qualified],
        ["Contract Signed","Lost","Negotiation","Meeting Scheduled","Qualified"],
        default="New"
    )
    cohort["Stage"] = pd.Categorical(
        stage, 
        categories=["New","Qualified","Meeting Scheduled","Negotiation","Contract Signed","Lost"], 
        ordered=True
    )

    # precedence: Signed > Lost > Negotiation > Meeting > Qualified > New
    funnel = (cohort.groupby("Stage").size()
             .reindex(cohort["Stage"].cat.categories, fill_value=0)
             .reset_index(name="Count"))

    new_total = max(int(funnel.loc[funnel["Stage"].eq("New"), "Count"].iloc[0]), 1)
    funnel["Label"] = (funnel["Count"] / new_total * 100).round(1).astype(str) + "%"

    fig = px.funnel(funnel, x="Count", y="Stage", text="Label", color="Stage",
                    color_discrete_sequence=["#4A90E2","#50E3C2","#2D9CDB","#FFA500","#7CFC00","#E74C3C"])
    fig.update_traces(textposition="inside", textfont_color="black")
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # aggregate and plot
    st.markdown("---")
    st.subheader("Top markets")
    if countries is not None and d.get("leads") is not None and "CountryId" in d["leads"].columns and "countryname_e" in countries.columns:
        L = d["leads"].copy()
        L["CountryId"] = pd.to_numeric(L["CountryId"], errors="coerce").astype("Int64")
        L["LeadStatusId"] = pd.to_numeric(L.get("LeadStatusId", pd.NA), errors="coerce").astype("Int64")

        # Top markets
        mom = pd.DataFrame({"CountryId": pd.Series(dtype="Int64"), "momentum": pd.Series(dtype=float)})
        if "CreatedOn" in L.columns and L["CreatedOn"].notna().any():
            W = L.copy()
            W["week"] = W["CreatedOn"].dt.to_period("W").apply(lambda p: p.start_time.date())
            wk = sorted(W["week"].dropna().unique())
            if len(wk) >= 8:
                last4 = set(wk[-4:])
                prev4 = set(wk[-8:-4])
                a = W[W["week"].isin(last4)].groupby("CountryId").size().reset_index(name="leads_last4")
                b = W[W["week"].isin(prev4)].groupby("CountryId").size().reset_index(name="leads_prev4")
                mom = a.merge(b, on="CountryId", how="outer").fillna(0)
                mom["momentum"] = (mom["leads_last4"] - mom["leads_prev4"]) / mom["leads_prev4"].replace(0, np.nan)
                mom["momentum"] = mom["momentum"].replace([np.inf,-np.inf], 0).fillna(0.0)
                mom = mom[["CountryId","momentum"]]
            else:
                tmp = W.groupby("CountryId").size().reset_index(name="leads_last")
                tmp["momentum"] = 0.0
                mom = tmp[["CountryId","momentum"]]

        # Momentum (last 4 weeks vs prior 4; initialize with key)
        name_map = {}
        if statuses is not None and {"leadstatusid","statusname_e"}.issubset(statuses.columns):
            name_map = dict(zip(statuses["leadstatusid"].astype(int), statuses["statusname_e"].astype(str)))

        leads = leads.copy()
        leads["Status"] = leads["leadstatusid"].map(name_map).fillna(leads.get("leadstatusid", pd.Series(dtype="Int64")).astype(str))
        leads["CreatedOn"] = pd.to_datetime(leads.get("CreatedOn"), errors="coerce")
        cutoff = leads["CreatedOn"].max() if "CreatedOn" in leads.columns else pd.Timestamp.today()
        leads["agedays"] = (cutoff - leads["CreatedOn"]).dt.days.astype("Int64")

        # Map status ids → names
        conn_rate = pd.DataFrame({"Status": pd.Series(dtype=str), "connect_rate": pd.Series(dtype=float)})
        if calls is not None and len(calls):
            C = calls.copy()
            C["CallDateTime"] = pd.to_datetime(C.get("CallDateTime"), errors="coerce")
            C = C.merge(leads[["LeadId","Status"]], on="LeadId", how="left")
            g = C.groupby("Status").agg(total=("LeadCallId","count"), connects=("CallStatusId", lambda s: (s==1).sum())).reset_index()
            g["connect_rate"] = (g["connects"]/g["total"]).fillna(0.0)
            conn_rate = g[["Status","connect_rate"]]

        # Connect rate per status
        base = leads.groupby("Status").agg({
            "LeadId": "count",
            "agedays": "mean",
            "EstimatedBudget": "sum"
        }).reset_index()
        base.columns = ["Status","Leads","AvgAgeDays","Pipeline"]
        total_leads = float(base["Leads"].sum()) if len(base) else 0.0
        base["Share"] = (base["Leads"]/total_leads*100.0).round(1) if total_leads>0 else 0.0

        breakdown = base.merge(meet_rate, on="Status", how="left").merge(conn_rate, on="Status", how="left")
        breakdown["meet_leads"] = breakdown["meet_leads"].fillna(0.0)
        breakdown["MeetingRate"] = (breakdown["meet_leads"]/breakdown["Leads"]*100.0).replace([np.inf, -np.inf], 0).fillna(0.0).round(1)
        breakdown["connect_rate"] = breakdown["connect_rate"].fillna(0.0).round(2)
        breakdown["AvgAgeDays"] = breakdown["AvgAgeDays"].fillna(0.0).round(1)
        breakdown = breakdown.sort_values(["Leads","Pipeline"], ascending=False)

        st.dataframe(breakdown[["Status","Leads","Share","AvgAgeDays","MeetingRate","connect_rate","Pipeline"]], 
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Leads": st.column_config.NumberColumn("Leads", format="%,d"),
                        "Share": st.column_config.ProgressColumn("Share", min_value=0.0, max_value=100.0, format="%.1f%%"),
                        "AvgAgeDays": st.column_config.NumberColumn("Avg age (days)", format="%.1f"),
                        "MeetingRate": st.column_config.ProgressColumn("Meeting rate", min_value=0.0, max_value=100.0, format="%.1f%%"),
                        "connect_rate": st.column_config.ProgressColumn("Connect rate", min_value=0.0, max_value=1.0, format="%.2f"),
                        "Pipeline": st.column_config.NumberColumn("Pipeline", format="%.0f"),
                    })

        # Aggregate breakdown

# Additional helper functions for AI insights and other pages
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
        "DurationSeconds": "mean" if "DurationSeconds" in x.columns else lambda s: x.groupby("LeadId")["LeadId"].count()
    }).reset_index()
    g.columns = ["LeadId","n","connected","mean_dur"]
    last = x.groupby("LeadId")[when_col].max().reset_index().rename(columns={when_col:"last_dt"})
    g = g.merge(last, on="LeadId", how="left")
    g["last_days"] = (cutoff - g["last_dt"]).dt.days.fillna(999)
    return g.drop(columns=["last_dt"], errors="ignore")

def weekly_meeting_series(meets):
    if meets is None or len(meets)==0:
        return pd.DataFrame({"week":[], "meetings":[]})
    m = meets.copy()
    dt_col = "StartDateTime" if "StartDateTime" in m.columns else ("startdatetime" if "startdatetime" in m.columns else None)
    if dt_col is None:
        return pd.DataFrame({"week":[], "meetings":[]})
    m["dt"] = pd.to_datetime(m[dt_col], errors="coerce")
    sid = "MeetingStatusId" if "MeetingStatusId" in m.columns else ("meetingstatusid" if "meetingstatusid" in m.columns else None)
    if sid is not None:
        m = m[m[sid].isin([1,6])]
    return m.groupby(pd.Grouper(key="dt", freq="W-MON")).size().reset_index(name="meetings").rename(columns={"dt":"week"})

# Lead Status page
def show_lead_status(d):
    leads = d.get("leads")
    statuses = d.get("lead_statuses")
    calls = d.get("calls")
    meets = d.get("agent_meeting_assignment")
    
    if leads is None or len(leads)==0:
        st.info("No lead status data in the selected range.")
        return

    # Won status id
    won_id = 9
    if statuses is not None and {"statusname_e","leadstatusid"}.issubset(statuses.columns):
        m = statuses.loc[statuses["statusname_e"].str.lower() == "won"]
        if not m.empty:
            won_id = int(m.iloc[0]["leadstatusid"])

    by_country = L.groupby("CountryId", dropna=True).size().reset_index(name="Leads")
    won_by_country = L.loc[L["LeadStatusId"].eq(won_id)].groupby("CountryId", dropna=True).size().reset_index(name="Won")
    view = (by_country.merge(won_by_country, on="CountryId", how="left")
           .merge(countries.rename(columns={"countryid":"CountryId","countryname_e":"Country"})[["CountryId","Country"]], on="CountryId", how="left")
           .fillna({"Won": 0})
           .sort_values(["Leads","Won"], ascending=False)[["Country","Leads","Won"]])

    st.dataframe(view, use_container_width=True, hide_index=True,
                column_config={
                    "Leads": st.column_config.NumberColumn("Leads", format="%d"),
                    "Won": st.column_config.NumberColumn("Won/Signed", format="%d")
                })
    else:
        st.info("Country data unavailable to build the market list.")
    
    # resolve Won id from master
    st.dataframe(view[["Country","Leads","Won","win_rate","meet_rate","connect_rate","Pipeline","opportunity_score","recommendation","action"]], 
                use_container_width=True, hide_index=True,
                column_config={
                    "win_rate": st.column_config.NumberColumn("Win rate", format="%.1f%%"),
                    "meet_rate": st.column_config.NumberColumn("Meet rate", format="%.1f%%"),
                    "connect_rate": st.column_config.NumberColumn("Connect rate", format="%.1f%%"),
                    "Pipeline": st.column_config.NumberColumn("Pipeline", format="%.0f"),
                    "opportunity_score": st.column_config.ProgressColumn("Opportunity", min_value=0.0, max_value=1.0, format="%.3f"),
                })

    # Normalize score
    if "period" not in leads.columns and "CreatedOn" in leads.columns:
        leads["period"] = leads["CreatedOn"].dt.to_period("M").apply(lambda p: p.start_time.date())

    if "period" in leads.columns:
        normalize = st.checkbox("Normalize to 100% per period", value=False, key="ls_norm")
        trend = leads.groupby(["period","Status"]).size().reset_index(name="count")
        if normalize:
            totals = trend.groupby("period")["count"].transform("sum").replace(0, np.nan)
            trend["count"] = (trend["count"]/totals*100).round(1)
            y_title = "Share (%)"
        else:
            y_title = "Count"

        fig_stack = px.bar(trend.sort_values("period"), x="period", y="count", color="Status", barmode="stack", 
                          title=f"Status mix by period ({y_title})")
        fig_stack.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", 
                               height=360, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_stack, use_container_width=True)

    # Optional stacked trend by period (uses fdata "period" already set)
    counts = leads["Status"].value_counts().reset_index()
    counts.columns = ["Status","count"]
    c1,c2 = st.columns([2,1])
    
    with c1:
        fig = px.pie(counts, names="Status", values="count", hole=0.35, 
                    color_discrete_sequence=px.colors.sequential.Viridis, 
                    title="Lead Status Share")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.metric("Total Leads", f"{len(leads):,}")
        won_id = None
        if statuses is not None and "statusname_e" in statuses.columns:
            m = statuses.loc[statuses["statusname_e"].str.lower() == "won"]
            if not m.empty:
                won_id = int(m.iloc[0]["leadstatusid"])
        won = int(leads.get("LeadStatusId", pd.Series(dtype="Int64")).astype("Int64").eq(won_id).sum()) if won_id is not None else 0
        st.metric("Won", f"{won:,}")

    # Summary donut and headline KPIs kept
    meet_rate = pd.DataFrame({"CountryId": pd.Series(dtype="Int64"), "meet_rate": pd.Series(dtype=float)})
    if meets is not None and len(meets):
        M = meets.copy()
        M.columns = M.columns.str.lower()
        dt_c = "startdatetime" if "startdatetime" in M.columns else None
        if dt_c is not None:
            if "meetingstatusid" in M.columns:
                M = M[M["meetingstatusid"].isin([1,6])]  # Scheduled/Rescheduled
            mm = M.merge(L[["LeadId","CountryId"]], left_on="leadid", right_on="LeadId", how="left")
            mr = mm.groupby("CountryId")["leadid"].nunique().reset_index(name="meet_leads")
            meet_rate = perf_base[["CountryId","Leads"]].merge(mr, on="CountryId", how="left").fillna({"meet_leads": 0})
            meet_rate["meet_rate"] = (meet_rate["meet_leads"]/meet_rate["Leads"]).fillna(0.0)
            meet_rate = meet_rate[["CountryId","meet_rate"]]

    # Meeting intent (Scheduled/Rescheduled; initialize with key to avoid KeyError)
    if isinstance(conn, set) or not isinstance(conn, pd.DataFrame):
        conn = g[["CountryId","connect_rate"]]  # ensure correct cols if set was used mistakenly

# AI Calls page
def show_calls(d):
    st.subheader("AI Call Activity")
    calls = d.get("calls")
    if calls is None or len(calls)==0:
        st.info("No call data in the selected range.")
        return

    meet_rate = pd.DataFrame({"Status": pd.Series(dtype=str), "meet_leads": pd.Series(dtype=float)})
    if meets is not None and len(meets):
        M = meets.copy()
        M.columns = M.columns.str.lower()
        dt_c = "startdatetime" if "startdatetime" in M.columns else None
        if dt_c is not None:
            if "meetingstatusid" in M.columns:
                M = M[M["meetingstatusid"].isin([1,6])]  # Scheduled/Rescheduled
            mm = M.merge(leads[["LeadId","Status"]], left_on="leadid", right_on="LeadId", how="left")
            meet_rate = mm.groupby("Status")["leadid"].nunique().reset_index(name="meet_leads")

    # Meeting intent per status (Scheduled/Rescheduled)
    C = calls.copy()
    if "CallDateTime" in C.columns:
        C["CallDateTime"] = pd.to_datetime(C["CallDateTime"], errors="coerce")

    # Normalize types

# Navigation and page routing
if HAS_OPTION_MENU:
    if selected=="Executive":
        show_executive_summary(fdata)
    elif selected=="Lead Status":
        show_lead_status(fdata)
    elif selected=="AI Calls":
        show_calls(fdata)
    elif selected=="AI Insights":
        show_ai_insights(fdata)
    elif selected=="Conversion":
        show_conversions(fdata)  # Conversion page
    elif selected=="Geo AI":
        show_geo_ai(fdata)
else:
    with tabs[0]: show_executive_summary(fdata)
    with tabs[1]: show_lead_status(fdata)
    with tabs[2]: show_calls(fdata)
    with tabs[3]: show_ai_insights(fdata)
    with tabs[4]: show_conversions(fdata)  # Conversion page
    with tabs[5]: show_geo_ai(fdata)

# Placeholder functions for missing pages (implement as needed)
def show_ai_insights(d):
    st.subheader("AI Insights")
    st.info("AI Insights page - Implementation in progress")

def show_conversions(d):
    st.subheader("Conversion Analysis")
    st.info("Conversion page - Implementation in progress")

def show_geo_ai(d):
    st.subheader("Geo AI Analysis")
    st.info("Geo AI page - Implementation in progress")


