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

# Page config and theme
st.set_page_config(page_title="DAR Global - Executive Dashboard", layout="wide", initial_sidebar_state="expanded")

EXEC_PRIMARY = "#DAA520"
EXEC_BLUE = "#1E90FF"
EXEC_GREEN = "#32CD32"
EXEC_DANGER = "#DC143C"
EXEC_BG = "#1a1a1a"
EXEC_SURFACE = "#2d2d2d"

# CSS for theme and styling (truncated for brevity)
st.markdown(f"""
<style>
:root {{
  --exec-bg: {EXEC_BG};
  --exec-surface: {EXEC_SURFACE};
  --exec-primary: {EXEC_PRIMARY};
  --exec-blue: {EXEC_BLUE};
  --exec-green: {EXEC_GREEN};
}}

/* Add your full CSS here as in previous code */
</style>
""", unsafe_allow_html=True)

# Helpers

def norm(df):
    if df is None:
        return None
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()
    return out

# -----------------------------------------------------------------------------
# Loader for ./data (robust to plural/singular file names)
# -----------------------------------------------------------------------------
# -------------------------------------------------------------------
# Database loader (SQL Server via st.connection) — drop-in replacement
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(data_dir: str | None = None):
    """
    Pulls all sheets from SQL using Streamlit's SQLConnection.
    Optional table overrides in secrets.toml:
    [connections.sql.tables]
    leads="dbo.Lead"
    agents="dbo.Agents"
    calls="dbo.LeadCallRecord"
    schedules="dbo.LeadSchedule"
    transactions="dbo.LeadTransaction"
    countries="dbo.Country"
    lead_stages="dbo.LeadStage"
    lead_statuses="dbo.LeadStatus"
    lead_sources="dbo.LeadSource"
    lead_scoring="dbo.LeadScoring"
    call_statuses="dbo.CallStatus"
    sentiments="dbo.CallSentiment"
    task_types="dbo.TaskType"
    task_statuses="dbo.TaskStatus"
    city_region="dbo.CityRegion"
    timezone_info="dbo.TimezoneInfo"
    priority="dbo.Priority"
    meeting_status="dbo.MeetingStatus"
    agent_meeting_assignment="dbo.AgentMeetingAssignment"
    """
    from streamlit.connections import SQLConnection

    # Build connection (configured in secrets.toml under [connections.sql])
    conn = st.connection("sql", type=SQLConnection)

    # Quick connectivity check (also shows server/db in the UI)
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

    # Helper to SELECT * from a table (with TTL cache)
    def fetch(table_fqn, label=None, limit=None):
        label = label or table_fqn
        top = f"TOP {int(limit)} " if limit else ""
        try:
            return conn.query(f"SELECT {top}* FROM {table_fqn}", ttl=600)
        except Exception as e:
            st.error(f"Query failed for {label}: {e}")
            return None

    # --- Pull raw tables ---
    ds = {}
    ds["leads"]        = fetch(T("leads","dbo.Lead"), "leads")
    ds["agents"]       = fetch(T("agents","dbo.Agents"), "agents")
    ds["calls"]        = fetch(T("calls","dbo.LeadCallRecord"), "calls")
    ds["schedules"]    = fetch(T("schedules","dbo.LeadSchedule"), "schedules")
    ds["transactions"] = fetch(T("transactions","dbo.LeadTransaction"), "transactions")

    for k, default in [
        ("countries","dbo.Country"), ("lead_stages","dbo.LeadStage"), ("lead_statuses","dbo.LeadStatus"),
        ("lead_sources","dbo.LeadSource"), ("lead_scoring","dbo.LeadScoring"),
        ("call_statuses","dbo.CallStatus"), ("sentiments","dbo.CallSentiment"),
        ("task_types","dbo.TaskType"), ("task_statuses","dbo.TaskStatus"),
        ("city_region","dbo.CityRegion"), ("timezone_info","dbo.TimezoneInfo"),
        ("priority","dbo.Priority"), ("meeting_status","dbo.MeetingStatus"),
        ("agent_meeting_assignment","dbo.AgentMeetingAssignment"),
    ]:
        ds[k] = fetch(T(k, default), k)

    # --- Normalization helpers (mirror your CSV logic) ---
    def norm_lower(df):
        if df is None: return None
        out = df.copy()
        out.columns = out.columns.str.strip().str.replace(r"[^\w]+","_",regex=True).str.lower()
        return out

    def rename(df, mapping):
        # Only rename if source column exists
        cols = {src: dst for src, dst in mapping.items() if src in df.columns}
        return df.rename(columns=cols)

    # --- Leads ---
    if ds["leads"] is not None:
        df = norm_lower(ds["leads"])
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
        df["CreatedOn"] = pd.to_datetime(df.get("CreatedOn"), errors="coerce")
        df["EstimatedBudget"] = pd.to_numeric(df.get("EstimatedBudget"), errors="coerce").fillna(0.0)
        ds["leads"] = df

    # --- Agents ---
    if ds["agents"] is not None:
        df = norm_lower(ds["agents"])
        df = rename(df, {"agentid":"AgentId","firstname":"FirstName","first_name":"FirstName",
                         "lastname":"LastName","last_name":"LastName","isactive":"IsActive"})
        for c, d in [("FirstName",""),("LastName",""),("Role",""),("IsActive",1)]:
            if c not in df.columns: df[c]=d
        ds["agents"] = df

    # --- Calls ---
    if ds["calls"] is not None:
        df = norm_lower(ds["calls"])
        df = rename(df, {
            "leadcallid":"LeadCallId","lead_id":"LeadId","leadid":"LeadId",
            "callstatusid":"CallStatusId","calldatetime":"CallDateTime","call_datetime":"CallDateTime",
            "durationseconds":"DurationSeconds","sentimentid":"SentimentId",
            "assignedagentid":"AssignedAgentId","calldirection":"CallDirection","direction":"CallDirection"
        })
        if "CallDateTime" in df.columns:
            df["CallDateTime"] = pd.to_datetime(df["CallDateTime"], errors="coerce")
        ds["calls"] = df

    # --- Schedules ---
    if ds["schedules"] is not None:
        df = norm_lower(ds["schedules"])
        df = rename(df, {"scheduleid":"ScheduleId","leadid":"LeadId","tasktypeid":"TaskTypeId",
                         "scheduleddate":"ScheduledDate","taskstatusid":"TaskStatusId",
                         "assignedagentid":"AssignedAgentId","completeddate":"CompletedDate","isfollowup":"IsFollowUp"})
        if "ScheduledDate" in df.columns: df["ScheduledDate"] = pd.to_datetime(df["ScheduledDate"], errors="coerce")
        if "CompletedDate" in df.columns: df["CompletedDate"] = pd.to_datetime(df["CompletedDate"], errors="coerce")
        ds["schedules"] = df

    # --- Transactions ---
    if ds["transactions"] is not None:
        df = norm_lower(ds["transactions"])
        df = rename(df, {"transactionid":"TransactionId","leadid":"LeadId","tasktypeid":"TaskTypeId","transactiondate":"TransactionDate"})
        if "TransactionDate" in df.columns: df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
        ds["transactions"] = df

    # --- Lookups (lowercase field names are fine for joins) ---
    for lk in ["countries","lead_stages","lead_statuses","lead_sources","lead_scoring",
               "call_statuses","sentiments","task_types","task_statuses","city_region",
               "timezone_info","priority","meeting_status","agent_meeting_assignment"]:
        if ds.get(lk) is not None: ds[lk] = norm_lower(ds[lk])

    return ds

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Filters")
    grain = st.radio("Time grain", ["Week","Month","Year"], index=1, horizontal=True)

data = load_data("data")

def filter_by_date(datasets, grain_sel: str):
    out = dict(datasets)
    cands=[]
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns: cands.append(pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce"))
    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns: cands.append(pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce"))
    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns: cands.append(pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce"))

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

    # Leads
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns:
        dt=pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce")
        mask=dt.dt.date.between(date_start, date_end)
        out["leads"]=out["leads"].loc[mask].copy()
        out["leads"]["period"]=add_period(dt.loc[mask])

    # Calls
    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns:
        dt=pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce")
        mask=dt.dt.date.between(date_start, date_end)
        out["calls"]=out["calls"].loc[mask].copy()
        out["calls"]["period"]=add_period(dt.loc[mask])

    # Schedules
    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns:
        dt=pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce")
        mask=dt.dt.date.between(date_start, date_end)
        out["schedules"]=out["schedules"].loc[mask].copy()
        out["schedules"]["period"]=add_period(dt.loc[mask])

    # Transactions
    if out.get("transactions") is not None and "TransactionDate" in out["transactions"].columns:
        dt=pd.to_datetime(out["transactions"]["TransactionDate"], errors="coerce")
        mask=dt.dt.date.between(date_start, date_end)
        out["transactions"]=out["transactions"].loc[mask].copy()
        out["transactions"]["period"]=add_period(dt.loc[mask])

    # AgentMeetingAssignment — align to window via StartDateTime
    if out.get("agent_meeting_assignment") is not None:
        ama = out["agent_meeting_assignment"].copy()
        cols_lower = {c.lower(): c for c in ama.columns}
        if "startdatetime" in cols_lower:
            dtcol = cols_lower["startdatetime"]
            dt = pd.to_datetime(ama[dtcol], errors="coerce")
            mask = dt.dt.date.between(date_start, date_end)
            out["agent_meeting_assignment"] = ama.loc[mask].copy()

    return out

fdata = filter_by_date(data, grain)

# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
NAV = [
    ("Executive","speedometer2","🎯 Executive Summary"),
    ("Lead Status","people","📈 Lead Status"),
    ("AI Calls","telephone","📞 AI Call Activity"),
    ("AI Insights","robot","🤖 AI Insights"),
    ("Conversion","bar-chart-line","📊 Conversion"),  # <-- comma added and icon fixed
    ("Geo AI","globe","🌍 Geo AI")
]
if HAS_OPTION_MENU:
    selected = option_menu(None, [n[0] for n in NAV], icons=[n[1] for n in NAV], orientation="horizontal", default_index=0,
                           styles={"container":{"padding":"0!important","background-color":"#0f1116"},
                                   "icon":{"color":EXEC_PRIMARY,"font-size":"35px"},
                                   "nav-link":{"font-size":"35px","color":"#d0d0d0","--hover-color":"#21252b"},
                                   "nav-link-selected":{"background-color":EXEC_SURFACE}})
else:
    tabs = st.tabs([n[2] for n in NAV])
    selected=None

# -----------------------------------------------------------------------------
# Executive Summary (Performance KPIs, trends, funnel, top markets)
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

    # Resolve Won status ID
    won_status_id = 9
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        match = lead_statuses.loc[lead_statuses["statusname_e"].str.lower() == "won"]
        if not match.empty and "leadstatusid" in match.columns:
            won_status_id = int(match.iloc[0]["leadstatusid"])

    # Performance KPIs
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
    meetings = d.get("agent_meeting_assignment")
    
    for (label, (start, end)), col in zip(date_ranges.items(), cols):
        leads_period = leads.loc[
            (pd.to_datetime(leads["CreatedOn"], errors="coerce") >= pd.Timestamp(start)) &
            (pd.to_datetime(leads["CreatedOn"], errors="coerce") <= pd.Timestamp(end))
        ] if "CreatedOn" in leads.columns else pd.DataFrame()
        
        if meetings is not None and len(meetings) > 0:
            m = meetings.copy()
            m.columns = m.columns.str.lower()
            date_col = "startdatetime" if "startdatetime" in m.columns else None
            if date_col is not None:
                m["dt"] = pd.to_datetime(m[date_col], errors="coerce")
                m = m[(m["dt"] >= pd.Timestamp(start)) & (m["dt"] <= pd.Timestamp(end))]
                if "meetingstatusid" in m.columns:
                    m = m[m["meetingstatusid"].isin({1, 6})]
                meetings_period = m
            else:
                meetings_period = pd.DataFrame()
        else:
            meetings_period = pd.DataFrame()
        
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

    # Trend at a glance
    st.markdown("---")
    st.subheader("Trend at a glance")
    trend_style = st.radio("Trend style", ["Line", "Bars", "Bullet"], index=0, horizontal=True, key="__trend_style_exec")
    
    leads_local = leads.copy()
    if "period" not in leads_local.columns:
        dt = pd.to_datetime(leads_local.get("CreatedOn"), errors="coerce")
        leads_local["period"] = dt.dt.to_period("M").apply(lambda p: p.start_time.date())
    
    leads_ts = leads_local.groupby("period").size().reset_index(name="value")
    
    if "LeadStatusId" in leads_local.columns:
        per_leads = leads_local.groupby("period").size().rename("total")
        per_won = leads_local.loc[leads_local["LeadStatusId"].eq(won_status_id)].groupby("period").size().rename("won")
        conv_ts = pd.concat([per_leads, per_won], axis=1).fillna(0.0).reset_index()
        conv_ts["value"] = (conv_ts["won"] / conv_ts["total"] * 100).round(1)
    else:
        conv_ts = pd.DataFrame({"period": [], "value": []})
    
    meetings = d.get("agent_meeting_assignment")
    if meetings is not None and len(meetings) > 0:
        m = meetings.copy()
        m.columns = m.columns.str.lower()
        date_col = "startdatetime" if "startdatetime" in m.columns else None
        if date_col is not None:
            m["period"] = pd.to_datetime(m[date_col], errors="coerce").dt.to_period("W").apply(lambda p: p.start_time.date())
            if "meetingstatusid" in m.columns:
                m = m[m["meetingstatusid"].isin({1, 6})]
            meet_ts = m.groupby("period").size().reset_index(name="value").rename(columns={"period": "period"})
        else:
            meet_ts = pd.DataFrame({"period": [], "value": []})
    else:
        meet_ts = pd.DataFrame({"period": [], "value": []})
    
    def _index(df):
        df = df.copy()
        if df.empty:
            df["idx"] = []
            return df
        base = df["value"].iloc[0] if df["value"].iloc[0] != 0 else 1.0
        df["idx"] = (df["value"] / base) * 100.0
        return df
    
    leads_ts = _index(leads_ts)
    conv_ts = _index(conv_ts)
    meet_ts = _index(meet_ts)
    
    def _apply_axes(fig, ys, title):
        ymin = float(pd.Series(ys).min()) if len(ys) else 0
        ymax = float(pd.Series(ys).max()) if len(ys) else 1
        pad = max(1.0, (ymax - ymin) * 0.12)
        rng = [ymin - pad, ymax + pad]
        fig.update_layout(
            height=180,
            title=dict(text=title, x=0.01, font=dict(size=12, color="#cfcfcf")),
            margin=dict(l=6, r=6, t=24, b=8),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=4, ticks="outside")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=3, ticks="outside", range=rng)
        return fig
    
    def tile_line(df, color, title):
        df = df.dropna().sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["period"], y=df["idx"], mode="lines+markers", line=dict(color=color, width=3, shape="spline"), marker=dict(size=5, color=color)))
        return _apply_axes(fig, df["idx"], title)
    
    def tile_bar(df, color, title):
        df = df.dropna().sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["period"], y=df["idx"], marker=dict(color=color, line=dict(color="rgba(255,255,255,0.15)", width=0.5)), opacity=0.9))
        return _apply_axes(fig, df["idx"], title)
    
    def tile_bullet(df, title, bar_color):
        if df.empty:
            fig = go.Figure()
            return _apply_axes(fig, [0, 1], title)
        cur = float(df["idx"].iloc[-1])
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=cur,
            number={'valueformat': ".0f"},
            delta={'reference': 100},
            gauge={
                'shape': "bullet",
                'axis': {'range': [80, 120]},
                'steps': [
                    {'range': [80, 95], 'color': "rgba(220,20,60,0.35)"},
                    {'range': [95, 105], 'color': "rgba(255,215,0,0.35)"},
                    {'range': [105, 120], 'color': "rgba(50,205,50,0.35)"}
                ],
                'bar': {'color': bar_color},
                'threshold': {'line': {'color': '#fff', 'width': 2}, 'value': 100}
            }
        ))
        fig.update_layout(height=120, margin=dict(l=8, r=8, t=26, b=8), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        return fig
    
    s1, s2, s3 = st.columns(3)
    if trend_style == "Line":
        with s1:
            st.plotly_chart(tile_line(leads_ts, EXEC_BLUE, "Leads trend (indexed)"), use_container_width=True)
        with s2:
            st.plotly_chart(tile_line(conv_ts, EXEC_GREEN, "Conversion rate (indexed)"), use_container_width=True)
        with s3:
            st.plotly_chart(tile_line(meet_ts, EXEC_PRIMARY, "Meeting scheduled (indexed)"), use_container_width=True)
    elif trend_style == "Bars":
        with s1:
            st.plotly_chart(tile_bar(leads_ts, EXEC_BLUE, "Leads trend (indexed)"), use_container_width=True)
        with s2:
            st.plotly_chart(tile_bar(conv_ts, EXEC_GREEN, "Conversion rate (indexed)"), use_container_width=True)
        with s3:
            st.plotly_chart(tile_bar(meet_ts, EXEC_PRIMARY, "Meeting scheduled (indexed)"), use_container_width=True)
    else:
        with s1:
            st.plotly_chart(tile_bullet(leads_ts, "Leads index", EXEC_BLUE), use_container_width=True)
        with s2:
            st.plotly_chart(tile_bullet(conv_ts, "Conversion index", EXEC_GREEN), use_container_width=True)
        with s3:
            st.plotly_chart(tile_bullet(meet_ts, "Meetings index", EXEC_PRIMARY), use_container_width=True)

    # Lead conversion snapshot (funnel)
    st.markdown("---")
    st.subheader("Lead conversion snapshot")

    # 1) Define the helper FIRST (fully indented body, no imports here)
    def render_funnel_and_markets(d):
    EXEC_PRIMARY = "#DAA520"
    EXEC_BLUE = "#1E90FF"
    EXEC_GREEN = "#32CD32"
    EXEC_DANGER = "#DC143C"
    stage_order = ["New", "Qualified", "Meeting Scheduled", "Negotiation", "Contract Signed", "Lost"]

    leads = norm(d.get("leads"))
    statuses = norm(d.get("lead_statuses"))
    ama = norm(d.get("agent_meeting_assignment"))
    countries = norm(d.get("countries"))

    if leads is None or "leadid" not in leads.columns:
        st.info("Leads data unavailable.")
        return

    def ids_from_status_names(names):
        if statuses is None or not {"leadstatusid", "statusname_e"}.issubset(statuses.columns):
            return set()
        return set(
            statuses.loc[statuses["statusname_e"].str.lower().isin([n.lower() for n in names]), "leadstatusid"]
            .dropna()
            .astype(int)
        )

    cohort = leads.copy()
    cohort.index = cohort["leadid"]
    cohort["hasmeeting"] = False
    if ama is not None and {"leadid", "meetingstatusid", "startdatetime"}.issubset(ama.columns):
        amavalid = ama.loc[
            ama["leadid"].isin(cohort["leadid"])
            & ama["meetingstatusid"].isin({1, 6})
            & (pd.to_datetime(ama["startdatetime"], errors="coerce") <= pd.Timestamp.now())
        ]
        if not amavalid.empty:
            cohort.loc[amavalid["leadid"], "hasmeeting"] = True

    qualified_ids = ids_from_status_names(["Qualified"])
    won_ids = ids_from_status_names(["Won"])
    lost_ids = ids_from_status_names(["Lost"])
    negotiation_ids = ids_from_status_names(["Negotiation", "On Hold", "Awaiting Budget"])

    cohort["qualified"] = cohort["leadstatusid"].isin(qualified_ids)
    cohort["won"] = cohort["leadstatusid"].isin(won_ids)
    cohort["lost"] = cohort["leadstatusid"].isin(lost_ids)
    cohort["negotiation"] = cohort["leadstatusid"].isin(negotiation_ids)

    def stage_resolver(row):
        if row["won"]:
            return "Contract Signed"
        if row["lost"]:
            return "Lost"
        if row["negotiation"]:
            return "Negotiation"
        if row["hasmeeting"]:
            return "Meeting Scheduled"
        if row["qualified"]:
            return "Qualified"
        return "New"

    cohort["Stage"] = cohort.apply(stage_resolver, axis=1)
    cohort["Stage"] = pd.Categorical(cohort["Stage"], categories=stage_order, ordered=True)

    funnel = cohort.groupby("Stage").size().reindex(stage_order, fill_value=0).reset_index(name="Count")
    funnel.columns = ["Stage", "Count"]
    stage_counts = dict(zip(funnel["Stage"], funnel["Count"]))
    new_total = max(int(stage_counts.get("New", 0)), 1)

    fig = px.funnel(
        funnel,
        x="Count",
        y="Stage",
        category_orders={"Stage": stage_order},
        color_discrete_sequence=[EXEC_BLUE, EXEC_GREEN, EXEC_PRIMARY, "#FFA500", "#7CFC00", EXEC_DANGER],
        text="Count",
    )
    fig.update_traces(textposition="inside", textfont_color="black", textinfo="value+percent initial")
    fig.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    if countries is not None and "countryid" in leads.columns and "countryname_e" in countries.columns:
        bycountry = (
            leads.groupby("countryid", dropna=True)
            .size()
            .reset_index(name="Leads")
            .merge(
                countries[["countryid", "countryname_e"]].rename(columns={"countryname_e": "Country"}),
                on="countryid",
                how="left",
            )
        )
        total = float(bycountry["Leads"].sum())
        bycountry["Share"] = (bycountry["Leads"] / total * 100.0).round(1) if total > 0 else 0.0
        top5 = bycountry.sort_values(["Share", "Leads"], ascending=False).head(5)
        st.subheader("Top markets")
        st.dataframe(
            top5[["Country", "Leads", "Share"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Share": st.column_config.ProgressColumn(
                    "Share", format="%.1f%%", min_value=0, max_value=100
                )
            },
        )
    else:
        st.info("Country data unavailable to build Top markets.")

    # 2) Now call the helper AFTER it is defined
    render_funnel_and_markets(d)

# -----------------------------------------------------------------------------
# Lead Status page
# -----------------------------------------------------------------------------
def show_lead_status(d):
    leads = d.get("leads"); statuses = d.get("lead_statuses")
    calls  = d.get("calls"); meets = d.get("agent_meeting_assignment")

    if leads is None or len(leads)==0:
        st.info("No lead status data in the selected range."); 
        return

    # Map status ids -> names
    name_map = {}
    if statuses is not None and {"leadstatusid","statusname_e"}.issubset(statuses.columns):
        name_map = dict(zip(statuses["leadstatusid"].astype(int), statuses["statusname_e"].astype(str)))
    leads = leads.copy()
    leads["Status"] = leads["LeadStatusId"].map(name_map).fillna(leads.get("LeadStatusId", pd.Series(dtype="Int64")).astype(str))
    leads["CreatedOn"] = pd.to_datetime(leads.get("CreatedOn"), errors="coerce")
    cutoff = leads["CreatedOn"].max() if "CreatedOn" in leads.columns else pd.Timestamp.today()
    leads["age_days"] = (cutoff - leads["CreatedOn"]).dt.days.astype("Int64")

    # Summary donut and headline KPIs (kept)
    counts = leads["Status"].value_counts().reset_index()
    counts.columns = ["Status","count"]
    c1,c2 = st.columns([2,1])
    with c1:
        fig = px.pie(counts, names="Status", values="count", hole=0.35, color_discrete_sequence=px.colors.sequential.Viridis,
                     title="Lead Status Share")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Total Leads", f"{len(leads):,}")
        won_id = None
        if statuses is not None and "statusname_e" in statuses.columns:
            m = statuses.loc[statuses["statusname_e"].str.lower()=="won"]
            if not m.empty: won_id = int(m.iloc[0]["leadstatusid"])
        won = int((leads.get("LeadStatusId", pd.Series(dtype="Int64")).astype("Int64")==won_id).sum()) if won_id is not None else 0
        st.metric("Won", f"{won:,}")

    # -------------- Lead Distribution Status --------------
    st.markdown("---"); st.subheader("Lead Distribution Status")

    # Horizontal bar distribution
    dist_sorted = counts.sort_values("count", ascending=True)
    fig_bar = px.bar(dist_sorted, x="count", y="Status", orientation="h",
                     title="Leads by status (sorted)", color="Status", color_discrete_sequence=px.colors.qualitative.Dark24)
    fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=360,
                          showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Optional stacked trend by period (uses fdata period already set)
    if "period" not in leads.columns and "CreatedOn" in leads.columns:
        leads["period"] = leads["CreatedOn"].dt.to_period("M").apply(lambda p: p.start_time.date())
    if "period" in leads.columns:
        normalize = st.checkbox("Normalize to 100% per period", value=False, key="ls_norm")
        trend = leads.groupby(["period","Status"]).size().reset_index(name="count")
        if normalize:
            totals = trend.groupby("period")["count"].transform("sum").replace(0, np.nan)
            trend["count"] = (trend["count"]/totals*100).round(1)
            y_title = "Share %"
        else:
            y_title = "Count"
        fig_stack = px.bar(trend.sort_values("period"), x="period", y="count", color="Status", barmode="stack",
                           title=f"Status mix by period ({y_title})")
        fig_stack.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                                height=360, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_stack, use_container_width=True)

    # -------------- Detailed Lead Breakdown --------------
    st.markdown("---"); st.subheader("Detailed Lead Breakdown")

    # Meeting intent per status (Scheduled/Rescheduled)
    meet_rate = pd.DataFrame({"Status": pd.Series(dtype="str"), "meet_leads": pd.Series(dtype="float")})
    if meets is not None and len(meets):
        M = meets.copy(); M.columns = M.columns.str.lower()
        dtc = "startdatetime" if "startdatetime" in M.columns else None
        if dtc is not None:
            if "meetingstatusid" in M.columns:
                M = M[M["meetingstatusid"].isin({1,6})]  # Scheduled/Rescheduled
            mm = M.merge(leads[["LeadId","Status"]], left_on="leadid", right_on="LeadId", how="left")
            meet_rate = mm.groupby("Status")["leadid"].nunique().reset_index(name="meet_leads")

    # Connect rate per status
    conn_rate = pd.DataFrame({"Status": pd.Series(dtype="str"), "connect_rate": pd.Series(dtype="float")})
    if calls is not None and len(calls):
        C = calls.copy()
        C["CallDateTime"] = pd.to_datetime(C.get("CallDateTime"), errors="coerce")
        C = C.merge(leads[["LeadId","Status"]], on="LeadId", how="left")
        g = C.groupby("Status").agg(total=("LeadCallId","count"),
                                    connects=("CallStatusId", lambda s:(s==1).sum())).reset_index()
        g["connect_rate"] = (g["connects"]/g["total"]).fillna(0.0)
        conn_rate = g[["Status","connect_rate"]]

    # Aggregate breakdown
    base = leads.groupby("Status").agg(
        Leads=("LeadId","count"),
        Avg_Age_Days=("age_days","mean"),
        Pipeline=("EstimatedBudget","sum")
    ).reset_index()
    total_leads = float(base["Leads"].sum()) if len(base) else 0.0
    base["Share_%"] = (base["Leads"]/total_leads*100.0).round(1) if total_leads>0 else 0.0

    breakdown = (base.merge(meet_rate, on="Status", how="left")
                      .merge(conn_rate, on="Status", how="left"))
    breakdown["meet_leads"] = breakdown["meet_leads"].fillna(0.0)
    breakdown["Meeting_Rate_%"] = (breakdown["meet_leads"]/breakdown["Leads"]*100.0).replace([np.inf, -np.inf], 0).fillna(0.0).round(1)
    breakdown["connect_rate"] = breakdown["connect_rate"].fillna(0.0).round(2)
    breakdown["Avg_Age_Days"] = breakdown["Avg_Age_Days"].fillna(0.0).round(1)
    breakdown = breakdown.sort_values(["Leads","Pipeline"], ascending=False)

    st.dataframe(
        breakdown[["Status","Leads","Share_%","Avg_Age_Days","Meeting_Rate_%","connect_rate","Pipeline"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Leads": st.column_config.NumberColumn("Leads", format="%,d"),
            "Share_%": st.column_config.ProgressColumn("Share", min_value=0.0, max_value=100.0, format="%.1f%%"),
            "Avg_Age_Days": st.column_config.NumberColumn("Avg age (days)", format="%.1f"),
            "Meeting_Rate_%": st.column_config.ProgressColumn("Meeting rate", min_value=0.0, max_value=100.0, format="%.1f%%"),
            "connect_rate": st.column_config.ProgressColumn("Connect rate", min_value=0.0, max_value=1.0, format="%.2f"),
            "Pipeline": st.column_config.NumberColumn("Pipeline", format="%.0f"),
        }
    )

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if HAS_OPTION_MENU:
    if selected=="Executive":          show_executive_summary(fdata)
    elif selected=="Lead Status":       show_lead_status(fdata)
else:
    with tabs[0]: show_executive_summary(fdata)
    with tabs[1]: show_lead_status(fdata)
