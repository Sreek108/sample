# app.py — DAR Global CEO Dashboard (with AI/ML Insights + Geo AI)

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
# Secrets / driver visibility (diagnostics; optional to remove for prod)
# -----------------------------------------------------------------------------
with st.expander("Debug: connection config and drivers", expanded=False):
    cfg = st.secrets.get("connections", {}).get("sql", {})
    url_present = isinstance(cfg.get("url"), str) and len(cfg.get("url")) > 0
    st.write("connections.sql keys:", list(cfg.keys()))
    st.write("Using single-line url:", url_present)
    if url_present:
        st.write("url prefix:", cfg["url"][:80] + " ...")
    try:
        import pyodbc
        st.write("ODBC drivers:", pyodbc.drivers())
    except Exception as e:
        st.write("pyodbc import/driver list error:", e)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def format_number(v):
    if pd.isna(v): return "0"
    return f"{v/1e6:.1f}M" if v>=1e6 else (f"{v/1e3:.1f}K" if v>=1e3 else f"{v:,.0f}")

def ensure_funnel_df(df_like):
    """
    Returns a well-formed funnel DataFrame with columns ['Stage','Count']
    in the order: New, Qualified, Meeting Scheduled, Negotiation, Contract Signed, Lost.
    Missing stages are added with 0 to avoid KeyErrors downstream.
    """
    expected_order = ["New","Qualified","Meeting Scheduled","Negotiation","Contract Signed","Lost"]
    if isinstance(df_like, pd.DataFrame):
        f = df_like.copy()
    elif isinstance(df_like, dict):
        f = pd.DataFrame(df_like)
    elif isinstance(df_like, list):
        f = pd.DataFrame(df_like)
    else:
        f = pd.DataFrame(columns=["Stage","Count"])
    cols_lower = {c.lower(): c for c in f.columns}
    stage_col = cols_lower.get("stage")
    count_col = cols_lower.get("count")
    if stage_col is None:
        for alt in ["status","label","bucket","name","phase"]:
            if alt in cols_lower: stage_col = cols_lower[alt]; break
    if count_col is None:
        for alt in ["value","values","total","n","cnt","size"]:
            if alt in cols_lower: count_col = cols_lower[alt]; break
    if stage_col is None or count_col is None:
        clean = pd.DataFrame({"Stage": expected_order, "Count": [0]*len(expected_order)})
    else:
        tmp = f[[stage_col, count_col]].rename(columns={stage_col:"Stage", count_col:"Count"})
        tmp["Stage"] = tmp["Stage"].astype(str).str.strip()
        tmp["Count"] = pd.to_numeric(tmp["Count"], errors="coerce").fillna(0).astype(int)
        clean = (
            tmp.groupby("Stage", as_index=False)["Count"].sum()
               .set_index("Stage").reindex(expected_order, fill_value=0).reset_index()
        )
    return clean

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

    tbl_cfg = st.secrets.get("connections", {}).get("sql", {}).get("tables", {})
    def T(key, default): return tbl_cfg.get(key, default)

    def fetch(table_fqn, label=None, limit=None):
        label = label or table_fqn
        top = f"TOP {int(limit)} " if limit else ""
        try:
            return conn.query(f"SELECT {top}* FROM {table_fqn}", ttl=600)
        except Exception as e:
            st.error(f"Query failed for {label}: {e}")
            return None

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

    def norm_lower(df):
        if df is None: return None
        out = df.copy()
        out.columns = out.columns.str.strip().str.replace(r"[^\w]+","_",regex=True).str.lower()
        return out

    def rename(df, mapping):
        cols = {src: dst for src, dst in mapping.items() if src in df.columns}
        return df.rename(columns=cols)

    # Leads
    if ds["leads"] is not None:
        df = norm_lower(ds["leads"])
        df = rename(df, {
            "leadid":"LeadId","lead_id":"LeadId","leadcode":"LeadCode",
            "leadstageid":"LeadStageId","leadstatusid":"LeadStatusId","leadscoringid":"LeadScoringId",
            "assignedagentid":"AssignedAgentId","createdon":"CreatedOn","isactive":"IsActive",
            "countryid":"CountryId","cityregionid":"CityRegionId","estimatedbudget":"EstimatedBudget","budget":"EstimatedBudget"
        })
        for col, default in [("EstimatedBudget",0.0),("LeadStageId",pd.NA),("LeadStatusId",pd.NA),
                             ("AssignedAgentId",pd.NA),("CreatedOn",pd.NaT),("IsActive",1)]:
            if col not in df.columns: df[col]=default
        df["CreatedOn"] = pd.to_datetime(df.get("CreatedOn"), errors="coerce")
        df["EstimatedBudget"] = pd.to_numeric(df.get("EstimatedBudget"), errors="coerce").fillna(0.0)
        ds["leads"] = df

    # Agents
    if ds["agents"] is not None:
        df = norm_lower(ds["agents"])
        df = rename(df, {"agentid":"AgentId","firstname":"FirstName","first_name":"FirstName",
                         "lastname":"LastName","last_name":"LastName","isactive":"IsActive"})
        for c, d in [("FirstName",""),("LastName",""),("Role",""),("IsActive",1)]:
            if c not in df.columns: df[c]=d
        ds["agents"] = df

    # Calls
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

    # Schedules
    if ds["schedules"] is not None:
        df = norm_lower(ds["schedules"])
        df = rename(df, {"scheduleid":"ScheduleId","leadid":"LeadId","tasktypeid":"TaskTypeId",
                         "scheduleddate":"ScheduledDate","taskstatusid":"TaskStatusId",
                         "assignedagentid":"AssignedAgentId","completeddate":"CompletedDate","isfollowup":"IsFollowUp"})
        if "ScheduledDate" in df.columns: df["ScheduledDate"] = pd.to_datetime(df["ScheduledDate"], errors="coerce")
        if "CompletedDate" in df.columns: df["CompletedDate"] = pd.to_datetime(df["CompletedDate"], errors="coerce")
        ds["schedules"] = df

    # Transactions
    if ds["transactions"] is not None:
        df = norm_lower(ds["transactions"])
        df = rename(df, {"transactionid":"TransactionId","leadid":"LeadId","tasktypeid":"TaskTypeId","transactiondate":"TransactionDate"})
        if "TransactionDate" in df.columns: df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
        ds["transactions"] = df

    # Lookups
    for lk in ["countries","lead_stages","lead_statuses","lead_sources","lead_scoring",
               "call_statuses","sentiments","task_types","task_statuses","city_region",
               "timezone_info","priority","meeting_status","agent_meeting_assignment"]:
        if ds.get(lk) is not None: ds[lk] = norm_lower(ds[lk])

    return ds

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
  <h1> DAR Global — Executive Dashboard</h1>
  <h3>AI‑Powered Analytics</h3>
  <p style="margin:6px 0 0 0; color:{EXEC_GREEN};">Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
</div>
""", unsafe_allow_html=True)

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

# -----------------------------------------------------------------------------
# Executive Summary (KPIs + PX funnel)
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    leads=d.get("leads"); calls=d.get("calls"); lead_statuses=d.get("lead_statuses")
    schedules=d.get("schedules"); ama=d.get("agent_meeting_assignment")
    if leads is None or len(leads)==0:
        st.info("No data available in the selected range."); return

    won_status_id = 9
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        m = lead_statuses.loc[lead_statuses["statusname_e"].str.lower()=="won"]
        if not m.empty and "leadstatusid" in m.columns: won_status_id = int(m.iloc[0]["leadstatusid"])

    st.subheader("Performance KPIs")
    today = pd.Timestamp.today().normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)
    date_ranges = {"Week to Date": (week_start, today), "Month to Date": (month_start, today), "Year to Date": (year_start, today)}
    cols = st.columns(3)

    def _safe_ids(df, col):
        try:
            return set(df[col].dropna().astype(int).unique())
        except Exception:
            return set()

    for (label, (start, end)), col in zip(date_ranges.items(), cols):
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

        if ("CreatedOn" in leads.columns) and ("LeadId" in leads.columns):
            lp_mask = (pd.to_datetime(leads["CreatedOn"], errors="coerce") >= start_ts) & \
                      (pd.to_datetime(leads["CreatedOn"], errors="coerce") <= end_ts)
            leads_period = leads.loc[lp_mask].copy()
            created_ids = _safe_ids(leads_period, "LeadId")
        else:
            leads_period = pd.DataFrame()
            created_ids = set()

        call_ids = set()
        if (d.get("calls") is not None) and {"CallDateTime","LeadId"}.issubset(d["calls"].columns):
            cc = d["calls"].copy()
            c_mask = (pd.to_datetime(cc["CallDateTime"], errors="coerce") >= start_ts) & \
                     (pd.to_datetime(cc["CallDateTime"], errors="coerce") <= end_ts)
            call_ids = _safe_ids(cc.loc[c_mask], "LeadId")

        sched_ids = set()
        if (d.get("schedules") is not None) and {"ScheduledDate","LeadId"}.issubset(d["schedules"].columns):
            ss = d["schedules"].copy()
            s_mask = (pd.to_datetime(ss["ScheduledDate"], errors="coerce") >= start_ts) & \
                     (pd.to_datetime(ss["ScheduledDate"], errors="coerce") <= end_ts)
            sched_ids = _safe_ids(ss.loc[s_mask], "LeadId")
        elif d.get("agent_meeting_assignment") is not None:
            ama_df = d["agent_meeting_assignment"].copy()
            ama_df.columns = ama_df.columns.str.lower()
            if {"startdatetime","leadid"}.issubset(ama_df.columns):
                ama_df["_dt"] = pd.to_datetime(ama_df["startdatetime"], errors="coerce")
                m_mask = (ama_df["_dt"] >= start_ts) & (ama_df["_dt"] <= end_ts)
                if "meetingstatusid" in ama_df.columns:
                    ama_df = ama_df.loc[m_mask & ama_df["meetingstatusid"].isin({1,6})]
                else:
                    ama_df = ama_df.loc[m_mask]
                try:
                    sched_ids = set(ama_df["leadid"].dropna().astype(int).unique())
                except Exception:
                    sched_ids = set()

        active_ids = created_ids | call_ids | sched_ids
        total_leads = len(created_ids) if len(created_ids) > 0 else len(active_ids)
        meetings_cnt = len(sched_ids)

        if "LeadStatusId" in leads.columns and "LeadId" in leads.columns:
            base_ids = created_ids if len(created_ids) > 0 else active_ids
            if len(base_ids):
                base_df = leads.loc[leads["LeadId"].astype("Int64").isin(list(base_ids))].copy()
                won_leads = int((base_df["LeadStatusId"] == won_status_id).sum()) if "LeadStatusId" in base_df.columns else 0
                conversion_rate = (won_leads / len(base_ids) * 100.0) if len(base_ids) else 0.0
            else:
                conversion_rate = 0.0
        else:
            conversion_rate = 0.0

        with col:
            st.markdown(f"#### {label}")
            st.markdown("**Total Leads**"); st.markdown(f"<span style='font-size:2rem;'>{int(total_leads)}</span>", unsafe_allow_html=True)
            st.markdown("**Conversion Rate**"); st.markdown(f"<span style='font-size:2rem;'>{conversion_rate:.1f}%</span>", unsafe_allow_html=True)
            st.markdown("**Meetings Scheduled**"); st.markdown(f"<span style='font-size:2rem;'>{int(meetings_cnt)}</span>", unsafe_allow_html=True)

    # Funnel derived from IDs (guarded)
    cohort_ids = pd.Index(leads["LeadId"].dropna().astype(int).unique()) if {"LeadId"}.issubset(leads.columns) else pd.Index([])
    qualified_ids = cohort_ids  # placeholder if no explicit stage mapping; replace with your logic if needed
    meet_ids = pd.Index(sched_ids) if 'sched_ids' in locals() else pd.Index([])
    neg_ids = pd.Index([])  # placeholder
    signed_ids = pd.Index(leads.loc[leads.get("LeadStatusId", pd.Series()).eq(won_status_id), "LeadId"].dropna().astype(int).unique()) if "LeadStatusId" in leads.columns else pd.Index([])
    lost_ids = pd.Index([])

    funnel_df = pd.DataFrame({
        "Stage": ["New","Qualified","Meeting Scheduled","Negotiation","Contract Signed","Lost"],
        "Count": [int(len(cohort_ids)), int(len(qualified_ids)), int(len(meet_ids)), int(len(neg_ids)), int(len(signed_ids)), int(len(lost_ids))]
    })
    funnel_df = ensure_funnel_df(funnel_df)

    fig_funnel = px.funnel(funnel_df, x="Count", y="Stage",
                           color_discrete_sequence=["#1E90FF", "#32CD32", "#DAA520", "#FFA500", "#7CFC00", "#DC143C"],
                           title="Lead Conversion Funnel")
    fig_funnel.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig_funnel, use_container_width=True)

# -----------------------------------------------------------------------------
# Lead Status page (Donut)
# -----------------------------------------------------------------------------
def show_lead_status(d):
    leads = d.get("leads"); statuses = d.get("lead_statuses")
    schedules = d.get("schedules"); ama = d.get("agent_meeting_assignment")
    if leads is None or len(leads) == 0 or statuses is None or len(statuses) == 0:
        st.info("No lead/status data in the selected range."); return

    def pick_col(df, candidates):
        m = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in m: return m[c.lower()]
        return None

    def safe_ids(df, col):
        try:
            return set(pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique())
        except Exception:
            return set()

    s = statuses.copy(); id_col = pick_col(s, ["leadstatusid","lead_status_id","statusid","status_id"])
    name_col = pick_col(s, ["statusname_e","statusname","status_name_e","status_name","name"])
    if id_col is None or name_col is None or "LeadStatusId" not in leads.columns:
        st.warning("Required status fields missing."); return

    s_slim = s[[id_col, name_col]].rename(columns={id_col: "LeadStatusId", name_col: "StatusName"})
    df = leads.merge(s_slim, on="LeadStatusId", how="left")
    df["StatusName"] = df["StatusName"].astype(str).str.strip()
    all_ids = safe_ids(df, "LeadId")
    if not all_ids:
        st.info("No LeadId values present after normalization."); return

    s_lc = s.copy(); s_lc.columns = s_lc.columns.str.lower(); name_lc = name_col.lower()
    def status_ids_by_names(names):
        if name_lc not in s_lc.columns or "leadstatusid" not in s_lc.columns: return set()
        mask = s_lc[name_lc].astype(str).str.lower().isin([n.lower() for n in names])
        return set(pd.to_numeric(s_lc.loc[mask, "leadstatusid"], errors="coerce").dropna().astype(int).unique())

    won_sid  = status_ids_by_names(["won","closed won","contract signed","signed","sale","converted","booked","deal won"])
    lost_sid = status_ids_by_names(["lost","closed lost","dead","rejected","not interested","no response","no-show","cancelled"])

    interested_sid = set()
    if name_lc in s_lc.columns:
        interested_sid = set(
            pd.to_numeric(
                s_lc.loc[
                    s_lc[name_lc].astype(str).str.contains(
                        r"\binterested\b|\binquiry\b|\benquiry\b|^lead$|^prospect$", case=False, na=False
                    ),
                    "leadstatusid"
                ],
                errors="coerce",
            ).dropna().astype(int).unique()
        )

    new_sid = set()
    if name_lc in s_lc.columns:
        new_sid = set(
            pd.to_numeric(
                s_lc.loc[s_lc[name_lc].astype(str).str.contains(r"\bnew\b|^fresh|^unassigned", case=False, na=False), "leadstatusid"],
                errors="coerce",
            ).dropna().astype(int).unique()
        )

    meeting_ids = set()
    if schedules is not None and {"LeadId"}.issubset(schedules.columns):
        meeting_ids |= safe_ids(schedules, "LeadId")
    if ama is not None:
        m = ama.copy(); m.columns = m.columns.str.lower()
        if "leadid" in m.columns:
            if "meetingstatusid" in m.columns and m["meetingstatusid"].notna().any():
                m = m[m["meetingstatusid"].isin({1,6})]
            meeting_ids |= set(pd.to_numeric(m["leadid"], errors="coerce").dropna().astype(int).unique())
    meeting_status_sid = status_ids_by_names(["meeting scheduled","appointment","meeting fixed","visit scheduled","demo scheduled","site visit"])
    if meeting_status_sid:
        meeting_ids |= set(pd.to_numeric(df.loc[df["LeadStatusId"].isin(meeting_status_sid), "LeadId"], errors="coerce").dropna().astype(int).unique())

    by_status = {}
    for label, sid_set in [
        ("Contract Signed", won_sid),
        ("Lost",            lost_sid),
        ("Interested",      interested_sid),
        ("New",             new_sid),
    ]:
        if sid_set:
            ids = set(pd.to_numeric(df.loc[df["LeadStatusId"].isin(sid_set), "LeadId"], errors="coerce").dropna().astype(int).unique())
        else:
            ids = set()
        by_status[label] = ids

    bucket_by_id = {}
    for lid in all_ids:
        if lid in by_status["Contract Signed"]:
            bucket_by_id[lid] = "Contract Signed"
        elif lid in by_status["Lost"]:
            bucket_by_id[lid] = "Lost"
        elif lid in meeting_ids:
            bucket_by_id[lid] = "Meeting Scheduled"
        elif lid in by_status["Interested"]:
            bucket_by_id[lid] = "Interested"
        elif lid in by_status["New"]:
            bucket_by_id[lid] = "New"
        else:
            bucket_by_id[lid] = "Interested"

    order = ["New", "Interested", "Meeting Scheduled", "Contract Signed", "Lost"]
    vc = pd.Series(list(bucket_by_id.values())).value_counts()
    counts = pd.DataFrame({"Bucket": order, "Count": [int(vc.get(k, 0)) for k in order]})

    total_leads = int(len(all_ids))
    signed_count = int(vc.get("Contract Signed", 0))
    lost_count   = int(vc.get("Lost", 0))
    active_leads = int(total_leads - signed_count - lost_count)
    win_rate = (signed_count / (signed_count + lost_count) * 100.0) if (signed_count + lost_count) > 0 else 0.0
    conversion_rate = (signed_count / total_leads * 100.0) if total_leads > 0 else 0.0

    left, right = st.columns([2, 1], gap="large")
    with left:
        fig = px.pie(
            counts,
            names="Bucket",
            values="Count",
            hole=0.55,
            category_orders={"Bucket": order},
            color="Bucket",
            color_discrete_map={
                "New": "#1E90FF",
                "Interested": "#43A047",
                "Meeting Scheduled": "#DAA520",
                "Contract Signed": "#7CFC00",
                "Lost": "#DC143C",
            },
            title="Lead Distribution by Status",
        )
        fig.update_traces(textposition="inside", textinfo="label+percent")
        fig.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Lead Metrics")
        st.metric("Total Leads", f"{total_leads:,}")
        st.metric("Active Leads", f"{active_leads:,}")
        st.metric("Win Rate", f"{win_rate:.1f}%")
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
# -----------------------------------------------------------------------------
# Calls page (basic)
# -----------------------------------------------------------------------------
def show_calls(d):
    st.subheader("AI Call Activity")

    calls = d.get("calls")
    if calls is None or len(calls)==0:
        st.info("No call data in the selected range.")
        return

    # Normalize types
    C = calls.copy()
    if "CallDateTime" in C.columns:
        C["CallDateTime"] = pd.to_datetime(C["CallDateTime"], errors="coerce")

    # ---------------- Daily calls and success rate ----------------
    if {"CallDateTime","LeadCallId","CallStatusId"}.issubset(C.columns):
        daily = C.groupby(C["CallDateTime"].dt.date).agg(
            Total=("LeadCallId","count"),
            Connected=("CallStatusId", lambda x: (x==1).sum())
        ).reset_index().rename(columns={"CallDateTime":"Date"})
        daily["SuccessRate"] = (daily["Connected"]/daily["Total"]*100).round(1)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Total"], mode="lines+markers",
                                     line=dict(color="#1E90FF", width=3)))
            fig.update_layout(title="Daily Calls",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["SuccessRate"], mode="lines+markers",
                                     line=dict(color="#32CD32", width=3)))
            fig.update_layout(title="Success Rate (%)",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Missing fields to plot daily calls or success rate (need CallDateTime, LeadCallId, CallStatusId).")

    # ---------------- Call Status Distribution (donut + trimmed KPIs) ----------------
    st.markdown("---"); st.subheader("Call Status Distribution")

    cs = d.get("call_statuses")
    name_map = {}
    if cs is not None and {"callstatusid","statusname_e"}.issubset(cs.columns):
        name_map = dict(zip(cs["callstatusid"].astype(int), cs["statusname_e"].astype(str)))  # labels from master [file:345]

    if "CallStatusId" in C.columns:
        dist = C.copy()
        dist["Status"] = dist["CallStatusId"].map(name_map).fillna(dist["CallStatusId"].astype(str))
        donut = dist["Status"].value_counts().reset_index()
        donut.columns = ["Status","count"]

        # Bigger donut
        fig = px.pie(
            donut,
            names="Status",
            values="count",
            hole=0.35,
            color_discrete_sequence=px.colors.sequential.RdPu,
            title="Outcomes"
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=60, b=80),   # <- larger bottom margin creates space
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
                font=dict(size=14),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white"
        ) 
        st.plotly_chart(fig, use_container_width=True)

        # Spacer between chart and KPI row
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)# uses full width for an even bigger feel [file:324]

        # KPIs in one row
        total_calls = int(len(C))
        connected_calls = int((C["CallStatusId"]==1).sum())
        connect_rate = (connected_calls/total_calls*100.0) if total_calls else 0.0
        avg_duration = float(pd.to_numeric(C.get("DurationSeconds", pd.Series(dtype=float)), errors="coerce").dropna().mean()) if "DurationSeconds" in C.columns else 0.0  # [file:324]

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.metric("Total calls", f"{total_calls:,}")                                    # [file:324]
        with k2: st.metric("Connected calls", f"{connected_calls:,}")                            # [file:324]
        with k3: st.metric("Connect rate", f"{connect_rate:.1f}%")                               # [file:324]
        with k4: st.metric("Avg duration (sec)", f"{avg_duration:.1f}")                          # [file:324]
    else:
        st.info("CallStatusId not available to render distribution and KPIs.") 
    # ---------------- Effectiveness by attempt number (kept) ----------------
    st.markdown("---"); st.subheader("Effectiveness by attempt number")
    if {"LeadId","CallDateTime","CallStatusId","LeadCallId"}.issubset(C.columns):
        A = C.copy().sort_values(["LeadId","CallDateTime"])
        A["attempt_no"] = A.groupby("LeadId").cumcount() + 1
        curve = A.groupby("attempt_no").agg(
            total=("LeadCallId","count"),
            connects=("CallStatusId", lambda s: (s==1).sum())
        ).reset_index()
        curve["connect_rate"] = (curve["connects"]/curve["total"]).fillna(0.0).round(3)
        fig = px.line(curve, x="attempt_no", y="connect_rate", markers=True,
                      title="Connect rate by attempt number")
        fig.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", xaxis_title="Attempt #", yaxis_title="Connect rate")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(curve, use_container_width=True, hide_index=True)
    else:
        st.info("Insufficient fields to analyze attempts (need LeadId, CallDateTime, CallStatusId, LeadCallId).")


def show_conversions(d):
    st.subheader("Conversion — Wins vs Dropped (no revenue/pipeline)")

    leads = d.get("leads")
    statuses = d.get("lead_statuses")
    meets = d.get("agent_meeting_assignment")

    if leads is None or len(leads) == 0:
        st.info("No data available in the selected range.")
        return

    # ---------- Resolve status ids (robust) ----------
    won_id, lost_id = 9, 10
    if statuses is not None and {"statusname_e", "leadstatusid"}.issubset(statuses.columns):
        s = statuses.copy()
        s["statusname_e_norm"] = s["statusname_e"].astype(str).str.strip().str.lower()
        s["leadstatusid"] = pd.to_numeric(s["leadstatusid"], errors="coerce").astype("Int64")
        w = s.loc[s["statusname_e_norm"].eq("won"), "leadstatusid"].dropna()
        l = s.loc[s["statusname_e_norm"].eq("lost"), "leadstatusid"].dropna()
        if not w.empty: won_id = int(w.iloc[0])
        if not l.empty: lost_id = int(l.iloc[0])

    # ---------- Prepare leads with period ----------
    L = leads.copy()
    L["LeadId"] = pd.to_numeric(L.get("LeadId"), errors="coerce").astype("Int64")
    L["LeadStatusId"] = pd.to_numeric(L.get("LeadStatusId"), errors="coerce").astype("Int64")
    if "period" not in L.columns:
        dt = pd.to_datetime(L.get("CreatedOn"), errors="coerce")
        L["period"] = dt.dt.to_period("M").apply(lambda p: p.start_time.date())

    # ---------- Monthly conversions vs dropped ----------
    per_total = L.groupby("period").size().rename("total")
    per_won   = L.loc[L["LeadStatusId"].eq(won_id)].groupby("period").size().rename("won")
    per_lost  = L.loc[L["LeadStatusId"].eq(lost_id)].groupby("period").size().rename("lost")
    conv = pd.concat([per_total, per_won, per_lost], axis=1).fillna(0.0).reset_index()
    conv["conv_rate"] = (conv["won"]/conv["total"]*100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            conv.sort_values("period"),
            x="period", y=["won","lost"],
            barmode="group",
            color_discrete_sequence=["#32CD32","#DC143C"],
            title="Monthly conversions vs dropped"
        )
        fig.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.line(
            conv.sort_values("period"),
            x="period", y="conv_rate", markers=True,
            title="Conversion rate trend (%)"
        )
        fig.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ---------- KPIs (no revenue/pipeline) ----------
    total = int(L["LeadId"].nunique()) if "LeadId" in L.columns else int(len(L))
    wins = int(L["LeadStatusId"].eq(won_id).sum())
    losses = int(L["LeadStatusId"].eq(lost_id).sum())
    conv_rate_overall = (wins/total*100.0) if total else 0.0
    drop_rate_overall = (losses/total*100.0) if total else 0.0

    # YTD conversions (based on CreatedOn)
    ytd_wins = 0
    if "CreatedOn" in L.columns:
        dt = pd.to_datetime(L["CreatedOn"], errors="coerce")
        today = pd.Timestamp.today()
        ystart = pd.Timestamp(year=today.year, month=1, day=1)
        ytd_wins = int(L.loc[(dt>=ystart) & L["LeadStatusId"].eq(won_id)].shape[0])

    k1,k2,k3 = st.columns(3)
    with k1: st.metric("YTD conversions", f"{ytd_wins:,}")
    with k2: st.metric("Conversion rate", f"{conv_rate_overall:.1f}%")
    with k3: st.metric("Drop rate", f"{drop_rate_overall:.1f}%")

    # ---------- Conversion Funnel (counts only) ----------
    st.markdown("---"); st.subheader("Conversion Funnel (counts)")

    # Qualified: all statuses in stage 2; Negotiation: On Hold/Awaiting Budget
    qualified_ids, nego_ids = set(), set()
    if statuses is not None:
        s = statuses.copy()
        s["leadstageid"]  = pd.to_numeric(s.get("leadstageid"), errors="coerce").astype("Int64")
        s["leadstatusid"] = pd.to_numeric(s.get("leadstatusid"), errors="coerce").astype("Int64")
        s["statusname_e_norm"] = s.get("statusname_e", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
        qualified_ids = set(s.loc[s["leadstageid"].eq(2), "leadstatusid"].dropna().astype(int).tolist())
        nego_mask = s["statusname_e_norm"].isin(["on hold","awaiting budget"])
        nego_ids = set(s.loc[nego_mask, "leadstatusid"].dropna().astype(int).tolist())

    # Initialize all stage counters to ensure no NameError
    new_count = int(L["LeadId"].nunique())
    qualified_count = int(L.loc[L["LeadStatusId"].isin(qualified_ids), "LeadId"].nunique()) if qualified_ids else 0

    meet_leads = set()
    meeting_count = 0
    if meets is not None and len(meets):
        M = meets.copy(); M.columns = M.columns.str.lower()
        if "startdatetime" in M.columns:
            if "meetingstatusid" in M.columns:
                M = M[M["meetingstatusid"].isin({1,6})]
            meet_leads = set(pd.to_numeric(M["leadid"], errors="coerce").dropna().astype(int).tolist())
            meeting_count = int(len(meet_leads))

    nego_count = int(L.loc[L["LeadStatusId"].isin(nego_ids) & L["LeadId"].isin(meet_leads), "LeadId"].nunique()) if nego_ids else 0
    won_count  = wins
    lost_count = losses

    funnel_df = pd.DataFrame({
        "Stage":["New","Qualified","Meeting Scheduled","Negotiation","Won","Lost"],
        "Count":[new_count, qualified_count, meeting_count, nego_count, won_count, lost_count]
    })

    # Optionally filter out zero-count stages to avoid awkward shapes
    show_zero_stages = False
    if not show_zero_stages:
        funnel_df = funnel_df[funnel_df["Count"]>0]

    fig = px.funnel(
        funnel_df, x="Count", y="Stage",
        color_discrete_sequence=["#1E90FF","#32CD32","#DAA520","#FFA500","#7CFC00","#DC143C"],
        title="Conversion Funnel (counts)"
    )
    fig.update_traces(textposition="inside", textinfo="value+percent initial")
    fig.update_layout(
        height=420,
        margin=dict(l=30, r=30, t=70, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Geo AI page (performance + AI recommendations)
# -----------------------------------------------------------------------------
def show_geo_ai(d):
    st.subheader("Geo AI — Market performance and opportunity")

    leads = d.get("leads")
    countries = d.get("countries")
    statuses = d.get("lead_statuses")
    calls = d.get("calls")
    meets = d.get("agent_meeting_assignment")

    if leads is None or len(leads)==0 or countries is None:
        st.info("No geo data available in the selected range.")
        return

    # Resolve Won id from master
    won_id = 9
    if statuses is not None and "statusname_e" in statuses.columns:
        m = statuses.loc[statuses["statusname_e"].str.lower()=="won"]
        if not m.empty and "leadstatusid" in m.columns:
            won_id = int(m.iloc[0]["leadstatusid"])

    L = leads.copy()
    L["CountryId"] = pd.to_numeric(L.get("CountryId"), errors="coerce").astype("Int64")
    L["LeadStatusId"] = pd.to_numeric(L.get("LeadStatusId"), errors="coerce").astype("Int64")
    L["EstimatedBudget"] = pd.to_numeric(L.get("EstimatedBudget"), errors="coerce").fillna(0.0)
    L["CreatedOn"] = pd.to_datetime(L.get("CreatedOn"), errors="coerce")

    # ---------- Section 1: Current market performance (table + map) ----------
    # Keep a base with CountryId for downstream AI merges
    perf_base = L.groupby("CountryId").agg(
        Leads=("LeadId","count"),
        Pipeline=("EstimatedBudget","sum")
    ).reset_index()
    won = L.loc[L["LeadStatusId"]==won_id].groupby("CountryId").size().reset_index(name="Won")
    perf_base = perf_base.merge(won, on="CountryId", how="left").fillna({"Won":0})
    total_pipe = float(perf_base["Pipeline"].sum())
    perf_base["Share"] = (perf_base["Pipeline"]/total_pipe*100.0).round(1) if total_pipe>0 else 0.0

    CTRY = countries.rename(columns={"countryid":"CountryId","countryname_e":"Country"})
    perf_view = perf_base.merge(CTRY[["CountryId","Country"]], on="CountryId", how="left") \
                         .sort_values(["Pipeline","Won","Leads"], ascending=False)[["Country","Leads","Won","Pipeline","Share"]]

    st.markdown("#### Current market performance")
    st.dataframe(
        perf_view, use_container_width=True, hide_index=True,
        column_config={
            "Pipeline": st.column_config.NumberColumn("Pipeline", format="%.0f"),
            "Share": st.column_config.ProgressColumn("Share", min_value=0.0, max_value=100.0, format="%.1f%%")
        }
    )

    # Performance map (choose a metric to color)
    st.markdown("##### Performance map")
    perf_metric = st.selectbox(
        "Color by",
        ["Pipeline","Leads","Won","Share"],
        index=0,
        key="geo_perf_metric"
    )

    perf_map = perf_view.copy()
    perf_map[perf_metric] = pd.to_numeric(perf_map[perf_metric], errors="coerce")

    try:
        if perf_metric == "Share":
            rng = [0, 100]
            fig_perf = px.choropleth(
                perf_map,
                locations="Country",
                locationmode="country names",
                color=perf_metric,
                hover_name="Country",
                hover_data={"Leads":":,", "Won":":,", "Pipeline":":,", "Share":".1f"},
                range_color=rng,
                color_continuous_scale="Reds",
                title=f"Current market performance — {perf_metric}"
            )
        else:
            fig_perf = px.choropleth(
                perf_map,
                locations="Country",
                locationmode="country names",
                color=perf_metric,
                hover_name="Country",
                hover_data={"Leads":":,", "Won":":,", "Pipeline":":,", "Share":".1f"},
                color_continuous_scale="Reds",
                title=f"Current market performance — {perf_metric}"
            )

        fig_perf.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=0,r=0,t=30,b=0)
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    except Exception:
        st.info("Map rendering skipped (requires valid country names).")

    # ---------- Section 2: Geo AI recommendations ----------
    st.markdown("---")
    st.markdown("#### Country opportunity (AI)")

    # Meeting intent (Scheduled/Rescheduled) — initialize with key to avoid KeyError
    meet_rate = pd.DataFrame({"CountryId": pd.Series(dtype="Int64"), "meet_rate": pd.Series(dtype="float")})
    if meets is not None and len(meets):
        M = meets.copy(); M.columns = M.columns.str.lower()
        dtc = "startdatetime" if "startdatetime" in M.columns else None
        if dtc is not None:
            if "meetingstatusid" in M.columns: M = M[M["meetingstatusid"].isin({1,6})]
            mm = M.merge(L[["LeadId","CountryId"]], left_on="leadid", right_on="LeadId", how="left")
            mr = mm.groupby("CountryId")["leadid"].nunique().reset_index(name="meet_leads")
            meet_rate = perf_base[["CountryId","Leads"]].merge(mr, on="CountryId", how="left").fillna({"meet_leads":0})
            meet_rate["meet_rate"] = (meet_rate["meet_leads"]/meet_rate["Leads"]).fillna(0.0)
            meet_rate = meet_rate[["CountryId","meet_rate"]]

    # Connect efficiency — initialize with key
    conn = pd.DataFrame({"CountryId": pd.Series(dtype="Int64"), "connect_rate": pd.Series(dtype="float")})
    if calls is not None and len(calls):
        C = calls.copy()
        C["CallDateTime"] = pd.to_datetime(C.get("CallDateTime"), errors="coerce")
        C = C.merge(L[["LeadId","CountryId"]], on="LeadId", how="left")
        g = C.groupby("CountryId").agg(
            total=("LeadCallId","count"),
            connects=("CallStatusId", lambda s:(s==1).sum())
        ).reset_index()
        g["connect_rate"] = (g["connects"]/g["total"]).fillna(0.0)
        conn = g[["CountryId","connect_rate"]]  # list, not set

        # ensure correct cols if set was used mistakenly
        if isinstance(conn, set) or not isinstance(conn, pd.DataFrame):
            conn = g[["CountryId","connect_rate"]]

    # Momentum (last 4 weeks vs prior 4) — initialize with key
    mom = pd.DataFrame({"CountryId": pd.Series(dtype="Int64"), "momentum": pd.Series(dtype="float")})
    if "CreatedOn" in L.columns and L["CreatedOn"].notna().any():
        W = L.copy()
        W["week"] = W["CreatedOn"].dt.to_period("W").apply(lambda p: p.start_time.date())
        wk = sorted(W["week"].dropna().unique())
        if len(wk)>=8:
            last4 = set(wk[-4:]); prev4 = set(wk[-8:-4])
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

    # Combine features on CountryId (safe merges even when frames are empty)
    df = perf_base.merge(meet_rate, on="CountryId", how="left") \
                  .merge(conn, on="CountryId", how="left") \
                  .merge(mom, on="CountryId", how="left") \
                  .fillna({"meet_rate":0.0,"connect_rate":0.0,"momentum":0.0})
    df["win_rate"] = (df["Won"]/df["Leads"]).fillna(0.0)

    # Normalize + score
    def mm(s):
        s = s.astype(float)
        lo, hi = s.min(), s.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi==lo:
            return pd.Series(0.0, index=s.index)
        return (s-lo)/(hi-lo)

    total_pipe = float(df["Pipeline"].sum())
    df["pipeline_share"] = df["Pipeline"]/total_pipe if total_pipe>0 else 0.0

    w = {"win_rate":0.35,"pipeline_share":0.30,"momentum":0.20,"connect_rate":0.10,"meet_rate":0.05}
    df["opportunity_score"] = (
        w["win_rate"]*mm(df["win_rate"]) +
        w["pipeline_share"]*mm(df["pipeline_share"]) +
        w["momentum"]*mm(df["momentum"]) +
        w["connect_rate"]*mm(df["connect_rate"]) +
        w["meet_rate"]*mm(df["meet_rate"])
    ).round(3)

    q75, q50 = df["opportunity_score"].quantile(0.75), df["opportunity_score"].quantile(0.50)
    def reco(r):
        if r["opportunity_score"]>=q75 and r["win_rate"]>=df["win_rate"].median(): return "Invest"
        if r["opportunity_score"]>=q50: return "Protect"
        if r["momentum"]>0: return "Explore"
        return "Deprioritize"
    df["recommendation"] = df.apply(reco, axis=1)

    def action(r):
        if r["recommendation"]=="Invest": return "Add senior closer capacity; accelerate meetings; allocate budget"
        if r["recommendation"]=="Protect": return "Maintain capacity; tighten SLA; defend share"
        if r["recommendation"]=="Explore": return "Low‑cost tests, partner outreach, targeted campaigns"
        return "Reduce spend; nurture via AI agent only"
    df["action"] = df.apply(action, axis=1)

    view = df.merge(CTRY[["CountryId","Country"]], on="CountryId", how="left") \
             .sort_values(["opportunity_score","Pipeline","Won"], ascending=False)

    st.dataframe(
        view[["Country","Leads","Won","win_rate","meet_rate","connect_rate","Pipeline","opportunity_score","recommendation","action"]],
        use_container_width=True, hide_index=True,
        column_config={
            "win_rate": st.column_config.NumberColumn("Win rate", format="%.1f"),
            "meet_rate": st.column_config.NumberColumn("Meet rate", format="%.1f"),
            "connect_rate": st.column_config.NumberColumn("Connect rate", format="%.1f"),
            "Pipeline": st.column_config.NumberColumn("Pipeline", format="%.0f"),
            "opportunity_score": st.column_config.ProgressColumn("Opportunity", min_value=0.0, max_value=1.0, format="%.3f"),
        }
    )

    # Opportunity map
    try:
        fig = px.choropleth(
            view, locations="Country", locationmode="country names",
            color="opportunity_score", hover_name="Country",
            color_continuous_scale="YlGnBu", title="Country opportunity map"
        )
        fig.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                          margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Map rendering skipped (requires valid country names).")

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if HAS_OPTION_MENU:
    if selected=="Executive":          show_executive_summary(fdata)
    elif selected=="Lead Status":       show_lead_status(fdata)
    elif selected=="AI Calls":          show_calls(fdata)
    elif selected=="AI Insights":       show_ai_insights(fdata)
    elif selected=="Conversion":        show_conversions(fdata)   # Conversion page
    elif selected=="Geo AI":            show_geo_ai(fdata)
else:
    with tabs[0]: show_executive_summary(fdata)
    with tabs[1]: show_lead_status(fdata)
    with tabs[2]: show_calls(fdata)
    with tabs[3]: show_ai_insights(fdata)
    with tabs[4]: show_conversions(fdata)  # Conversion page
    with tabs[5]: show_geo_ai(fdata)
