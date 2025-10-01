# app.py — DAR Global CEO Dashboard (SQL Server via st.connection, PX funnel integrated and fixed)

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
def format_currency(v):
    if pd.isna(v): return "$0"
    return f"${v/1e9:.1f}B" if v>=1e9 else (f"${v/1e6:.1f}M" if v>=1e6 else f"${v:,.0f}")

def format_number(v):
    if pd.isna(v): return "0"
    return f"{v/1e6:.1f}M" if v>=1e6 else (f"{v/1e3:.1f}K" if v>=1e3 else f"{v:,.0f}")

# -----------------------------------------------------------------------------
# Database loader (table names configurable via Secrets; errors surfaced)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(_unused: str = "data"):
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

    def norm(df):
        if df is None: return None
        out = df.copy()
        out.columns = out.columns.str.strip().str.replace(r"[^\w]+","_", regex=True)
        return out

    def coerce_dt(s): return pd.to_datetime(s, errors="coerce")

    if ds["leads"] is not None:
        df = norm(ds["leads"])
        for c in ["CreatedOn","CreatedDate","CreateDate","CreatedAt","created_on","createddate","created"]:
            if c in df.columns:
                if c != "CreatedOn": df["CreatedOn"] = df[c]
                break
        if "EstimatedBudget" in df.columns:
            df["EstimatedBudget"] = pd.to_numeric(df["EstimatedBudget"], errors="coerce").fillna(0.0)
        if "CreatedOn" in df.columns:
            df["CreatedOn"] = coerce_dt(df["CreatedOn"])
        ds["leads"] = df

    if ds["calls"] is not None:
        df = norm(ds["calls"])
        for c in ["CallDateTime","CallDatetime","CallTime","CallDate","created_at","createdon"]:
            if c in df.columns:
                if c != "CallDateTime": df["CallDateTime"] = df[c]
                break
        if "CallDateTime" in df.columns:
            df["CallDateTime"] = coerce_dt(df["CallDateTime"])
        ds["calls"] = df

    if ds["schedules"] is not None:
        df = norm(ds["schedules"])
        for c in ["ScheduledDate","ScheduleDate","StartDate","StartDateTime","DueDate"]:
            if c in df.columns:
                if c != "ScheduledDate": df["ScheduledDate"] = df[c]
                break
        if "ScheduledDate" in df.columns: df["ScheduledDate"] = coerce_dt(df["ScheduledDate"])
        if "CompletedDate" in df.columns: df["CompletedDate"] = coerce_dt(df["CompletedDate"])
        ds["schedules"] = df

    if ds["transactions"] is not None:
        df = norm(ds["transactions"])
        for c in ["TransactionDate","TxnDate","CreatedOn","CreatedDate","Date"]:
            if c in df.columns:
                if c != "TransactionDate": df["TransactionDate"] = df[c]
                break
        if "TransactionDate" in df.columns: df["TransactionDate"] = coerce_dt(df["TransactionDate"])
        ds["transactions"] = df

    for lk in ["countries","lead_stages","lead_statuses","lead_sources","lead_scoring",
               "call_statuses","sentiments","task_types","task_statuses","city_region",
               "timezone_info","priority","meeting_status","agent_meeting_assignment"]:
        if ds.get(lk) is not None:
            ds[lk] = norm(ds[lk])

    return ds

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
  <h1> DAR Global — Executive Dashboard</h1>
  <h3>AI‑Powered Analytics</h3>
  <p style="margin: 6px 0 0 0; color: {EXEC_GREEN};">Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
</div>
""", unsafe_allow_html=True)

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
# Executive Summary (Performance KPIs + PX funnel)
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
    date_ranges = {"Week to Date": (week_start, today), "Month to Date": (month_start, today), "Year to Date": (year_start, today))
    cols = st.columns(3)
    for (label, (start, end)), col in zip(date_ranges.items(), cols):
        leads_period = leads.loc[(pd.to_datetime(leads["CreatedOn"], errors="coerce") >= pd.Timestamp(start)) & (pd.to_datetime(leads["CreatedOn"], errors="coerce") <= pd.Timestamp(end))] if "CreatedOn" in leads.columns else pd.DataFrame()
        meetings_cnt = 0
        if schedules is not None and "ScheduledDate" in schedules.columns:
            s = schedules.copy(); s["_dt"] = pd.to_datetime(s["ScheduledDate"], errors="coerce"); s = s[(s["_dt"]>=pd.Timestamp(start)) & (s["_dt"]<=pd.Timestamp(end))]
            meetings_cnt = int(s["LeadId"].nunique()) if "LeadId" in s.columns else len(s)
        elif ama is not None:
            m = ama.copy(); m.columns = m.columns.str.lower()
            if "startdatetime" in m.columns:
                m["_dt"] = pd.to_datetime(m["startdatetime"], errors="coerce"); m = m[(m["_dt"]>=pd.Timestamp(start)) & (m["_dt"]<=pd.Timestamp(end))]
                if "meetingstatusid" in m.columns: m = m[m["meetingstatusid"].isin({1,6})]
                meetings_cnt = int(m["leadid"].nunique()) if "leadid" in m.columns else len(m)
        total_leads = int(len(leads_period))
        won_leads = int((leads_period["LeadStatusId"]==won_status_id).sum()) if "LeadStatusId" in leads_period.columns else 0
        conversion_rate = (won_leads/total_leads*100.0) if total_leads else 0.0
        with col:
            st.markdown(f"#### {label}")
            st.markdown("**Total Leads**"); st.markdown(f"<span style='font-size:2rem;'>{total_leads}</span>", unsafe_allow_html=True)
            st.markdown("**Conversion Rate**"); st.markdown(f"<span style='font-size:2rem;'>{conversion_rate:.1f}%</span>", unsafe_allow_html=True)
            st.markdown("**Meetings Scheduled**"); st.markdown(f"<span style='font-size:2rem;'>{meetings_cnt}</span>", unsafe_allow_html=True)

    st.markdown("---"); st.subheader("Trend at a glance")
    trend_style = st.radio("Trend style", ["Line","Bars","Bullet"], index=0, horizontal=True, key="__trend_style_exec")

    dt_leads = pd.to_datetime(leads.get("CreatedOn"), errors="coerce")
    leads2 = leads.copy(); leads2["period"]=dt_leads.dt.to_period("M").apply(lambda p: p.start_time.date())
    won_mask_all = leads2["LeadStatusId"].eq(won_status_id) if "LeadStatusId" in leads2.columns else pd.Series(False, index=leads2.index)

    leads_ts = leads2.groupby("period").size().reset_index(name="value")
    pipeline_ts = leads2.groupby("period")["EstimatedBudget"].sum().reset_index(name="value") if "EstimatedBudget" in leads2.columns else pd.DataFrame({"period":[], "value":[]})
    rev_ts = leads2.loc[won_mask_all].groupby("period")["EstimatedBudget"].sum().reset_index(name="value") if "EstimatedBudget" in leads2.columns else pd.DataFrame({"period":[], "value":[]})

    if calls is not None and len(calls)>0 and "CallDateTime" in calls.columns:
        c=calls.copy(); c["period"]=pd.to_datetime(c["CallDateTime"], errors="coerce").dt.to_period("W").apply(lambda p: p.start_time.date())
        calls_ts=c.groupby("period").agg(total=("LeadCallId","count"), connected=("CallStatusId", lambda x: (x==1).sum())).reset_index()
        calls_ts["value"]=(calls_ts["connected"]/calls_ts["total"]*100).round(1)
    else:
        calls_ts=pd.DataFrame({"period":[], "value":[]})

    def _index(df):
        df=df.copy()
        if df.empty: df["idx"]=[]; return df
        base = df["value"].iloc[0] if df["value"].iloc[0]!=0 else 1.0
        df["idx"]=(df["value"]/base)*100.0
        return df

    leads_ts=_index(leads_ts); pipeline_ts=_index(pipeline_ts); rev_ts=_index(rev_ts); calls_ts=_index(calls_ts)

    def _apply_axes(fig, ys, title):
        ymin=float(pd.Series(ys).min()) if len(ys) else 0
        ymax=float(pd.Series(ys).max()) if len(ys) else 1
        pad=max(1.0,(ymax-ymin)*0.12); rng=[ymin-pad, ymax+pad]
        fig.update_layout(height=180, title=dict(text=title, x=0.01, font=dict(size=12, color="#cfcfcf")),
                          margin=dict(l=6,r=6,t=24,b=8), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=4, ticks="outside")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=3, ticks="outside", range=rng)
        return fig

    def tile_line(df,color,title):
        df=df.dropna().sort_values("period"); fig=go.Figure()
        fig.add_trace(go.Scatter(x=df["period"], y=df["idx"], mode="lines+markers", line=dict(color=color, width=3, shape="spline"), marker=dict(size=5,color=color)))
        return _apply_axes(fig, df["idx"], title)

    def tile_bar(df,color,title):
        df=df.dropna().sort_values("period"); fig=go.Figure()
        fig.add_trace(go.Bar(x=df["period"], y=df["idx"], marker=dict(color=color, line=dict(color="rgba(255,255,255,0.15)", width=0.5)), opacity=0.9))
        return _apply_axes(fig, df["idx"], title)

    def tile_bullet(df,title,bar_color):
        if df.empty:
            fig=go.Figure()
            return _apply_axes(fig, [0,1], title)
        cur=float(df["idx"].iloc[-1])
        fig=go.Figure(go.Indicator(mode="number+gauge+delta", value=cur, number={'valueformat':".0f"}, delta={'reference':100},
                                   gauge={'shape':"bullet",'axis':{'range':[80,120]},
                                          'steps':[{'range':[80,95],'color':"rgba(220,20,60,0.35)"},{'range':[95,105],'color':"rgba(255,215,0,0.35)"},
                                                   {'range':[105,120],'color':"rgba(50,205,50,0.35)"}],
                                          'bar':{'color':bar_color},'threshold':{'line':{'color':'#fff','width':2},'value':100}}))
        fig.update_layout(height=120, margin=dict(l=8,r=8,t=26,b=8), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        return fig

    s1,s2,s3,s4 = st.columns(4)
    if trend_style=="Line":
        with s1: st.plotly_chart(tile_line(leads_ts,EXEC_BLUE,"Leads trend (indexed)"), use_container_width=True)
        with s2: st.plotly_chart(tile_line(pipeline_ts,EXEC_PRIMARY,"Pipeline trend (indexed)"), use_container_width=True)
        with s3: st.plotly_chart(tile_line(rev_ts,EXEC_GREEN,"Revenue trend (indexed)"), use_container_width=True)
        with s4: st.plotly_chart(tile_line(calls_ts,"#7dd3fc","Call success trend (indexed)"), use_container_width=True)
    elif trend_style=="Bars":
        with s1: st.plotly_chart(tile_bar(leads_ts,EXEC_BLUE,"Leads trend (indexed)"), use_container_width=True)
        with s2: st.plotly_chart(tile_bar(pipeline_ts,EXEC_PRIMARY,"Pipeline trend (indexed)"), use_container_width=True)
        with s3: st.plotly_chart(tile_bar(rev_ts,EXEC_GREEN,"Revenue trend (indexed)"), use_container_width=True)
        with s4: st.plotly_chart(tile_bar(calls_ts,"#7dd3fc","Call success trend (indexed)"), use_container_width=True)
    else:
        with s1: st.plotly_chart(tile_bullet(leads_ts,"Leads index",EXEC_BLUE), use_container_width=True)
        with s2: st.plotly_chart(tile_bullet(pipeline_ts,"Pipeline index",EXEC_PRIMARY), use_container_width=True)
        with s3: st.plotly_chart(tile_bullet(rev_ts,"Revenue index",EXEC_GREEN), use_container_width=True)
        with s4: st.plotly_chart(tile_bullet(calls_ts,"Call success index","#7dd3fc"), use_container_width=True)

    # ---------------- Plotly Express Funnel (DataFrame-driven, ordered stages) ----------------
    st.markdown("---")
    st.subheader("Lead conversion funnel")

    # Helper functions for statuses
    def have(df, cols): return (df is not None) and set(cols).issubset(df.columns)
    def status_ids_by_stage(stage_no):
        if lead_statuses is None: return set()
        s = lead_statuses.copy(); s.columns = s.columns.str.lower()
        if not {"leadstageid","leadstatusid"}.issubset(s.columns): return set()
        return set(s.loc[s["leadstageid"].astype("Int64")==stage_no, "leadstatusid"].astype(int).tolist())
    def status_ids_by_name(names):
        if lead_statuses is None: return set()
        s = lead_statuses.copy(); s.columns = s.columns.str.lower()
        if not {"statusname_e","leadstatusid"}.issubset(s.columns): return set()
        return set(s.loc[s["statusname_e"].str.lower().isin([n.lower() for n in names]), "leadstatusid"].astype(int).tolist())

    # Determine cohorts
    cohort_ids = pd.Index(leads["LeadId"].dropna().astype(int).unique()) if have(leads, ["LeadId"]) else pd.Index([])
    qualified_sid = status_ids_by_stage(2)
    qualified_ids = pd.Index(
        leads.loc[have(leads, ["LeadStatusId","LeadId"]) & leads["LeadStatusId"].isin(qualified_sid), "LeadId"]
        .dropna().astype(int).unique()
    ).intersection(cohort_ids)

    meet_ids = pd.Index([])
    if ama is not None:
        m = ama.copy(); m.columns = m.columns.str.lower()
        if "leadid" in m.columns:
            if "meetingstatusid" in m.columns:
                scheduled_mask = m["meetingstatusid"].isin({1,6}) if m["meetingstatusid"].notna().any() else m["meetingstatusid"].notna()
                m = m.loc[scheduled_mask]
            meet_ids = pd.Index(m["leadid"].dropna().astype(int).unique()).intersection(qualified_ids)

    neg_sid = status_ids_by_name(["On Hold","Awaiting Budget"])
    neg_ids = pd.Index(
        leads.loc[have(leads, ["LeadStatusId","LeadId"]) & leads["LeadStatusId"].isin(neg_sid), "LeadId"]
        .dropna().astype(int).unique()
    ).intersection(meet_ids)

    won_sid = status_ids_by_name(["Won"])
    signed_ids = pd.Index(
        leads.loc[have(leads, ["LeadStatusId","LeadId"]) & leads["LeadStatusId"].isin(won_sid), "LeadId"]
        .dropna().astype(int).unique()
    ).intersection(meet_ids)

    lost_sid = status_ids_by_name(["Lost"])
    lost_ids = pd.Index(
        leads.loc[have(leads, ["LeadStatusId","LeadId"]) & leads["LeadStatusId"].isin(lost_sid), "LeadId"]
        .dropna().astype(int).unique()
    ).intersection(meet_ids)

    # Counts per stage (length must match stages list)
    stages = ["New", "Qualified", "Meeting Scheduled", "Negotiation", "Contract Signed", "Lost"]
    new_count       = int(cohort_ids.size)
    qualified_count = int(qualified_ids.size)
    meeting_count   = int(meet_ids.size)
    neg_count       = int(neg_ids.size)
    signed_count    = int(signed_ids.size)
    lost_count      = int(lost_ids.size)

    funnel_df = pd.DataFrame({
        "Stage": stages,
        "Count": [new_count, qualified_count, meeting_count, neg_count, signed_count, lost_count],
    })

    fig_funnel = px.funnel(
        funnel_df,
        x="Count",
        y="Stage",
        color_discrete_sequence=["#1E90FF", "#32CD32", "#DAA520", "#FFA500", "#7CFC00", "#DC143C"],
        title="Lead Conversion Funnel",
    )
    fig_funnel.update_layout(
        height=340,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    # ---------------- Top markets ----------------
    st.markdown("---"); st.subheader("Top markets")
    countries=d.get("countries")
    if countries is not None and "CountryId" in leads.columns and "countryname_e" in countries.columns:
        geo = leads.groupby("CountryId").size().reset_index(name="Leads")
        geo = geo.merge(countries[["countryid","countryname_e"]].rename(columns={"countryid":"CountryId","countryname_e":"Country"}), on="CountryId", how="left")
        if "EstimatedBudget" in leads.columns and leads["EstimatedBudget"].sum()>0:
            geo_pipe = leads.groupby("CountryId")["EstimatedBudget"].sum().reset_index(name="Pipeline")
            geo = geo.merge(geo_pipe, on="CountryId", how="left"); total = float(geo["Pipeline"].sum()); geo["Share"] = (geo["Pipeline"]/total*100).round(1) if total>0 else 0.0
        else:
            total = float(geo["Leads"].sum()); geo["Share"] = (geo["Leads"]/total*100).round(1) if total>0 else 0.0
        top5 = geo.sort_values(["Share","Leads"], ascending=False).head(5)[["Country","Leads","Share"]]
        st.dataframe(top5, use_container_width=True, column_config={"Share": st.column_config.ProgressColumn("Share", format="%.1f%%", min_value=0.0, max_value=100.0)}, hide_index=True)
    else:
        st.info("Country data unavailable to build the markets table.")

    st.markdown("---"); st.subheader("🤖 AI-Powered Strategic Insights")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="insight-box">
          <h4>🔮 Predictive Signals</h4>
          <ul>
            <li>Use LeadStatus 'Won' as conversion anchor across all KPIs</li>
            <li>Call 'Connected' defines success for operational trends</li>
            <li>Weekly grain smooths volatility for short‑term monitoring</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="insight-box">
          <h4>🎯 Actions</h4>
          <ul>
            <li>Set budgets in Leads for pipeline KPIs; else fallback uses counts</li>
            <li>Upload data/marketing_spend.csv to compute ROI</li>
            <li>Coach with call outcomes and add sentiment for richer insights</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Lead Status page (robust to schema differences)
# -----------------------------------------------------------------------------
def show_lead_status(d):
    statuses = d.get("lead_statuses")
    leads = d.get("leads")
    if leads is None or len(leads) == 0 or statuses is None or len(statuses) == 0:
        st.info("No lead status data in the selected range."); return

    def pick_col(df, candidates):
        m = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in m: return m[c.lower()]
        return None

    s = statuses.copy()
    id_col = pick_col(s, ["leadstatusid","lead_status_id","statusid","status_id"])
    name_col = pick_col(s, ["statusname_e","statusname","status_name_e","status_name","name"])
    if id_col is None or name_col is None:
        st.warning(f"Lead status columns not found (have: {list(s.columns)})"); return

    lbl_map = dict(zip(s[id_col].astype(int), s[name_col].astype(str)))
    if "LeadStatusId" not in leads.columns:
        st.info("LeadStatusId column not present on leads after normalization."); return

    counts = leads["LeadStatusId"].value_counts(dropna=False).reset_index()
    counts.columns = ["LeadStatusId","count"]
    counts["label"] = counts["LeadStatusId"].map(lbl_map).fillna(counts["LeadStatusId"].astype(str))

    c1,c2 = st.columns([2,1])
    with c1:
        fig = px.pie(counts, names="label", values="count", hole=0.35, color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Total Leads", format_number(len(leads)))
        won_ids = s.loc[s[name_col].str.lower().eq("won"), id_col].astype(int).tolist()
        won = int(leads["LeadStatusId"].isin(won_ids).sum()) if won_ids else 0
        discuss_mask = s[name_col].str.contains("discussion", case=False, na=False)
        discuss_ids = s.loc[discuss_mask, id_col].astype(int).tolist()
        in_discuss = int(leads["LeadStatusId"].isin(discuss_ids).sum()) if discuss_ids else 0
        st.metric("Won", format_number(won))
        st.metric("In Discussion", format_number(in_discuss))

# -----------------------------------------------------------------------------
# Calls page (basic)
# -----------------------------------------------------------------------------
def show_calls(d):
    calls=d.get("calls")
    if calls is None or len(calls)==0:
        st.info("No call data in the selected range."); return
    if "CallDateTime" in calls.columns:
        c=calls.copy(); c["CallDateTime"]=pd.to_datetime(c["CallDateTime"], errors="coerce")
        daily=c.groupby(c["CallDateTime"].dt.date).agg(Total=("LeadCallId","count"), Connected=("CallStatusId", lambda x:(x==1).sum())).reset_index()
        daily["SuccessRate"]=(daily["Connected"]/daily["Total"]*100).round(1)
        col1,col2=st.columns(2)
        with col1:
            fig=go.Figure(); fig.add_trace(go.Scatter(x=daily["CallDateTime"], y=daily["Total"], mode="lines+markers", line=dict(color=EXEC_BLUE,width=3)))
            fig.update_layout(title="Daily Calls", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig=go.Figure(); fig.add_trace(go.Scatter(x=daily["CallDateTime"], y=daily["SuccessRate"], mode="lines+markers", line=dict(color=EXEC_GREEN,width=3)))
            fig.update_layout(title="Success Rate (%)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
    st.dataframe(calls.head(1000), use_container_width=True)

# -----------------------------------------------------------------------------
# Navigation (single path; no duplicates)
# -----------------------------------------------------------------------------
NAV = [("Executive","speedometer2","🎯 Executive Summary"),
       ("Lead Status","people","📈 Lead Status"),
       ("AI Calls","telephone","📞 AI Call Activity")]

if HAS_OPTION_MENU:
    selected = option_menu(
        None, [n[0] for n in NAV], icons=[n[1] for n in NAV],
        orientation="horizontal", default_index=0,
        styles={"container":{"padding":"0!important","background-color":"#0f1116"},
                "icon":{"color":EXEC_PRIMARY,"font-size":"16px"},
                "nav-link":{"font-size":"14px","color":"#d0d0d0","--hover-color":"#21252b"},
                "nav-link-selected":{"background-color":EXEC_SURFACE}}
    )
    if selected=="Executive":        show_executive_summary(fdata)
    elif selected=="Lead Status":    show_lead_status(fdata)
    elif selected=="AI Calls":       show_calls(fdata)
else:
    t1, t2, t3 = st.tabs([n[2] for n in NAV])
    with t1: show_executive_summary(fdata)
    with t2: show_lead_status(fdata)
    with t3: show_calls(fdata)
