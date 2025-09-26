# app.py — DAR Global CEO Dashboard (SQL Server backend)

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

# -----------------------------------------------------------------------------
# Page config and theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DAR Global - Executive Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    if pd.isna(v):
        return "$0"
    return f"${v/1e9:.1f}B" if v>=1e9 else (f"${v/1e6:.1f}M" if v>=1e6 else f"${v:,.0f}")

def format_number(v):
    if pd.isna(v):
        return "0"
    return f"{v/1e6:.1f}M" if v>=1e6 else (f"{v/1e3:.1f}K" if v>=1e3 else f"{v:,.0f}")

# -----------------------------------------------------------------------------
# Database loader (replaces CSV loader)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(_unused: str = "data"):
    """
    Load all required tables from SQL Server using Streamlit's st.connection.
    Connection details come from .streamlit/secrets.toml under [connections.sql].
    """
    conn = st.connection("sql")  # Uses Streamlit SQLConnection with secrets

    ds = {}
    def fetch(table):
        try:
            return conn.query(f"SELECT * FROM dbo.{table}", ttl=600)
        except Exception:
            return None

    # Base tables
    ds["leads"]        = fetch("Lead")
    ds["agents"]       = fetch("Agents")
    ds["calls"]        = fetch("LeadCallRecord")
    ds["schedules"]    = fetch("LeadSchedule")
    ds["transactions"] = fetch("LeadTransaction")

    # Lookups
    ds["countries"]    = fetch("Country")
    ds["lead_stages"]  = fetch("LeadStage")
    ds["lead_statuses"]= fetch("LeadStatus")
    ds["lead_sources"] = fetch("LeadSource")
    ds["lead_scoring"] = fetch("LeadScoring")
    ds["call_statuses"]= fetch("CallStatus")
    ds["sentiments"]   = fetch("CallSentiment")
    ds["task_types"]   = fetch("TaskType")
    ds["task_statuses"]= fetch("TaskStatus")
    ds["city_region"]  = fetch("CityRegion")
    ds["timezone_info"]= fetch("TimezoneInfo")
    ds["priority"]     = fetch("Priority")
    ds["meeting_status"]=fetch("MeetingStatus")
    ds["agent_meeting_assignment"] = fetch("AgentMeetingAssignment")

    # ---- Normalize columns / types to match original app expectations ----
    def norm(df):
        if df is None:
            return None
        out = df.copy()
        out.columns = out.columns.str.strip().str.replace(r"[^\w]+","_",regex=True).str.lower()
        return out

    def rename(df, mapping):
        if df is None:
            return None
        return df.rename(columns={c: mapping[c] for c in mapping if c in df.columns})

    # Leads
    if ds["leads"] is not None:
        df = norm(ds["leads"])
        df = rename(df,{
            "leadid":"LeadId","lead_id":"LeadId","leadcode":"LeadCode",
            "leadstageid":"LeadStageId","leadstatusid":"LeadStatusId","leadscoringid":"LeadScoringId",
            "assignedagentid":"AssignedAgentId","createdon":"CreatedOn","isactive":"IsActive",
            "countryid":"CountryId","cityregionid":"CityRegionId",
            "estimatedbudget":"EstimatedBudget","budget":"EstimatedBudget"
        })
        for col, default in [("EstimatedBudget",0.0),("LeadStageId",pd.NA),("LeadStatusId",pd.NA),
                             ("AssignedAgentId",pd.NA),("CreatedOn",pd.NaT),("IsActive",1)]:
            if col not in df.columns:
                df[col]=default
        df["CreatedOn"]=pd.to_datetime(df["CreatedOn"], errors="coerce")
        df["EstimatedBudget"]=pd.to_numeric(df["EstimatedBudget"], errors="coerce").fillna(0.0)
        ds["leads"]=df

    # Agents
    if ds["agents"] is not None:
        df=norm(ds["agents"])
        df=rename(df,{"agentid":"AgentId","firstname":"FirstName","first_name":"FirstName",
                      "lastname":"LastName","last_name":"LastName","isactive":"IsActive"})
        for c, d in [("FirstName",""),("LastName",""),("Role",""),("IsActive",1)]:
            if c not in df.columns:
                df[c]=d
        ds["agents"]=df

    # Calls
    if ds["calls"] is not None:
        df=norm(ds["calls"])
        df=rename(df,{"leadcallid":"LeadCallId","lead_id":"LeadId","leadid":"LeadId",
                      "callstatusid":"CallStatusId","calldatetime":"CallDateTime","call_datetime":"CallDateTime",
                      "durationseconds":"DurationSeconds","sentimentid":"SentimentId",
                      "assignedagentid":"AssignedAgentId","calldirection":"CallDirection","direction":"CallDirection"})
        if "CallDateTime" in df.columns:
            df["CallDateTime"]=pd.to_datetime(df["CallDateTime"], errors="coerce")
        ds["calls"]=df

    # Schedules
    if ds["schedules"] is not None:
        df=norm(ds["schedules"])
        df=rename(df,{"scheduleid":"ScheduleId","leadid":"LeadId","tasktypeid":"TaskTypeId","scheduleddate":"ScheduledDate",
                      "taskstatusid":"TaskStatusId","assignedagentid":"AssignedAgentId","completeddate":"CompletedDate","isfollowup":"IsFollowUp"})
        if "ScheduledDate" in df.columns:
            df["ScheduledDate"]=pd.to_datetime(df["ScheduledDate"], errors="coerce")
        if "CompletedDate" in df.columns:
            df["CompletedDate"]=pd.to_datetime(df["CompletedDate"], errors="coerce")
        ds["schedules"]=df

    # Transactions
    if ds["transactions"] is not None:
        df=norm(ds["transactions"])
        df=rename(df,{"transactionid":"TransactionId","leadid":"LeadId","tasktypeid":"TaskTypeId","transactiondate":"TransactionDate"})
        if "TransactionDate" in df.columns:
            df["TransactionDate"]=pd.to_datetime(df["TransactionDate"], errors="coerce")
        ds["transactions"]=df

    # Lookups
    for lk in ["countries","lead_stages","lead_statuses","lead_sources","lead_scoring","call_statuses","sentiments",
               "task_types","task_statuses","city_region","timezone_info","priority","meeting_status","agent_meeting_assignment"]:
        if ds.get(lk) is not None:
            ds[lk]=norm(ds[lk])

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

data = load_data("data")  # param kept for signature compatibility

# Optional debug to verify ranges; collapse by default
with st.expander("Debug: row counts and ranges", expanded=False):
    def mm(s):
        s = pd.to_datetime(s, errors="coerce")
        return f"{s.min()} → {s.max()}"
    st.write({k: (0 if (v is None) else len(v)) for k,v in data.items()})
    if data.get("leads") is not None and "CreatedOn" in data["leads"].columns:
        st.write("Leads CreatedOn:", mm(data["leads"]["CreatedOn"]))
    if data.get("calls") is not None and "CallDateTime" in data["calls"].columns:
        st.write("Calls CallDateTime:", mm(data["calls"]["CallDateTime"]))
    if data.get("schedules") is not None and "ScheduledDate" in data["schedules"].columns:
        st.write("Schedules ScheduledDate:", mm(data["schedules"]["ScheduledDate"]))

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
        if preset=="Last 7 days":
            default_start, default_end = max(gmin, today-timedelta(days=6)), today
        elif preset=="Last 30 days":
            default_start, default_end = max(gmin, today-timedelta(days=29)), today
        elif preset=="Last 90 days":
            default_start, default_end = max(gmin, today-timedelta(days=89)), today
        elif preset=="MTD":
            default_start, default_end = max(gmin, today.replace(day=1)), today
        elif preset=="YTD":
            default_start, default_end = max(gmin, date(today.year,1,1)), today
        else:
            default_start, default_end = gmin, gmax
        step = timedelta(days=1 if grain_sel in ["Week","Month"] else 7)
        date_start, date_end = st.slider("Date range", min_value=gmin, max_value=gmax, value=(default_start, default_end), step=step)

    def add_period(dt):
        if grain_sel=="Week":
            return dt.dt.to_period("W").apply(lambda p: p.start_time.date())
        if grain_sel=="Month":
            return dt.dt.to_period("M").apply(lambda p: p.start_time.date())
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

    # AgentMeetingAssignment optional date filter (StartDateTime if present)
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
    ("AI Calls","telephone","📞 AI Call Activity")
]
if HAS_OPTION_MENU:
    selected = option_menu(None, [n[0] for n in NAV], icons=[n[1] for n in NAV], orientation="horizontal", default_index=0,
                           styles={"container":{"padding":"0!important","background-color":"#0f1116"},
                                   "icon":{"color":EXEC_PRIMARY,"font-size":"16px"},
                                   "nav-link":{"font-size":"14px","color":"#d0d0d0","--hover-color":"#21252b"},
                                   "nav-link-selected":{"background-color":EXEC_SURFACE}})
else:
    tabs = st.tabs([n[2] for n in NAV])
    selected=None

# -----------------------------------------------------------------------------
# Executive Summary (Performance KPIs only)
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    leads=d.get("leads"); agents=d.get("agents"); calls=d.get("calls")
    lead_statuses=d.get("lead_statuses"); countries=d.get("countries")
    schedules=d.get("schedules"); ama=d.get("agent_meeting_assignment")

    if leads is None or len(leads)==0:
        st.info("No data available in the selected range."); return

    # Determine 'Won' status id by name if available; fallback to 9
    won_status_id = 9
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        match = lead_statuses.loc[lead_statuses["statusname_e"].str.lower()=="won"]
        if not match.empty and "leadstatusid" in match.columns:
            won_status_id = int(match.iloc[0]["leadstatusid"])

    st.subheader("Performance KPIs")
    today = pd.Timestamp.today().normalize()
    week_start = today - pd.Timedelta(days=today.weekday())  # Monday
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)
    date_ranges = {
        "Week to Date": (week_start, today),
        "Month to Date": (month_start, today),
        "Year to Date": (year_start, today),
    }
    cols = st.columns(3)
    for (label, (start, end)), col in zip(date_ranges.items(), cols):
        # Leads in period
        leads_period = leads.loc[
            (pd.to_datetime(leads["CreatedOn"], errors="coerce") >= pd.Timestamp(start)) &
            (pd.to_datetime(leads["CreatedOn"], errors="coerce") <= pd.Timestamp(end))
        ] if "CreatedOn" in leads.columns else pd.DataFrame()

        # Meetings scheduled (prefer LeadSchedule, fallback AMA)
        meetings_cnt = 0
        if schedules is not None and "ScheduledDate" in schedules.columns:
            s = schedules.copy()
            s["_dt"] = pd.to_datetime(s["ScheduledDate"], errors="coerce")
            s = s[(s["_dt"]>=pd.Timestamp(start)) & (s["_dt"]<=pd.Timestamp(end))]
            meetings_cnt = int(s["LeadId"].nunique()) if "LeadId" in s.columns else len(s)
        elif ama is not None:
            m = ama.copy(); m.columns = m.columns.str.lower()
            if "startdatetime" in m.columns:
                m["_dt"] = pd.to_datetime(m["startdatetime"], errors="coerce")
                m = m[(m["_dt"]>=pd.Timestamp(start)) & (m["_dt"]<=pd.Timestamp(end))]
                if "meetingstatusid" in m.columns:
                    m = m[m["meetingstatusid"].isin({1,6})]
                meetings_cnt = int(m["leadid"].nunique()) if "leadid" in m.columns else len(m)

        total_leads = int(len(leads_period))
        won_leads = int((leads_period["LeadStatusId"]==won_status_id).sum()) if "LeadStatusId" in leads_period.columns else 0
        conversion_rate = (won_leads/total_leads*100.0) if total_leads else 0.0

        with col:
            st.markdown(f"#### {label}")
            st.markdown("**Total Leads**")
            st.markdown(f"<span style='font-size:2rem;'>{total_leads}</span>", unsafe_allow_html=True)
            st.markdown("**Conversion Rate**")
            st.markdown(f"<span style='font-size:2rem;'>{conversion_rate:.1f}%</span>", unsafe_allow_html=True)
            st.markdown("**Meetings Scheduled**")
            st.markdown(f"<span style='font-size:2rem;'>{meetings_cnt}</span>", unsafe_allow_html=True)

    # ---------------- Trend at a glance (indexed) ----------------
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
        if df.empty:
            df["idx"]=[]
            return df
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

    # ---------------- Lead conversion snapshot (funnel) ----------------
    st.markdown("---")
    st.subheader("Lead conversion snapshot")

    leads_df   = d.get("leads").copy()
    statuses   = d.get("lead_statuses")
    meetings   = d.get("agent_meeting_assignment")

    def have(df, cols):
        return (df is not None) and set(cols).issubset(df.columns)

    def status_ids_by_name(names):
        if statuses is None: return set()
        s = statuses.copy(); s.columns = s.columns.str.lower()
        if not {"statusname_e","leadstatusid"}.issubset(s.columns): return set()
        return set(s.loc[s["statusname_e"].str.lower().isin([n.lower() for n in names]),"leadstatusid"].astype(int).tolist())

    def status_ids_by_stage(stage_no):
        if statuses is None: return set()
        s = statuses.copy(); s.columns = s.columns.str.lower()
        if not {"leadstageid","leadstatusid"}.issubset(s.columns): return set()
        return set(s.loc[s["leadstageid"].astype("Int64")==stage_no, "leadstatusid"].astype(int).tolist())

    cohort_ids = pd.Index(leads_df["LeadId"].dropna().astype(int).unique()) if have(leads_df, ["LeadId"]) else pd.Index([])
    new_count  = int(cohort_ids.size)

    qualified_sid = status_ids_by_stage(2)
    qualified_ids = pd.Index(
        leads_df.loc[
            have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(qualified_sid),
            "LeadId"
        ].dropna().astype(int).unique()
    ).intersection(cohort_ids)
    qualified_count = int(qualified_ids.size)

    meet_ids = pd.Index([])
    if meetings is not None:
        m = meetings.copy(); m.columns = m.columns.str.lower()
        if "leadid" in m.columns:
            if "meetingstatusid" in m.columns:
                scheduled_mask = m["meetingstatusid"].isin({1,6}) if m["meetingstatusid"].notna().any() else m["meetingstatusid"].notna()
                m = m.loc[scheduled_mask]
            meet_ids = pd.Index(m["leadid"].dropna().astype(int).unique()).intersection(qualified_ids)
    meeting_count = int(meet_ids.size)

    neg_sid = status_ids_by_name(["On Hold","Awaiting Budget"])
    neg_ids = pd.Index(
        leads_df.loc[
            have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(neg_sid),
            "LeadId"
        ].dropna().astype(int).unique()
    ).intersection(meet_ids)
    neg_count = int(neg_ids.size)

    won_sid = status_ids_by_name(["Won"])
    signed_ids = pd.Index(
        leads_df.loc[
            have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(won_sid),
            "LeadId"
        ].dropna().astype(int).unique()
    ).intersection(meet_ids)
    signed_count = int(signed_ids.size)

    lost_sid = status_ids_by_name(["Lost"])
    lost_ids = pd.Index(
        leads_df.loc[
            have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(lost_sid),
            "LeadId"
        ].dropna().astype(int).unique()
    ).intersection(meet_ids)
    lost_count = int(lost_ids.size)

    funnel_df = pd.DataFrame({
        "Stage": ["New","Qualified","Meeting Scheduled","Negotiation","Contract Signed","Lost"],
        "Count": [new_count, qualified_count, meeting_count, neg_count, signed_count, lost_count]
    })

    fig_funnel = px.funnel(
        funnel_df, x="Count", y="Stage",
        color_discrete_sequence=[EXEC_BLUE, EXEC_GREEN, EXEC_PRIMARY, "#FFA500", "#7CFC00", EXEC_DANGER]
    )
    fig_funnel.update_layout(
        height=340,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=0, r=0, t=10, b=10)
    )
    st.plotly_chart(fig_funnel, use_container_width=True)

    # ---------------- Top markets ----------------
    st.markdown("---"); st.subheader("Top markets")
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
# Lead Status page
# -----------------------------------------------------------------------------
def show_lead_status(d):
    leads=d.get("leads"); statuses=d.get("lead_statuses")
    if leads is None or len(leads)==0 or "LeadStatusId" not in leads.columns:
        st.info("No lead status data in the selected range."); return
    lbl_map={}
    if statuses is not None and "leadstatusid" in statuses.columns:
        name_col = "statusname_e" if "statusname_e" in statuses.columns else None
        for _,r in statuses.iterrows():
            lbl_map[int(r["leadstatusid"])]= str(r[name_col]) if name_col else f"Status {int(r['leadstatusid'])}"
    counts = leads["LeadStatusId"].value_counts().reset_index()
    counts.columns=["LeadStatusId","count"]
    counts["label"] = counts["LeadStatusId"].map(lbl_map).fillna(counts["LeadStatusId"].astype(str))
    c1,c2 = st.columns([2,1])
    with c1:
        fig = px.pie(counts, names="label", values="count", hole=0.35, color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Total Leads", format_number(len(leads)))
        won_id = None
        if statuses is not None and "statusname_e" in statuses.columns:
            m = statuses.loc[statuses["statusname_e"].str.lower()=="won"]
            if not m.empty: won_id = int(m.iloc[0]["leadstatusid"])
        won = int((leads["LeadStatusId"]==won_id).sum()) if won_id is not None else 0
        disc_ids = statuses.loc[statuses["statusname_e"].str.contains("discussion", case=False, na=False), "leadstatusid"].tolist() if statuses is not None else []
        in_discuss = int(leads["LeadStatusId"].isin(disc_ids).sum()) if len(disc_ids) else 0
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
# Router
# -----------------------------------------------------------------------------
if HAS_OPTION_MENU:
    if selected=="Executive":
        show_executive_summary(fdata)
    elif selected=="Lead Status":
        show_lead_status(fdata)
    elif selected=="AI Calls":
        show_calls(fdata)
else:
    with tabs[0]:
        show_executive_summary(fdata)
    with tabs[1]:
        show_lead_status(fdata)
    with tabs[2]:
        show_calls(fdata)
