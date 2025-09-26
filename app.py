# app.py — DAR Global CEO Dashboard (works with 2‑year synthetic dataset in ./data)

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
# Loader for ./data (robust to plural/singular file names)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(data_dir="data"):
    def pick(*names):
        for n in names:
            p=os.path.join(data_dir,n)
            if os.path.exists(p): return p
        return None
    def read(path):
        try: return pd.read_csv(path) if path else None
        except: return None
    def norm(df):
        df=df.copy(); df.columns=(df.columns.str.strip().str.replace(r"[^\w]+","_",regex=True).str.lower()); return df
    def rename(df,map_):
        return df.rename(columns={c:map_[c] for c in map_ if c in df.columns})

    ds={}
    ds["leads"]       = read(pick("Leads.csv","Lead.csv"))
    ds["agents"]      = read(pick("Agents.csv"))
    ds["calls"]       = read(pick("LeadCallRecord.csv"))
    ds["schedules"]   = read(pick("LeadSchedule.csv"))
    ds["transactions"]= read(pick("LeadTransaction.csv"))
    ds["countries"]   = read(pick("Country.csv"))
    ds["lead_stages"] = read(pick("LeadStage.csv"))
    ds["lead_statuses"]=read(pick("LeadStatus.csv"))
    ds["lead_sources"]= read(pick("LeadSource.csv"))
    ds["lead_scoring"]= read(pick("LeadScoring.csv"))
    ds["call_statuses"]= read(pick("CallStatus.csv"))
    ds["sentiments"]  = read(pick("CallSentiment.csv"))
    ds["task_types"]  = read(pick("TaskType.csv"))
    ds["task_statuses"]=read(pick("TaskStatus.csv"))
    ds["city_region"] = read(pick("CityRegion.csv"))
    ds["timezone_info"]=read(pick("TimezoneInfo.csv"))
    ds["priority"]    = read(pick("Priority.csv"))
    ds["meeting_status"]=read(pick("MeetingStatus.csv"))
    ds["agent_meeting_assignment"]=read(pick("AgentMeetingAssignment.csv"))

    # Normalize and align to canonical names used in pages
    if ds["leads"] is not None:
        df=norm(ds["leads"])
        df=rename(df,{
            "leadid":"LeadId","lead_id":"LeadId","leadcode":"LeadCode",
            "leadstageid":"LeadStageId","leadstatusid":"LeadStatusId","leadscoringid":"LeadScoringId",
            "assignedagentid":"AssignedAgentId","createdon":"CreatedOn","isactive":"IsActive",
            "countryid":"CountryId","cityregionid":"CityRegionId",
            "estimatedbudget":"EstimatedBudget","budget":"EstimatedBudget"
        })
        for col, default in [("EstimatedBudget",0.0),("LeadStageId",pd.NA),("LeadStatusId",pd.NA),
                             ("AssignedAgentId",pd.NA),("CreatedOn",pd.NaT),("IsActive",1)]:
            if col not in df.columns: df[col]=default
        df["CreatedOn"]=pd.to_datetime(df["CreatedOn"], errors="coerce")
        df["EstimatedBudget"]=pd.to_numeric(df["EstimatedBudget"], errors="coerce").fillna(0.0)
        ds["leads"]=df

    if ds["agents"] is not None:
        df=norm(ds["agents"])
        df=rename(df,{"agentid":"AgentId","firstname":"FirstName","first_name":"FirstName","lastname":"LastName","last_name":"LastName","isactive":"IsActive"})
        for c, d in [("FirstName",""),("LastName",""),("Role",""),("IsActive",1)]: 
            if c not in df.columns: df[c]=d
        ds["agents"]=df

    if ds["calls"] is not None:
        df=norm(ds["calls"])
        df=rename(df,{
            "leadcallid":"LeadCallId","lead_id":"LeadId","leadid":"LeadId",
            "callstatusid":"CallStatusId","calldatetime":"CallDateTime","call_datetime":"CallDateTime",
            "durationseconds":"DurationSeconds","sentimentid":"SentimentId",
            "assignedagentid":"AssignedAgentId","calldirection":"CallDirection","direction":"CallDirection"
        })
        if "calldatetime" in df.columns: df["CallDateTime"]=pd.to_datetime(df["calldatetime"], errors="coerce")
        if "CallDateTime" in df.columns: df["CallDateTime"]=pd.to_datetime(df["CallDateTime"], errors="coerce")
        ds["calls"]=df

    if ds["schedules"] is not None:
        df=norm(ds["schedules"])
        df=rename(df,{"scheduleid":"ScheduleId","leadid":"LeadId","tasktypeid":"TaskTypeId","scheduleddate":"ScheduledDate","taskstatusid":"TaskStatusId","assignedagentid":"AssignedAgentId","completeddate":"CompletedDate","isfollowup":"IsFollowUp"})
        if "ScheduledDate" in df.columns: df["ScheduledDate"]=pd.to_datetime(df["ScheduledDate"], errors="coerce")
        if "CompletedDate" in df.columns: df["CompletedDate"]=pd.to_datetime(df["CompletedDate"], errors="coerce")
        ds["schedules"]=df

    if ds["transactions"] is not None:
        df=norm(ds["transactions"])
        df=rename(df,{"transactionid":"TransactionId","leadid":"LeadId","tasktypeid":"TaskTypeId","transactiondate":"TransactionDate"})
        if "TransactionDate" in df.columns: df["TransactionDate"]=pd.to_datetime(df["TransactionDate"], errors="coerce")
        ds["transactions"]=df

    for lk in ["countries","lead_stages","lead_statuses","lead_sources","lead_scoring","call_statuses","sentiments","task_types","task_statuses","city_region","timezone_info","priority","meeting_status","agent_meeting_assignment"]:
        if ds.get(lk) is not None: ds[lk]=norm(ds[lk])

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
    selected=None

# -----------------------------------------------------------------------------
# Executive Summary (Performance KPIs)
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    leads=d.get("leads"); agents=d.get("agents"); calls=d.get("calls")
    lead_statuses=d.get("lead_statuses"); countries=d.get("countries")

    if leads is None or len(leads)==0:
        st.info("No data available in the selected range."); return

    # 'Won' status id
    won_status_id = 9
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        match = lead_statuses.loc[lead_statuses["statusname_e"].str.lower()=="won"]
        if not match.empty and "leadstatusid" in match.columns:
            won_status_id = int(match.iloc[0]["leadstatusid"])

    total_leads = len(leads)
    won_mask = leads["LeadStatusId"].eq(won_status_id) if "LeadStatusId" in leads.columns else pd.Series(False, index=leads.index)
    won_leads = int(won_mask.sum())
    conversion_rate = (won_leads/total_leads*100) if total_leads else 0.0

    active_pipeline = leads["EstimatedBudget"].sum() if "EstimatedBudget" in leads.columns else 0.0
    won_revenue = leads.loc[won_mask, "EstimatedBudget"].sum() if ("EstimatedBudget" in leads.columns) else 0.0

    total_calls = len(calls) if calls is not None else 0
    connected_calls = int((calls["CallStatusId"]==1).sum()) if (calls is not None and "CallStatusId" in calls.columns) else 0
    call_success_rate = (connected_calls/total_calls*100) if total_calls else 0.0

    active_agents = int(agents[agents["IsActive"]==1].shape[0]) if (agents is not None and "IsActive" in agents.columns) else (len(agents) if agents is not None else 0)
    assigned_leads = int(leads["AssignedAgentId"].notna().sum()) if "AssignedAgentId" in leads.columns else 0
    agent_utilization = (assigned_leads/active_agents) if active_agents else 0.0

    st.subheader("Performance KPIs")
    today = pd.Timestamp.today().normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)
    date_ranges = {"Week to Date":(week_start,today),"Month to Date":(month_start,today),"Year to Date":(year_start,today)}
    cols = st.columns(3)
    meetings = d.get("agent_meeting_assignment")

    for (label, (start, end)), col in zip(date_ranges.items(), cols):
        leads_period = leads.loc[(pd.to_datetime(leads["CreatedOn"], errors="coerce")>=pd.Timestamp(start)) &
                                 (pd.to_datetime(leads["CreatedOn"], errors="coerce")<=pd.Timestamp(end))] if "CreatedOn" in leads.columns else pd.DataFrame()
        if meetings is not None and len(meetings)>0:
            m = meetings.copy(); m.columns = m.columns.str.lower()
            date_col = "startdatetime" if "startdatetime" in m.columns else None
            if date_col is not None:
                m["_dt"] = pd.to_datetime(m[date_col], errors="coerce")
                m = m[(m["_dt"]>=pd.Timestamp(start)) & (m["_dt"]<=pd.Timestamp(end))]
                if "meetingstatusid" in m.columns: m = m[m["meetingstatusid"].isin({1,6})]
                meetings_period = m
            else: meetings_period = pd.DataFrame()
        else: meetings_period = pd.DataFrame()

        total_leads_p = int(len(leads_period))
        won_leads_p = int((leads_period["LeadStatusId"]==won_status_id).sum()) if "LeadStatusId" in leads_period.columns else 0
        conv_rate_p = (won_leads_p/total_leads_p*100.0) if total_leads_p else 0.0
        meetings_scheduled = int(meetings_period["leadid"].nunique()) if "leadid" in meetings_period.columns else 0

        with col:
            st.markdown(f"#### {label}")
            st.markdown("Total Leads")
            st.markdown(f"<span style='font-size:2rem;'>{total_leads_p}</span>", unsafe_allow_html=True)
            st.markdown("Conversion Rate")
            st.markdown(f"<span style='font-size:2rem;'>{conv_rate_p:.1f}%</span>", unsafe_allow_html=True)
            st.markdown("Meetings Scheduled")
            st.markdown(f"<span style='font-size:2rem;'>{meetings_scheduled}</span>", unsafe_allow_html=True)

    # Trend at a glance (Leads, Conversion Rate, Meeting Scheduled)
    st.markdown("---"); st.subheader("Trend at a glance")
    trend_style = st.radio("Trend style", ["Line","Bars","Bullet"], index=0, horizontal=True, key="__trend_style_exec")

    if "period" not in leads.columns:
        dt=pd.to_datetime(leads.get("CreatedOn"), errors="coerce")
        leads=leads.copy(); leads["period"]=dt.dt.to_period("M").apply(lambda p: p.start_time.date())

    leads_ts = leads.groupby("period").size().reset_index(name="value")
    if "LeadStatusId" in leads.columns:
        per_leads = leads.groupby("period").size().rename("total")
        per_won = leads.loc[leads["LeadStatusId"].eq(won_status_id)].groupby("period").size().rename("won")
        conv_ts = pd.concat([per_leads, per_won], axis=1).fillna(0.0).reset_index()
        conv_ts["value"] = (conv_ts["won"]/conv_ts["total"]*100).round(1)
    else:
        conv_ts = pd.DataFrame({"period":[], "value":[]})

    meetings = d.get("agent_meeting_assignment")
    if meetings is not None and len(meetings)>0:
        m = meetings.copy(); m.columns = m.columns.str.lower()
        date_col = "startdatetime" if "startdatetime" in m.columns else None
        if date_col is not None:
            m["_period"] = pd.to_datetime(m[date_col], errors="coerce").dt.to_period("W").apply(lambda p: p.start_time.date())
            if "meetingstatusid" in m.columns: m = m[m["meetingstatusid"].isin({1,6})]
            meet_ts = m.groupby("_period").size().reset_index(name="value").rename(columns={"_period":"period"})
        else:
            meet_ts = pd.DataFrame({"period":[], "value":[]})
    else:
        meet_ts = pd.DataFrame({"period":[], "value":[]})

    def _index(df):
        df=df.copy()
        if df.empty: df["idx"]=[]; return df
        base = df["value"].iloc[0] if df["value"].iloc[0]!=0 else 1.0
        df["idx"]=(df["value"]/base)*100.0
        return df

    leads_ts = _index(leads_ts); conv_ts = _index(conv_ts); meet_ts = _index(meet_ts)

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
        if df.empty: fig=go.Figure(); return _apply_axes(fig, [0,1], title)
        cur=float(df["idx"].iloc[-1])
        fig=go.Figure(go.Indicator(mode="number+gauge+delta", value=cur, number={'valueformat':".0f"}, delta={'reference':100},
                                   gauge={'shape':"bullet",'axis':{'range':[80,120]},
                                          'steps':[{'range':[80,95],'color':"rgba(220,20,60,0.35)"},{'range':[95,105],'color':"rgba(255,215,0,0.35)"},
                                                   {'range':[105,120],'color':"rgba(50,205,50,0.35)"}],
                                          'bar':{'color':bar_color},'threshold':{'line':{'color':'#fff','width':2},'value':100}}))
        fig.update_layout(height=120, margin=dict(l=8,r=8,t=26,b=8), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        return fig

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

    # Lead conversion snapshot (funnel)
    st.markdown("---"); st.subheader("Lead conversion snapshot")

    leads_df   = d.get("leads").copy()
    statuses   = d.get("lead_statuses")
    ama        = d.get("agent_meeting_assignment")

    def have(df, cols): 
        return (df is not None) and set(cols).issubset(df.columns)
    def status_ids_by_name(names):
        if statuses is None: return set()
        s = statuses.copy(); s.columns = s.columns.str.lower()
        if not {"statusname_e","leadstatusid"}.issubset(s.columns): return set()
        return set(s.loc[s["statusname_e"].str.lower().isin([n.lower() for n in names]), "leadstatusid"].astype(int).tolist())
    def status_ids_by_stage(stage_no):
        if statuses is None: return set()
        s = statuses.copy(); s.columns = s.columns.str.lower()
        if not {"leadstageid","leadstatusid"}.issubset(s.columns): return set()
        return set(s.loc[s["leadstageid"].astype("Int64")==stage_no, "leadstatusid"].astype(int).tolist())

    cohort_ids = pd.Index(leads_df["LeadId"].dropna().astype(int).unique()) if have(leads_df, ["LeadId"]) else pd.Index([])
    new_count  = int(cohort_ids.size)

    q_sid   = status_ids_by_stage(2)
    qual_ids = pd.Index(leads_df.loc[have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(q_sid),"LeadId"].dropna().astype(int).unique()).intersection(cohort_ids)
    qualified_count = int(qual_ids.size)

    meet_ids = pd.Index([])
    if ama is not None:
        m = ama.copy(); m.columns = m.columns.str.lower()
        if {"leadid","meetingstatusid"}.issubset(m.columns):
            m = m[m["leadid"].isin(qual_ids)]
            m = m[m["meetingstatusid"].isin({1,6})]
            meet_ids = pd.Index(m["leadid"].dropna().astype(int).unique())
    meeting_count = int(meet_ids.size)

    neg_sid = status_ids_by_name(["On Hold","Awaiting Budget"])
    neg_ids = pd.Index(leads_df.loc[have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(neg_sid),"LeadId"].dropna().astype(int).unique()).intersection(meet_ids)
    neg_count = int(neg_ids.size)

    won_sid = status_ids_by_name(["Won"])
    signed_ids = pd.Index(leads_df.loc[have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(won_sid),"LeadId"].dropna().astype(int).unique()).intersection(meet_ids)
    signed_count = int(signed_ids.size)

    lost_sid = status_ids_by_name(["Lost"])
    lost_ids = pd.Index(leads_df.loc[have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(lost_sid),"LeadId"].dropna().astype(int).unique()).intersection(meet_ids)
    lost_count = int(lost_ids.size)

    funnel_df = pd.DataFrame({"Stage":["New","Qualified","Meeting Scheduled","Negotiation","Contract Signed","Lost"],
                              "Count":[new_count, qualified_count, meeting_count, neg_count, signed_count, lost_count]})
    fig = px.funnel(funnel_df, x="Count", y="Stage",
                    color_discrete_sequence=[EXEC_BLUE, EXEC_GREEN, EXEC_PRIMARY, "#FFA500", "#7CFC00", EXEC_DANGER])
    fig.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                      margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # -------------------- Top markets: Country, Leads, Won/Signed --------------------
    st.markdown("---"); st.subheader("Top markets")
    if countries is not None and "CountryId" in leads_df.columns and "countryname_e" in countries.columns:
        # Leads per country
        by_country = leads_df.groupby("CountryId").size().reset_index(name="Leads")
        # Won per country (Signed contracts)
        if "LeadStatusId" in leads_df.columns:
            won_by_country = (leads_df.loc[leads_df["LeadStatusId"].astype("Int64")==won_status_id]
                              .groupby("CountryId").size().reset_index(name="Won"))
        else:
            won_by_country = pd.DataFrame({"CountryId":[], "Won":[]})
        # Join names and present
        final = (by_country.merge(won_by_country, on="CountryId", how="left")
                           .merge(countries[["countryid","countryname_e"]]
                                  .rename(columns={"countryid":"CountryId","countryname_e":"Country"}),
                                  on="CountryId", how="left")
                           .fillna({"Won":0})
                           .sort_values(["Leads","Won"], ascending=False)[["Country","Leads","Won"]])
        st.dataframe(final, use_container_width=True, hide_index=True,
                     column_config={
                         "Leads": st.column_config.NumberColumn("Leads", format="%d"),
                         "Won":   st.column_config.NumberColumn("Won/Signed", format="%d"),
                     })
    else:
        st.info("Country data unavailable to build the market list.")

    # Insights
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
# AI Insights page (Lead propensity + Forecast + Actions)
# -----------------------------------------------------------------------------
def _recent_agg(df, when_col, cutoff, days=14):
    if df is None or len(df)==0 or when_col not in df: 
        return pd.DataFrame({"LeadId":[], "n":[], "connected":[], "mean_dur":[], "last_days":[]})
    x = df.copy()
    x[when_col] = pd.to_datetime(x[when_col], errors="coerce")
    window = cutoff - pd.Timedelta(days=days)
    x = x[(x[when_col]>=window) & (x[when_col]<=cutoff)]
    g = x.groupby("LeadId").agg(
        n=("LeadId","count"),
        connected=("CallStatusId", lambda s: (s==1).mean() if "CallStatusId" in x.columns else 0.0),
        mean_dur=("DurationSeconds", "mean") if "DurationSeconds" in x.columns else ("LeadId","count")
    ).reset_index()
    last = x.groupby("LeadId")[when_col].max().reset_index().rename(columns={when_col:"last_dt"})
    g = g.merge(last, on="LeadId", how="left")
    g["last_days"] = (cutoff - g["last_dt"]).dt.days.fillna(999)
    return g.drop(columns=["last_dt"], errors="ignore")

def _weekly_meeting_series(meets):
    if meets is None or len(meets)==0: 
        return pd.DataFrame({"week":[], "meetings":[]})
    m = meets.copy()
    dt_col = "StartDateTime" if "StartDateTime" in m.columns else ("startdatetime" if "startdatetime" in m.columns else None)
    if dt_col is None: return pd.DataFrame({"week":[], "meetings":[]})
    m["dt"] = pd.to_datetime(m[dt_col], errors="coerce")
    sid = "MeetingStatusId" if "MeetingStatusId" in m.columns else ("meetingstatusid" if "meetingstatusid" in m.columns else None)
    if sid is not None: m = m[m[sid].isin([1,6])]
    g = m.groupby(pd.Grouper(key="dt", freq="W-MON")).size().reset_index(name="meetings").rename(columns={"dt":"week"})
    return g

def _weekly_wins_series(leads):
    if leads is None or len(leads)==0: 
        return pd.DataFrame({"week":[], "wins":[]})
    l = leads.copy()
    l["dt"] = pd.to_datetime(l["CreatedOn"], errors="coerce")
    if "LeadStatusId" in l.columns:
        l = l[l["LeadStatusId"].astype("Int64")==9]
    g = l.groupby(pd.Grouper(key="dt", freq="W-MON")).size().reset_index(name="wins").rename(columns={"dt":"week"})
    return g

def _sma_forecast(vals, k=4):
    if len(vals)==0: return [0.0]*k
    s = pd.Series(vals)
    last = s.rolling(min_periods=1, window=min(4,len(s))).mean().iloc[-1]
    return [float(last)]*k

def show_ai_insights(d):
    st.subheader("Lead win propensity and next-best-actions")
    leads = d.get("leads"); calls = d.get("calls"); meets = d.get("agent_meeting_assignment")
    if leads is None or len(leads)==0:
        st.info("No data available to train insights."); return

    # Prepare label
    won_id = 9
    leads = leads.copy()
    leads["CreatedOn"] = pd.to_datetime(leads["CreatedOn"], errors="coerce")
    leads["label"] = (leads.get("LeadStatusId", pd.Series(index=leads.index).astype("Int64")).astype("Int64")==won_id).astype(int)
    cutoff = leads["CreatedOn"].max() if "CreatedOn" in leads.columns else pd.Timestamp.today()

    # Feature engineering (14-day recency windows)
    calls_14 = _recent_agg(calls, "CallDateTime", cutoff, 14)
    meet_norm = None
    if meets is not None and len(meets):
        m = meets.copy()
        dtc = "StartDateTime" if "StartDateTime" in m.columns else ("startdatetime" if "startdatetime" in m.columns else None)
        if dtc is not None:
            m[dtc] = pd.to_datetime(m[dtc], errors="coerce")
            sid = "MeetingStatusId" if "MeetingStatusId" in m.columns else ("meetingstatusid" if "meetingstatusid" in m.columns else None)
            if sid is not None: m = m[m[sid].isin([1,6])]
            m = m.rename(columns={dtc:"When"})
            meet_norm = _recent_agg(m, "When", cutoff, 14).rename(columns={"n":"meet_n","connected":"meet_connected","mean_dur":"meet_mean_dur","last_days":"meet_last_days"})
    if meet_norm is None:
        meet_norm = pd.DataFrame({"LeadId":[], "meet_n":[], "meet_connected":[], "meet_mean_dur":[], "meet_last_days":[]})

    # Assemble X
    X = leads[["LeadId","LeadStageId","LeadStatusId","AssignedAgentId"]].fillna(0).copy()
    X["age_days"] = (cutoff - leads["CreatedOn"]).dt.days.fillna(0)
    X = X.merge(calls_14, on="LeadId", how="left").merge(meet_norm, on="LeadId", how="left").fillna(0)
    y = leads["label"].values

    # Train model or fallback
    if SKLEARN_OK and len(leads)>=10 and y.sum()>=1:
        X_fit = X.drop(columns=["LeadId"])
        X_tr, X_te, y_tr, y_te = train_test_split(X_fit, y, test_size=0.25, random_state=42, stratify=y if y.sum()>0 else None)
        base = GradientBoostingClassifier(random_state=42)
        model = CalibratedClassifierCV(base, cv=3)
        model.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1]) if len(np.unique(y_te))>1 else np.nan
        win_prob = model.predict_proba(X_fit)[:,1]
        st.metric("Validation AUC", f"{auc:.3f}" if not np.isnan(auc) else "—")
    else:
        # Heuristic fallback
        st.info("Using heuristic scoring (ML library unavailable or insufficient data).")
        win_prob = (
            0.15
            + 0.25*(X["LeadStatusId"].astype(int).isin([6,7,8]).astype(float))
            + 0.25*(X["meet_n"].clip(0,3)/3.0)
            + 0.20*(X["connected"].clip(0,1))
            + 0.15*(1.0/(1.0+X["age_days"]/30.0))
        ).clip(0,1).values

    # Score + actions
    scored = leads[["LeadId","LeadStatusId"]].copy()
    scored["win_prob"] = win_prob

    tmp = X.merge(scored, on="LeadId", how="left")
    def nba(r):
        if r["win_prob"]>=0.60 and r.get("meet_n",0)==0: return "Book meeting within 72h"
        if 0.30<=r["win_prob"]<0.60 and r.get("connected",0)<0.30: return "Nurture call + brochure"
        if r["win_prob"]<0.30 and r.get("n",0)>=2: return "Switch to AI Agent sequence"
        return "Maintain cadence"
    scored["next_action"] = [nba(r) for _,r in tmp.iterrows()]
    st.dataframe(scored.sort_values("win_prob", ascending=False).reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    # Forecasts
    st.markdown("---"); st.subheader("4‑week outlook")
    wm = _weekly_meeting_series(meets)
    ww = _weekly_wins_series(leads)
    f_meet = _sma_forecast(wm["meetings"] if len(wm) else [], 4)
    f_wins = _sma_forecast(ww["wins"] if len(ww) else [], 4)
    c1,c2 = st.columns(2)
    with c1: st.metric("Forecast avg meetings / week (next 4)", f"{np.mean(f_meet):.1f}")
    with c2: st.metric("Forecast avg wins / week (next 4)", f"{np.mean(f_wins):.1f}")

    # Small history charts
    col1, col2 = st.columns(2)
    with col1:
        hist = wm.copy()
        hist["type"]="Meetings"
        fig = px.line(hist, x="week", y="meetings", markers=True, title="Weekly meetings")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=260)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        hist = ww.copy()
        fig = px.line(hist, x="week", y="wins", markers=True, title="Weekly wins")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=260)
        st.plotly_chart(fig, use_container_width=True)

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
    if selected=="Executive": show_executive_summary(fdata)
    elif selected=="Lead Status": show_lead_status(fdata)
    elif selected=="AI Calls": show_calls(fdata)
    elif selected=="AI Insights": show_ai_insights(fdata)
else:
    with tabs[0]: show_executive_summary(fdata)
    with tabs[1]: show_lead_status(fdata)
    with tabs[2]: show_calls(fdata)
    with tabs[3]: show_ai_insights(fdata)
