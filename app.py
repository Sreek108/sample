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

# Page config and theme
st.set_page_config(page_title="DAR Global - Executive Dashboard", layout="wide", initial_sidebar_state="collapsed")

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
[data-testid="stSidebar"] {{
    display: none;
}}
</style>
""", unsafe_allow_html=True)

# Utility to normalize column names
def norm(df):
    if df is None:
        return None
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()
    return out

# Data loading
@st.cache_data(show_spinner=False)
def load_data(data_dir: str | None = None):
    from streamlit.connections import SQLConnection
    conn = st.connection("sql", type=SQLConnection)
    
    try:
        info = conn.query("SELECT @@SERVERNAME AS server, DB_NAME() AS db", ttl=60)
        st.caption(f"Connected to {info.iloc[0]['server']} / {info.iloc[0]['db']}")
    except Exception as e:
        st.error(f"Connectivity check failed: {e}")
        return {}

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

    ds = {}
    ds["leads"] = fetch(T("leads", "dbo.Lead"), "leads")
    ds["agents"] = fetch(T("agents", "dbo.Agents"), "agents")
    ds["calls"] = fetch(T("calls", "dbo.LeadCallRecord"), "calls")
    ds["schedules"] = fetch(T("schedules", "dbo.LeadSchedule"), "schedules")
    ds["transactions"] = fetch(T("transactions", "dbo.LeadTransaction"), "transactions")

    for k, default in [
        ("countries", "dbo.Country"), ("lead_stages", "dbo.LeadStage"), ("lead_statuses", "dbo.LeadStatus"),
        ("lead_sources", "dbo.LeadSource"), ("lead_scoring", "dbo.LeadScoring"),
        ("call_statuses", "dbo.CallStatus"), ("sentiments", "dbo.CallSentiment"),
        ("task_types", "dbo.TaskType"), ("task_statuses", "dbo.TaskStatus"),
        ("city_region", "dbo.CityRegion"), ("timezone_info", "dbo.TimezoneInfo"),
        ("priority", "dbo.Priority"), ("meeting_status", "dbo.MeetingStatus"),
        ("agent_meeting_assignment", "dbo.AgentMeetingAssignment"),
    ]:
        ds[k] = fetch(T(k, default), k)

    def norm_lower(df):
        if df is None:
            return None
        out = df.copy()
        out.columns = out.columns.str.strip().str.replace(r"[^\w]+", "_", regex=True).str.lower()
        return out

    def rename(df, mapping):
        cols = {src: dst for src, dst in mapping.items() if src in df.columns}
        return df.rename(columns=cols)

    if ds["leads"] is not None:
        df = norm_lower(ds["leads"])
        df = rename(df, {
            "leadid": "LeadId", "lead_id": "LeadId", "leadcode": "LeadCode",
            "leadstageid": "LeadStageId", "leadstatusid": "LeadStatusId", "leadscoringid": "LeadScoringId",
            "assignedagentid": "AssignedAgentId", "createdon": "CreatedOn", "isactive": "IsActive",
            "countryid": "CountryId", "cityregionid": "CityRegionId",
            "estimatedbudget": "EstimatedBudget", "budget": "EstimatedBudget"
        })
        for col, default in [("EstimatedBudget", 0.0), ("LeadStageId", pd.NA), ("LeadStatusId", pd.NA),
                             ("AssignedAgentId", pd.NA), ("CreatedOn", pd.NaT), ("IsActive", 1)]:
            if col not in df.columns:
                df[col] = default
        df["CreatedOn"] = pd.to_datetime(df.get("CreatedOn"), errors="coerce")
        df["EstimatedBudget"] = pd.to_numeric(df.get("EstimatedBudget"), errors="coerce").fillna(0.0)
        ds["leads"] = df

    if ds["agents"] is not None:
        df = norm_lower(ds["agents"])
        df = rename(df, {"agentid": "AgentId", "firstname": "FirstName", "first_name": "FirstName",
                         "lastname": "LastName", "last_name": "LastName", "isactive": "IsActive"})
        ds["agents"] = df

    if ds["calls"] is not None:
        df = norm_lower(ds["calls"])
        df = rename(df, {
            "leadcallid": "LeadCallId", "lead_id": "LeadId", "leadid": "LeadId",
            "callstatusid": "CallStatusId", "calldatetime": "CallDateTime", "call_datetime": "CallDateTime",
            "durationseconds": "DurationSeconds", "sentimentid": "SentimentId",
            "assignedagentid": "AssignedAgentId", "calldirection": "CallDirection", "direction": "CallDirection"
        })
        if "CallDateTime" in df.columns:
            df["CallDateTime"] = pd.to_datetime(df["CallDateTime"], errors="coerce")
        ds["calls"] = df

    if ds["schedules"] is not None:
        df = norm_lower(ds["schedules"])
        df = rename(df, {"scheduleid": "ScheduleId", "leadid": "LeadId", "tasktypeid": "TaskTypeId",
                         "scheduleddate": "ScheduledDate", "taskstatusid": "TaskStatusId",
                         "assignedagentid": "AssignedAgentId", "completeddate": "CompletedDate", "isfollowup": "IsFollowUp"})
        if "ScheduledDate" in df.columns:
            df["ScheduledDate"] = pd.to_datetime(df["ScheduledDate"], errors="coerce")
        if "CompletedDate" in df.columns:
            df["CompletedDate"] = pd.to_datetime(df["CompletedDate"], errors="coerce")
        ds["schedules"] = df

    if ds["transactions"] is not None:
        df = norm_lower(ds["transactions"])
        df = rename(df, {"transactionid": "TransactionId", "leadid": "LeadId", "tasktypeid": "TaskTypeId", "transactiondate": "TransactionDate"})
        if "TransactionDate" in df.columns:
            df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
        ds["transactions"] = df

    for lk in ["countries", "lead_stages", "lead_statuses", "lead_sources", "lead_scoring",
               "call_statuses", "sentiments", "task_types", "task_statuses", "city_region",
               "timezone_info", "priority", "meeting_status", "agent_meeting_assignment"]:
        if ds.get(lk) is not None:
            ds[lk] = norm_lower(ds[lk])

    return ds

# Set default grain and load data
grain = "Month"
data = load_data("data")

# CORRECTED PROGRESSIVE FUNNEL (New at top, Lost at bottom, count only)
def render_funnel_and_markets(d):
    stage_order = ["Lost", "Won", "Negotiation", "Meeting Scheduled", "Interested", "New"]

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

    interested_ids = ids_from_status_names(["Interested", "Qualified", "Hot", "Warm"])
    negotiation_ids = ids_from_status_names(["Negotiation", "On Hold", "Awaiting Budget", "Proposal Sent"])
    won_ids = ids_from_status_names(["Won", "Closed Won", "Contract Signed"])
    lost_ids = ids_from_status_names(["Lost", "Closed Lost", "Dead", "Not Interested"])

    total_leads = len(leads)
    interested_leads = leads[leads["leadstatusid"].isin(interested_ids | negotiation_ids | won_ids)]
    interested_count = len(interested_leads)
    
    meeting_lead_ids = set()
    if ama is not None and {"leadid", "meetingstatusid", "startdatetime"}.issubset(ama.columns):
        meeting_lead_ids = set(
            ama.loc[ama["meetingstatusid"].isin({1, 6}), "leadid"].dropna().astype(int).unique()
        )
    
    meeting_scheduled_leads = leads[
        (leads["leadid"].isin(meeting_lead_ids)) | 
        (leads["leadstatusid"].isin(negotiation_ids | won_ids))
    ]
    meeting_count = len(meeting_scheduled_leads)
    
    negotiation_leads = leads[leads["leadstatusid"].isin(negotiation_ids | won_ids)]
    negotiation_count = len(negotiation_leads)
    
    won_leads = leads[leads["leadstatusid"].isin(won_ids)]
    won_count = len(won_leads)
    
    lost_leads = leads[leads["leadstatusid"].isin(lost_ids)]
    lost_count = len(lost_leads)

    funnel_data = [
        {"Stage": "Lost", "Count": lost_count if lost_count > 0 else 1},
        {"Stage": "Won", "Count": won_count if won_count > 0 else 1},
        {"Stage": "Negotiation", "Count": negotiation_count if negotiation_count > 0 else 1},
        {"Stage": "Meeting Scheduled", "Count": meeting_count if meeting_count > 0 else 1},
        {"Stage": "Interested", "Count": interested_count if interested_count > 0 else 1},
        {"Stage": "New", "Count": total_leads}
    ]
    
    funnel_df = pd.DataFrame(funnel_data)

    fig = px.funnel(
        funnel_df,
        x="Count",
        y="Stage",
        category_orders={"Stage": stage_order},
        color_discrete_sequence=[EXEC_DANGER, "#7CFC00", "#FFA500", EXEC_PRIMARY, EXEC_GREEN, EXEC_BLUE]
    )
    
    fig.update_traces(
        textposition="inside", 
        textfont_color="white",
        textfont_size=16,
        textinfo="value"
    )
    
    fig.update_layout(
        height=450,
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

# Executive Summary with Date Slicer (3 columns only)
def show_executive_summary(d):
    all_leads = data.get("leads")
    lead_statuses = d.get("lead_statuses")
    
    if all_leads is None or len(all_leads) == 0:
        st.info("No data available.")
        return

    won_status_id = 9
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        match = lead_statuses.loc[lead_statuses["statusname_e"].str.lower() == "won"]
        if not match.empty and "leadstatusid" in match.columns:
            won_status_id = int(match.iloc[0]["leadstatusid"])

    st.subheader("Performance KPIs")
    
    # DATE RANGE SLICER
    col_date1, col_date2, col_date3 = st.columns([1, 1, 2])
    
    if "CreatedOn" in all_leads.columns:
        all_dates = pd.to_datetime(all_leads["CreatedOn"], errors="coerce").dropna()
        min_date = all_dates.min().date() if len(all_dates) > 0 else date.today() - timedelta(days=365)
        max_date = all_dates.max().date() if len(all_dates) > 0 else date.today()
    else:
        min_date = date.today() - timedelta(days=365)
        max_date = date.today()
    
    with col_date1:
        date_from = st.date_input("From Date", value=min_date, min_value=min_date, max_value=max_date, key="date_from")
    
    with col_date2:
        date_to = st.date_input("To Date", value=max_date, min_value=min_date, max_value=max_date, key="date_to")
    
    with col_date3:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Apply Date Range", type="primary"):
            st.rerun()
    
    st.markdown("---")
    
    # Filter leads by date range
    filtered_leads = all_leads.loc[
        (pd.to_datetime(all_leads["CreatedOn"], errors="coerce").dt.date >= date_from) &
        (pd.to_datetime(all_leads["CreatedOn"], errors="coerce").dt.date <= date_to)
    ] if "CreatedOn" in all_leads.columns else all_leads.copy()
    
    today = pd.Timestamp.today().normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)

    # 3 columns only (removed Selected Range)
    cols = st.columns(3)
    all_meetings = data.get("agent_meeting_assignment")

    # Only Week, Month, Year (removed Selected Range)
    periods = [
        ("Week to Date", week_start, today),
        ("Month to Date", month_start, today),
        ("Year to Date", year_start, today)
    ]

    for (label, start, end), col in zip(periods, cols):
        leads_period = filtered_leads.loc[
            (pd.to_datetime(filtered_leads["CreatedOn"], errors="coerce") >= start) &
            (pd.to_datetime(filtered_leads["CreatedOn"], errors="coerce") <= end)
        ] if "CreatedOn" in filtered_leads.columns else pd.DataFrame()
        
        if all_meetings is not None and len(all_meetings) > 0:
            m = all_meetings.copy()
            m.columns = m.columns.str.lower()
            date_col = "startdatetime" if "startdatetime" in m.columns else None
            
            if date_col is not None:
                m["dt"] = pd.to_datetime(m[date_col], errors="coerce")
                m = m[(m["dt"] >= start) & (m["dt"] <= end)]
                
                if "meetingstatusid" in m.columns:
                    m = m[m["meetingstatusid"].isin({1, 6})]
                meetings_period = m
            else:
                meetings_period = pd.DataFrame()
        else:
            meetings_period = pd.DataFrame()
        
        total_leads_p = int(len(leads_period))
        won_leads_p = int((leads_period["LeadStatusId"] == won_status_id).sum()) if "LeadStatusId" in leads_period.columns and len(leads_period) > 0 else 0
        conv_rate_p = (won_leads_p / total_leads_p * 100.0) if total_leads_p > 0 else 0.0
        meetings_scheduled = int(meetings_period["leadid"].nunique()) if "leadid" in meetings_period.columns and len(meetings_period) > 0 else 0
        
        with col:
            st.markdown(f"#### {label}")
            st.markdown("**Total Leads**")
            st.markdown(f"<span style='font-size:2rem;'>{total_leads_p:,}</span>", unsafe_allow_html=True)
            st.markdown("**Conversion Rate**")
            st.markdown(f"<span style='font-size:2rem;'>{conv_rate_p:.1f}%</span>", unsafe_allow_html=True)
            st.markdown("**Meetings Scheduled**")
            st.markdown(f"<span style='font-size:2rem;'>{meetings_scheduled:,}</span>", unsafe_allow_html=True)

    # Trend at a glance
    leads = filtered_leads.copy()
    st.markdown("---")
    st.subheader("Trend at a glance")
    trend_style = st.radio("Trend style", ["Line", "Bars", "Bullet"], index=0, horizontal=True, key="__trend_style_exec")
    
    leads_local = leads.copy()
    
    if "period" not in leads_local.columns and "CreatedOn" in leads_local.columns:
        dt = pd.to_datetime(leads_local["CreatedOn"], errors="coerce")
        if grain == "Week":
            leads_local["period"] = dt.dt.to_period("W").apply(lambda p: p.start_time.date())
        elif grain == "Month":
            leads_local["period"] = dt.dt.to_period("M").apply(lambda p: p.start_time.date())
        else:
            leads_local["period"] = dt.dt.to_period("Y").apply(lambda p: p.start_time.date())
    
    leads_ts = leads_local.groupby("period").size().reset_index(name="value")
    
    if "LeadStatusId" in leads_local.columns:
        per_leads = leads_local.groupby("period").size()
        per_won = leads_local.loc[leads_local["LeadStatusId"].eq(won_status_id)].groupby("period").size()
        conv_ts = pd.DataFrame({"period": per_leads.index, "total": per_leads.values})
        conv_ts = conv_ts.merge(
            pd.DataFrame({"period": per_won.index, "won": per_won.values}),
            on="period", how="left"
        ).fillna(0)
        conv_ts["value"] = (conv_ts["won"] / conv_ts["total"] * 100).round(1)
    else:
        conv_ts = pd.DataFrame({"period": [], "value": []})
    
    meetings = d.get("agent_meeting_assignment")
    if meetings is not None and len(meetings) > 0:
        m = meetings.copy()
        m.columns = m.columns.str.lower()
        date_col = "startdatetime" if "startdatetime" in m.columns else None
        if date_col is not None:
            m["dt"] = pd.to_datetime(m[date_col], errors="coerce")
            if grain == "Week":
                m["period"] = m["dt"].dt.to_period("W").apply(lambda p: p.start_time.date())
            elif grain == "Month":
                m["period"] = m["dt"].dt.to_period("M").apply(lambda p: p.start_time.date())
            else:
                m["period"] = m["dt"].dt.to_period("Y").apply(lambda p: p.start_time.date())
            
            if "meetingstatusid" in m.columns:
                m = m[m["meetingstatusid"].isin({1, 6})]
            meet_ts = m.groupby("period")["leadid"].nunique().reset_index(name="value")
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
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=6, ticks="outside")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=4, ticks="outside", range=rng)
        return fig
    
    def tile_line(df, color, title):
        df = df.dropna().sort_values("period")
        if len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return _apply_axes(fig, [0, 1], title)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["period"], y=df["idx"], mode="lines+markers", 
                                 line=dict(color=color, width=3, shape="spline"), 
                                 marker=dict(size=6, color=color)))
        return _apply_axes(fig, df["idx"], title)
    
    def tile_bar(df, color, title):
        df = df.dropna().sort_values("period")
        if len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return _apply_axes(fig, [0, 1], title)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["period"], y=df["idx"], 
                             marker=dict(color=color, line=dict(color="rgba(255,255,255,0.15)", width=0.5)), 
                             opacity=0.9))
        return _apply_axes(fig, df["idx"], title)
    
    def tile_bullet(df, title, bar_color):
        if df.empty or len(df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
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

    st.markdown("---")
    st.subheader("Lead conversion snapshot")
    
    d_filtered = dict(d)
    d_filtered["leads"] = filtered_leads
    render_funnel_and_markets(d_filtered)

# Lead Status with VERTICAL bar chart (removed status mix by period)
def show_lead_status(d):
    leads = d.get("leads")
    statuses = d.get("lead_statuses")
    calls = d.get("calls")
    meets = d.get("agent_meeting_assignment")

    if leads is None or len(leads) == 0:
        st.info("No lead status data in the selected range.")
        return

    name_map = {}
    if statuses is not None and {"leadstatusid", "statusname_e"}.issubset(statuses.columns):
        name_map = dict(zip(statuses["leadstatusid"].astype(int), statuses["statusname_e"].astype(str)))
    leads = leads.copy()
    leads["Status"] = leads["LeadStatusId"].map(name_map).fillna(leads.get("LeadStatusId", pd.Series(dtype="Int64")).astype(str))
    leads["CreatedOn"] = pd.to_datetime(leads.get("CreatedOn"), errors="coerce")
    cutoff = leads["CreatedOn"].max() if "CreatedOn" in leads.columns else pd.Timestamp.today()
    leads["age_days"] = (cutoff - leads["CreatedOn"]).dt.days.astype("Int64")

    counts = leads["Status"].value_counts().reset_index()
    counts.columns = ["Status", "count"]
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.pie(counts, names="Status", values="count", hole=0.35, color_discrete_sequence=px.colors.sequential.Viridis,
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
        won = int((leads.get("LeadStatusId", pd.Series(dtype="Int64")).astype("Int64") == won_id).sum()) if won_id is not None else 0
        st.metric("Won", f"{won:,}")

    st.markdown("---")
    st.subheader("Lead Distribution Status")

    # VERTICAL BAR CHART
    dist_sorted = counts.sort_values("count", ascending=False)
    fig_bar = px.bar(
        dist_sorted, 
        x="Status",
        y="count",
        title="Leads by status",
        color="Status", 
        color_discrete_sequence=px.colors.qualitative.Set3,
        text="count"
    )
    fig_bar.update_traces(textposition='outside', textfont_size=12)
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)", 
        font_color="white", 
        height=400,
        showlegend=False, 
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="Status",
        yaxis_title="Count"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Removed "Status mix by period" section
    
    st.markdown("---")
    st.subheader("Detailed Lead Breakdown")

    meet_rate = pd.DataFrame({"Status": pd.Series(dtype="str"), "meet_leads": pd.Series(dtype="float")})
    if meets is not None and len(meets):
        M = meets.copy()
        M.columns = M.columns.str.lower()
        dtc = "startdatetime" if "startdatetime" in M.columns else None
        if dtc is not None:
            if "meetingstatusid" in M.columns:
                M = M[M["meetingstatusid"].isin({1, 6})]
            mm = M.merge(leads[["LeadId", "Status"]], left_on="leadid", right_on="LeadId", how="left")
            meet_rate = mm.groupby("Status")["leadid"].nunique().reset_index(name="meet_leads")

    conn_rate = pd.DataFrame({"Status": pd.Series(dtype="str"), "connect_rate": pd.Series(dtype="float")})
    if calls is not None and len(calls):
        C = calls.copy()
        C["CallDateTime"] = pd.to_datetime(C.get("CallDateTime"), errors="coerce")
        C = C.merge(leads[["LeadId", "Status"]], on="LeadId", how="left")
        g = C.groupby("Status").agg(total=("LeadCallId", "count"),
                                    connects=("CallStatusId", lambda s: (s == 1).sum())).reset_index()
        g["connect_rate"] = (g["connects"] / g["total"]).fillna(0.0)
        conn_rate = g[["Status", "connect_rate"]]

    base = leads.groupby("Status").agg(
        Leads=("LeadId", "count"),
        Avg_Age_Days=("age_days", "mean"),
        Pipeline=("EstimatedBudget", "sum")
    ).reset_index()
    total_leads = float(base["Leads"].sum()) if len(base) else 0.0
    base["Share_%"] = (base["Leads"] / total_leads * 100.0).round(1) if total_leads > 0 else 0.0

    breakdown = (base.merge(meet_rate, on="Status", how="left")
                      .merge(conn_rate, on="Status", how="left"))
    breakdown["meet_leads"] = breakdown["meet_leads"].fillna(0.0)
    breakdown["Meeting_Rate_%"] = (breakdown["meet_leads"] / breakdown["Leads"] * 100.0).replace([np.inf, -np.inf], 0).fillna(0.0).round(1)
    breakdown["connect_rate"] = breakdown["connect_rate"].fillna(0.0).round(2)
    breakdown["Avg_Age_Days"] = breakdown["Avg_Age_Days"].fillna(0.0).round(1)
    breakdown = breakdown.sort_values(["Leads", "Pipeline"], ascending=False)

    st.dataframe(
        breakdown[["Status", "Leads", "Share_%", "Avg_Age_Days", "Meeting_Rate_%", "connect_rate", "Pipeline"]],
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

# Create a simple filtered dataset for navigation
fdata = {"leads": data.get("leads"), "lead_statuses": data.get("lead_statuses"), 
         "agent_meeting_assignment": data.get("agent_meeting_assignment"),
         "countries": data.get("countries"), "calls": data.get("calls")}

# Navigation
NAV = [
    ("Executive", "speedometer2", "🎯 Executive Summary"),
    ("Lead Status", "people", "📈 Lead Status"),
]

if HAS_OPTION_MENU:
    selected = option_menu(
        None,
        [n[0] for n in NAV],
        icons=[n[1] for n in NAV],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0f1116"},
            "icon": {"color": EXEC_PRIMARY, "font-size": "16px"},
            "nav-link": {"font-size": "14px", "color": "#d0d0d0", "--hover-color": "#21252b"},
            "nav-link-selected": {"background-color": EXEC_SURFACE}
        }
    )
    if selected == "Executive":
        show_executive_summary(fdata)
    elif selected == "Lead Status":
        show_lead_status(fdata)
else:
    tabs = st.tabs([n[2] for n in NAV])
    with tabs[0]:
        show_executive_summary(fdata)
    with tabs[1]:
        show_lead_status(fdata)
