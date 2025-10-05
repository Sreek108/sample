```python
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

# Page config
st.set_page_config(
    page_title="DAR Global - Executive Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Light theme palette
PRIMARY_GOLD = "#DAA520"
ACCENT_BLUE  = "#1E90FF"   # Week
ACCENT_GREEN = "#32CD32"   # Month
ACCENT_AMBER = "#F59E0B"   # Year
ACCENT_RED   = "#DC143C"

BG_PAGE     = "#F6F7FB"    # page background
BG_SURFACE  = "#FFFFFF"    # cards/surfaces
TEXT_MAIN   = "#111827"    # near-black
TEXT_MUTED  = "#475467"    # muted gray
BORDER_COL  = "rgba(0,0,0,0.10)"
DIVIDER_COL = "rgba(0,0,0,0.12)"
GRID_COL    = "rgba(0,0,0,0.06)"

# Global light styles (escape braces for f-string)
st.markdown(f"""
<style>
:root {{
  --bg-page: {BG_PAGE};
  --bg-surface: {BG_SURFACE};
  --text-main: {TEXT_MAIN};
  --text-muted: {TEXT_MUTED};
  --border-col: {BORDER_COL};
  --divider-col: {DIVIDER_COL};
}}

/* Page bg */
section.main > div.block-container {{
  background: var(--bg-page);
}}

/* Hide sidebar */
[data-testid="stSidebar"] {{ display: none; }}

/* Period card with colored top accent and inner separators (light) */
.kpi-pane {{
  position: relative;
  border: 1px solid var(--border-col);
  border-radius: 12px;
  background: var(--bg-surface);
  padding: 14px 16px 12px 16px;
  height: 150px;
  display: flex; flex-direction: column; justify-content: space-between;
}}
.kpi-pane::before {{
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: var(--accent, #3B82F6);
  border-top-left-radius: 12px; border-top-right-radius: 12px;
}}
.kpi-title {{
  margin: 2px 0 10px 0; font-size: 1.05rem; font-weight: 700; color: var(--text-main);
}}
.kpi-row {{
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; align-items: center;
}}
.kpi-cell {{ text-align: center; padding: 6px 8px; }}
.kpi-cell + .kpi-cell {{ border-left: 1px solid var(--divider-col); }}
.kpi-label {{
  font-size: .80rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px;
}}
.kpi-value {{
  font-size: 1.6rem; font-weight: 700; color: var(--text-main); line-height: 1.1;
}}

/* Headings color */
h1, h2, h3, h4, h5, h6, label, p, span, div, .st-emotion-cache-10trblm {{
  color: var(--text-main);
}}
</style>
""", unsafe_allow_html=True)

# Utility: render KPI group (title + 3 KPI cells) with an accent color
def render_kpi_group(title: str, total_leads: int, conv_rate_pct: float, meetings: int, accent: str):
    conv_txt = f"{conv_rate_pct:.0f}%" if abs(conv_rate_pct - round(conv_rate_pct)) < 1e-9 else f"{conv_rate_pct:.1f}%"
    html = f"""
    <div class="kpi-pane" style="--accent:{accent}">
      <div class="kpi-title">{title}</div>
      <div class="kpi-row">
        <div class="kpi-cell">
          <div class="kpi-label">Total Leads</div>
          <div class="kpi-value">{total_leads:,}</div>
        </div>
        <div class="kpi-cell">
          <div class="kpi-label">Conversion Rate</div>
          <div class="kpi-value">{conv_txt}</div>
        </div>
        <div class="kpi-cell">
          <div class="kpi-label">Meeting Scheduled</div>
          <div class="kpi-value">{meetings:,}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Utility: bordered trend tile using Streamlit native container
def trend_box(fig):
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

# Utility to normalize column names
def norm(df):
    if df is None:
        return None
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()
    return out

# Data loading with Streamlit connection or direct SQLAlchemy fallback
@st.cache_data(show_spinner=False)
def load_data(_=None):
    # 1) Try Streamlit SQL connection
    from streamlit.connections import SQLConnection
    conn = None
    try:
        conn = st.connection("sql", type=SQLConnection)
        info = conn.query("SELECT @@SERVERNAME AS [server], DB_NAME() AS [db]", ttl=60)
        st.caption(f"Connected to {info.iloc[0]['server']} / {info.iloc[0]['db']}")
    except Exception:
        conn = None

    # 2) Fallback to SQLAlchemy using provided creds
    engine = None
    run_sql = None
    if conn is None:
        try:
            import sqlalchemy as sa
            from urllib.parse import quote_plus

            s = st.secrets.get("connections", {}).get("sql", {})
            server   = s.get("server",   "auto.resourceplus.app")
            database = s.get("database", "Data_Lead")
            username = s.get("username", "sa")
            password = s.get("password", "test!serv!123")
            driver   = s.get("driver",   "ODBC Driver 18 for SQL Server")
            encrypt  = s.get("encrypt",  "no")
            tsc      = s.get("TrustServerCertificate", "yes")

            odbc_params = f"driver={quote_plus(driver)}&Encrypt={encrypt}&TrustServerCertificate={tsc}"
            url = f"mssql+pyodbc://{quote_plus(username)}:{quote_plus(password)}@{server}:1433/{quote_plus(database)}?{odbc_params}"
            engine = sa.create_engine(url, fast_executemany=True)

            @st.cache_data(ttl=600, show_spinner=False)
            def _run(sql: str):
                return pd.read_sql(sql, engine)

            # test connectivity
            _ = _run("SELECT @@SERVERNAME AS [server], DB_NAME() AS [db]")
            st.caption(f"Connected to {server} / {database}")
            run_sql = _run
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return {}

    # unified query
    def q(sql: str, ttl=600):
        if conn is not None:
            return conn.query(sql, ttl=ttl)
        return run_sql(sql)

    # table mapping from secrets if present
    tbl_cfg = st.secrets.get("connections", {}).get("sql", {}).get("tables", {})
    def T(key, default):
        return tbl_cfg.get(key, default)

    def fetch(table_fqn, label=None, limit=None):
        label = label or table_fqn
        top = f"TOP {int(limit)} " if limit else ""
        try:
            return q(f"SELECT {top}* FROM {table_fqn}", ttl=600)
        except Exception as e:
            st.error(f"Query failed for {label}: {e}")
            return None

    ds = {}
    ds["leads"]        = fetch(T("leads",        "dbo.Lead"),             "leads")
    ds["agents"]       = fetch(T("agents",       "dbo.Agents"),           "agents")
    ds["calls"]        = fetch(T("calls",        "dbo.LeadCallRecord"),   "calls")
    ds["schedules"]    = fetch(T("schedules",    "dbo.LeadSchedule"),     "schedules")
    ds["transactions"] = fetch(T("transactions", "dbo.LeadTransaction"),  "transactions")

    for k, default in [
        ("countries",                "dbo.Country"),
        ("lead_stages",              "dbo.LeadStage"),
        ("lead_statuses",            "dbo.LeadStatus"),
        ("lead_sources",             "dbo.LeadSource"),
        ("lead_scoring",             "dbo.LeadScoring"),
        ("call_statuses",            "dbo.CallStatus"),
        ("sentiments",               "dbo.CallSentiment"),
        ("task_types",               "dbo.TaskType"),
        ("task_statuses",            "dbo.TaskStatus"),
        ("city_region",              "dbo.CityRegion"),
        ("timezone_info",            "dbo.TimezoneInfo"),
        ("priority",                 "dbo.Priority"),
        ("meeting_status",           "dbo.MeetingStatus"),
        ("agent_meeting_assignment", "dbo.AgentMeetingAssignment"),
    ]:
        ds[k] = fetch(T(k, default), k)

    # normalize
    def norm_lower(df):
        if df is None: return None
        out = df.copy()
        out.columns = out.columns.str.strip().str.replace(r"[^\w]+", "_", regex=True).str.lower()
        return out

    def rename(df, mapping):
        cols = {src: dst for src, dst in mapping.items() if src in df.columns}
        return df.rename(columns=cols)

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
            if col not in df.columns: df[col] = default
        df["CreatedOn"]      = pd.to_datetime(df.get("CreatedOn"), errors="coerce")
        df["EstimatedBudget"]= pd.to_numeric(df.get("EstimatedBudget"), errors="coerce").fillna(0.0)
        ds["leads"] = df

    if ds["agents"] is not None:
        df = norm_lower(ds["agents"])
        df = rename(df, {"agentid":"AgentId","firstname":"FirstName","first_name":"FirstName",
                         "lastname":"LastName","last_name":"LastName","isactive":"IsActive"})
        ds["agents"] = df

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

    if ds["schedules"] is not None:
        df = norm_lower(ds["schedules"])
        df = rename(df, {"scheduleid":"ScheduleId","leadid":"LeadId","tasktypeid":"TaskTypeId",
                         "scheduleddate":"ScheduledDate","taskstatusid":"TaskStatusId",
                         "assignedagentid":"AssignedAgentId","completeddate":"CompletedDate","isfollowup":"IsFollowUp"})
        if "ScheduledDate" in df.columns:
            df["ScheduledDate"] = pd.to_datetime(df["ScheduledDate"], errors="coerce")
        if "CompletedDate" in df.columns:
            df["CompletedDate"] = pd.to_datetime(df["CompletedDate"], errors="coerce")
        ds["schedules"] = df

    if ds["transactions"] is not None:
        df = norm_lower(ds["transactions"])
        df = rename(df, {"transactionid":"TransactionId","leadid":"LeadId","tasktypeid":"TaskTypeId","transactiondate":"TransactionDate"})
        if "TransactionDate" in df.columns:
            df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
        ds["transactions"] = df

    for lk in ["countries","lead_stages","lead_statuses","lead_sources","lead_scoring","call_statuses",
               "sentiments","task_types","task_statuses","city_region","timezone_info","priority","meeting_status",
               "agent_meeting_assignment"]:
        if ds.get(lk) is not None:
            ds[lk] = norm_lower(ds[lk])

    return ds

grain = "Month"
data  = load_data()

# Funnel and markets
def render_funnel_and_markets(d):
    stage_order = ["Lost","Won","Negotiation","Meeting Scheduled","Interested","New"]
    leads      = norm(d.get("leads"))
    statuses   = norm(d.get("lead_statuses"))
    ama        = norm(d.get("agent_meeting_assignment"))
    countries  = norm(d.get("countries"))

    if leads is None or "leadid" not in leads.columns:
        st.info("Leads data unavailable.")
        return

    def ids_from(names):
        if statuses is None or not {"leadstatusid","statusname_e"}.issubset(statuses.columns):
            return set()
        return set(statuses.loc[
            statuses["statusname_e"].str.lower().isin([n.lower() for n in names]),
            "leadstatusid"
        ].dropna().astype(int).unique())

    interested_ids  = ids_from(["Interested","Qualified","Hot","Warm"])
    negotiation_ids = ids_from(["Negotiation","On Hold","Awaiting Budget","Proposal Sent"])
    won_ids         = ids_from(["Won","Closed Won","Contract Signed"])
    lost_ids        = ids_from(["Lost","Closed Lost","Dead","Not Interested"])

    total_leads = len(leads)
    interested_count  = len(leads[leads["leadstatusid"].isin(interested_ids | negotiation_ids | won_ids)])

    meeting_ids = set()
    if ama is not None and {"leadid","meetingstatusid","startdatetime"}.issubset(ama.columns):
        meeting_ids = set(ama.loc[ama["meetingstatusid"].isin({1,6}), "leadid"].dropna().astype(int).unique())

    meeting_count      = len(leads[(leads["leadid"].isin(meeting_ids)) | (leads["leadstatusid"].isin(negotiation_ids | won_ids))])
    negotiation_count  = len(leads[leads["leadstatusid"].isin(negotiation_ids | won_ids)])
    won_count          = len(leads[leads["leadstatusid"].isin(won_ids)])
    lost_count         = len(leads[leads["leadstatusid"].isin(lost_ids)])

    funnel_df = pd.DataFrame([
        {"Stage":"Lost","Count":max(lost_count,1)},
        {"Stage":"Won","Count":max(won_count,1)},
        {"Stage":"Negotiation","Count":max(negotiation_count,1)},
        {"Stage":"Meeting Scheduled","Count":max(meeting_count,1)},
        {"Stage":"Interested","Count":max(interested_count,1)},
        {"Stage":"New","Count":total_leads},
    ])

    fig = px.funnel(
        funnel_df, x="Count", y="Stage",
        category_orders={"Stage":stage_order},
        color_discrete_sequence=[ACCENT_RED,"#34D399","#F59E0B",PRIMARY_GOLD,ACCENT_GREEN,ACCENT_BLUE]
    )
    # Light theme text
    fig.update_traces(textposition="inside", textfont_color=TEXT_MAIN, textfont_size=16, textinfo="value")
    fig.update_layout(
        height=450, margin=dict(l=0,r=0,t=10,b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT_MAIN
    )
    st.plotly_chart(fig, use_container_width=True)

    if countries is not None and "countryid" in leads.columns and "countryname_e" in countries.columns:
        bycountry = (
            leads.groupby("countryid", dropna=True).size().reset_index(name="Leads")
            .merge(countries[["countryid","countryname_e"]].rename(columns={"countryname_e":"Country"}), on="countryid", how="left")
        )
        total = float(bycountry["Leads"].sum())
        bycountry["Share"] = (bycountry["Leads"]/total*100.0).round(1) if total>0 else 0.0
        top5 = bycountry.sort_values(["Share","Leads"], ascending=False).head(5)
        st.subheader("Top markets")
        st.dataframe(
            top5[["Country","Leads","Share"]],
            use_container_width=True, hide_index=True,
            column_config={"Share": st.column_config.ProgressColumn("Share", format="%.1f%%", min_value=0, max_value=100)}
        )
    else:
        st.info("Country data unavailable to build Top markets.")

# Executive summary with KPI panes and bordered trend tiles (date fixes applied)
def show_executive_summary(d):
    leads_all = d.get("leads")
    lead_statuses = d.get("lead_statuses")

    if leads_all is None or len(leads_all) == 0:
        st.info("No data available.")
        return

    # Resolve Won status id safely
    won_status_id = 9
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        m = lead_statuses.loc[lead_statuses["statusname_e"].str.lower() == "won"]
        if not m.empty and "leadstatusid" in m.columns:
            won_status_id = int(m.iloc[0]["leadstatusid"])

    st.subheader("Performance KPIs")

    # Date slicer defaults from data
    c1, c2, c3 = st.columns([1, 1, 2])
    if "CreatedOn" in leads_all.columns:
        all_dates = pd.to_datetime(leads_all["CreatedOn"], errors="coerce").dropna()
        gmin = all_dates.min().date() if len(all_dates) else date.today() - timedelta(days=365)
        gmax = all_dates.max().date() if len(all_dates) else date.today()
    else:
        gmin = date.today() - timedelta(days=365)
        gmax = date.today()

    with c1:
        date_from = st.date_input("From Date", value=gmin, min_value=gmin, max_value=gmax, key="date_from")
    with c2:
        date_to = st.date_input("To Date", value=gmax, min_value=gmin, max_value=gmax, key="date_to")
    with c3:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Apply Date Range", type="primary"):
            st.rerun()

    st.markdown("---")

    # Normalize endpoints and filter once for downstream visuals
    start_day = pd.Timestamp(date_from)
    end_day = pd.Timestamp(date_to)
    if "CreatedOn" in leads_all.columns:
        created = pd.to_datetime(leads_all["CreatedOn"], errors="coerce")
        filtered_leads = leads_all.loc[(created.dt.date >= start_day.date()) & (created.dt.date <= end_day.date())].copy()
    else:
        filtered_leads = leads_all.copy()

    # Helper to clamp a period start to the selected From Date
    def clamp_start(period_start):
        return max(pd.Timestamp(period_start), start_day)

    # Compute period anchors based on To Date (not system today)
    week_start_raw = end_day - pd.Timedelta(days=end_day.weekday())
    month_start_raw = end_day.replace(day=1)
    year_start_raw = end_day.replace(month=1, day=1)

    week_start = clamp_start(week_start_raw)
    month_start = clamp_start(month_start_raw)
    year_start = clamp_start(year_start_raw)

    meetings_all = d.get("agent_meeting_assignment")

    def metrics_between(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
        # Leads in period
        if "CreatedOn" in filtered_leads.columns:
            dt = pd.to_datetime(filtered_leads["CreatedOn"], errors="coerce")
            lp = filtered_leads.loc[(dt >= start_ts) & (dt <= end_ts)].copy()
        else:
            lp = pd.DataFrame()

        # Meetings in period
        if meetings_all is not None and len(meetings_all) > 0:
            m = meetings_all.copy()
            m.columns = m.columns.str.lower()
            dtcol = "startdatetime" if "startdatetime" in m.columns else None
            if dtcol is not None:
                m["dt"] = pd.to_datetime(m[dtcol], errors="coerce")
                m = m[(m["dt"] >= start_ts) & (m["dt"] <= end_ts)]
                if "meetingstatusid" in m.columns:
                    m = m[m["meetingstatusid"].isin({1, 6})]
                mp = m
            else:
                mp = pd.DataFrame()
        else:
            mp = pd.DataFrame()

        total = int(len(lp))
        won = int((lp.get("LeadStatusId", pd.Series(dtype="Int64")) == won_status_id).sum()) if total else 0
        conv = (won / total * 100.0) if total else 0.0
        meet = int(mp["leadid"].nunique()) if "leadid" in mp.columns and len(mp) > 0 else 0
        return total, conv, meet

    # Three panes anchored to To Date and clamped to From Date
    g1, g2, g3 = st.columns(3)
    with g1:
        t, c, m = metrics_between(week_start, end_day)
        render_kpi_group("Week To Date", t, c, m, accent=ACCENT_BLUE)
    with g2:
        t, c, m = metrics_between(month_start, end_day)
        render_kpi_group("Month To Date", t, c, m, accent=ACCENT_GREEN)
    with g3:
        t, c, m = metrics_between(year_start, end_day)
        render_kpi_group("Year To Date", t, c, m, accent=ACCENT_AMBER)

    # Trend at a glance (light layout)
    st.markdown("---")
    st.subheader("Trend at a glance")
    trend_style = st.radio("Trend style", ["Line","Bars","Bullet"], index=0, horizontal=True, key="__trend_style_exec")

    leads_local = filtered_leads.copy()
    if "period" not in leads_local.columns and "CreatedOn" in leads_local.columns:
        dt = pd.to_datetime(leads_local["CreatedOn"], errors="coerce")
        if grain=="Week":
            leads_local["period"] = dt.dt.to_period("W").apply(lambda p: p.start_time.date())
        elif grain=="Month":
            leads_local["period"] = dt.dt.to_period("M").apply(lambda p: p.start_time.date())
        else:
            leads_local["period"] = dt.dt.to_period("Y").apply(lambda p: p.start_time.date())

    leads_ts = leads_local.groupby("period").size().reset_index(name="value") if "period" in leads_local.columns else pd.DataFrame({"period":[],"value":[]})
    if "LeadStatusId" in leads_local.columns and "period" in leads_local.columns:
        per_total = leads_local.groupby("period").size()
        per_won   = leads_local.loc[leads_local["LeadStatusId"].eq(won_status_id)].groupby("period").size()
        conv_ts   = pd.DataFrame({"period": per_total.index, "total": per_total.values}).merge(
            pd.DataFrame({"period": per_won.index, "won": per_won.values}), on="period", how="left"
        ).fillna(0.0)
        conv_ts["value"] = (conv_ts["won"]/conv_ts["total"]*100).round(1)
    else:
        conv_ts = pd.DataFrame({"period":[],"value":[]})

    meetings = d.get("agent_meeting_assignment")
    if meetings is not None and len(meetings)>0:
        m = meetings.copy(); m.columns = m.columns.str.lower()
        date_col = "startdatetime" if "startdatetime" in m.columns else None
        if date_col is not None:
            m["dt"] = pd.to_datetime(m[date_col], errors="coerce")
            if grain=="Week":
                m["period"] = m["dt"].dt.to_period("W").apply(lambda p: p.start_time.date())
            elif grain=="Month":
                m["period"] = m["dt"].dt.to_period("M").apply(lambda p: p.start_time.date())
            else:
                m["period"] = m["dt"].dt.to_period("Y").apply(lambda p: p.start_time.date())
            if "meetingstatusid" in m.columns: m = m[m["meetingstatusid"].isin({1,6})]
            meet_ts = m.groupby("period")["leadid"].nunique().reset_index(name="value")
        else:
            meet_ts = pd.DataFrame({"period":[],"value":[]})
    else:
        meet_ts = pd.DataFrame({"period":[],"value":[]})

    def _index(df):
        df = df.copy()
        if df.empty: df["idx"] = []; return df
        base = df["value"].iloc[0] if df["value"].iloc[0]!=0 else 1.0
        df["idx"] = (df["value"]/base)*100.0
        return df

    leads_ts = _index(leads_ts)
    conv_ts  = _index(conv_ts)
    meet_ts  = _index(meet_ts)

    def _apply_axes(fig, ys, title):
        ymin = float(pd.Series(ys).min()) if len(ys) else 0
        ymax = float(pd.Series(ys).max()) if len(ys) else 1
        pad  = max(1.0, (ymax-ymin)*0.12)
        fig.update_layout(
            height=180,
            title=dict(text=title, x=0.01, font=dict(size=12, color=TEXT_MUTED)),
            margin=dict(l=6,r=6,t=24,b=8),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_MAIN,
            showlegend=False
        )
        fig.update_xaxes(showgrid=True, gridcolor=GRID_COL, tickfont=dict(color=TEXT_MUTED, size=10), nticks=6, ticks="outside")
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COL, tickfont=dict(color=TEXT_MUTED, size=10), nticks=4, ticks="outside", range=[ymin-pad, ymax+pad])
        return fig

    def tile_line(df, color, title):
        df = df.dropna().sort_values("period")
        if len(df)==0:
            fig = go.Figure(); fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED))
            return _apply_axes(fig,[0,1],title)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["period"], y=df["idx"], mode="lines+markers",
                                 line=dict(color=color, width=3, shape="spline"), marker=dict(size=6, color=color)))
        return _apply_axes(fig, df["idx"], title)

    def tile_bar(df, color, title):
        df = df.dropna().sort_values("period")
        if len(df)==0:
            fig = go.Figure(); fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED))
            return _apply_axes(fig,[0,1],title)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["period"], y=df["idx"],
                             marker=dict(color=color, line=dict(color="rgba(0,0,0,0.08)", width=0.5)), opacity=0.95))
        return _apply_axes(fig, df["idx"], title)

    def tile_bullet(df, title, bar_color):
        if df.empty or len(df)==0:
            fig = go.Figure(); fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED))
            return _apply_axes(fig,[0,1],title)
        cur = float(df["idx"].iloc[-1])
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta", value=cur, number={'valueformat': ".0f", 'font': {'color': TEXT_MAIN}},
            delta={'reference': 100},
            gauge={{'shape':"bullet",'axis':{{'range':[80,120]}},'steps':[
                    {{'range':[80,95], 'color':"rgba(239,68,68,0.20)"}},
                    {{'range':[95,105],'color':"rgba(234,179,8,0.20)"}},
                    {{'range':[105,120],'color':"rgba(34,197,94,0.20)"}}
                ],
                'bar':{{'color':bar_color}},
                'threshold':{{'line':{{'color':'#111827','width':2}}, 'value':100}}
            }}
        ))
        fig.update_layout(height=120, margin=dict(l=8,r=8,t=26,b=8), paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT_MAIN)
        return fig

    s1, s2, s3 = st.columns(3)
    if trend_style=="Line":
        with s1: trend_box(tile_line(leads_ts, ACCENT_BLUE,   "Leads trend (indexed)"))
        with s2: trend_box(tile_line(conv_ts,  ACCENT_GREEN,  "Conversion rate (indexed)"))
        with s3: trend_box(tile_line(meet_ts,  PRIMARY_GOLD,  "Meeting scheduled (indexed)"))
    elif trend_style=="Bars":
        with s1: trend_box(tile_bar(leads_ts, ACCENT_BLUE,   "Leads trend (indexed)"))
        with s2: trend_box(tile_bar(conv_ts,  ACCENT_GREEN,  "Conversion rate (indexed)"))
        with s3: trend_box(tile_bar(meet_ts,  PRIMARY_GOLD,  "Meeting scheduled (indexed)"))
    else:
        with s1: trend_box(tile_bullet(leads_ts, "Leads index", ACCENT_BLUE))
        with s2: trend_box(tile_bullet(conv_ts,  "Conversion index", ACCENT_GREEN))
        with s3: trend_box(tile_bullet(meet_ts,  "Meetings index", PRIMARY_GOLD))

    st.markdown("---")
    st.subheader("Lead conversion snapshot")
    d2 = dict(d)
    d2["leads"] = filtered_leads
    render_funnel_and_markets(d2)

# Lead Status page
def show_lead_status(d):
    leads  = d.get("leads")
    stats  = d.get("lead_statuses")
    calls  = d.get("calls")
    meets  = d.get("agent_meeting_assignment")

    if leads is None or len(leads)==0:
        st.info("No lead status data in the selected range.")
        return

    name_map = {}
    if stats is not None and {"leadstatusid","statusname_e"}.issubset(stats.columns):
        name_map = dict(zip(stats["leadstatusid"].astype(int), stats["statusname_e"].astype(str)))

    L = leads.copy()
    L["Status"]    = L["LeadStatusId"].map(name_map).fillna(L.get("LeadStatusId", pd.Series(dtype="Int64")).astype(str))
    L["CreatedOn"] = pd.to_datetime(L.get("CreatedOn"), errors="coerce")
    cutoff         = L["CreatedOn"].max() if "CreatedOn" in L.columns else pd.Timestamp.today()
    L["age_days"]  = (cutoff - L["CreatedOn"]).dt.days.astype("Int64")

    counts = L["Status"].value_counts().reset_index()
    counts.columns = ["Status","count"]
    c1, c2 = st.columns([2,1])
    with c1:
        fig = px.pie(counts, names="Status", values="count", hole=0.35, color_discrete_sequence=px.colors.sequential.Viridis,
                     title="Lead Status Share")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT_MAIN)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Total Leads", f"{len(L):,}")
        won_id = None
        if stats is not None and "statusname_e" in stats.columns:
            m = stats.loc[stats["statusname_e"].str.lower()=="won"]
            if not m.empty: won_id = int(m.iloc[0]["leadstatusid"])
        won = int((L.get("LeadStatusId", pd.Series(dtype="Int64")).astype("Int64")==won_id).sum()) if won_id is not None else 0
        st.metric("Won", f"{won:,}")

    st.markdown("---")
    st.subheader("Lead Distribution Status")
    dist_sorted = counts.sort_values("count", ascending=False)
    fig_bar = px.bar(dist_sorted, x="Status", y="count", title="Leads by status",
                     color="Status", color_discrete_sequence=px.colors.qualitative.Set3, text="count")
    fig_bar.update_traces(textposition='outside', textfont_size=12)
    fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color=TEXT_MAIN, height=400, showlegend=False,
                          margin=dict(l=0,r=0,t=40,b=0), xaxis_title="Status", yaxis_title="Count")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Lead Breakdown")

    meet_rate = pd.DataFrame({"Status":pd.Series(dtype="str"), "meet_leads":pd.Series(dtype="float")})
    if meets is not None and len(meets):
        M = meets.copy(); M.columns = M.columns.str.lower()
        dtc = "startdatetime" if "startdatetime" in M.columns else None
        if dtc is not None:
            if "meetingstatusid" in M.columns: M = M[M["meetingstatusid"].isin({1,6})]
            mm = M.merge(L[["LeadId","Status"]], left_on="leadid", right_on="LeadId", how="left")
            meet_rate = mm.groupby("Status")["leadid"].nunique().reset_index(name="meet_leads")

    conn_rate = pd.DataFrame({"Status":pd.Series(dtype="str"), "connect_rate":pd.Series(dtype="float")})
    if calls is not None and len(calls):
        C = calls.copy()
        C["CallDateTime"] = pd.to_datetime(C.get("CallDateTime"), errors="coerce")
        C = C.merge(L[["LeadId","Status"]], on="LeadId", how="left")
        g = C.groupby("Status").agg(total=("LeadCallId","count"),
                                    connects=("CallStatusId", lambda s: (s==1).sum())).reset_index()
        g["connect_rate"] = (g["connects"]/g["total"]).fillna(0.0)
        conn_rate = g[["Status","connect_rate"]]

    base = L.groupby("Status").agg(
        Leads=("LeadId","count"),
        Avg_Age_Days=("age_days","mean"),
        Pipeline=("EstimatedBudget","sum")
    ).reset_index()
    total_leads = float(base["Leads"].sum()) if len(base) else 0.0
    base["Share_%"] = (base["Leads"]/total_leads*100.0).round(1) if total_leads>0 else 0.0

    breakdown = (base.merge(meet_rate, on="Status", how="left")
                      .merge(conn_rate, on="Status", how="left"))
    breakdown["meet_leads"]     = breakdown["meet_leads"].fillna(0.0)
    breakdown["Meeting_Rate_%"] = (breakdown["meet_leads"]/breakdown["Leads"]*100.0).replace([np.inf,-np.inf],0).fillna(0.0).round(1)
    breakdown["connect_rate"]   = breakdown["connect_rate"].fillna(0.0).round(2)
    breakdown["Avg_Age_Days"]   = breakdown["Avg_Age_Days"].fillna(0.0).round(1)
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

# Dataset wrapper for pages
fdata = {
    "leads": data.get("leads"),
    "lead_statuses": data.get("lead_statuses"),
    "agent_meeting_assignment": data.get("agent_meeting_assignment"),
    "countries": data.get("countries"),
    "calls": data.get("calls"),
}

# Navigation
NAV = [
    ("Executive", "speedometer2", "🎯 Executive Summary"),
    ("Lead Status", "people", "📈 Lead Status"),
]

if HAS_OPTION_MENU:
    selected = option_menu(
        None, [n[0] for n in NAV], icons=[n[1] for n in NAV],
        orientation="horizontal", default_index=0,
        styles={
            "container": {"padding":"0!important","background-color": BG_PAGE},
            "icon": {"color": PRIMARY_GOLD, "font-size": "16px"},
            "nav-link": {"font-size": "14px", "color": TEXT_MUTED, "--hover-color": "#EEF2FF"},
            "nav-link-selected": {"background-color": BG_SURFACE, "color": TEXT_MAIN, "border-bottom": f"2px solid {PRIMARY_GOLD}"},
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
```
