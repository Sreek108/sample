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
ACCENT_BLUE  = "#1E90FF"
ACCENT_GREEN = "#32CD32"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED   = "#DC143C"

BG_PAGE     = "#F6F7FB"
BG_SURFACE  = "#FFFFFF"
TEXT_MAIN   = "#111827"
TEXT_MUTED  = "#475467"
BORDER_COL  = "rgba(0,0,0,0.10)"
DIVIDER_COL = "rgba(0,0,0,0.12)"
GRID_COL    = "rgba(0,0,0,0.06)"

# Global styles
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

section.main > div.block-container {{
  background: var(--bg-page);
}}

[data-testid="stSidebar"] {{ display: none; }}

h1, h2, h3, h4, h5, h6, label, p, span, div, .st-emotion-cache-10trblm {{
  color: var(--text-main);
}}

/* KPI Metric Cards */
.metric-card {{
    background: white;
    padding: 20px 16px;
    border-radius: 8px;
    border: 1px solid #E5E7EB;
    text-align: center;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: box-shadow 0.2s;
}}
.metric-card:hover {{
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
.metric-label {{
    font-size: 0.875rem;
    color: #6B7280;
    font-weight: 500;
    margin-bottom: 8px;
    text-transform: capitalize;
}}
.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    line-height: 1;
}}
.period-header {{
    font-size: 0.95rem;
    font-weight: 600;
    color: #374151;
    margin: 20px 0 12px 0;
    padding-left: 4px;
}}
</style>
""", unsafe_allow_html=True)

# Utility functions
def trend_box(fig):
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)

def norm(df):
    if df is None:
        return None
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()
    return out

# Database connection
def get_connection():
    """Establish database connection (Streamlit SQL or SQLAlchemy)"""
    from streamlit.connections import SQLConnection
    
    try:
        conn = st.connection("sql", type=SQLConnection)
        info = conn.query("SELECT @@SERVERNAME AS [server], DB_NAME() AS [db]", ttl=60)
        st.caption(f"✅ Connected to {info.iloc[0]['server']} / {info.iloc[0]['db']}")
        return conn, None
    except Exception:
        pass
    
    # Fallback to SQLAlchemy
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
        
        @st.cache_data(ttl=300, show_spinner=False)
        def _run(sql: str):
            return pd.read_sql(sql, engine)
        
        _ = _run("SELECT @@SERVERNAME AS [server], DB_NAME() AS [db]")
        st.caption(f"✅ Connected to {server} / {database}")
        return None, _run
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        return None, None

# Optimized data loading
@st.cache_data(ttl=3600, show_spinner=False)
def load_lookup_tables(_conn, _runner):
    """Load static reference tables (cached for 1 hour)"""
    
    def q(sql):
        if _conn:
            return _conn.query(sql, ttl=3600)
        return _runner(sql)
    
    return {
        "countries": q("""
            SELECT CountryId, CountryName_E 
            FROM dbo.Country 
            WHERE IsActive = 1
        """),
        "lead_statuses": q("""
            SELECT LeadStatusId, StatusName_E 
            FROM dbo.LeadStatus 
            WHERE IsActive = 1
        """),
        "lead_stages": q("""
            SELECT LeadStageId, StageName_E 
            FROM dbo.LeadStage 
            WHERE IsActive = 1
        """),
        "meeting_status": q("""
            SELECT MeetingStatusId, StatusName_E 
            FROM dbo.MeetingStatus 
            WHERE IsActive = 1
        """),
        "call_statuses": q("""
            SELECT CallStatusId, StatusName_E 
            FROM dbo.CallStatus 
            WHERE IsActive = 1
        """),
    }

@st.cache_data(ttl=60, show_spinner=False)
def load_transactional_data(_conn, _runner):
    """Load frequently changing data (cached for 1 minute)"""
    
    def q(sql):
        if _conn:
            return _conn.query(sql, ttl=60)
        return _runner(sql)
    
    leads = q("""
        SELECT 
            LeadId, LeadCode, LeadStageId, LeadStatusId, LeadScoringId,
            AssignedAgentId, CountryId, CityRegionId, CreatedOn, IsActive
        FROM dbo.Lead 
        WHERE IsActive = 1
    """)
    
    meetings = q("""
        SELECT 
            AssignmentId, LeadId, StartDateTime, EndDateTime, 
            MeetingStatusId, AgentId
        FROM dbo.AgentMeetingAssignment
    """)
    
    calls = q("""
        SELECT 
            LeadCallId, LeadId, CallDateTime, DurationSeconds,
            CallStatusId, SentimentId, AssignedAgentId, CallDirection
        FROM dbo.LeadCallRecord
    """)
    
    return {
        "leads": leads,
        "agent_meeting_assignment": meetings,
        "calls": calls,
    }

# Initialize connection
conn, runner = get_connection()

if conn is None and runner is None:
    st.stop()

# Load data
lookups = load_lookup_tables(conn, runner)
transactions = load_transactional_data(conn, runner)
data = {**lookups, **transactions}

# Normalize all dataframes
for key in data:
    if data[key] is not None:
        df = data[key].copy()
        df.columns = df.columns.str.strip().str.lower()
        
        if key == "leads":
            df = df.rename(columns={
                "leadid": "LeadId", "leadcode": "LeadCode", "leadstageid": "LeadStageId",
                "leadstatusid": "LeadStatusId", "leadscoringid": "LeadScoringId",
                "assignedagentid": "AssignedAgentId", "createdon": "CreatedOn",
                "isactive": "IsActive", "countryid": "CountryId", "cityregionid": "CityRegionId",
            })
            df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], errors="coerce")
            
        elif key == "agent_meeting_assignment":
            df = df.rename(columns={
                "assignmentid": "AssignmentId", "leadid": "LeadId",
                "startdatetime": "StartDateTime", "enddatetime": "EndDateTime",
                "meetingstatusid": "MeetingStatusId", "agentid": "AgentId",
            })
            df["StartDateTime"] = pd.to_datetime(df["StartDateTime"], errors="coerce")
            
        elif key == "calls":
            df = df.rename(columns={
                "leadcallid": "LeadCallId", "leadid": "LeadId",
                "calldatetime": "CallDateTime", "durationseconds": "DurationSeconds",
                "callstatusid": "CallStatusId", "sentimentid": "SentimentId",
                "assignedagentid": "AssignedAgentId", "calldirection": "CallDirection",
            })
            df["CallDateTime"] = pd.to_datetime(df["CallDateTime"], errors="coerce")
            
        elif key == "countries":
            df = df.rename(columns={"countryid": "CountryId", "countryname_e": "CountryName_E"})
            
        elif key == "lead_statuses":
            df = df.rename(columns={"leadstatusid": "LeadStatusId", "statusname_e": "StatusName_E"})
        
        data[key] = df

grain = "Month"

# Funnel and markets
def render_funnel_and_markets(d):
    stage_order = ["Lost","Won","Negotiation","Meeting Scheduled","Interested","New"]
    leads      = d.get("leads")
    statuses   = d.get("lead_statuses")
    ama        = d.get("agent_meeting_assignment")
    countries  = d.get("countries")

    if leads is None or "LeadId" not in leads.columns:
        st.info("Leads data unavailable.")
        return

    def ids_from(names):
        if statuses is None or "StatusName_E" not in statuses.columns:
            return set()
        return set(statuses.loc[
            statuses["StatusName_E"].str.lower().isin([n.lower() for n in names]),
            "LeadStatusId"
        ].dropna().astype(int).unique())

    interested_ids  = ids_from(["Interested","Qualified","Hot","Warm"])
    negotiation_ids = ids_from(["Negotiation","On Hold","Awaiting Budget","Proposal Sent"])
    won_ids         = ids_from(["Won","Closed Won","Contract Signed"])
    lost_ids        = ids_from(["Lost","Closed Lost","Dead","Not Interested"])

    total_leads = len(leads)
    interested_count  = len(leads[leads["LeadStatusId"].isin(interested_ids | negotiation_ids | won_ids)])

    meeting_ids = set()
    if ama is not None and "LeadId" in ama.columns:
        meeting_ids = set(ama.loc[ama["MeetingStatusId"].isin({1,6}), "LeadId"].dropna().astype(int).unique())

    meeting_count      = len(leads[(leads["LeadId"].isin(meeting_ids)) | (leads["LeadStatusId"].isin(negotiation_ids | won_ids))])
    negotiation_count  = len(leads[leads["LeadStatusId"].isin(negotiation_ids | won_ids)])
    won_count          = len(leads[leads["LeadStatusId"].isin(won_ids)])
    lost_count         = len(leads[leads["LeadStatusId"].isin(lost_ids)])

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
    fig.update_traces(textposition="inside", textfont_color=TEXT_MAIN, textfont_size=16, textinfo="value")
    fig.update_layout(
        height=450, margin=dict(l=0,r=0,t=10,b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT_MAIN
    )
    st.plotly_chart(fig, use_container_width=True)

    if countries is not None and "CountryId" in leads.columns and "CountryName_E" in countries.columns:
        bycountry = (
            leads.groupby("CountryId", dropna=True).size().reset_index(name="Leads")
            .merge(countries[["CountryId","CountryName_E"]].rename(columns={"CountryName_E":"Country"}), on="CountryId", how="left")
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
        st.info("Country data unavailable.")

# Executive summary
def show_executive_summary(d):
    leads_all = d.get("leads")
    lead_statuses = d.get("lead_statuses")

    if leads_all is None or len(leads_all) == 0:
        st.info("No data available.")
        return

    # Resolve Won status id
    won_status_id = 9
    if lead_statuses is not None and "StatusName_E" in lead_statuses.columns:
        m = lead_statuses.loc[lead_statuses["StatusName_E"].str.lower() == "won"]
        if not m.empty and "LeadStatusId" in m.columns:
            won_status_id = int(m.iloc[0]["LeadStatusId"])

    st.subheader("Performance KPIs")

    # Date slicer
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

    # WTD/MTD/YTD calculations
    today = pd.Timestamp.today().normalize()
    week_start_wtd = today - pd.Timedelta(days=today.weekday())
    month_start_mtd = today.replace(day=1)
    year_start_ytd = today.replace(month=1, day=1)

    meetings_all = d.get("agent_meeting_assignment")

    def metrics_full_dataset(start_ts: pd.Timestamp, end_ts: pd.Timestamp):
        if "CreatedOn" in leads_all.columns:
            dt = pd.to_datetime(leads_all["CreatedOn"], errors="coerce")
            lp = leads_all.loc[(dt >= start_ts) & (dt <= end_ts)].copy()
        else:
            lp = pd.DataFrame()

        if meetings_all is not None and len(meetings_all) > 0:
            m = meetings_all.copy()
            if "StartDateTime" in m.columns:
                m["dt"] = pd.to_datetime(m["StartDateTime"], errors="coerce")
                m = m[(m["dt"] >= start_ts) & (m["dt"] <= end_ts)]
                if "MeetingStatusId" in m.columns:
                    m = m[m["MeetingStatusId"].isin({1, 6})]
                mp = m
            else:
                mp = pd.DataFrame()
        else:
            mp = pd.DataFrame()

        total = int(len(lp))
        won = int((lp.get("LeadStatusId", pd.Series(dtype="Int64")) == won_status_id).sum()) if total else 0
        conv_pct = (won / total * 100.0) if total else 0.0
        meet = int(mp["LeadId"].nunique()) if "LeadId" in mp.columns and len(mp) > 0 else 0
        return total, conv_pct, meet, won

    # Get metrics
    week_total, week_conv, week_meet, week_won = metrics_full_dataset(week_start_wtd, today)
    month_total, month_conv, month_meet, month_won = metrics_full_dataset(month_start_mtd, today)
    year_total, year_conv, year_meet, year_won = metrics_full_dataset(year_start_ytd, today)

    # Format large numbers
    def format_number(num):
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return f"{num:,}"

    # Week row
    st.markdown('<div class="period-header">📅 Week To Date</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Leads</div>
            <div class="metric-value">{format_number(week_total)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conversion Rate</div>
            <div class="metric-value">{week_conv:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Meeting Scheduled</div>
            <div class="metric-value">{format_number(week_meet)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conversion Count</div>
            <div class="metric-value">{format_number(week_won)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Month row
    st.markdown('<div class="period-header">📅 Month To Date</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Leads</div>
            <div class="metric-value">{format_number(month_total)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conversion Rate</div>
            <div class="metric-value">{month_conv:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Meeting Scheduled</div>
            <div class="metric-value">{format_number(month_meet)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conversion Count</div>
            <div class="metric-value">{format_number(month_won)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Year row
    st.markdown('<div class="period-header">📅 Year To Date</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Leads</div>
            <div class="metric-value">{format_number(year_total)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conversion Rate</div>
            <div class="metric-value">{year_conv:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Meeting Scheduled</div>
            <div class="metric-value">{format_number(year_meet)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conversion Count</div>
            <div class="metric-value">{format_number(year_won)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Trends
    st.markdown("---")
    st.subheader("Trend at a glance")
    trend_style = st.radio("Trend style", ["Line","Bars","Bullet"], index=0, horizontal=True, key="__trend_style_exec")

    start_day = pd.Timestamp(date_from)
    end_day = pd.Timestamp(date_to)
    if "CreatedOn" in leads_all.columns:
        created = pd.to_datetime(leads_all["CreatedOn"], errors="coerce")
        filtered_leads = leads_all.loc[(created.dt.date >= start_day.date()) & (created.dt.date <= end_day.date())].copy()
    else:
        filtered_leads = leads_all.copy()

    leads_local = filtered_leads.copy()
    if "CreatedOn" in leads_local.columns:
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
        m = meetings.copy()
        if "StartDateTime" in m.columns:
            m["dt"] = pd.to_datetime(m["StartDateTime"], errors="coerce")
            if grain=="Week":
                m["period"] = m["dt"].dt.to_period("W").apply(lambda p: p.start_time.date())
            elif grain=="Month":
                m["period"] = m["dt"].dt.to_period("M").apply(lambda p: p.start_time.date())
            else:
                m["period"] = m["dt"].dt.to_period("Y").apply(lambda p: p.start_time.date())
            if "MeetingStatusId" in m.columns: m = m[m["MeetingStatusId"].isin({1,6})]
            meet_ts = m.groupby("period")["LeadId"].nunique().reset_index(name="value")
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
            mode="number+gauge+delta",
            value=cur,
            number={'valueformat': ".0f", 'font': {'color': TEXT_MAIN}},
            delta={'reference': 100},
            gauge={
                'shape': "bullet",
                'axis': {'range': [80, 120]},
                'steps': [
                    {'range': [80, 95], 'color': "rgba(239,68,68,0.20)"},
                    {'range': [95, 105], 'color': "rgba(234,179,8,0.20)"},
                    {'range': [105, 120], 'color': "rgba(34,197,94,0.20)"},
                ],
                'bar': {'color': bar_color},
                'threshold': {'line': {'color': '#111827', 'width': 2}, 'value': 100}
            }
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
        st.info("No lead status data.")
        return

    name_map = {}
    if stats is not None and "StatusName_E" in stats.columns:
        name_map = dict(zip(stats["LeadStatusId"].astype(int), stats["StatusName_E"].astype(str)))

    L = leads.copy()
    L["Status"] = L["LeadStatusId"].map(name_map).fillna(L.get("LeadStatusId", pd.Series(dtype="Int64")).astype(str))
    L["CreatedOn"] = pd.to_datetime(L.get("CreatedOn"), errors="coerce")
    cutoff = L["CreatedOn"].max() if "CreatedOn" in L.columns else pd.Timestamp.today()
    L["age_days"] = (cutoff - L["CreatedOn"]).dt.days.astype("Int64")

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
        if stats is not None and "StatusName_E" in stats.columns:
            m = stats.loc[stats["StatusName_E"].str.lower()=="won"]
            if not m.empty: won_id = int(m.iloc[0]["LeadStatusId"])
        won = int((L.get("LeadStatusId", pd.Series(dtype="Int64")).astype("Int64")==won_id).sum()) if won_id is not None else 0
        st.metric("Won", f"{won:,}")

    st.markdown("---")
    st.subheader("Lead Distribution Status")

    dark_palette = ["#1f2937","#374151","#4b5563","#111827","#0f766e","#7c2d12","#1e3a8a","#065f46","#6b21a8","#7f1d1d"]

    dist_sorted = counts.sort_values("count", ascending=False)
    fig_bar = px.bar(dist_sorted, x="Status", y="count", title="Leads by status",
                     color="Status", color_discrete_sequence=dark_palette, text="count")
    fig_bar.update_traces(textposition='outside', textfont_size=12, marker_line=dict(color="rgba(255,255,255,0.6)", width=0.5))
    fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color=TEXT_MAIN, height=400, showlegend=False,
                          margin=dict(l=0, r=0, t=40, b=0), xaxis_title="Status", yaxis_title="Count")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Lead Breakdown")

    meet_rate = pd.DataFrame({"Status":pd.Series(dtype="str"), "meet_leads":pd.Series(dtype="float")})
    if meets is not None and len(meets):
        M = meets.copy()
        if "MeetingStatusId" in M.columns and "LeadId" in M.columns:
            M = M[M["MeetingStatusId"].isin({1,6})]
            mm = M.merge(L[["LeadId","Status"]], on="LeadId", how="left")
            meet_rate = mm.groupby("Status")["LeadId"].nunique().reset_index(name="meet_leads")

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
        Avg_Age_Days=("age_days","mean")
    ).reset_index()
    
    total_leads = float(base["Leads"].sum()) if len(base) else 0.0
    base["Share_%"] = (base["Leads"]/total_leads*100.0).round(1) if total_leads>0 else 0.0

    breakdown = (base.merge(meet_rate, on="Status", how="left")
                      .merge(conn_rate, on="Status", how="left"))
    breakdown["meet_leads"]     = breakdown["meet_leads"].fillna(0.0)
    breakdown["Meeting_Rate_%"] = (breakdown["meet_leads"]/breakdown["Leads"]*100.0).replace([np.inf,-np.inf],0).fillna(0.0).round(1)
    breakdown["connect_rate"]   = breakdown["connect_rate"].fillna(0.0).round(2)
    breakdown["Avg_Age_Days"]   = breakdown["Avg_Age_Days"].fillna(0.0).round(1)
    breakdown = breakdown.sort_values("Leads", ascending=False)

    st.dataframe(
        breakdown[["Status","Leads","Share_%","Avg_Age_Days","Meeting_Rate_%","connect_rate"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Leads": st.column_config.NumberColumn("Leads", format="%,d"),
            "Share_%": st.column_config.ProgressColumn("Share", min_value=0.0, max_value=100.0, format="%.1f%%"),
            "Avg_Age_Days": st.column_config.NumberColumn("Avg age (days)", format="%.1f"),
            "Meeting_Rate_%": st.column_config.ProgressColumn("Meeting rate", min_value=0.0, max_value=100.0, format="%.1f%%"),
            "connect_rate": st.column_config.ProgressColumn("Connect rate", min_value=0.0, max_value=1.0, format="%.2f"),
        }
    )

# Navigation
fdata = data

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
