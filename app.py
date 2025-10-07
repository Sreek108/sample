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

# Global styles with Inter font
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --bg-page: {BG_PAGE};
  --bg-surface: {BG_SURFACE};
  --text-main: {TEXT_MAIN};
  --text-muted: {TEXT_MUTED};
  --border-col: {BORDER_COL};
  --divider-col: {DIVIDER_COL};
}}

* {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}}

section.main > div.block-container {{
  background: var(--bg-page);
}}

[data-testid="stSidebar"] {{ display: none; }}

h1, h2, h3, h4, h5, h6, label, p, span, div, .st-emotion-cache-10trblm {{
  color: var(--text-main);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}}

/* KPI Metric Cards - Professional Dashboard Style with Inter Font */
.metric-card {{
    background: transparent;
    padding: 16px 12px;
    border-radius: 0px;
    border: none;
    text-align: center;
    min-height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}}
.metric-label {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    font-size: 0.8125rem;
    color: #9CA3AF;
    font-weight: 400;
    margin-bottom: 6px;
    text-transform: none;
    letter-spacing: 0;
}}
.metric-value {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    font-size: 1.875rem;
    font-weight: 600;
    color: #111827;
    line-height: 1.2;
    font-feature-settings: 'tnum' 1;
}}
.period-header {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #374151;
    margin: 16px 0 8px 0;
    padding-left: 0px;
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
            SELECT LeadStageId, StageName_E, SortOrder 
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
            AssignedAgentId, CountryId, CityRegionId, CreatedOn, ModifiedOn, IsActive
        FROM dbo.Lead 
        WHERE IsActive = 1
    """)
    
    meetings = q("""
        SELECT 
            AssignmentId, LeadId, StartDateTime, EndDateTime, 
            MeetingStatusId, AgentId
        FROM dbo.AgentMeetingAssignment 
        WHERE MeetingStatusId IN (1, 6)
    """)
    
    calls = q("""
        SELECT 
            LeadCallId, LeadId, CallDateTime, DurationSeconds,
            CallStatusId, SentimentId, AssignedAgentId, CallDirection
        FROM dbo.LeadCallRecord 
        WHERE CallDateTime >= DATEADD(MONTH, -12, GETDATE())
    """)
    
    stage_audit = q("""
        SELECT 
            AuditId, LeadId, StageId, CreatedOn
        FROM dbo.LeadStageAudit 
        WHERE CreatedOn >= DATEADD(MONTH, -12, GETDATE())
    """)
    
    return {
        "leads": leads,
        "agent_meeting_assignment": meetings,
        "calls": calls,
        "lead_stage_audit": stage_audit,
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
                "modifiedon": "ModifiedOn", "isactive": "IsActive", 
                "countryid": "CountryId", "cityregionid": "CityRegionId",
            })
            df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], errors="coerce")
            df["ModifiedOn"] = pd.to_datetime(df["ModifiedOn"], errors="coerce")
            
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
            
        elif key == "lead_stage_audit":
            df = df.rename(columns={
                "auditid": "AuditId", "leadid": "LeadId",
                "stageid": "StageId", "createdon": "CreatedOn"
            })
            df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], errors="coerce")
            
        elif key == "countries":
            df = df.rename(columns={"countryid": "CountryId", "countryname_e": "CountryName_E"})
            
        elif key == "lead_statuses":
            df = df.rename(columns={"leadstatusid": "LeadStatusId", "statusname_e": "StatusName_E"})
        
        elif key == "lead_stages":
            df = df.rename(columns={
                "leadstageid": "LeadStageId", 
                "stagename_e": "StageName_E",
                "sortorder": "SortOrder"
            })
        
        data[key] = df

grain = "Month"

# Funnel and markets
def render_funnel_and_markets(d):
    leads      = d.get("leads")
    stages     = d.get("lead_stages")
    audit      = d.get("lead_stage_audit")
    countries  = d.get("countries")

    if leads is None or "LeadId" not in leads.columns:
        st.info("Leads data unavailable.")
        return

    total_leads = len(leads)

    # Use LeadStageAudit for funnel (unique leads per stage)
    if audit is not None and stages is not None and "StageId" in audit.columns:
        funnel_query = audit.merge(
            stages[["LeadStageId", "StageName_E", "SortOrder"]],
            left_on="StageId",
            right_on="LeadStageId",
            how="left"
        )
        
        funnel_df = (
            funnel_query.groupby(["SortOrder", "StageName_E"], as_index=False)["LeadId"]
            .nunique()
            .rename(columns={"LeadId": "Count"})
            .sort_values("SortOrder", ascending=True)
        )
        
        funnel_df = funnel_df[funnel_df["StageName_E"].str.lower() != "lost"]
        funnel_df["Stage"] = funnel_df["StageName_E"]
        
    else:
        st.info("LeadStageAudit unavailable - showing basic funnel")
        funnel_df = pd.DataFrame([{"Stage": "New", "Count": total_leads, "SortOrder": 1}])

    # Create funnel chart
    fig = go.Figure(go.Funnel(
        name='Sales Funnel',
        y=funnel_df['Stage'],
        x=funnel_df['Count'],
        textposition="inside",
        textinfo="value+percent initial",
        textfont=dict(color="white", size=16, family="Inter"),
        marker={
            "color": ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"][:len(funnel_df)],
            "line": {"width": 2, "color": "white"}
        },
        connector={"line": {"color": "#34495e", "width": 3}}
    ))
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)", 
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT_MAIN, 
        font_family="Inter",
        title={
            'text': f"Sales Funnel - From {total_leads:,} Total Leads",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': TEXT_MAIN, 'family': 'Inter'}
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Top markets
    if countries is not None and "CountryId" in leads.columns and "CountryName_E" in countries.columns:
        bycountry = (
            leads.groupby("CountryId", dropna=True, as_index=False).size()
            .rename(columns={"size": "Leads"})
            .merge(countries[["CountryId","CountryName_E"]].rename(columns={"CountryName_E":"Country"}), 
                   on="CountryId", how="left")
        )
        total = float(bycountry["Leads"].sum())
        bycountry["Share"] = (bycountry["Leads"]/total*100.0).round(1) if total>0 else 0.0
        top5 = bycountry.nlargest(5, ["Share", "Leads"])
        
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
    if 'won_status_id' not in st.session_state:
        won_status_id = 9
        if lead_statuses is not None and "StatusName_E" in lead_statuses.columns:
            m = lead_statuses.loc[lead_statuses["StatusName_E"].str.lower() == "won"]
            if not m.empty and "LeadStatusId" in m.columns:
                won_status_id = int(m.iloc[0]["LeadStatusId"])
        st.session_state.won_status_id = won_status_id
    else:
        won_status_id = st.session_state.won_status_id

    st.subheader("Performance KPIs")

    # Get date range from session state (set by nav filter) or defaults
    if 'date_from' in st.session_state and 'date_to' in st.session_state:
        date_from = st.session_state.date_from
        date_to = st.session_state.date_to
    else:
        # Default to last 30 days
        today = date.today()
        date_from = today - timedelta(days=30)
        date_to = today
        st.session_state.date_from = date_from
        st.session_state.date_to = date_to

    # WTD/MTD/YTD calculations
    today_ts = pd.Timestamp.today().normalize()
    week_start_wtd = today_ts - pd.Timedelta(days=today_ts.weekday())
    month_start_mtd = today_ts.replace(day=1)
    year_start_ytd = today_ts.replace(month=1, day=1)

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
    week_total, week_conv, week_meet, week_won = metrics_full_dataset(week_start_wtd, today_ts)
    month_total, month_conv, month_meet, month_won = metrics_full_dataset(month_start_mtd, today_ts)
    year_total, year_conv, year_meet, year_won = metrics_full_dataset(year_start_ytd, today_ts)

    # Format large numbers
    def format_number(num):
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return f"{num:,}"

    # Week row
    st.markdown('<div class="period-header">Week To Date</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="period-header">Month To Date</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="period-header">Year To Date</div>', unsafe_allow_html=True)
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
        mask = (created.dt.date >= start_day.date()) & (created.dt.date <= end_day.date())
        filtered_leads = leads_all.loc[mask].copy()
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

    leads_ts = leads_local.groupby("period", as_index=False).size().rename(columns={"size": "value"}) if "period" in leads_local.columns else pd.DataFrame({"period":[],"value":[]})
    
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
            meet_ts = m.groupby("period", as_index=False)["LeadId"].nunique().rename(columns={"LeadId": "value"})
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
            title=dict(text=title, x=0.01, font=dict(size=12, color=TEXT_MUTED, family="Inter")),
            margin=dict(l=6,r=6,t=24,b=8),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_MAIN, family="Inter"),
            showlegend=False
        )
        fig.update_xaxes(showgrid=True, gridcolor=GRID_COL, tickfont=dict(color=TEXT_MUTED, size=10, family="Inter"), nticks=6, ticks="outside")
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COL, tickfont=dict(color=TEXT_MUTED, size=10, family="Inter"), nticks=4, ticks="outside", range=[ymin-pad, ymax+pad])
        return fig

    def tile_line(df, color, title):
        df = df.dropna().sort_values("period")
        if len(df)==0:
            fig = go.Figure(); fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED, family="Inter"))
            return _apply_axes(fig,[0,1],title)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["period"], y=df["idx"], mode="lines+markers",
                                 line=dict(color=color, width=3, shape="spline"), marker=dict(size=6, color=color)))
        return _apply_axes(fig, df["idx"], title)

    def tile_bar(df, color, title):
        df = df.dropna().sort_values("period")
        if len(df)==0:
            fig = go.Figure(); fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED, family="Inter"))
            return _apply_axes(fig,[0,1],title)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["period"], y=df["idx"],
                             marker=dict(color=color, line=dict(color="rgba(0,0,0,0.08)", width=0.5)), opacity=0.95))
        return _apply_axes(fig, df["idx"], title)

    def tile_bullet(df, title, bar_color):
        if df.empty or len(df)==0:
            fig = go.Figure(); fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED, family="Inter"))
            return _apply_axes(fig,[0,1],title)
        cur = float(df["idx"].iloc[-1])
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=cur,
            number={'valueformat': ".0f", 'font': {'color': TEXT_MAIN, 'family': 'Inter'}},
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
        fig.update_layout(height=120, margin=dict(l=8,r=8,t=26,b=8), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MAIN, family="Inter"))
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
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MAIN, family="Inter"))
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
                          font=dict(color=TEXT_MAIN, family="Inter"), height=400, showlegend=False,
                          margin=dict(l=0, r=0, t=40, b=0), xaxis_title="Status", yaxis_title="Count")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Lead Breakdown")

    meet_rate = pd.DataFrame({"Status":pd.Series(dtype="str"), "meet_leads":pd.Series(dtype="float")})
    if meets is not None and len(meets):
        M = meets.copy()
        if "LeadId" in M.columns:
            mm = M.merge(L[["LeadId","Status"]], on="LeadId", how="left")
            meet_rate = mm.groupby("Status", as_index=False)["LeadId"].nunique().rename(columns={"LeadId": "meet_leads"})

    conn_rate = pd.DataFrame({"Status":pd.Series(dtype="str"), "connect_rate":pd.Series(dtype="float")})
    if calls is not None and len(calls):
        C = calls.copy()
        C["CallDateTime"] = pd.to_datetime(C.get("CallDateTime"), errors="coerce")
        C = C.merge(L[["LeadId","Status"]], on="LeadId", how="left")
        g = C.groupby("Status", as_index=False).agg(
            total=("LeadCallId","count"),
            connects=("CallStatusId", lambda s: (s==1).sum())
        )
        g["connect_rate"] = (g["connects"]/g["total"]).fillna(0.0)
        conn_rate = g[["Status","connect_rate"]]

    base = L.groupby("Status", as_index=False).agg(
        Leads=("LeadId","count"),
        Avg_Age_Days=("age_days","mean")
    )
    
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

# Navigation with COMPACT Date Filter on the right
fdata = data

NAV = [
    ("Executive", "speedometer2", "🎯 Executive Summary"),
    ("Lead Status", "people", "📈 Lead Status"),
]

if HAS_OPTION_MENU:
    # Create container for nav and date filter in same row
    nav_col, filter_col = st.columns([4, 0.8])
    
    with nav_col:
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
    
    with filter_col:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        if st.button("📅 Filter", type="secondary", use_container_width=True, key="toggle_date_filter"):
            st.session_state.show_date_filter = not st.session_state.get('show_date_filter', False)
    
    # Show date filter popup if toggled
    if st.session_state.get('show_date_filter', False):
        with st.container(border=True):
            filter_type = st.radio(
                "Select Time Period",
                ["Week", "Month", "Year", "Custom"],
                horizontal=True,
                key="date_filter_type_nav"
            )
            
            today = date.today()
            
            if filter_type == "Week":
                st.session_state.date_from = today - timedelta(days=7)
                st.session_state.date_to = today
                st.info(f"📊 Last 7 days: {st.session_state.date_from.strftime('%Y-%m-%d')} to {st.session_state.date_to.strftime('%Y-%m-%d')}")
                
            elif filter_type == "Month":
                st.session_state.date_from = today - timedelta(days=30)
                st.session_state.date_to = today
                st.info(f"📊 Last 30 days: {st.session_state.date_from.strftime('%Y-%m-%d')} to {st.session_state.date_to.strftime('%Y-%m-%d')}")
                
            elif filter_type == "Year":
                st.session_state.date_from = today - timedelta(days=365)
                st.session_state.date_to = today
                st.info(f"📊 Last 365 days: {st.session_state.date_from.strftime('%Y-%m-%d')} to {st.session_state.date_to.strftime('%Y-%m-%d')}")
                
            else:  # Custom
                c1, c2, c3 = st.columns([1, 1, 1])
                
                with c1:
                    custom_from = st.date_input(
                        "From Date", 
                        value=st.session_state.get('date_from', today - timedelta(days=30)),
                        key="custom_date_from_nav"
                    )
                
                with c2:
                    custom_to = st.date_input(
                        "To Date", 
                        value=st.session_state.get('date_to', today),
                        key="custom_date_to_nav"
                    )
                
                with c3:
                    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
                    if st.button("Apply", type="primary", key="apply_custom_date_nav"):
                        st.session_state.date_from = custom_from
                        st.session_state.date_to = custom_to
                        st.rerun()
    
    st.markdown("---")
    
    if selected == "Executive":
        show_executive_summary(fdata)
    elif selected == "Lead Status":
        show_lead_status(fdata)
        
else:
    # Fallback for standard tabs
    nav_col, filter_col = st.columns([4, 0.8])
    
    with nav_col:
        tabs = st.tabs([n[2] for n in NAV])
    
    with filter_col:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        if st.button("📅 Filter", type="secondary", use_container_width=True, key="toggle_date_filter_fallback"):
            st.session_state.show_date_filter = not st.session_state.get('show_date_filter', False)
    
    if st.session_state.get('show_date_filter', False):
        with st.container(border=True):
            filter_type = st.radio(
                "Select Time Period",
                ["Week", "Month", "Year", "Custom"],
                horizontal=True,
                key="date_filter_type_fallback"
            )
            
            today = date.today()
            
            if filter_type == "Week":
                st.session_state.date_from = today - timedelta(days=7)
                st.session_state.date_to = today
                st.info(f"📊 Last 7 days")
                
            elif filter_type == "Month":
                st.session_state.date_from = today - timedelta(days=30)
                st.session_state.date_to = today
                st.info(f"📊 Last 30 days")
                
            elif filter_type == "Year":
                st.session_state.date_from = today - timedelta(days=365)
                st.session_state.date_to = today
                st.info(f"📊 Last 365 days")
                
            else:
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    custom_from = st.date_input("From Date", value=st.session_state.get('date_from', today - timedelta(days=30)), key="custom_from_fb")
                with c2:
                    custom_to = st.date_input("To Date", value=st.session_state.get('date_to', today), key="custom_to_fb")
                with c3:
                    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
                    if st.button("Apply", type="primary", key="apply_fb"):
                        st.session_state.date_from = custom_from
                        st.session_state.date_to = custom_to
                        st.rerun()
    
    st.markdown("---")
    
    with tabs[0]:
        show_executive_summary(fdata)
    with tabs[1]:
        show_lead_status(fdata)
