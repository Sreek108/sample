import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, date
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional horizontal nav
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False
    logger.warning("streamlit_option_menu not available, using fallback navigation")

# Page config
st.set_page_config(
    page_title="DAR Global - Executive Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="📊"
)

# Professional color palette
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

# Professional CSS with SIMPLE KPI LAYOUT
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
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
}}

section.main > div.block-container {{
  background: var(--bg-page);
  padding-top: 1rem;
}}

[data-testid="stSidebar"] {{ 
  display: none; 
}}

h1, h2, h3, h4, h5, h6, label, p, span, div {{
  color: var(--text-main);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
}}

/* Simple KPI Cards - Clean Layout */
.kpi-container {{
    text-align: left;
    padding: 12px 8px;
    background: transparent;
}}

/* KPI Labels - Small, Gray, Top */
.kpi-label {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
    font-size: 13px;
    font-weight: 400;
    color: #9CA3AF;
    margin-bottom: 8px;
    line-height: 1.2;
}}

/* KPI Values - Large, Bold, Bottom */
.kpi-value {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
    font-size: 32px;
    font-weight: 600;
    color: {TEXT_MAIN};
    line-height: 1.2;
}}

/* Period Divider - NO BORDER LINES */
.period-divider {{
    margin: 32px 0 16px 0;
    padding-top: 0px;
}}

.period-header {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: #6B7280;
    margin-bottom: 16px;
}}

/* Loading spinner */
.stSpinner > div {{
  border-top-color: {PRIMARY_GOLD} !important;
}}

/* Hide Streamlit branding */
#MainMenu {{
  visibility: hidden;
}}

footer {{
  visibility: hidden;
}}

/* Smooth scrolling */
html {{
  scroll-behavior: smooth;
}}
</style>
""", unsafe_allow_html=True)

# Utility functions
def trend_box(fig):
    """Display chart in bordered container"""
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def validate_dataframe(df: pd.DataFrame, required_columns: list, df_name: str = "DataFrame") -> bool:
    """Validate dataframe has required columns and data"""
    if df is None or df.empty:
        st.warning(f"⚠️ {df_name} is empty or unavailable")
        return False
    
    missing = set(required_columns) - set(df.columns)
    if missing:
        st.error(f"❌ {df_name} missing required columns: {missing}")
        return False
    
    logger.info(f"✅ {df_name} validation passed: {len(df):,} records")
    return True

def format_number(num: int) -> str:
    """Format numbers with appropriate suffixes"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,}"

# OPTIMIZED Database Connection with Connection Pooling
@st.cache_resource
def get_connection() -> Tuple[Optional[Any], Optional[callable]]:
    """Establish optimized database connection with connection pooling"""
    from streamlit.connections import SQLConnection
    
    # Try Streamlit SQL connection first
    try:
        conn = st.connection("sql", type=SQLConnection)
        info = conn.query("SELECT @@SERVERNAME AS [server], DB_NAME() AS [db]", ttl=60)
        st.caption(f"✅ Connected to {info.iloc[0]['server']} / {info.iloc[0]['db']}")
        logger.info(f"Connected via Streamlit SQL to {info.iloc[0]['server']}")
        return conn, None
    except Exception as e1:
        logger.warning(f"Streamlit SQL connection failed: {e1}")
    
    # Fallback to SQLAlchemy with connection pooling
    try:
        import sqlalchemy as sa
        from urllib.parse import quote_plus
        
        s = st.secrets.get("connections", {}).get("sql", {})
        
        required_keys = ["server", "database", "username", "password"]
        missing = [k for k in required_keys if k not in s]
        if missing:
            st.error(f"❌ Missing required connection keys in secrets: {missing}")
            st.info("Please configure database credentials in `.streamlit/secrets.toml`")
            return None, None
        
        server = s["server"]
        database = s["database"]
        username = s["username"]
        password = s["password"]
        driver = s.get("driver", "ODBC Driver 18 for SQL Server")
        encrypt = s.get("encrypt", "no")
        tsc = s.get("TrustServerCertificate", "yes")
        
        odbc_params = f"driver={quote_plus(driver)}&Encrypt={encrypt}&TrustServerCertificate={tsc}"
        url = f"mssql+pyodbc://{quote_plus(username)}:{quote_plus(password)}@{server}:1433/{quote_plus(database)}?{odbc_params}"
        
        engine = sa.create_engine(
            url, 
            fast_executemany=True,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600,
            connect_args={
                "timeout": 30,
                "command_timeout": 300
            }
        )
        
        @st.cache_data(ttl=300, show_spinner=False, max_entries=50)
        def _run(sql: str):
            try:
                start_time = datetime.now()
                result = pd.read_sql(sql, engine)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Query executed in {duration:.2f}s, returned {len(result):,} records")
                return result
            except Exception as e:
                logger.error(f"Query failed: {e}")
                raise
        
        test = _run("SELECT @@SERVERNAME AS [server], DB_NAME() AS [db]")
        st.caption(f"✅ Connected to {server} / {database} (SQLAlchemy + Pool)")
        logger.info(f"Connected via SQLAlchemy to {server}/{database}")
        return None, _run
        
    except Exception as e2:
        st.error(f"❌ Database connection failed: {e2}")
        logger.error(f"Database connection error: {e2}")
        st.info("💡 Please check your connection settings in `.streamlit/secrets.toml`")
        return None, None

# OPTIMIZED Data Loading
@st.cache_data(ttl=3600, show_spinner=False, max_entries=10)
def load_lookup_tables(_conn, _runner) -> Dict[str, pd.DataFrame]:
    """Load static reference tables (cached for 1 hour)"""
    logger.info("Loading lookup tables...")
    
    def q(sql: str, table_name: str = ""):
        try:
            if _conn:
                return _conn.query(sql, ttl=3600)
            return _runner(sql)
        except Exception as e:
            logger.error(f"Failed to load {table_name}: {e}")
            return pd.DataFrame()
    
    tables = {
        "countries": q("""
            SELECT CountryId, CountryName_E 
            FROM dbo.Country 
            WHERE IsActive = 1
            ORDER BY CountryName_E
        """, "countries"),
        
        "lead_statuses": q("""
            SELECT LeadStatusId, StatusName_E 
            FROM dbo.LeadStatus 
            WHERE IsActive = 1
            ORDER BY LeadStatusId
        """, "lead_statuses"),
        
        "lead_stages": q("""
            SELECT LeadStageId, StageName_E, SortOrder 
            FROM dbo.LeadStage 
            WHERE IsActive = 1
            ORDER BY SortOrder
        """, "lead_stages"),
        
        "meeting_status": q("""
            SELECT MeetingStatusId, StatusName_E 
            FROM dbo.MeetingStatus 
            WHERE IsActive = 1
            ORDER BY MeetingStatusId
        """, "meeting_status"),
        
        "call_statuses": q("""
            SELECT CallStatusId, StatusName_E 
            FROM dbo.CallStatus 
            WHERE IsActive = 1
            ORDER BY CallStatusId
        """, "call_statuses")
    }
    
    logger.info("✅ Lookup tables loaded successfully")
    return tables

@st.cache_data(ttl=60, show_spinner=False, max_entries=20)
def load_transactional_data(_conn, _runner, months_back: int = 12) -> Dict[str, pd.DataFrame]:
    """Load transactional data with date filtering"""
    logger.info(f"Loading transactional data for last {months_back} months...")
    
    def q(sql: str, table_name: str = ""):
        try:
            if _conn:
                return _conn.query(sql, ttl=60)
            return _runner(sql)
        except Exception as e:
            logger.error(f"Failed to load {table_name}: {e}")
            return pd.DataFrame()
    
    tables = {
        "leads": q(f"""
            SELECT 
                LeadId, LeadCode, LeadStageId, LeadStatusId, LeadScoringId,
                AssignedAgentId, CountryId, CityRegionId, CreatedOn, IsActive
            FROM dbo.Lead 
            WHERE IsActive = 1 
            AND CreatedOn >= DATEADD(MONTH, -{months_back}, GETDATE())
            ORDER BY CreatedOn DESC
        """, "leads"),
        
        "agent_meeting_assignment": q(f"""
            SELECT 
                AssignmentId, LeadId, StartDateTime, EndDateTime, 
                MeetingStatusId, AgentId
            FROM dbo.AgentMeetingAssignment
            WHERE StartDateTime >= DATEADD(MONTH, -{months_back}, GETDATE())
            ORDER BY StartDateTime DESC
        """, "meetings"),
        
        "calls": q(f"""
            SELECT 
                LeadCallId, LeadId, CallDateTime, DurationSeconds,
                CallStatusId, SentimentId, AssignedAgentId, CallDirection
            FROM dbo.LeadCallRecord
            WHERE CallDateTime >= DATEADD(MONTH, -{months_back}, GETDATE())
            ORDER BY CallDateTime DESC
        """, "calls"),
        
        "lead_stage_audit": q(f"""
            SELECT 
                AuditId, LeadId, StageId, CreatedOn
            FROM dbo.LeadStageAudit
            WHERE CreatedOn >= DATEADD(MONTH, -{months_back}, GETDATE())
            ORDER BY CreatedOn DESC
        """, "stage_audit")
    }
    
    for name, df in tables.items():
        if not df.empty:
            logger.info(f"✅ {name}: {len(df):,} records loaded")
    
    return tables

# Data Normalization
def normalize_dataframes(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Normalize dataframe columns and data types"""
    logger.info("Normalizing dataframes...")
    
    try:
        for key, df in data.items():
            if df is None or df.empty:
                continue
                
            df.columns = df.columns.str.strip().str.lower()
            
            if key == "leads":
                column_map = {
                    "leadid": "LeadId", "leadcode": "LeadCode", "leadstageid": "LeadStageId",
                    "leadstatusid": "LeadStatusId", "leadscoringid": "LeadScoringId",
                    "assignedagentid": "AssignedAgentId", "createdon": "CreatedOn",
                    "isactive": "IsActive", "countryid": "CountryId", "cityregionid": "CityRegionId"
                }
                df = df.rename(columns=column_map)
                df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], errors="coerce")
                
            elif key == "agent_meeting_assignment":
                column_map = {
                    "assignmentid": "AssignmentId", "leadid": "LeadId",
                    "startdatetime": "StartDateTime", "enddatetime": "EndDateTime",
                    "meetingstatusid": "MeetingStatusId", "agentid": "AgentId"
                }
                df = df.rename(columns=column_map)
                df["StartDateTime"] = pd.to_datetime(df["StartDateTime"], errors="coerce")
                
            elif key == "calls":
                column_map = {
                    "leadcallid": "LeadCallId", "leadid": "LeadId",
                    "calldatetime": "CallDateTime", "durationseconds": "DurationSeconds",
                    "callstatusid": "CallStatusId", "sentimentid": "SentimentId",
                    "assignedagentid": "AssignedAgentId", "calldirection": "CallDirection"
                }
                df = df.rename(columns=column_map)
                df["CallDateTime"] = pd.to_datetime(df["CallDateTime"], errors="coerce")
                
            elif key == "lead_stage_audit":
                column_map = {
                    "auditid": "AuditId", "leadid": "LeadId",
                    "stageid": "StageId", "createdon": "CreatedOn"
                }
                df = df.rename(columns=column_map)
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
            
    except Exception as e:
        logger.error(f"Data normalization error: {e}")
        
    logger.info("✅ Data normalization completed")
    return data

# Initialize dashboard
@st.cache_data(ttl=60)
def initialize_dashboard():
    """Initialize dashboard data"""
    with st.spinner("🔄 Connecting to database..."):
        conn, runner = get_connection()
    
    if conn is None and runner is None:
        st.error("❌ Cannot connect to database. Please check your configuration.")
        st.stop()
    
    with st.spinner("📊 Loading dashboard data..."):
        try:
            lookups = load_lookup_tables(conn, runner)
            transactions = load_transactional_data(conn, runner, months_back=12)
            
            data = {**lookups, **transactions}
            data = normalize_dataframes(data)
            
            if not validate_dataframe(data.get("leads", pd.DataFrame()), ["LeadId", "CreatedOn"], "Leads"):
                st.warning("⚠️ Lead data validation failed - some features may not work properly")
            
            return data
            
        except Exception as e:
            logger.error(f"Dashboard initialization failed: {e}")
            st.error(f"❌ Failed to load dashboard data: {e}")
            st.stop()

# Load data
data = initialize_dashboard()
grain = "Month"

# Funnel and Markets
def render_funnel_and_markets(d: Dict[str, pd.DataFrame]):
    """Render funnel chart and top markets"""
    leads = d.get("leads", pd.DataFrame())
    stages = d.get("lead_stages", pd.DataFrame())
    audit = d.get("lead_stage_audit", pd.DataFrame())
    countries = d.get("countries", pd.DataFrame())

    if not validate_dataframe(leads, ["LeadId"], "Leads"):
        return

    total_leads = len(leads)
    
    try:
        if not audit.empty and not stages.empty and "StageId" in audit.columns:
            funnel_query = audit.merge(
                stages[["LeadStageId", "StageName_E", "SortOrder"]],
                left_on="StageId",
                right_on="LeadStageId",
                how="inner"
            )
            
            funnel_df = (
                funnel_query.groupby(["SortOrder", "StageName_E"], as_index=False)["LeadId"]
                .nunique()
                .rename(columns={"LeadId": "Count"})
                .sort_values("SortOrder", ascending=True)
            )
            
            stage_rename = {
                "New": "New Leads", 
                "Qualified": "Qualified", 
                "Followup Process": "Follow-up",
                "Meeting Scheduled": "Meetings", 
                "Negotiation": "Negotiation", 
                "Won": "Won"
            }
            
            funnel_df["Stage"] = funnel_df["StageName_E"].map(stage_rename).fillna(funnel_df["StageName_E"])
            funnel_df = funnel_df[funnel_df["Stage"] != "Lost"].reset_index(drop=True)
            
        else:
            st.info("📊 Using simplified funnel")
            funnel_df = pd.DataFrame([{"Stage": "Total Leads", "Count": total_leads, "SortOrder": 1}])

        colors = [PRIMARY_GOLD, ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, "#9b59b6"]
        
        fig = go.Figure(go.Funnel(
            name='Sales Funnel',
            y=funnel_df['Stage'].tolist(),
            x=funnel_df['Count'].tolist(),
            textposition="inside",
            textinfo="value+percent initial",
            textfont=dict(color="white", size=14),
            marker=dict(
                color=colors[:len(funnel_df)],
                line=dict(width=2, color="white")
            ),
            connector=dict(line=dict(color="#34495e", width=2))
        ))
        
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=60, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_MAIN),
            title=dict(
                text=f"",
                x=0.5, 
                xanchor='center',
                font=dict(size=20, color=TEXT_MAIN)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        if not countries.empty and "CountryId" in leads.columns:
            try:
                market_analysis = (
                    leads.groupby("CountryId", dropna=True, as_index=False)
                    .size()
                    .rename(columns={"size": "Leads"})
                    .merge(
                        countries[["CountryId", "CountryName_E"]].rename(columns={"CountryName_E": "Country"}), 
                        on="CountryId", 
                        how="inner"
                    )
                )
                
                if not market_analysis.empty:
                    total = market_analysis["Leads"].sum()
                    market_analysis["Share"] = (market_analysis["Leads"] / total * 100.0).round(1)
                    top_markets = market_analysis.nlargest(5, "Leads")
                    
                    st.subheader("🌍 Top Markets")
                    
                    # Format Leads as text for left alignment
                    top_markets['Leads_Display'] = top_markets['Leads'].apply(lambda x: str(x))
                    
                    st.dataframe(
                        top_markets[["Country", "Leads_Display", "Share"]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Country": st.column_config.TextColumn("Country", width="medium"),
                            "Leads_Display": st.column_config.TextColumn("Leads", width="small"),
                            "Share": st.column_config.ProgressColumn(
                                "Market Share", 
                                format="%.1f%%", 
                                min_value=0, 
                                max_value=100
                            )
                        }
                    )
                else:
                    st.info("📍 No market data available")
                    
            except Exception as e:
                logger.error(f"Market analysis error: {e}")
                st.info("📍 Market data temporarily unavailable")
        else:
            st.info("📍 Country data unavailable")
                    
            except Exception as e:
                logger.error(f"Market analysis error: {e}")
                st.info("📍 Market data temporarily unavailable")
        else:
            st.info("📍 Country data unavailable")
            
    except Exception as e:
        logger.error(f"Funnel rendering error: {e}")
        st.error("📊 Error rendering funnel chart")

# Executive Summary with CLEAN KPI LAYOUT
def show_executive_summary(d: Dict[str, pd.DataFrame]):
    """Display executive summary with clean KPI layout"""
    leads_all = d.get("leads", pd.DataFrame())
    lead_statuses = d.get("lead_statuses", pd.DataFrame())

    if not validate_dataframe(leads_all, ["LeadId", "CreatedOn"], "Leads"):
        return

    # Resolve Won status ID
    won_status_id = 9
    if not lead_statuses.empty and "StatusName_E" in lead_statuses.columns:
        won_mask = lead_statuses["StatusName_E"].str.lower() == "won"
        if won_mask.any():
            won_status_id = int(lead_statuses.loc[won_mask, "LeadStatusId"].iloc[0])
    
    st.session_state.won_status_id = won_status_id

    st.subheader("📊 Performance KPIs")

    # Get date range
    today = date.today()
    date_from = st.session_state.get('date_from', today - timedelta(days=30))
    date_to = st.session_state.get('date_to', today)

    # Period calculations
    today_ts = pd.Timestamp.today().normalize()
    periods = {
        'week': (today_ts - pd.Timedelta(days=today_ts.weekday()), today_ts),
        'month': (today_ts.replace(day=1), today_ts),
        'year': (today_ts.replace(month=1, day=1), today_ts)
    }

    meetings_all = d.get("agent_meeting_assignment", pd.DataFrame())

    @st.cache_data(ttl=300)
    def calculate_period_metrics(leads_data: pd.DataFrame, meetings_data: pd.DataFrame, 
                                start_ts: pd.Timestamp, end_ts: pd.Timestamp, won_id: int):
        try:
            if "CreatedOn" in leads_data.columns:
                dt_mask = (
                    (pd.to_datetime(leads_data["CreatedOn"], errors="coerce") >= start_ts) & 
                    (pd.to_datetime(leads_data["CreatedOn"], errors="coerce") <= end_ts)
                )
                period_leads = leads_data[dt_mask].copy()
            else:
                period_leads = pd.DataFrame()

            period_meetings = pd.DataFrame()
            if not meetings_data.empty and "StartDateTime" in meetings_data.columns:
                dt_mask = (
                    (pd.to_datetime(meetings_data["StartDateTime"], errors="coerce") >= start_ts) & 
                    (pd.to_datetime(meetings_data["StartDateTime"], errors="coerce") <= end_ts)
                )
                valid_meetings = meetings_data[dt_mask]
                if "MeetingStatusId" in valid_meetings.columns:
                    period_meetings = valid_meetings[valid_meetings["MeetingStatusId"].isin([1, 6])]

            total = len(period_leads)
            won = int((period_leads.get("LeadStatusId", pd.Series(dtype="int64")) == won_id).sum()) if total > 0 else 0
            conv_pct = (won / total * 100.0) if total > 0 else 0.0
            meetings = int(period_meetings["LeadId"].nunique()) if "LeadId" in period_meetings.columns and len(period_meetings) > 0 else 0
            
            return total, conv_pct, meetings, won
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return 0, 0.0, 0, 0

    # Calculate metrics
    metrics = {}
    for period_name, (start_ts, end_ts) in periods.items():
        metrics[period_name] = calculate_period_metrics(leads_all, meetings_all, start_ts, end_ts, won_status_id)

    # Display CLEAN KPI cards
    for period_name, period_label in [('week', 'Week To Date'), ('month', 'Month To Date'), ('year', 'Year To Date')]:
        total, conv_pct, meetings, won = metrics[period_name]
        
        # Period header with divider
        st.markdown(f'<div class="period-divider"><div class="period-header">{period_label}</div></div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Total Leads</div>
                <div class="kpi-value">{format_number(total)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Conversion Rate</div>
                <div class="kpi-value">{conv_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Meetings Scheduled</div>
                <div class="kpi-value">{format_number(meetings)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Won Deals</div>
                <div class="kpi-value">{format_number(won)}</div>
            </div>
            """, unsafe_allow_html=True)

    # Performance Trends
    st.markdown("---")
    st.subheader("📈 Performance Trends")
    
    trend_col1, trend_col2 = st.columns([2, 1])
    with trend_col1:
        trend_style = st.radio(
            "Chart Style", 
            ["Line", "Bars", "Bullet"], 
            index=0, 
            horizontal=True, 
            key="trend_style_exec"
        )
    with trend_col2:
        st.caption("")
    
    start_day = pd.Timestamp(date_from)
    end_day = pd.Timestamp(date_to)
    
    if "CreatedOn" in leads_all.columns:
        date_mask = (
            (pd.to_datetime(leads_all["CreatedOn"], errors="coerce").dt.date >= start_day.date()) & 
            (pd.to_datetime(leads_all["CreatedOn"], errors="coerce").dt.date <= end_day.date())
        )
        filtered_leads = leads_all[date_mask].copy()
    else:
        filtered_leads = pd.DataFrame()
    
    if not filtered_leads.empty:
        leads_local = filtered_leads.copy()
        if "CreatedOn" in leads_local.columns:
            dt = pd.to_datetime(leads_local["CreatedOn"], errors="coerce")
            if grain == "Week":
                leads_local["period"] = dt.dt.to_period("W").apply(lambda p: p.start_time.date())
            elif grain == "Month":
                leads_local["period"] = dt.dt.to_period("M").apply(lambda p: p.start_time.date())
            else:
                leads_local["period"] = dt.dt.to_period("Y").apply(lambda p: p.start_time.date())

        leads_ts = leads_local.groupby("period").size().reset_index(name="value") if "period" in leads_local.columns else pd.DataFrame()
        
        if "LeadStatusId" in leads_local.columns and "period" in leads_local.columns:
            per_total = leads_local.groupby("period").size()
            per_won = leads_local.loc[leads_local["LeadStatusId"].eq(won_status_id)].groupby("period").size()
            conv_ts = pd.DataFrame({"period": per_total.index, "total": per_total.values}).merge(
                pd.DataFrame({"period": per_won.index, "won": per_won.values}), on="period", how="left"
            ).fillna(0.0)
            conv_ts["value"] = (conv_ts["won"] / conv_ts["total"] * 100).round(1)
        else:
            conv_ts = pd.DataFrame()

        if not meetings_all.empty and "StartDateTime" in meetings_all.columns:
            m = meetings_all.copy()
            m["dt"] = pd.to_datetime(m["StartDateTime"], errors="coerce")
            if grain == "Week":
                m["period"] = m["dt"].dt.to_period("W").apply(lambda p: p.start_time.date())
            elif grain == "Month":
                m["period"] = m["dt"].dt.to_period("M").apply(lambda p: p.start_time.date())
            else:
                m["period"] = m["dt"].dt.to_period("Y").apply(lambda p: p.start_time.date())
            
            if "MeetingStatusId" in m.columns:
                m = m[m["MeetingStatusId"].isin([1, 6])]
            meet_ts = m.groupby("period")["LeadId"].nunique().reset_index(name="value")
        else:
            meet_ts = pd.DataFrame()

        def _index(df):
            df = df.copy()
            if df.empty:
                df["idx"] = []
                return df
            base = df["value"].iloc[0] if len(df) > 0 and df["value"].iloc[0] != 0 else 1.0
            df["idx"] = (df["value"] / base) * 100.0
            return df

        leads_ts = _index(leads_ts)
        conv_ts = _index(conv_ts)
        meet_ts = _index(meet_ts)

        def _apply_axes(fig, ys, title):
            ymin = float(pd.Series(ys).min()) if len(ys) else 0
            ymax = float(pd.Series(ys).max()) if len(ys) else 1
            pad = max(1.0, (ymax - ymin) * 0.12)
            fig.update_layout(
                height=220,
                title=dict(text=title, x=0.01, font=dict(size=13, color=TEXT_MUTED)),
                margin=dict(l=10, r=10, t=35, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_MAIN),
                showlegend=False
            )
            fig.update_xaxes(showgrid=True, gridcolor=GRID_COL, tickfont=dict(color=TEXT_MUTED, size=10), nticks=6)
            fig.update_yaxes(showgrid=True, gridcolor=GRID_COL, tickfont=dict(color=TEXT_MUTED, size=10), nticks=5, range=[ymin - pad, ymax + pad])
            return fig

        def tile_line(df, color, title):
            df = df.dropna().sort_values("period")
            if len(df) == 0:
                fig = go.Figure()
                fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED))
                return _apply_axes(fig, [0, 1], title)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["period"], y=df["idx"], 
                mode="lines+markers",
                line=dict(color=color, width=3, shape="spline"),
                marker=dict(size=8, color=color, line=dict(color="white", width=2))
            ))
            return _apply_axes(fig, df["idx"], title)

        def tile_bar(df, color, title):
            df = df.dropna().sort_values("period")
            if len(df) == 0:
                fig = go.Figure()
                fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED))
                return _apply_axes(fig, [0, 1], title)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["period"], y=df["idx"],
                marker=dict(color=color, line=dict(color="rgba(255,255,255,0.6)", width=1)),
                opacity=0.9
            ))
            return _apply_axes(fig, df["idx"], title)

        def tile_bullet(df, title, bar_color):
            if df.empty or len(df) == 0:
                fig = go.Figure()
                fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(color=TEXT_MUTED))
                return _apply_axes(fig, [0, 1], title)
            
            cur = float(df["idx"].iloc[-1])
            fig = go.Figure(go.Indicator(
                mode="number+gauge+delta",
                value=cur,
                number={'valueformat': ".0f", 'font': {'color': TEXT_MAIN, 'size': 20}},
                delta={'reference': 100, 'valueformat': '.1f'},
                gauge={
                    'shape': "bullet",
                    'axis': {'range': [80, 120]},
                    'steps': [
                        {'range': [80, 95], 'color': "rgba(239,68,68,0.15)"},
                        {'range': [95, 105], 'color': "rgba(234,179,8,0.15)"},
                        {'range': [105, 120], 'color': "rgba(34,197,94,0.15)"},
                    ],
                    'bar': {'color': bar_color, 'thickness': 0.75},
                    'threshold': {'line': {'color': '#111827', 'width': 2}, 'value': 100}
                }
            ))
            fig.update_layout(height=160, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
            return fig

        s1, s2, s3 = st.columns(3)
        
        if trend_style == "Line":
            with s1: 
                st.plotly_chart(tile_line(leads_ts, ACCENT_BLUE, "Leads Trend"), use_container_width=True, config={'displayModeBar': False})
            with s2: 
                st.plotly_chart(tile_line(conv_ts, ACCENT_GREEN, "Conversion Rate Trend"), use_container_width=True, config={'displayModeBar': False})
            with s3: 
                st.plotly_chart(tile_line(meet_ts, PRIMARY_GOLD, "Meetings Trend"), use_container_width=True, config={'displayModeBar': False})
                    
        elif trend_style == "Bars":
            with s1: 
                st.plotly_chart(tile_bar(leads_ts, ACCENT_BLUE, "Leads Trend"), use_container_width=True, config={'displayModeBar': False})
            with s2: 
                st.plotly_chart(tile_bar(conv_ts, ACCENT_GREEN, "Conversion Rate Trend"), use_container_width=True, config={'displayModeBar': False})
            with s3: 
                st.plotly_chart(tile_bar(meet_ts, PRIMARY_GOLD, "Meetings Trend"), use_container_width=True, config={'displayModeBar': False})
                    
        else:  # Bullet
            with s1: 
                st.plotly_chart(tile_bullet(leads_ts, "Leads Index", ACCENT_BLUE), use_container_width=True, config={'displayModeBar': False})
            with s2: 
                st.plotly_chart(tile_bullet(conv_ts, "Conversion Index", ACCENT_GREEN), use_container_width=True, config={'displayModeBar': False})
            with s3: 
                st.plotly_chart(tile_bullet(meet_ts, "Meetings Index", PRIMARY_GOLD), use_container_width=True, config={'displayModeBar': False})
    
    else:
        st.warning("⚠️ No leads found in the selected date range")

    st.markdown("---")
    st.subheader("🎯 Lead Conversion Snapshot")
    render_funnel_and_markets({**d, "leads": filtered_leads if not filtered_leads.empty else leads_all})

# Lead Status Analysis
def show_lead_status(d: Dict[str, pd.DataFrame]):
    """Display enhanced lead status analysis with professional visualizations"""
    leads = d.get("leads", pd.DataFrame())
    stats = d.get("lead_statuses", pd.DataFrame())
    calls = d.get("calls", pd.DataFrame())
    meets = d.get("agent_meeting_assignment", pd.DataFrame())
    stages = d.get("lead_stages", pd.DataFrame())
    audit = d.get("lead_stage_audit", pd.DataFrame())

    if not validate_dataframe(leads, ["LeadId", "LeadStatusId"], "Leads"):
        return

    try:
        # Build status mapping
        name_map = {}
        if not stats.empty and "StatusName_E" in stats.columns:
            name_map = dict(zip(stats["LeadStatusId"].astype(int), stats["StatusName_E"].astype(str)))

        L = leads.copy()
        L["Status"] = L["LeadStatusId"].map(name_map).fillna(L["LeadStatusId"].astype(str))
        L["CreatedOn"] = pd.to_datetime(L.get("CreatedOn"), errors="coerce")
        
        cutoff = L["CreatedOn"].max() if "CreatedOn" in L.columns else pd.Timestamp.today()
        L["age_days"] = (cutoff - L["CreatedOn"]).dt.days.fillna(0).astype(int)

        # Get Won status
        won_status_id = 9
        if not stats.empty and "StatusName_E" in stats.columns:
            won_status = stats[stats["StatusName_E"].str.lower() == "won"]
            if not won_status.empty:
                won_status_id = int(won_status.iloc[0]["LeadStatusId"])

        # Calculate key metrics
        total_leads = len(L)
        won_count = int((L["LeadStatusId"] == won_status_id).sum())
        conversion_rate = (won_count / total_leads * 100) if total_leads > 0 else 0

        # ===== FEATURE 1: STATUS KPIs =====
        st.subheader("📊 Lead Status Overview")
        
        hot_statuses = ['Meeting Scheduled', 'Negotiating Terms', 'Meeting Confirmed', 'Awaiting Budget']
        cold_statuses = ['Follow up Needed', 'Interested', 'Attempted Contact', 'Callback Scheduled', 'In Discussion']
        
        hot_leads = len(L[L["Status"].isin(hot_statuses)])
        cold_leads = len(L[L["Status"].isin(cold_statuses)])
        avg_days_to_close = L[L["LeadStatusId"] == won_status_id]["age_days"].mean() if won_count > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">🔥 Hot Leads</div>
                <div class="kpi-value">{format_number(hot_leads)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">❄️ Cold Leads</div>
                <div class="kpi-value">{format_number(cold_leads)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">🎯 Conversion Rate</div>
                <div class="kpi-value">{conversion_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">⏱️ Avg Days to Close</div>
                <div class="kpi-value">{avg_days_to_close:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        # ===== FEATURE 2: STATUS TREND CHART =====
        st.markdown("---")
        st.subheader("📈 Status Trends Over Time")
        
        if "CreatedOn" in L.columns and not L.empty:
            L_trend = L.copy()
            L_trend["Month"] = pd.to_datetime(L_trend["CreatedOn"]).dt.to_period('M').astype(str)
            trend_data = L_trend.groupby(['Month', 'Status']).size().reset_index(name='Count')
            
            top_statuses = L["Status"].value_counts().head(6).index.tolist()
            trend_filtered = trend_data[trend_data["Status"].isin(top_statuses)]
            
            if len(trend_filtered) > 0:
                fig_trend = px.line(
                    trend_filtered, 
                    x='Month', 
                    y='Count', 
                    color='Status',
                    markers=True,
                    title='Lead Count by Status Over Time'
                )
                
                fig_trend.update_layout(
                    height=400,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=TEXT_MAIN),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

        # ===== FEATURE 3: AGENT PERFORMANCE =====
        st.markdown("---")
        st.subheader("👥 Agent Performance by Status")
        
        if "AssignedAgentId" in L.columns:
            L_with_agents = L[(L["AssignedAgentId"].notna()) & (L["AssignedAgentId"] != 0)].copy()
            
            if len(L_with_agents) > 0:
                agent_stats = L_with_agents.groupby("AssignedAgentId").agg(
                    Total_Leads=("LeadId", "count"),
                    Won_Leads=("LeadStatusId", lambda x: (x == won_status_id).sum()),
                    Avg_Days=("age_days", "mean")
                ).reset_index()
                
                agent_stats["Conversion_Rate"] = (agent_stats["Won_Leads"] / agent_stats["Total_Leads"] * 100).round(1)
                agent_stats["Avg_Days"] = agent_stats["Avg_Days"].round(1)
                agent_stats = agent_stats.sort_values("Conversion_Rate", ascending=False).head(10)
                
                agent_stats["Performance"] = agent_stats["Conversion_Rate"].apply(
                    lambda x: '🟢 High' if x >= 5.0 else ('🟡 Medium' if x >= 2.0 else '🔴 Low')
                )
                
                agent_stats = agent_stats.rename(columns={"AssignedAgentId": "Agent_ID"})
                
                st.dataframe(
                    agent_stats,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Agent_ID": st.column_config.NumberColumn("Agent ID", format="%d"),
                        "Total_Leads": st.column_config.NumberColumn("Total Leads", format="%d"),
                        "Won_Leads": st.column_config.NumberColumn("Won", format="%d"),
                        "Conversion_Rate": st.column_config.ProgressColumn("Conversion Rate", format="%.1f%%", min_value=0, max_value=20),
                        "Avg_Days": st.column_config.NumberColumn("Avg Days", format="%.1f"),
                        "Performance": st.column_config.TextColumn("Performance")
                    }
                )
                
                total_with_agents = len(L_with_agents)
                total_without_agents = len(L) - total_with_agents
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("👥 Leads with Assigned Agents", f"{total_with_agents:,}")
                with col_b:
                    st.metric("❓ Unassigned Leads", f"{total_without_agents:,}")
            else:
                st.warning("⚠️ No leads have been assigned to agents yet.")
                st.info(f"📊 Total leads available for assignment: **{len(L):,}**")

        # ===== PROFESSIONAL STATUS DISTRIBUTION WITH MATCHING CARD STYLES =====
        st.markdown("---")
        
        status_counts = L["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        
        st.subheader("📊 Lead Status Overview")
        
        col_chart, col_stats = st.columns([2, 1])
        
        with col_chart:
            # Professional horizontal bar chart
            status_counts_sorted = status_counts.sort_values("Count", ascending=True)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=status_counts_sorted["Status"],
                x=status_counts_sorted["Count"],
                orientation='h',
                marker=dict(color=ACCENT_BLUE, line=dict(color='white', width=1)),
                text=status_counts_sorted["Count"],
                textposition='outside',
                textfont=dict(size=11, color=TEXT_MAIN),
                hovertemplate='<b>%{y}</b><br>Leads: %{x:,}<extra></extra>'
            ))
            
            fig_bar.update_layout(
                title=dict(text="Lead Distribution by Status", font=dict(size=16, color=TEXT_MAIN)),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT_MAIN),
                xaxis=dict(title="Number of Leads", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                yaxis=dict(title=""),
                margin=dict(l=10, r=50, t=50, b=40)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        
        with col_stats:
            # CSS for matching white card style
            st.markdown("""
            <style>
            .info-card {
                background: white;
                padding: 20px;
                border-radius: 12px;
                border: 2px solid #E5E7EB;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                margin-bottom: 16px;
            }
            .info-card h4 {
                color: #111827;
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 12px;
                border-bottom: 2px solid #1E90FF;
                padding-bottom: 6px;
            }
            .info-item {
                padding: 8px 0;
                border-bottom: 1px solid #F3F4F6;
                font-size: 14px;
                color: #374151;
            }
            .info-item:last-child {
                border-bottom: none;
            }
            .info-item strong {
                color: #111827;
                font-weight: 600;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Key Metrics Card - WHITE CARD STYLE
            metrics_items = f"""
            <div class="info-item"><strong>📊 Total Leads:</strong> {total_leads:,}</div>
            <div class="info-item"><strong>🎯 Won Deals:</strong> {won_count:,}</div>
            <div class="info-item"><strong>📈 Win Rate:</strong> {conversion_rate:.1f}%</div>
            """
            
            key_metrics_html = f"""
            <div class="info-card">
                <h4>📊 Key Metrics</h4>
                {metrics_items}
            </div>
            """
            
            st.markdown(key_metrics_html, unsafe_allow_html=True)
            
            # Top 3 Statuses Card - WHITE CARD STYLE
            top_3 = status_counts.head(3)
            status_items = ""
            for idx, row in top_3.iterrows():
                pct = (row['Count'] / total_leads * 100)
                status_items += f'<div class="info-item"><strong>{row["Status"]}:</strong> {row["Count"]:,} ({pct:.1f}%)</div>'
            
            top_status_html = f"""
            <div class="info-card">
                <h4>🏆 Top 3 Statuses</h4>
                {status_items}
            </div>
            """
            
            st.markdown(top_status_html, unsafe_allow_html=True)

        # ===== STATUS COMPARISON MATRIX =====
        st.markdown("---")
        st.subheader("📊 Status Comparison Matrix")
        
        comparison = L.groupby('Status').agg(
            Total_Leads=('LeadId', 'count'),
            Avg_Age=('age_days', 'mean'),
            Won_Count=('LeadStatusId', lambda x: (x == won_status_id).sum())
        ).reset_index()
        
        comparison['Win_Rate'] = (comparison['Won_Count'] / comparison['Total_Leads'] * 100).round(1)
        comparison['Avg_Age'] = comparison['Avg_Age'].round(0)
        
        if not meets.empty:
            meeting_stats = meets.merge(L[['LeadId', 'Status']], on='LeadId', how='inner')
            meeting_counts = meeting_stats.groupby('Status')['LeadId'].nunique().reset_index(name='Meetings')
            comparison = comparison.merge(meeting_counts, on='Status', how='left')
            comparison['Meetings'] = comparison['Meetings'].fillna(0).astype(int)
        else:
            comparison['Meetings'] = 0
        
        comparison = comparison.sort_values('Total_Leads', ascending=False)
        
        comparison['Health_Score'] = (
            (comparison['Win_Rate'] * 0.4) + 
            ((100 - comparison['Avg_Age'].clip(0, 100)) * 0.3) +
            ((comparison['Meetings'] / comparison['Total_Leads'].clip(1) * 100).clip(0, 100) * 0.3)
        ).round(0)
        
        st.dataframe(
            comparison[['Status', 'Total_Leads', 'Win_Rate', 'Avg_Age', 'Meetings', 'Health_Score']],
            use_container_width=True,
            hide_index=True,
            height=450,
            column_config={
                "Status": st.column_config.TextColumn("Status", width="large"),
                "Total_Leads": st.column_config.NumberColumn("Leads", format="%,d"),
                "Win_Rate": st.column_config.ProgressColumn("Win Rate", format="%.1f%%", min_value=0, max_value=20),
                "Avg_Age": st.column_config.NumberColumn("Avg Age (Days)", format="%.0f"),
                "Meetings": st.column_config.NumberColumn("Meetings", format="%d"),
                "Health_Score": st.column_config.ProgressColumn("Health Score", format="%.0f", min_value=0, max_value=100)
            }
        )


    except Exception as e:
        logger.error(f"Lead status analysis error: {e}")
        st.error(f"❌ Error analyzing lead status data: {e}")
# Navigation
NAV = [
    ("Executive", "speedometer2", "🎯 Executive Summary"),
    ("Lead Status", "people", "📈 Lead Status Analysis"),
]

if HAS_OPTION_MENU:
    nav_col, filter_col = st.columns([5.5, 0.5])
    
    with nav_col:
        selected = option_menu(
            None, [n[0] for n in NAV], icons=[n[1] for n in NAV],
            orientation="horizontal", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": BG_PAGE},
                "icon": {"color": PRIMARY_GOLD, "font-size": "16px"},
                "nav-link": {"font-size": "14px", "color": TEXT_MUTED, "font-weight": "500", "--hover-color": "#EEF2FF"},
                "nav-link-selected": {"background-color": BG_SURFACE, "color": TEXT_MAIN, "border-bottom": f"3px solid {PRIMARY_GOLD}", "font-weight": "600"},
            }
        )
    
    with filter_col:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        
        with st.popover("📅", use_container_width=True):
            st.markdown("### 📅 Date Filter")
            
            filter_type = st.radio("Time Period", ["Week", "Month", "Year", "Custom"], horizontal=False, key="date_filter_type_nav")
            
            today = date.today()
            
            if filter_type == "Week":
                st.session_state.date_from = today - timedelta(days=7)
                st.session_state.date_to = today
                st.success("✅ Last 7 days")
                st.info(f"From: {st.session_state.date_from.strftime('%Y-%m-%d')}")
                st.info(f"To: {st.session_state.date_to.strftime('%Y-%m-%d')}")
                
            elif filter_type == "Month":
                st.session_state.date_from = today - timedelta(days=30)
                st.session_state.date_to = today
                st.success("✅ Last 30 days")
                st.info(f"From: {st.session_state.date_from.strftime('%Y-%m-%d')}")
                st.info(f"To: {st.session_state.date_to.strftime('%Y-%m-%d')}")
                
            elif filter_type == "Year":
                st.session_state.date_from = today - timedelta(days=365)
                st.session_state.date_to = today
                st.success("✅ Last 365 days")
                st.info(f"From: {st.session_state.date_from.strftime('%Y-%m-%d')}")
                st.info(f"To: {st.session_state.date_to.strftime('%Y-%m-%d')}")
                
            elif filter_type == "Custom":
                st.markdown("#### 📆 Select Custom Range")
                
                if 'date_from' not in st.session_state:
                    st.session_state.date_from = today - timedelta(days=30)
                if 'date_to' not in st.session_state:
                    st.session_state.date_to = today
                
                custom_from = st.date_input("From Date", value=st.session_state.date_from, key="custom_date_from_nav")
                custom_to = st.date_input("To Date", value=st.session_state.date_to, key="custom_date_to_nav")
                
                if st.button("✅ Apply Custom Range", type="primary", use_container_width=True, key="apply_custom_btn"):
                    if custom_from <= custom_to:
                        st.session_state.date_from = custom_from
                        st.session_state.date_to = custom_to
                        st.success(f"✅ Applied: {custom_from} to {custom_to}")
                        st.rerun()
                    else:
                        st.error("❌ 'From Date' must be before 'To Date'")
                
                st.info(f"📅 Current: {st.session_state.date_from} to {st.session_state.date_to}")
    
    st.markdown("---")
    
    if selected == "Executive":
        show_executive_summary(data)
    elif selected == "Lead Status":
        show_lead_status(data)
        
else:
    nav_col, filter_col = st.columns([5.5, 0.5])
    
    with nav_col:
        tabs = st.tabs([n[2] for n in NAV])
    
    with filter_col:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        
        with st.popover("📅", use_container_width=True):
            st.markdown("### 📅 Date Filter")
            filter_type = st.radio("Time Period", ["Week", "Month", "Year", "Custom"], horizontal=False, key="date_filter_type_fallback")
            
            today = date.today()
            
            if filter_type == "Week":
                st.session_state.date_from = today - timedelta(days=7)
                st.session_state.date_to = today
                st.success("✅ Last 7 days")
            elif filter_type == "Month":
                st.session_state.date_from = today - timedelta(days=30)
                st.session_state.date_to = today
                st.success("✅ Last 30 days")
            elif filter_type == "Year":
                st.session_state.date_from = today - timedelta(days=365)
                st.session_state.date_to = today
                st.success("✅ Last 365 days")
            elif filter_type == "Custom":
                st.markdown("#### 📆 Select Custom Range")
                if 'date_from' not in st.session_state:
                    st.session_state.date_from = today - timedelta(days=30)
                if 'date_to' not in st.session_state:
                    st.session_state.date_to = today
                
                custom_from = st.date_input("From Date", value=st.session_state.date_from, key="custom_date_from_fb")
                custom_to = st.date_input("To Date", value=st.session_state.date_to, key="custom_date_to_fb")
                
                if st.button("✅ Apply Custom Range", type="primary", use_container_width=True, key="apply_custom_fb"):
                    if custom_from <= custom_to:
                        st.session_state.date_from = custom_from
                        st.session_state.date_to = custom_to
                        st.success(f"✅ Applied: {custom_from} to {custom_to}")
                        st.rerun()
                    else:
                        st.error("❌ 'From Date' must be before 'To Date'")
                
                st.info(f"📅 Current: {st.session_state.date_from} to {st.session_state.date_to}")
    
    st.markdown("---")
    
    with tabs[0]:
        show_executive_summary(data)
    with tabs[1]:
        show_lead_status(data)
