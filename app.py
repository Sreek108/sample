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

# Professional CSS with updated KPI specifications
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
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', Arial, sans-serif;
}}

section.main > div.block-container {{
  background: var(--bg-page);
  padding-top: 1rem;
}}

[data-testid="stSidebar"] {{ display: none; }}

h1, h2, h3, h4, h5, h6, label, p, span, div, .st-emotion-cache-10trblm {{
  color: var(--text-main);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', Arial, sans-serif;
}}

/* Optimized KPI Metric Cards */
.metric-card {{
    background: var(--bg-surface);
    padding: 20px 16px;
    border-radius: 8px;
    border: 1px solid var(--border-col);
    text-align: center;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', Arial, sans-serif;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s ease;
}}

.metric-card:hover {{
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}}

/* KPI Labels - Professional specs */
.metric-label {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', Arial, sans-serif;
    font-size: 15px;
    font-weight: 400;
    color: #6B7280;
    margin-bottom: 8px;
    text-transform: none;
    letter-spacing: 0;
    line-height: 1.4;
}}

/* KPI Values - Professional specs */
.metric-value {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', Arial, sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #111827;
    line-height: 1.2;
    font-feature-settings: 'tnum' 1;
}}

.period-header {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', Arial, sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: #374151;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid {PRIMARY_GOLD};
}}

/* Loading spinner */
.stSpinner > div {{
  border-top-color: {PRIMARY_GOLD} !important;
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
        # Test connection
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
        
        # SECURITY: Use secrets without hardcoded defaults
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
        
        # Build connection string
        odbc_params = f"driver={quote_plus(driver)}&Encrypt={encrypt}&TrustServerCertificate={tsc}"
        url = f"mssql+pyodbc://{quote_plus(username)}:{quote_plus(password)}@{server}:1433/{quote_plus(database)}?{odbc_params}"
        
        # OPTIMIZATION: Connection pooling for better performance
        engine = sa.create_engine(
            url, 
            fast_executemany=True,
            pool_pre_ping=True,      # Verify connections before using
            pool_size=5,             # Connection pool size
            max_overflow=10,         # Additional connections if needed
            pool_recycle=3600,       # Recycle connections every hour
            connect_args={
                "timeout": 30,       # Connection timeout
                "command_timeout": 300  # Query timeout
            }
        )
        
        @st.cache_data(ttl=300, show_spinner=False, max_entries=50)
        def _run(sql: str):
            """Execute SQL with caching and error handling"""
            try:
                start_time = datetime.now()
                result = pd.read_sql(sql, engine)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Query executed in {duration:.2f}s, returned {len(result):,} records")
                return result
            except Exception as e:
                logger.error(f"Query failed: {e}")
                raise
        
        # Test connection
        test = _run("SELECT @@SERVERNAME AS [server], DB_NAME() AS [db]")
        st.caption(f"✅ Connected to {server} / {database} (SQLAlchemy + Pool)")
        logger.info(f"Connected via SQLAlchemy to {server}/{database}")
        return None, _run
        
    except Exception as e2:
        st.error(f"❌ Database connection failed: {e2}")
        logger.error(f"Database connection error: {e2}")
        st.info("💡 Please check your connection settings in `.streamlit/secrets.toml`")
        return None, None

# OPTIMIZED Data Loading with Date Filtering
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
    """OPTIMIZED: Load transactional data with date filtering (cached for 1 minute)"""
    logger.info(f"Loading transactional data for last {months_back} months...")
    
    def q(sql: str, table_name: str = ""):
        try:
            if _conn:
                return _conn.query(sql, ttl=60)
            return _runner(sql)
        except Exception as e:
            logger.error(f"Failed to load {table_name}: {e}")
            return pd.DataFrame()
    
    # OPTIMIZATION: Only load recent data to reduce memory usage
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
    
    # Log data sizes for monitoring
    for name, df in tables.items():
        if not df.empty:
            logger.info(f"✅ {name}: {len(df):,} records loaded")
    
    return tables

# OPTIMIZED Data Normalization
def normalize_dataframes(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Normalize dataframe columns and data types efficiently"""
    logger.info("Normalizing dataframes...")
    
    try:
        for key, df in data.items():
            if df is None or df.empty:
                continue
                
            # Normalize column names once
            df.columns = df.columns.str.strip().str.lower()
            
            # Efficient renaming and type conversion
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

# Initialize connection and load data
@st.cache_data(ttl=60)
def initialize_dashboard():
    """Initialize dashboard data with error handling"""
    with st.spinner("🔄 Connecting to database..."):
        conn, runner = get_connection()
    
    if conn is None and runner is None:
        st.error("❌ Cannot connect to database. Please check your configuration.")
        st.stop()
    
    with st.spinner("📊 Loading dashboard data..."):
        try:
            # Load data with optimizations
            lookups = load_lookup_tables(conn, runner)
            transactions = load_transactional_data(conn, runner, months_back=12)
            
            # Combine and normalize
            data = {**lookups, **transactions}
            data = normalize_dataframes(data)
            
            # Validate critical tables
            if not validate_dataframe(data.get("leads", pd.DataFrame()), ["LeadId", "CreatedOn"], "Leads"):
                st.warning("⚠️ Lead data validation failed - some features may not work properly")
            
            return data
            
        except Exception as e:
            logger.error(f"Dashboard initialization failed: {e}")
            st.error(f"❌ Failed to load dashboard data: {e}")
            st.stop()

# Load dashboard data
data = initialize_dashboard()
grain = "Month"

# OPTIMIZED Funnel and Markets Analysis
def render_funnel_and_markets(d: Dict[str, pd.DataFrame]):
    """Render optimized funnel chart and top markets analysis"""
    leads = d.get("leads", pd.DataFrame())
    stages = d.get("lead_stages", pd.DataFrame())
    audit = d.get("lead_stage_audit", pd.DataFrame())
    countries = d.get("countries", pd.DataFrame())

    if not validate_dataframe(leads, ["LeadId"], "Leads"):
        return

    total_leads = len(leads)
    
    try:
        # Optimized funnel calculation
        if not audit.empty and not stages.empty and "StageId" in audit.columns:
            funnel_query = audit.merge(
                stages[["LeadStageId", "StageName_E", "SortOrder"]],
                left_on="StageId",
                right_on="LeadStageId",
                how="inner"
            )
            
            # Efficient groupby with proper sorting
            funnel_df = (
                funnel_query.groupby(["SortOrder", "StageName_E"], as_index=False)["LeadId"]
                .nunique()
                .rename(columns={"LeadId": "Count"})
                .sort_values("SortOrder", ascending=True)
            )
            
            # Clean stage names
            stage_rename = {
                "New": "New Leads", "Qualified": "Qualified", "Followup Process": "Follow-up",
                "Meeting Scheduled": "Meetings", "Negotiation": "Negotiation", "Won": "Won"
            }
            
            funnel_df["Stage"] = funnel_df["StageName_E"].map(stage_rename).fillna(funnel_df["StageName_E"])
            funnel_df = funnel_df[funnel_df["Stage"] != "Lost"].reset_index(drop=True)
            
        else:
            st.info("📊 Using simplified funnel - LeadStageAudit data unavailable")
            funnel_df = pd.DataFrame([{"Stage": "Total Leads", "Count": total_leads, "SortOrder": 1}])

        # Professional funnel chart
        colors = [PRIMARY_GOLD, ACCENT_BLUE, ACCENT_GREEN, ACCENT_AMBER, "#9b59b6"]
        
        fig = go.Figure(go.Funnel(
            name='Sales Funnel',
            y=funnel_df['Stage'].tolist(),
            x=funnel_df['Count'].tolist(),
            textposition="inside",
            textinfo="value+percent initial",
            textfont=dict(color="white", size=14, family="Inter", weight="bold"),
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
            font=dict(color=TEXT_MAIN, family="Inter"),
            title=dict(
                text=f"Sales Funnel - {total_leads:,} Total Leads",
                x=0.5, xanchor='center',
                font=dict(size=20, color=TEXT_MAIN, family="Inter", weight="bold")
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Top markets analysis
        if not countries.empty and "CountryId" in leads.columns:
            try:
                market_analysis = (
                    leads.groupby("CountryId", dropna=True, as_index=False).size()
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
                    st.dataframe(
                        top_markets[["Country", "Leads", "Share"]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Country": st.column_config.TextColumn("Country", width="medium"),
                            "Leads": st.column_config.NumberColumn("Leads", format="%,d"),
                            "Share": st.column_config.ProgressColumn("Market Share", format="%.1f%%", min_value=0, max_value=100)
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
        logger.error(f"Funnel rendering error: {e}")
        st.error("📊 Error rendering funnel chart")

# OPTIMIZED Executive Summary
def show_executive_summary(d: Dict[str, pd.DataFrame]):
    """Display executive summary with optimized calculations"""
    leads_all = d.get("leads", pd.DataFrame())
    lead_statuses = d.get("lead_statuses", pd.DataFrame())

    if not validate_dataframe(leads_all, ["LeadId", "CreatedOn"], "Leads"):
        return

    # Resolve Won status ID efficiently
    won_status_id = 9  # Default fallback
    if not lead_statuses.empty and "StatusName_E" in lead_statuses.columns:
        won_mask = lead_statuses["StatusName_E"].str.lower() == "won"
        if won_mask.any():
            won_status_id = int(lead_statuses.loc[won_mask, "LeadStatusId"].iloc[0])
    
    st.session_state.won_status_id = won_status_id

    st.subheader("📊 Performance KPIs")

    # Get date range from session state
    today = date.today()
    date_from = st.session_state.get('date_from', today - timedelta(days=30))
    date_to = st.session_state.get('date_to', today)

    # Efficient period calculations
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
        """Calculate metrics for a specific period efficiently"""
        try:
            # Filter leads by date
            if "CreatedOn" in leads_data.columns:
                dt_mask = (
                    (pd.to_datetime(leads_data["CreatedOn"], errors="coerce") >= start_ts) & 
                    (pd.to_datetime(leads_data["CreatedOn"], errors="coerce") <= end_ts)
                )
                period_leads = leads_data[dt_mask].copy()
            else:
                period_leads = pd.DataFrame()

            # Filter meetings by date
            period_meetings = pd.DataFrame()
            if not meetings_data.empty and "StartDateTime" in meetings_data.columns:
                dt_mask = (
                    (pd.to_datetime(meetings_data["StartDateTime"], errors="coerce") >= start_ts) & 
                    (pd.to_datetime(meetings_data["StartDateTime"], errors="coerce") <= end_ts)
                )
                valid_meetings = meetings_data[dt_mask]
                if "MeetingStatusId" in valid_meetings.columns:
                    period_meetings = valid_meetings[valid_meetings["MeetingStatusId"].isin([1, 6])]

            # Calculate metrics
            total = len(period_leads)
            won = int((period_leads.get("LeadStatusId", pd.Series(dtype="int64")) == won_id).sum()) if total > 0 else 0
            conv_pct = (won / total * 100.0) if total > 0 else 0.0
            meetings = int(period_meetings["LeadId"].nunique()) if "LeadId" in period_meetings.columns and len(period_meetings) > 0 else 0
            
            return total, conv_pct, meetings, won
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            return 0, 0.0, 0, 0

    # Calculate all period metrics efficiently
    metrics = {}
    for period_name, (start_ts, end_ts) in periods.items():
        metrics[period_name] = calculate_period_metrics(leads_all, meetings_all, start_ts, end_ts, won_status_id)

    # Display KPI cards with professional styling
    for period_name, period_label in [('week', 'Week To Date'), ('month', 'Month To Date'), ('year', 'Year To Date')]:
        total, conv_pct, meetings, won = metrics[period_name]
        
        st.markdown(f'<div class="period-header">{period_label}</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Leads</div>
                <div class="metric-value">{format_number(total)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Conversion Rate</div>
                <div class="metric-value">{conv_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Meetings Scheduled</div>
                <div class="metric-value">{format_number(meetings)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Won Deals</div>
                <div class="metric-value">{format_number(won)}</div>
            </div>
            """, unsafe_allow_html=True)

    # Trends section (simplified for performance)
    st.markdown("---")
    st.subheader("📈 Performance Trends")
    
    # Filter leads by selected date range
    start_day = pd.Timestamp(date_from)
    end_day = pd.Timestamp(date_to)
    
    if "CreatedOn" in leads_all.columns:
        date_mask = (
            (pd.to_datetime(leads_all["CreatedOn"], errors="coerce").dt.date >= start_day.date()) & 
            (pd.to_datetime(leads_all["CreatedOn"], errors="coerce").dt.date <= end_day.date())
        )
        filtered_leads = leads_all[date_mask].copy()
        
        if not filtered_leads.empty:
            st.info(f"📅 Showing {len(filtered_leads):,} leads from {date_from} to {date_to}")
        else:
            st.warning("⚠️ No leads found in the selected date range")
            filtered_leads = leads_all.tail(100)  # Show recent data as fallback
    else:
        filtered_leads = leads_all.copy()

    st.markdown("---")
    st.subheader("🎯 Lead Conversion Analysis")
    render_funnel_and_markets({**d, "leads": filtered_leads})

# OPTIMIZED Lead Status Analysis
def show_lead_status(d: Dict[str, pd.DataFrame]):
    """Display comprehensive lead status analysis"""
    leads = d.get("leads", pd.DataFrame())
    stats = d.get("lead_statuses", pd.DataFrame())
    calls = d.get("calls", pd.DataFrame())
    meets = d.get("agent_meeting_assignment", pd.DataFrame())

    if not validate_dataframe(leads, ["LeadId", "LeadStatusId"], "Leads"):
        return

    try:
        # Build status name mapping
        name_map = {}
        if not stats.empty and "StatusName_E" in stats.columns:
            name_map = dict(zip(stats["LeadStatusId"].astype(int), stats["StatusName_E"].astype(str)))

        # Process leads data
        L = leads.copy()
        L["Status"] = L["LeadStatusId"].map(name_map).fillna(L["LeadStatusId"].astype(str))
        L["CreatedOn"] = pd.to_datetime(L.get("CreatedOn"), errors="coerce")
        
        # Calculate lead age
        cutoff = L["CreatedOn"].max() if "CreatedOn" in L.columns else pd.Timestamp.today()
        L["age_days"] = (cutoff - L["CreatedOn"]).dt.days.fillna(0).astype(int)

        # Status distribution
        status_counts = L["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        # Professional visualizations
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Enhanced pie chart
            fig_pie = px.pie(
                status_counts, 
                names="Status", 
                values="Count",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="Lead Status Distribution"
            )
            fig_pie.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                textfont_size=12
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_MAIN, family="Inter"),
                title_font_size=16,
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

        with col2:
            # Key metrics
            total_leads = len(L)
            st.metric("📊 Total Leads", f"{total_leads:,}")
            
            # Won leads calculation
            won_count = 0
            if not stats.empty and "StatusName_E" in stats.columns:
                won_status = stats[stats["StatusName_E"].str.lower() == "won"]
                if not won_status.empty:
                    won_id = int(won_status.iloc[0]["LeadStatusId"])
                    won_count = int((L["LeadStatusId"] == won_id).sum())
            
            st.metric("🎯 Won Deals", f"{won_count:,}")
            
            if total_leads > 0:
                win_rate = (won_count / total_leads * 100)
                st.metric("📈 Win Rate", f"{win_rate:.1f}%")

        st.markdown("---")
        
        # Enhanced status breakdown with performance metrics
        st.subheader("📋 Detailed Status Analysis")

        # Calculate meeting rates
        meeting_rates = pd.DataFrame({"Status": [], "meeting_rate": []})
        if not meets.empty and "LeadId" in meets.columns:
            valid_meetings = meets[meets.get("MeetingStatusId", pd.Series()).isin([1, 6])]
            if not valid_meetings.empty:
                meeting_data = valid_meetings.merge(L[["LeadId", "Status"]], on="LeadId", how="inner")
                meeting_rates = (
                    meeting_data.groupby("Status")["LeadId"]
                    .nunique()
                    .reset_index(name="meetings")
                )

        # Calculate call connection rates
        connection_rates = pd.DataFrame({"Status": [], "connect_rate": []})
        if not calls.empty and "LeadId" in calls.columns:
            call_data = calls.merge(L[["LeadId", "Status"]], on="LeadId", how="inner")
            if not call_data.empty:
                conn_stats = call_data.groupby("Status").agg(
                    total_calls=("LeadCallId", "count"),
                    connected_calls=("CallStatusId", lambda x: (x == 1).sum())
                ).reset_index()
                conn_stats["connect_rate"] = (conn_stats["connected_calls"] / conn_stats["total_calls"]).fillna(0)
                connection_rates = conn_stats[["Status", "connect_rate"]]

        # Combine all metrics
        summary_stats = L.groupby("Status").agg(
            Leads=("LeadId", "count"),
            Avg_Age_Days=("age_days", "mean")
        ).reset_index()

        # Calculate market share
        total = summary_stats["Leads"].sum()
        summary_stats["Market_Share"] = (summary_stats["Leads"] / total * 100).round(1) if total > 0 else 0

        # Merge with meeting and connection rates
        if not meeting_rates.empty:
            summary_stats = summary_stats.merge(meeting_rates.rename(columns={"meetings": "Meeting_Count"}), on="Status", how="left")
            summary_stats["Meeting_Rate"] = (summary_stats["Meeting_Count"].fillna(0) / summary_stats["Leads"] * 100).round(1)
        else:
            summary_stats["Meeting_Rate"] = 0.0

        if not connection_rates.empty:
            summary_stats = summary_stats.merge(connection_rates, on="Status", how="left")
            summary_stats["connect_rate"] = summary_stats["connect_rate"].fillna(0).round(3)
        else:
            summary_stats["connect_rate"] = 0.0

        # Clean up data
        summary_stats["Avg_Age_Days"] = summary_stats["Avg_Age_Days"].fillna(0).round(1)
        summary_stats = summary_stats.sort_values("Leads", ascending=False)

        # Professional data table
        st.dataframe(
            summary_stats[["Status", "Leads", "Market_Share", "Avg_Age_Days", "Meeting_Rate", "connect_rate"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "Leads": st.column_config.NumberColumn("Lead Count", format="%,d"),
                "Market_Share": st.column_config.ProgressColumn("Market Share", format="%.1f%%", min_value=0, max_value=100),
                "Avg_Age_Days": st.column_config.NumberColumn("Avg Age (Days)", format="%.1f"),
                "Meeting_Rate": st.column_config.ProgressColumn("Meeting Rate", format="%.1f%%", min_value=0, max_value=100),
                "connect_rate": st.column_config.ProgressColumn("Call Connect Rate", format="%.1f%%", min_value=0, max_value=100)
            }
        )

    except Exception as e:
        logger.error(f"Lead status analysis error: {e}")
        st.error("❌ Error analyzing lead status data")

# PROFESSIONAL Navigation with Optimized Date Filter
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
                "nav-link": {
                    "font-size": "14px", "color": TEXT_MUTED, 
                    "font-weight": "500", "--hover-color": "#EEF2FF"
                },
                "nav-link-selected": {
                    "background-color": BG_SURFACE, "color": TEXT_MAIN, 
                    "border-bottom": f"3px solid {PRIMARY_GOLD}",
                    "font-weight": "600"
                },
            }
        )
    
    with filter_col:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        
        with st.popover("📅", use_container_width=True):
            st.markdown("### 📅 Date Filter")
            
            filter_type = st.radio(
                "Time Period", ["Week", "Month", "Year", "Custom"],
                horizontal=False, key="date_filter_type_nav"
            )
            
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
            else:
                st.markdown("#### Custom Range")
                custom_from = st.date_input(
                    "From", 
                    value=st.session_state.get('date_from', today - timedelta(days=30)),
                    key="custom_date_from_nav"
                )
                custom_to = st.date_input(
                    "To",
                    value=st.session_state.get('date_to', today),
                    key="custom_date_to_nav"
                )
                if st.button("Apply", type="primary", use_container_width=True):
                    st.session_state.date_from = custom_from
                    st.session_state.date_to = custom_to
                    st.success("✅ Applied")
                    st.rerun()
    
    st.markdown("---")
    
    # Route to selected page
    if selected == "Executive":
        show_executive_summary(data)
    elif selected == "Lead Status":
        show_lead_status(data)
        
else:
    # Fallback navigation
    nav_col, filter_col = st.columns([5.5, 0.5])
    
    with nav_col:
        tabs = st.tabs([n[2] for n in NAV])
    
    with filter_col:
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        with st.popover("📅", use_container_width=True):
            st.markdown("### 📅 Date Filter")
            filter_type = st.radio("Time Period", ["Week", "Month", "Year", "Custom"], horizontal=False)
            # Similar filter logic as above
    
    st.markdown("---")
    
    with tabs[0]:
        show_executive_summary(data)
    with tabs[1]:
        show_lead_status(data)

# Add footer with performance info
st.markdown("---")
st.caption(f"🚀 Dashboard loaded successfully | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: {len(data.get('leads', [])):,} leads")
