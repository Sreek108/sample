# DAR Global — CEO Executive Dashboard (SQL backend)

This app connects directly to Microsoft SQL Server using Streamlit’s st.connection, with credentials stored in `.streamlit/secrets.toml` and never committed to Git.

## Quick start (local)
1. Create a virtualenv and install:
   pip install -r requirements.txt
2. Create `.streamlit/secrets.toml`:
   [connections.sql]
   dialect = "mssql"
   driver  = "pyodbc"
   host    = "auto.resourceplus.app"
   database = "Data_Lead"
   username = "sa"
   password = "********"

   [connections.sql.query]
   driver = "ODBC Driver 17 for SQL Server"
   encrypt = "yes"
   trustservercertificate = "no"
3. Run:
   streamlit run app.py
4. If no data appears, widen the sidebar date range and confirm the Debug expander shows valid min/max dates.

## Deploy (Streamlit Community Cloud)
- Push code to GitHub (secrets.toml is ignored by .gitignore).
- In the app’s Settings → Advanced → Secrets, paste the same TOML values used locally.

- Connect the repo, set main file to `app.py`, and deploy. 
