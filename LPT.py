# app.py
"""
Student LeetCode Performance Dashboard
- Full-screen background wallpaper
- Polished, modern dashboard aesthetics
- Card-style KPIs and leaderboard visuals
- All original functionality intact
"""

import re
import time
from typing import Optional, Dict, Any, List
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import numpy as np
import random
import base64

# ------------------ Streamlit page config ------------------
st.set_page_config(page_title="Student LeetCode Tracker", layout="wide")

# ------------------ Add background wallpaper & CSS ------------------

st.markdown(f"""
    <style>
    /* App background */
    [data-testid="stAppViewContainer"] {{
        background-image: url("https://wallpapercave.com/wp/wp8018093.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}

    /* Card styling for dataframes, KPIs, plots - DARK THEME */
    div[data-testid="stDataFrame"],
    div[data-testid="stMetric"],
    div[data-testid="stPlotlyChart"] {{
        background: rgba(10, 10, 20, 0.8) !important; /* Dark semi-transparent bg */
        border-radius: 15px !important;
        padding: 12px !important;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important; /* Subtle border */
    }}

    /* Fine-tune metric and table text color for dark theme */
    div[data-testid="stMetric"] *,
    div[data-testid="stDataFrame"] * {{
        color: #E0E0E0 !important; /* Light grey text */
    }}

    /* Sidebar styling - DARK THEME */
    section[data-testid="stSidebar"] {{
        background: transparent !important;
    }}
    section[data-testid="stSidebar"] > div {{
        background: rgba(10, 10, 20, 0.8) !important; /* Dark semi-transparent bg */
        border-radius: 15px !important;
        padding: 15px !important;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.15) !important;
    }}

    /* Headers */
    [data-testid="stAppViewContainer"] h1 {{
        background: transparent !important;
        color: #FFFFFF !important; /* Solid white */
        font-weight: 700 !important; /* Bolder font */
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8) !important; /* Stronger, cleaner shadow */
        font-family: 'Arial', sans-serif !important;
        text-align: center;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        font-size: 3rem !important;
    }}

    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 {{
        background: transparent !important;
        color: #FFFFFF !important; /* White text for high contrast */
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.7) !important; /* Shadow for readability */
        font-family: 'Arial', sans-serif !important;
        text-align: center;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }}

    /* Custom styling for alert boxes */
    div[data-testid="stInfo"],
    div[data-testid="stSuccess"],
    div[data-testid="stWarning"],
    div[data-testid="stError"] {{
        border-radius: 15px !important;
        padding: 1rem !important;
        border: none !important;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2) !important;
    }}

    div[data-testid="stInfo"] {{
        background-color: rgba(52, 152, 219, 0.8) !important; /* Blue accent */
    }}

    div[data-testid="stSuccess"] {{
        background-color: rgba(46, 204, 113, 0.8) !important; /* Green accent */
    }}

    div[data-testid="stWarning"] {{
        background-color: rgba(241, 196, 15, 0.8) !important; /* Yellow accent */
    }}

    div[data-testid="stError"] {{
        background-color: rgba(231, 76, 60, 0.8) !important; /* Red accent */
    }}

    /* Ensure text inside alerts is light */
    div[data-testid="stInfo"] *,
    div[data-testid="stSuccess"] *,
    div[data-testid="stWarning"] *,
    div[data-testid="stError"] * {{
        color: white !important;
    }}

    /* Style the file uploader */
    [data-testid="stFileUploader"] {{
        background: transparent !important;
        border: none !important; /* Removed dashed border */
        border-radius: 15px !important;
    }}

    [data-testid="stFileUploader"] label {{
        color: #FFFFFF !important;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.7) !important;
    }}

    [data-testid="stFileUploader"] small {{
        color: rgba(255, 255, 255, 0.7) !important;
    }}

    [data-testid="stFileUploader"] button {{
        background: rgba(240, 242, 246, 0.9) !important;
        color: #2C3E50 !important;
        border: none !important;
    }}
    </style>
    """, 
    unsafe_allow_html=True
)

# ------------------ GraphQL config & helpers ------------------
GRAPHQL_URL = "https://leetcode.com/graphql"

QUERY_CHECK = """
query getUserProfile($username: String!) {{
  matchedUser(username: $username) {{
    username
  }}
}}
"""

QUERY_STATS = """
query getUserProfile($username: String!) {{
  matchedUser(username: $username) {{
    username
    submitStats {{
      acSubmissionNum {{
        difficulty
        count
        submissions
      }}
    }}
  }}
}}
"""

DEFAULT_DELAY = 0.6

def extract_username(url: str) -> Optional[str]:
    if not isinstance(url, str): return None
    url = url.strip()
    if url == "": return None
    m = re.search(r'leetcode\.com/(?:u/)?([^/?#]+)/?', url)
    if m: return m.group(1)
    if "/" in url: return url.rstrip("/").split("/")[-1]
    return url

def do_graphql_post(query: str, variables: Dict[str, Any], timeout: int = 10) -> Optional[Dict]:
    try:
        r = requests.post(GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=timeout)
        if r.status_code == 200: return r.json()
        return None
    except requests.RequestException:
        return None

@st.cache_data(show_spinner=False)
def check_profile_exists(username: str) -> bool:
    if not username: return False
    resp = do_graphql_post(QUERY_CHECK, {"username": username})
    if not resp: return False
    return resp.get("data", {}).get("matchedUser") is not None

@st.cache_data(show_spinner=False)
def fetch_user_stats(username: str) -> Dict[str, int]:
    if not username: return {"Easy":0,"Medium":0,"Hard":0,"Total":0}
    resp = do_graphql_post(QUERY_STATS, {"username": username})
    if not resp: return {"Easy":0,"Medium":0,"Hard":0,"Total":0}
    matched = resp.get("data", {}).get("matchedUser")
    if not matched: return {"Easy":0,"Medium":0,"Hard":0,"Total":0}
    arr = matched.get("submitStats", {}).get("acSubmissionNum", [])
    counts = {entry.get("difficulty"): entry.get("count",0) for entry in arr if entry.get("difficulty")}
    easy = counts.get("Easy",0); medium=counts.get("Medium",0); hard=counts.get("Hard",0)
    total = counts.get("All", easy+medium+hard)
    return {"Easy": easy,"Medium": medium,"Hard": hard,"Total": total}

def make_mock_dataset(n: int = 30) -> pd.DataFrame:
    students = [f"Student_{i}" for i in range(1,n+1)]
    urls = [f"https://leetcode.com/u/student_{i}/" for i in range(1,n+1)]
    data = {"Student": students, "LeetCode URL": urls,
            "Easy": [random.randint(5,100) for _ in range(n)],
            "Medium": [random.randint(0,80) for _ in range(n)],
            "Hard": [random.randint(0,30) for _ in range(n)]}
    df = pd.DataFrame(data); df["Total"] = df["Easy"] + df["Medium"] + df["Hard"]
    return df

st.title("〽️ Student LeetCode Performance Tracker")

# Sidebar
st.sidebar.header("Options")
use_mock = st.sidebar.checkbox("Use mock dataset", value=False)
workers = st.sidebar.number_input("Parallel Workers", 1, 20, 5, 1, help="Set the number of concurrent threads to fetch data. Higher values are faster but increase the risk of API rate-limiting.")
show_invalid = st.sidebar.checkbox("Show invalid links", True)
if st.sidebar.button("♻️ Clear Cache"):
    check_profile_exists.clear(); fetch_user_stats.clear(); st.success("Cache cleared.")

uploaded_file = st.file_uploader("Upload Excel/CSV with 'Student' and 'LeetCode URL'", type=["xlsx","xls","csv"])

# Mock dataset
if use_mock and uploaded_file is None:
    if st.button("Generate mock dataset"):
        df_uploaded = make_mock_dataset(30)
        st.session_state["uploaded_df"] = df_uploaded
        st.success("Mock dataset generated.")
        st.dataframe(df_uploaded.head(10))
else:
    if uploaded_file:
        try:
            if uploaded_file.name.endswith((".xls",".xlsx")): df_uploaded = pd.read_excel(uploaded_file)
            else: df_uploaded = pd.read_csv(uploaded_file)
        except Exception as e: st.error(f"Failed to read file: {e}"); df_uploaded=None
        if df_uploaded is not None:
            st.session_state["uploaded_df"] = df_uploaded
            st.subheader("Preview uploaded data")
            st.dataframe(df_uploaded.head(10))
    else:
        if "uploaded_df" not in st.session_state: st.session_state["uploaded_df"]=None

# ------------------ Automated Processing Pipeline ------------------
if st.session_state.get("uploaded_df") is not None and st.session_state.get("results_df") is None:
    src_df: pd.DataFrame = st.session_state.get("uploaded_df")
    
    # --- 1. Validation Stage ---
    with st.spinner("Step 1: Validating all user profiles..."):
        student_col = next((c for c in src_df.columns if "student" in c.lower()), src_df.columns[0])
        url_col = next((c for c in src_df.columns if "leetcode" in c.lower()), None)
        
        validated_df = None
        if url_col is None:
            st.error("No LeetCode URL column found.")
        else:
            def validate_row(row_tuple):
                idx, row = row_tuple
                student = row.get(student_col, f"Student_{idx + 1}")
                url_raw = row.get(url_col, "")
                username = extract_username(str(url_raw))
                row_out = {"Student": student, "LeetCode URL": url_raw, "LeetCode Username": username or "", "Profile Valid": False, "Reason": ""}
                if not username:
                    row_out["Reason"] = "Username extraction failed"; return False, row_out
                if use_mock or check_profile_exists(username):
                    row_out["Profile Valid"] = True; return True, row_out
                else:
                    row_out["Reason"] = "Profile not found"; return False, row_out

            rows_to_process = list(src_df.iterrows())
            valid_rows, invalid_rows = [], []
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(validate_row, row_tuple) for row_tuple in rows_to_process]
                for future in as_completed(futures):
                    is_valid, result_row = future.result()
                    if is_valid: valid_rows.append(result_row)
                    else: invalid_rows.append(result_row)
            
            validated_df = pd.DataFrame(valid_rows)
            st.success(f"Validation complete — Valid: {len(validated_df)} | Invalid: {len(invalid_rows)}")
            if show_invalid and invalid_rows:
                st.subheader("Invalid rows"); st.dataframe(pd.DataFrame(invalid_rows).head(50))

    # --- 2. Fetching Stage ---
    if validated_df is not None and not validated_df.empty:
        with st.spinner("Step 2: Fetching LeetCode stats for valid profiles..."):
            def fetch_row_stats(row):
                username = row.get("LeetCode Username")
                student = row.get("Student")
                url_raw = row.get("LeetCode URL")
                if use_mock:
                    src_df = st.session_state.get("uploaded_df")
                    if src_df is not None and all(c in src_df.columns for c in ["Easy", "Medium", "Hard"]):
                        matched = src_df[(src_df.get(student_col) == student) | (src_df.get(url_col) == url_raw)]
                        if not matched.empty:
                            r0 = matched.iloc[0]
                            stats = {"Easy": int(r0.get("Easy", 0)), "Medium": int(r0.get("Medium", 0)), "Hard": int(r0.get("Hard", 0))}
                            stats["Total"] = sum(stats.values())
                        else: stats = {"Easy": 0, "Medium": 0, "Hard": 0, "Total": 0}
                    else: stats = {"Easy": 0, "Medium": 0, "Hard": 0, "Total": 0}
                else:
                    stats = fetch_user_stats(username)
                return {"Student": student, "LeetCode URL": url_raw, "LeetCode Username": username,
                        "Easy": int(stats.get("Easy", 0)), "Medium": int(stats.get("Medium", 0)),
                        "Hard": int(stats.get("Hard", 0)), "Total": int(stats.get("Total", 0))}

            rows_to_process = [row for _, row in validated_df.iterrows()]
            results = []
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(fetch_row_stats, row) for row in rows_to_process]
                for future in as_completed(futures):
                    results.append(future.result())
            
            df_results = pd.DataFrame(results)
            st.session_state["results_df"] = df_results
            st.success("Fetched stats successfully!")
            st.rerun()

# ------------------ Display Results ------------------
if st.session_state.get("results_df") is not None:
    df = st.session_state["results_df"]
    st.subheader("✅ Processed Student Stats")
    st.dataframe(df.sort_values("Total",ascending=False).reset_index(drop=True))

    # ------------------ KPIs ------------------
    st.subheader("Key Performance Indicators")
    top = df.loc[df["Total"].idxmax()]
    bottom = df.loc[df["Total"].idxmin()]
    avg = df["Total"].mean()
    k1,k2,k3 = st.columns(3)
    k1.metric("🏆 Top Performer", f"{{top['Student']}} ({{top['Total']}})", delta_color="normal")
    k2.metric("📉 Weakest Performer", f"{{bottom['Student']}} ({{bottom['Total']}})", delta_color="inverse")
    k3.metric("📊 Average Solved", f"{{avg:.1f}}")

    # Scatter Plot: Medium vs Hard
    st.subheader("Analysis: Medium vs. Hard Problems")
    fig_scatter = px.scatter(df, x="Medium", y="Hard", 
                             title="Medium vs. Hard Problems Solved",
                             hover_data=['Student', 'Easy', 'Medium', 'Hard', 'Total'],
                             color="Total", color_continuous_scale="Viridis")
    fig_scatter.update_layout(xaxis_title="Medium Problems Solved", yaxis_title="Hard Problems Solved")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.write("---")
    # Leaderboard
    st.subheader("Leaderboard (Total Problems Solved)")
    fig_leader = px.bar(df.sort_values("Total",ascending=False), x="Student", y="Total",
                        title="Leaderboard - Total Problems Solved", text="Total",
                        color="Total", color_continuous_scale="Blues")
    fig_leader.update_traces(textposition="outside")
    st.plotly_chart(fig_leader,use_container_width=True)

    # Problems by difficulty (stacked)
    st.subheader("Problems by Difficulty")
    fig_stack = px.bar(df.sort_values("Total",ascending=False), x="Student", y=["Easy","Medium","Hard"],
                       barmode="stack", title="Stacked Difficulty", color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_stack,use_container_width=True)

    # Pie chart
    st.subheader("Overall Difficulty Distribution")
    totals={"Easy":int(df["Easy"].sum()),"Medium":int(df["Medium"].sum()),"Hard":int(df["Hard"].sum())}
    dist_df = pd.DataFrame({"Difficulty":list(totals.keys()),"Count":list(totals.values())})
    fig_pie = px.pie(dist_df,names="Difficulty",values="Count",title="Overall Difficulty Distribution",
                     color="Difficulty", color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_pie,use_container_width=True)

    # Filters & CSV export
    st.write("---")
    st.subheader("Interactive Filters & Export")
    col_a,col_b = st.columns([2,1])
    with col_a:
        top_n = st.number_input("Show top N students by Total (0=all)",0,1000,0,1)
        sort_by = st.selectbox("Sort by", ["Total","Easy","Medium","Hard","Student"],0)
    with col_b:
        download_name = st.text_input("Download file name", "leetcode_stats_processed.csv")
        if st.button("⬇️ Download CSV"):
            st.download_button("Save CSV", df.to_csv(index=False).encode("utf-8"), file_name=download_name, mime="text/csv")
    display_df = df.sort_values(sort_by, ascending=(sort_by=="Student"))
    if top_n>0: display_df=display_df.sort_values("Total",ascending=False).head(top_n)
    st.dataframe(display_df.reset_index(drop=True))

else:
    st.info("No results yet. Upload dataset and fetch stats.")

