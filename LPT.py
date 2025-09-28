# app.py
"""
Student LeetCode Performance Dashboard
- Full-screen background wallpaper
- Polished, modern dashboard aesthetics
- Card-style KPIs and leaderboard visuals
- Invalid rows are tracked with 'Profile Valid' + 'Reason'
"""

import re
from typing import Optional, Dict, Any
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------ Streamlit page config ------------------
st.set_page_config(page_title="Student LeetCode Tracker", layout="wide")

# ------------------ Add background wallpaper & CSS ------------------
st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("https://wallpapercave.com/wp/wp8018093.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    div[data-testid="stDataFrame"],
    div[data-testid="stMetric"],
    div[data-testid="stPlotlyChart"] {{
        background: rgba(10, 10, 20, 0.8) !important;
        border-radius: 15px !important;
        padding: 12px !important;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }}
    div[data-testid="stMetric"] *,
    div[data-testid="stDataFrame"] * {{
        color: #E0E0E0 !important;
    }}
    section[data-testid="stSidebar"] {{
        background: transparent !important;
    }}
    section[data-testid="stSidebar"] > div {{
        background: rgba(10, 10, 20, 0.8) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.15) !important;
    }}
    [data-testid="stAppViewContainer"] h1 {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8) !important;
        text-align: center;
        font-size: 3rem !important;
    }}
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3 {{
        color: #FFFFFF !important;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.7) !important;
        text-align: center;
    }}
    div[data-testid="stError"],
    div[data-testid="stSuccess"],
    div[data-testid="stWarning"] {{
        border-radius: 15px !important;
        padding: 1rem !important;
        border: none !important;
    }}
    [data-testid="stFileUploader"] label {{
        color: #FFFFFF !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ GraphQL config ------------------
GRAPHQL_URL = "https://leetcode.com/graphql"

QUERY_CHECK = """
query getUserProfile($username: String!) {
  matchedUser(username: $username) {
    username
  }
}
"""

QUERY_STATS = """
query getUserProfile($username: String!) {
  matchedUser(username: $username) {
    username
    submitStats {
      acSubmissionNum {
        difficulty
        count
      }
    }
  }
}
"""

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


# ------------------ Streamlit UI ------------------
st.title("〽️ Student LeetCode Performance Tracker")

# Sidebar
st.sidebar.header("Options")
workers = st.sidebar.number_input("Parallel Workers", 1, 20, 10, 1)
show_invalid = st.sidebar.checkbox("Show invalid links", True)
if st.sidebar.button("♻️ Clear Cache"):
    check_profile_exists.clear(); fetch_user_stats.clear(); st.success("Cache cleared.")

uploaded_file = st.file_uploader("Upload Excel/CSV with 'Student' and 'LeetCode URL'", type=["xlsx","xls","csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith((".xls",".xlsx")): df_uploaded = pd.read_excel(uploaded_file)
        else: df_uploaded = pd.read_csv(uploaded_file)
    except Exception as e: st.error(f"Failed to read file: {e}"); df_uploaded=None
    if df_uploaded is not None:
        st.session_state["uploaded_df"] = df_uploaded
else:
    if "uploaded_df" not in st.session_state: st.session_state["uploaded_df"]=None

# ------------------ Automated Processing Pipeline ------------------
if st.session_state.get("uploaded_df") is not None and st.session_state.get("results_df") is None:
    src_df: pd.DataFrame = st.session_state.get("uploaded_df")
    
    # --- 1. Validation Stage ---
    with st.spinner("Step 1: Validating all user profiles..."):
        student_col = next((c for c in src_df.columns if "student" in c.lower()), src_df.columns[0])
        url_col = next((c for c in src_df.columns if "leetcode" in c.lower()), None)

        if url_col is None:
            st.error("No LeetCode URL column found.")
        else:
            def validate_row(row_tuple):
                idx, row = row_tuple
                student = row.get(student_col, f"Student_{idx + 1}")
                url_raw = row.get(url_col, "")
                username = extract_username(str(url_raw))

                row_out = {
                    "Student": student,
                    "LeetCode URL": url_raw,
                    "LeetCode Username": username or "",
                    "Profile Valid": False,
                    "Reason": ""
                }

                if not username:
                    row_out["Reason"] = "Username extraction failed"
                elif check_profile_exists(username):
                    row_out["Profile Valid"] = True
                else:
                    row_out["Reason"] = "Profile not found"
                return row_out

            rows_to_process = list(src_df.iterrows())
            results = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(validate_row, row_tuple) for row_tuple in rows_to_process]
                for future in as_completed(futures):
                    results.append(future.result())

            validated_df = pd.DataFrame(results)
            st.session_state["validated_df"] = validated_df
            valid_df = validated_df[validated_df["Profile Valid"]]
            invalid_df = validated_df[~validated_df["Profile Valid"]]

            st.success(f"Validation complete — ✅ Valid: {len(valid_df)} | ❌ Invalid: {len(invalid_df)}")
            if show_invalid and not invalid_df.empty:
                st.subheader("Invalid rows (with reasons)")
                st.dataframe(invalid_df[["Student","LeetCode URL","Reason"]].head(100))

    # --- 2. Fetching Stage ---
    validated_df = st.session_state.get("validated_df")
    if validated_df is not None and not validated_df.empty:
        with st.spinner("Step 2: Fetching LeetCode stats for valid profiles..."):
            def fetch_row_stats(row):
                if not row["Profile Valid"]:
                    return {**row, "Easy":0,"Medium":0,"Hard":0,"Total":0}
                stats = fetch_user_stats(row["LeetCode Username"])
                return {**row,
                        "Easy": int(stats.get("Easy",0)),
                        "Medium": int(stats.get("Medium",0)),
                        "Hard": int(stats.get("Hard",0)),
                        "Total": int(stats.get("Total",0))}

            rows_to_process = [row for _, row in validated_df.iterrows()]
            results = []
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
    st.subheader("✅ Processed Student Stats (with validity info)")
    st.dataframe(df.sort_values(["Profile Valid","Total"],ascending=[False,False]).reset_index(drop=True))

    # Separate table for invalid URLs (post-processing view)
    if show_invalid:
        df_invalid = df[~df["Profile Valid"]]
        if not df_invalid.empty:
            st.subheader("❌ Invalid URLs")
            st.dataframe(df_invalid[["Student","LeetCode URL","Reason"]].reset_index(drop=True))
            with st.expander("Download invalid URLs as CSV"):
                invalid_download_name = st.text_input("File name for invalid URLs", "invalid_urls.csv", key="invalid_download_name")
                st.download_button(
                    "⬇️ Download Invalid URLs CSV",
                    df_invalid[["Student","LeetCode URL","Reason"]].to_csv(index=False).encode("utf-8"),
                    file_name=invalid_download_name,
                    mime="text/csv",
                    key="invalid_download_btn",
                )

    # Filter only valid for charts
    df_valid = df[df["Profile Valid"]]

    if not df_valid.empty:
        st.subheader("Key Performance Indicators")
        top = df_valid.loc[df_valid["Total"].idxmax()]
        bottom = df_valid.loc[df_valid["Total"].idxmin()]
        avg = df_valid["Total"].mean()
        k1,k2,k3 = st.columns(3)
        k1.metric("🏆 Top Performer", f"{top['Student']} ({top['Total']})")
        k2.metric("📉 Weakest Performer", f"{bottom['Student']} ({bottom['Total']})")
        k3.metric("📊 Average Solved", f"{avg:.1f}")

        st.subheader("Analysis: Medium vs. Hard Problems")
        fig_scatter = px.scatter(df_valid, x="Medium", y="Hard", 
                                 hover_data=['Student','Easy','Medium','Hard','Total'],
                                 color="Total", color_continuous_scale="Viridis")
        st.plotly_chart(fig_scatter,use_container_width=True)

        st.subheader("Leaderboard (Total Problems Solved)")
        fig_leader = px.bar(df_valid.sort_values("Total",ascending=False), x="Student", y="Total",
                            text="Total", color="Total", color_continuous_scale="Blues")
        fig_leader.update_traces(textposition="outside")
        st.plotly_chart(fig_leader,use_container_width=True)

        st.subheader("Problems by Difficulty")
        fig_stack = px.bar(df_valid.sort_values("Total",ascending=False), x="Student", y=["Easy","Medium","Hard"],
                           barmode="stack", color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig_stack,use_container_width=True)

        st.subheader("Overall Difficulty Distribution")
        totals={"Easy":int(df_valid["Easy"].sum()),"Medium":int(df_valid["Medium"].sum()),"Hard":int(df_valid["Hard"].sum())}
        dist_df = pd.DataFrame({"Difficulty":list(totals.keys()),"Count":list(totals.values())})
        fig_pie = px.pie(dist_df,names="Difficulty",values="Count",
                         color="Difficulty", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_pie,use_container_width=True)

        st.write("---")
        st.subheader("Interactive Filters & Export")
        col_a,col_b = st.columns([2,1])
        with col_a:
            top_n = st.number_input("Show top N students by Total (0=all)",0,1000,0,1)
            sort_by = st.selectbox("Sort by", ["Total","Easy","Medium","Hard","Student"],0)
        with col_b:
            download_name = st.text_input("Download file name", "leetcode_stats_processed.csv")
            if st.button("⬇️ Download CSV"):
                st.download_button("Save CSV", df.to_csv(index=False).encode("utf-8"),
                                   file_name=download_name, mime="text/csv")
        display_df = df.sort_values(sort_by, ascending=(sort_by=="Student"))
        if top_n>0: display_df=display_df.sort_values("Total",ascending=False).head(top_n)
        st.dataframe(display_df.reset_index(drop=True))

else:
    st.info("No results yet. Upload dataset and fetch stats.")
