# app.py
"""
Streamlit app: Student LeetCode Performance Dashboard
Features:
 - Upload Excel/CSV with columns: "Student" and "LeetCode URL"
 - Validate URLs (extract username + verify using LeetCode GraphQL)
 - Fetch per-user solved counts by difficulty (Easy, Medium, Hard) and Total
 - Caching of results to reduce repeated calls
 - Visualizations: Leaderboard, Stacked bar (difficulty), Pie chart, KPIs
 - Download processed CSV
 - Option to use a generated mock dataset for testing
"""

import re
import time
import io
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# ------------------ Configuration ------------------
GRAPHQL_URL = "https://leetcode.com/graphql"
# GraphQL queries
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
        submissions
      }
    }
    submitStatsGlobal {
      acSubmissionNum {
        difficulty
        count
        submissions
      }
    }
  }
}
"""
# Throttle delay between network calls (seconds)
DEFAULT_DELAY = 0.6

# ------------------ Helper functions ------------------

def extract_username(url: str) -> Optional[str]:
    """
    Extract LeetCode username from several URL formats:
      - https://leetcode.com/u/username/
      - https://leetcode.com/username/
      - username
    Returns username or None
    """
    if not isinstance(url, str):
        return None
    url = url.strip()
    if url == "":
        return None
    # try regex
    m = re.search(r'leetcode\.com/(?:u/)?([^/?#]+)/?', url)
    if m:
        return m.group(1)
    # fallback: last part
    if "/" in url:
        return url.rstrip("/").split("/")[-1]
    return url

def do_graphql_post(query: str, variables: Dict[str, Any], timeout: int = 10) -> Optional[Dict]:
    """Make a POST request to LeetCode GraphQL and return parsed JSON or None."""
    try:
        r = requests.post(GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        # else return None for non-200
        return None
    except requests.RequestException:
        return None

@st.cache_data(show_spinner=False)
def check_profile_exists(username: str) -> bool:
    """Return True if the LeetCode profile exists."""
    if not username:
        return False
    resp = do_graphql_post(QUERY_CHECK, {"username": username})
    if not resp:
        return False
    mu = resp.get("data", {}).get("matchedUser")
    return mu is not None

@st.cache_data(show_spinner=False)
def fetch_user_stats(username: str) -> Dict[str, int]:
    """
    Fetch solved counts by difficulty for a username.
    Returns dict like {"Easy": int, "Medium": int, "Hard": int, "Total": int}
    If fetch fails return zeros.
    """
    if not username:
        return {"Easy": 0, "Medium": 0, "Hard": 0, "Total": 0}
    resp = do_graphql_post(QUERY_STATS, {"username": username})
    if not resp:
        return {"Easy": 0, "Medium": 0, "Hard": 0, "Total": 0}
    matched = resp.get("data", {}).get("matchedUser")
    if not matched:
        return {"Easy": 0, "Medium": 0, "Hard": 0, "Total": 0}
    arr = matched.get("submitStats", {}).get("acSubmissionNum", [])
    # arr entries like {'difficulty': 'All'/'Easy'/'Medium'/'Hard', 'count': N, ...}
    counts = {entry.get("difficulty"): entry.get("count", 0) for entry in arr if entry.get("difficulty")}
    easy = counts.get("Easy", 0)
    medium = counts.get("Medium", 0)
    hard = counts.get("Hard", 0)
    total = counts.get("All", easy + medium + hard)
    return {"Easy": easy, "Medium": medium, "Hard": hard, "Total": total}

# ------------------ Mock dataset helper ------------------
def make_mock_dataset(n: int = 30) -> pd.DataFrame:
    """Return a mock dataset with n students (authentic-looking but fake usernames)."""
    import random
    students = [f"Student_{i}" for i in range(1, n+1)]
    urls = [f"https://leetcode.com/u/student_{i}/" for i in range(1, n+1)]
    data = {
        "Student": students,
        "LeetCode URL": urls,
        "Easy": [random.randint(5, 100) for _ in range(n)],
        "Medium": [random.randint(0, 80) for _ in range(n)],
        "Hard": [random.randint(0, 30) for _ in range(n)],
    }
    df = pd.DataFrame(data)
    df["Total"] = df["Easy"] + df["Medium"] + df["Hard"]
    return df

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Student LeetCode Dashboard", layout="wide")
st.title("📊 Student LeetCode Performance Dashboard")

# Sidebar options
st.sidebar.header("Options")
use_mock = st.sidebar.checkbox("Use mock dataset (no network calls)", value=False)
delay = st.sidebar.number_input("Delay between requests (s)", min_value=0.0, max_value=5.0, value=DEFAULT_DELAY, step=0.1)
show_invalid = st.sidebar.checkbox("Show invalid/unverified links after validation", value=True)
auto_fetch = st.sidebar.checkbox("Fetch stats automatically after validation", value=True)

# File uploader
uploaded_file = st.file_uploader("Upload Excel/CSV with columns: 'Student' and 'LeetCode URL' (or use mock dataset)", type=["xlsx", "xls", "csv"])

# If use mock, show button to generate
if use_mock and uploaded_file is None:
    if st.button("Generate mock dataset for testing"):
        df_uploaded = make_mock_dataset(30)
        st.success("Mock dataset generated — use it as input below.")
        st.dataframe(df_uploaded.head(10))
        # put df into session state for processing
        st.session_state["uploaded_df"] = df_uploaded
else:
    # if uploaded, read file
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith((".xls", ".xlsx")):
                df_uploaded = pd.read_excel(uploaded_file)
            else:
                df_uploaded = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            df_uploaded = None
        if df_uploaded is not None:
            # Require "Student" and "LeetCode URL" columns; allow variations
            # Normalize column names
            cols_lower = [c.lower() for c in df_uploaded.columns]
            # Determine Student column
            student_col = None
            url_col = None
            for c in df_uploaded.columns:
                if c.lower() in ["student", "name", "student name"]:
                    student_col = c
                if c.lower() in ["leetcode url", "leetcode", "profile", "profile url", "url"]:
                    url_col = c
            # fallbacks
            if student_col is None:
                # pick first column as student if not found
                student_col = df_uploaded.columns[0]
            if url_col is None and "leetcode" in "".join(cols_lower):
                # try to find column containing 'leetcode'
                for c in df_uploaded.columns:
                    if "leetcode" in c.lower():
                        url_col = c
                        break
            if url_col is None:
                st.warning("Could not find a 'LeetCode URL' column automatically. Please ensure your file has a column named 'LeetCode URL' or 'LeetCode'.")
            st.subheader("Preview uploaded data")
            st.dataframe(df_uploaded.head(10))
            # store into session state for later steps
            st.session_state["uploaded_df"] = df_uploaded
    else:
        # nothing uploaded yet
        if "uploaded_df" not in st.session_state:
            st.session_state["uploaded_df"] = None

# Main action buttons
st.write("---")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    validate_btn = st.button("🔍 Validate URLs")
with col2:
    fetch_btn = st.button("⚡ Fetch Stats (for validated users)")
with col3:
    reset_cache = st.button("♻️ Clear cached fetches")

if reset_cache:
    # Clear the caches we used
    try:
        check_profile_exists.clear()
        fetch_user_stats.clear()
        st.success("Cleared cached profile checks & stats.")
    except Exception:
        st.warning("Could not clear cache (function caching may have changed).")

# Validation step
validated_df = None
if validate_btn or (auto_fetch and fetch_btn):  # user pressed validation
    src_df: pd.DataFrame = st.session_state.get("uploaded_df")
    if src_df is None:
        st.warning("No uploaded dataset found. Upload a file or use the mock dataset option.")
    else:
        # Determine columns again
        cols = src_df.columns.tolist()
        # try to find columns
        student_col = next((c for c in cols if c.lower() in ["student", "student name", "name"]), cols[0])
        url_col = next((c for c in cols if c.lower() in ["leetcode url", "leetcode", "profile url", "profile", "url"]), None)
        if url_col is None:
            # try to find column containing 'leetcode'
            url_col = next((c for c in cols if "leetcode" in c.lower()), None)
        if url_col is None:
            st.error("Could not find a LeetCode URL column. Rename column to 'LeetCode URL' and re-upload.")
        else:
            st.info(f"Using columns -> Student: '{student_col}'  |  URL: '{url_col}'")
            rows = []
            invalid_rows = []
            progress = st.progress(0)
            total = len(src_df)
            i = 0
            placeholder = st.empty()
            for idx, row in src_df.iterrows():
                i += 1
                student = row.get(student_col, f"student_{i}")
                url_raw = row.get(url_col, "")
                username = extract_username(str(url_raw))
                # default structure
                row_out = {
                    "Student": student,
                    "LeetCode URL": url_raw,
                    "LeetCode Username": username or "",
                    "Profile Valid": False,
                    "Reason": ""
                }
                if not username:
                    row_out["Reason"] = "Could not extract username"
                    invalid_rows.append(row_out)
                else:
                    # if using mock dataset, we assume valid
                    if use_mock:
                        row_out["Profile Valid"] = True
                        rows.append(row_out)
                    else:
                        ok = check_profile_exists(username)
                        if ok:
                            row_out["Profile Valid"] = True
                            rows.append(row_out)
                        else:
                            row_out["Reason"] = "Profile not found"
                            invalid_rows.append(row_out)
                progress.progress(i/total)
                time.sleep(delay)  # polite delay
            progress.empty()
            placeholder.empty()
            validated_df = pd.DataFrame(rows)
            invalid_df = pd.DataFrame(invalid_rows)
            st.success(f"Validation complete — Valid: {len(validated_df)}  Invalid: {len(invalid_df)}")
            if show_invalid:
                st.subheader("Invalid or problematic rows")
                if invalid_df.empty:
                    st.write("None — all good!")
                else:
                    st.dataframe(invalid_df.head(50))
            # store validated
            st.session_state["validated_df"] = validated_df

# Fetching step (either pressed or auto_fetch + validated)
if fetch_btn or (auto_fetch and st.session_state.get("validated_df") is not None):
    vdf: pd.DataFrame = st.session_state.get("validated_df")
    if vdf is None or vdf.empty:
        st.warning("No validated profiles available. First upload and validate URLs.")
    else:
        st.info("Fetching LeetCode stats for validated profiles (uses cached results where available).")
        results: List[Dict[str, Any]] = []
        total = len(vdf)
        progress = st.progress(0)
        i = 0
        for idx, row in vdf.iterrows():
            i += 1
            username = row.get("LeetCode Username")
            student = row.get("Student")
            url_raw = row.get("LeetCode URL")
            if use_mock:
                # If mock dataset uploaded earlier (e.g., contains counts), preserve those
                # If uploaded_df already has counts, Try to grab them
                src_df = st.session_state.get("uploaded_df")
                if src_df is not None and all(c in src_df.columns for c in ["Easy", "Medium", "Hard"]):
                    matched = src_df[(src_df.get(student_col) == student) | (src_df.get(url_col) == url_raw)]
                    if not matched.empty:
                        # take first match
                        r0 = matched.iloc[0]
                        easy = int(r0.get("Easy", 0))
                        medium = int(r0.get("Medium", 0))
                        hard = int(r0.get("Hard", 0))
                        total_solved = easy + medium + hard
                        stats = {"Easy": easy, "Medium": medium, "Hard": hard, "Total": total_solved}
                    else:
                        stats = {"Easy": 0, "Medium": 0, "Hard": 0, "Total": 0}
                else:
                    # purely generated mock: random cache call (but we already created mock earlier)
                    stats = {"Easy": 0, "Medium": 0, "Hard": 0, "Total": 0}
            else:
                stats = fetch_user_stats(username)
                time.sleep(delay)  # polite delay between fetches

            results.append({
                "Student": student,
                "LeetCode URL": url_raw,
                "LeetCode Username": username,
                "Easy": int(stats.get("Easy", 0)),
                "Medium": int(stats.get("Medium", 0)),
                "Hard": int(stats.get("Hard", 0)),
                "Total": int(stats.get("Total", 0))
            })
            progress.progress(i/total)
        progress.empty()
        df_results = pd.DataFrame(results)
        # store
        st.session_state["results_df"] = df_results
        st.success("Fetched stats and built results table.")

# If results present, show KPIs and visualizations
if st.session_state.get("results_df") is not None:
    df = st.session_state["results_df"]
    st.subheader("✅ Processed Student Stats")
    st.dataframe(df.sort_values("Total", ascending=False).reset_index(drop=True))

    # KPIs
    st.subheader("Key Performance Indicators")
    top = df.loc[df["Total"].idxmax()]
    bottom = df.loc[df["Total"].idxmin()]
    avg = df["Total"].mean()
    k1, k2, k3 = st.columns(3)
    k1.metric("🏆 Top Performer", f"{top['Student']} ({top['Total']})")
    k2.metric("📉 Weakest Performer", f"{bottom['Student']} ({bottom['Total']})")
    k3.metric("📊 Average Solved", f"{avg:.1f}")

    st.write("---")
    # Visuals - Leaderboard
    st.subheader("Leaderboard (Total Problems Solved)")
    fig_leader = px.bar(df.sort_values("Total", ascending=False), x="Student", y="Total",
                        title="Leaderboard - Total Problems Solved", text="Total")
    fig_leader.update_traces(textposition="outside")
    st.plotly_chart(fig_leader, use_container_width=True)

    # Difficulty breakdown (stacked)
    st.subheader("Problems by Difficulty (per Student)")
    fig_stack = px.bar(df.sort_values("Total", ascending=False), x="Student", y=["Easy", "Medium", "Hard"],
                       title="Problems by Difficulty (stacked)", barmode="stack")
    st.plotly_chart(fig_stack, use_container_width=True)

    # Overall distribution pie
    st.subheader("Overall Difficulty Distribution")
    totals = {
        "Easy": int(df["Easy"].sum()),
        "Medium": int(df["Medium"].sum()),
        "Hard": int(df["Hard"].sum())
    }
    dist_df = pd.DataFrame({"Difficulty": list(totals.keys()), "Count": list(totals.values())})
    fig_pie = px.pie(dist_df, names="Difficulty", values="Count", title="Overall Difficulty Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Optional: allow user to pick top N
    st.write("---")
    st.subheader("Interactive filters & export")
    col_a, col_b = st.columns([2,1])
    with col_a:
        top_n = st.number_input("Show top N students by Total (0 = show all)", min_value=0, value=0, step=1)
        sort_by = st.selectbox("Sort table by", options=["Total", "Easy", "Medium", "Hard", "Student"], index=0)
    with col_b:
        download_name = st.text_input("Download file name", value="leetcode_stats_processed.csv")
        if st.button("⬇️ Download processed CSV"):
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Click to save CSV", data=csv_bytes, file_name=download_name, mime="text/csv")

    # show filtered table
    display_df = df.sort_values(by=sort_by, ascending=(sort_by == "Student"))
    if top_n > 0:
        display_df = display_df.sort_values("Total", ascending=False).head(top_n)
    st.dataframe(display_df.reset_index(drop=True))

    # Small help / next steps
    # st.write("---")
    # st.markdown("""
    # **Next steps & tips**
    # - To deploy: push this repo to GitHub and connect to Streamlit Community Cloud (share.streamlit.io).
    # - For large classes: run validation & fetch in batches or add a backend service to cache results periodically.
    # - Respect rate limits: if you see timeouts or errors, increase the `Delay between requests` in the sidebar.
    # - To track progress over time: store `results_df` with a timestamp into a CSV/DB on each run and visualize trends.
    # """)
    
        # 🔥 Extra Visuals (Power BI style)

    st.write("---")
    st.subheader("Additional Visuals")

    # Scatter Plot: Easy vs Hard solved
    st.write("📍 Easy vs Hard Problems Solved (per student)")
    fig_scatter = px.scatter(df, x="Easy", y="Hard", size="Total", color="Medium",
                             hover_name="Student",
                             title="Easy vs Hard Problems Solved")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Box Plot: Distribution of problems solved
    st.write("📦 Distribution of Problems Solved by Difficulty")
    fig_box = px.box(df.melt(id_vars=["Student"], value_vars=["Easy", "Medium", "Hard"]),
                     x="variable", y="value", points="all",
                     title="Distribution of Problems Solved (Easy, Medium, Hard)")
    st.plotly_chart(fig_box, use_container_width=True)

    # Treemap: Contribution of each student per difficulty
    st.write("🌳 Treemap of Student Contributions")
    treemap_df = df.melt(id_vars=["Student"], value_vars=["Easy", "Medium", "Hard"],
                         var_name="Difficulty", value_name="Count")
    fig_tree = px.treemap(treemap_df, path=["Difficulty", "Student"], values="Count",
                          color="Difficulty", title="Treemap of Contributions by Difficulty")
    st.plotly_chart(fig_tree, use_container_width=True)

    # Heatmap: Correlation between Easy, Medium, Hard, Total
    st.write("🔥 Correlation Heatmap")
    corr = df[["Easy", "Medium", "Hard", "Total"]].corr()
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto",
                         title="Correlation between Difficulties and Total Solved")
    st.plotly_chart(fig_heat, use_container_width=True)

    # Line Chart: Cumulative solved problems (simulated, if you track over time)
    st.write("📈 Simulated Progress Trend")
    # Fake cumulative data for demo (in real case: use timestamped data)
    trend_df = df[["Student", "Total"]].copy()
    trend_df = trend_df.sort_values("Total", ascending=False).head(5)  # top 5 students
    import numpy as np
    import random
    simulated = []
    for _, row in trend_df.iterrows():
        cum = np.cumsum([max(0, int(row["Total"]/10 + random.randint(-2, 5))) for _ in range(10)])
        simulated.append(pd.DataFrame({"Week": range(1, 11),
                                       "Solved": cum,
                                       "Student": row["Student"]}))
    trend_all = pd.concat(simulated)
    fig_line = px.line(trend_all, x="Week", y="Solved", color="Student",
                       title="(Simulated) Trend of Problems Solved Over Weeks")
    st.plotly_chart(fig_line, use_container_width=True)


else:
    st.info("No processed results yet. Upload a file, validate URLs, then fetch stats (or use Mock dataset).")
