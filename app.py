import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import re

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Bengaluru Metropolitan Transportation Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* General font and background */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f7f7f7;
    }
    /* Tab headers */
    .css-18ni7ap { font-size:20px; font-weight:bold; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# CONSTANTS
# -------------------------------
MAX_MAP_POINTS = 5000
SAMPLE_SEED = 42

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def extract_point(point):
    match = re.findall(r"POINT\s*\(([^)]+)\)", point)
    if match:
        lon, lat = match[0].split()
        return float(lon), float(lat)
    return np.nan, np.nan

def extract_linestring(ls):
    coords = re.findall(r"LINESTRING\s*\(([^)]+)\)", ls)
    if coords:
        pairs = coords[0].split(",")
        return [(float(p.split()[0]), float(p.split()[1])) for p in pairs]
    return []

def downsample_df(df: pd.DataFrame, n: int = MAX_MAP_POINTS, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    if df is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

# -------------------------------
# LOAD DATA FROM PATHS
# -------------------------------
# STOPS_PATH = "/Users/hemanth/Desktop/DataSets/routes/backend/analytics/bmtc_dashboard/stops.csv"
# AGGREGATED_PATH = "/Users/hemanth/Desktop/DataSets/routes/backend/analytics/bmtc_dashboard/aggregated.csv"
# ROUTES_PATH = "/Users/hemanth/Desktop/DataSets/routes/backend/analytics/bmtc_dashboard/routes.csv"

AGGREGATED_PATH = "https://raw.githubusercontent.com/HemanthGowdaaa/BMTC_DataAnalysis/main/aggregated.csv"
ROUTES_PATH = "https://raw.githubusercontent.com/HemanthGowdaaa/BMTC_DataAnalysis/main/routes.csv"
STOPS_PATH = "https://raw.githubusercontent.com/HemanthGowdaaa/BMTC_DataAnalysis/main/stops.csv"

@st.cache_data
def load_stops(path):
    df = pd.read_csv(path)
    df["lon"], df["lat"] = zip(*df["geometry"].apply(extract_point))
    return df

@st.cache_data
def load_aggregated(path):
    df = pd.read_csv(path)
    df["lon"], df["lat"] = zip(*df["geometry"].apply(extract_point))
    return df

@st.cache_data
def load_routes(path):
    df = pd.read_csv(path)
    df["coords"] = df["geometry"].apply(extract_linestring)
    return df

stops_df = load_stops(STOPS_PATH)
aggregated_df = load_aggregated(AGGREGATED_PATH)
routes_df = load_routes(ROUTES_PATH)

# -------------------------------
# PAGE TITLE
# -------------------------------
st.title("🚍 Bengaluru Metropolitan Transportation Analysis")
st.write("Interactive analytics for bus stops, aggregated summaries, and route geometries.")

# -------------------------------
# TABS
# -------------------------------
tabs = st.tabs([
    "📌 Overview", 
    "📊 Statistics", 
    "📈 Visualizations", 
    "🚌 Bus Stop Profiles",
    "🗺 Maps", 
     
    
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tabs[0]:
    st.header("📌 Dataset Overview")
    st.subheader("Stops Dataset")
    st.dataframe(stops_df.head())
    st.subheader("Aggregated Dataset")
    st.dataframe(aggregated_df.head())
    st.subheader("Routes Dataset")
    st.dataframe(routes_df.head())

# ============================================================
# TAB 2: STATISTICS
# ============================================================
# with tabs[1]:
#     st.header("📊 Statistical Summary")
#     st.subheader("Summary Statistics")
#     st.write(aggregated_df[["trip_count","route_count"]].describe())

#     st.subheader("Variability Metrics")
#     col1, col2 = st.columns(2)
#     with col1:
#         tc = aggregated_df["trip_count"]
#         st.write("### Trip Count")
#         st.write(f"Std Dev: {tc.std():.2f}")
#         st.write(f"MAD: {tc.mad():.2f}")
#         st.write(f"IQR: {tc.quantile(0.75) - tc.quantile(0.25):.2f}")
#     with col2:
#         rc = aggregated_df["route_count"]
#         st.write("### Route Count")
#         st.write(f"Std Dev: {rc.std():.2f}")
#         st.write(f"MAD: {rc.mad():.2f}")
#         st.write(f"IQR: {rc.quantile(0.75) - rc.quantile(0.25):.2f}")

# with tabs[1]:
#     st.header("📊 Statistical Summary")

#     # ============================
#     # Summary Stats
#     # ============================
#     st.subheader("Summary Statistics")
#     st.write(aggregated_df[["trip_count", "route_count"]].describe())

#     # ============================
#     # Additional Metrics
#     # ============================
#     st.subheader("📌 Additional Metrics")

#     colA, colB = st.columns(2)

#     # 🔹 Total Number of Bus Stops
#     with colA:
#         total_stops = stops_df.shape[0]
#         st.metric("Total Number of Bus Stops", total_stops)

#     # 🔹 Top 5 Routes With Most 'route_count'
#     with colB:
#         st.write("### Top 5 Routes With Most Routes")
#         top5_routes = aggregated_df.nlargest(5, "route_count")[["route_name", "route_count"]]
#         st.dataframe(top5_routes)

#     # ============================
#     # Variability Metrics
#     # ============================
#     st.subheader("Variability Metrics")
#     col1, col2 = st.columns(2)

#     with col1:
#         tc = aggregated_df["trip_count"]
#         st.write("### Trip Count")
#         st.write(f"Std Dev: {tc.std():.2f}")
#         st.write(f"MAD: {tc.mad():.2f}")
#         st.write(f"IQR: {tc.quantile(0.75) - tc.quantile(0.25):.2f}")

#     with col2:
#         rc = aggregated_df["route_count"]
#         st.write("### Route Count")
#         st.write(f"Std Dev: {rc.std():.2f}")
#         st.write(f"MAD: {rc.mad():.2f}")
#         st.write(f"IQR: {rc.quantile(0.75) - rc.quantile(0.25):.2f}")


with tabs[1]:
    st.header("📊 Statistical Summary")

    # ============================
    # Summary Statistics
    # ============================
    st.subheader("Summary Statistics")
    st.write(aggregated_df[["trip_count", "route_count"]].describe())

    # ============================
    # Additional Metrics
    # ============================
    st.subheader("📌 Additional Metrics")

    colA, colB = st.columns(2)

    # 🔹 Total Number of Bus Stops
    with colA:
        total_stops = stops_df.shape[0]
        st.metric("Total Number of Bus Stops", total_stops)

    # 🔹 Top 5 Routes With Highest route_count
    with colB:
        st.write("### Top 5 Routes With Most Routes")

        # Auto-detect route-name column
        possible_cols = ["name", "route_name", "route", "route_no", "route_id"]
        route_col = None
        for col in possible_cols:
            if col in aggregated_df.columns:
                route_col = col
                break

        if route_col is None:
            st.error("No route name column found in aggregated_df")
        else:
            top5_routes = aggregated_df.nlargest(5, "route_count")[[route_col, "route_count"]]
            st.dataframe(top5_routes)

    # ============================
    # Variability Metrics
    # ============================
    st.subheader("Variability Metrics")

    col1, col2 = st.columns(2)

    # ---- Trip Count Metrics ----
    with col1:
        tc = aggregated_df["trip_count"]
        st.write("### Trip Count")
        st.write(f"Std Dev: {tc.std():.2f}")
        st.write(f"MAD: {tc.mad():.2f}")
        st.write(f"IQR: {tc.quantile(0.75) - tc.quantile(0.25):.2f}")

    # ---- Route Count Metrics ----
    with col2:
        rc = aggregated_df["route_count"]
        st.write("### Route Count")
        st.write(f"Std Dev: {rc.std():.2f}")
        st.write(f"MAD: {rc.mad():.2f}")
        st.write(f"IQR: {rc.quantile(0.75) - rc.quantile(0.25):.2f}")

# ============================================================
# TAB 3: VISUALIZATIONS
# ============================================================
with tabs[2]:
    st.header("📈 Visualizations")
    min_trip, max_trip = st.slider(
        "Trip Count Range",
        int(aggregated_df["trip_count"].min()),
        int(aggregated_df["trip_count"].max()),
        (int(aggregated_df["trip_count"].min()), int(aggregated_df["trip_count"].max()))
    )
    filtered_df = aggregated_df[(aggregated_df["trip_count"]>=min_trip) & (aggregated_df["trip_count"]<=max_trip)]

    st.subheader("Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df[["trip_count","route_count"]], ax=ax)
    st.pyplot(fig)

    st.subheader("Histogram")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="route_count", y="trip_count", ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(filtered_df[["trip_count","route_count"]].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# ============================================================
# TAB 5: BUS STOP PROFILE
# ============================================================
with tabs[3]:
    st.header("🚌 Bus Stop Profiles")
    stop_n