 
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
    page_title="Bengaluru Metropolitan Transportation Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f7f7f7;
    }
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

# FIX — custom MAD function
def calc_mad(series):
    return (series - series.mean()).abs().mean()

# -------------------------------
# LOAD DATA
# -------------------------------
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
# TITLE
# -------------------------------
st.title("🚍 Bengaluru Metropolitan Transportation Data Analysis")
st.write("Interactive analytics for bus stops, aggregated summaries, and route geometries.")

# -------------------------------
# TABS
# -------------------------------
tabs = st.tabs([
    "📌 Overview", 
    "📊 Statistics", 
    "📈 Visualizations", 
    "🚌 Bus Stop Profiles", 
    "🛣 Route Explorer",
    "🗺 Maps"
])

# ============================================================
# TAB 1 — OVERVIEW
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
# TAB 2 — STATISTICS
# ============================================================
# with tabs[1]:
#     st.header("📊 Statistical Summary")
    
#     # Summary Statistics
#     st.subheader("Summary Statistics")
#     st.write(aggregated_df[["trip_count", "route_count"]].describe())

#     # Total Bus Stops
#     st.subheader("📍 Total Bus Stops")
#     st.metric("Total Bus Stops", len(stops_df))

#     # Top 5 Bus Stops by Routes
#     st.subheader("🏆 Top 5 Bus Stops With Highest Routes")
#     top5_routes = stops_df.nlargest(5, "route_count")[["name", "route_count"]]
#     st.dataframe(top5_routes)

#     # Variability Metrics for each column
#     st.subheader("📌 Variability Metrics")
#     col1, col2 = st.columns(2)

#     with col1:
#         tc = aggregated_df["trip_count"]
#         st.write("### Trip Count")
#         st.write(f"Std Dev: {tc.std():.2f}")
#         st.write(f"MAD: {calc_mad(tc):.2f}")
#         st.write(f"IQR: {(tc.quantile(0.75) - tc.quantile(0.25)):.2f}")

#     with col2:
#         rc = aggregated_df["route_count"]
#         st.write("### Route Count")
#         st.write(f"Std Dev: {rc.std():.2f}")
#         st.write(f"MAD: {calc_mad(rc):.2f}")
#         st.write(f"IQR: {(rc.quantile(0.75) - rc.quantile(0.25)):.2f}")

#     # Generalized Variability Table
#     st.subheader("📌 Variability Summary (All Numeric Columns)")
#     num_cols = aggregated_df.select_dtypes(include=np.number).columns
#     variability = []

#     for col in num_cols:
#         series = aggregated_df[col]
#         variability.append({
#             "Column": col,
#             "Std Dev": series.std(),
#             "MAD": calc_mad(series),
#             "IQR": series.quantile(0.75) - series.quantile(0.25)
#         })

#     st.dataframe(pd.DataFrame(variability))


with tabs[1]:
    st.header("📊 Statistical Summary")
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(aggregated_df[["trip_count", "route_count"]].describe())

    # Total Bus Stops
    st.subheader("📍 Total Bus Stops")
    st.metric("Total Bus Stops", len(stops_df))

    # Top 5 Bus Stops by Routes
    st.subheader("🏆 Top 5 Bus Stops With Highest Routes")
    top5_routes = stops_df.nlargest(5, "route_count")[["name", "route_count"]]
    st.dataframe(top5_routes)

    # Variability + Central Tendency Metrics
    st.subheader("📌 Variability & Central Tendency Metrics")
    col1, col2 = st.columns(2)

    with col1:
        tc = aggregated_df["trip_count"]
        st.write("### Trip Count")
        st.write(f"Mean: {tc.mean():.2f}")
        st.write(f"Median: {tc.median():.2f}")
        st.write(f"Mode: {tc.mode().iloc[0] if not tc.mode().empty else 'N/A'}")
        st.write(f"Std Dev: {tc.std():.2f}")
        st.write(f"MAD: {calc_mad(tc):.2f}")
        st.write(f"IQR: {(tc.quantile(0.75) - tc.quantile(0.25)):.2f}")

    with col2:
        rc = aggregated_df["route_count"]
        st.write("### Route Count")
        st.write(f"Mean: {rc.mean():.2f}")
        st.write(f"Median: {rc.median():.2f}")
        st.write(f"Mode: {rc.mode().iloc[0] if not rc.mode().empty else 'N/A'}")
        st.write(f"Std Dev: {rc.std():.2f}")
        st.write(f"MAD: {calc_mad(rc):.2f}")
        st.write(f"IQR: {(rc.quantile(0.75) - rc.quantile(0.25)):.2f}")

    # Generalized Summary Table
    st.subheader("📌 Statistical Summary (All Numeric Columns)")
    num_cols = aggregated_df.select_dtypes(include=np.number).columns
    stats = []

    for col in num_cols:
        series = aggregated_df[col]
        stats.append({
            "Column": col,
            "Mean": series.mean(),
            "Median": series.median(),
            "Mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
            "Std Dev": series.std(),
            "MAD": calc_mad(series),
            "IQR": series.quantile(0.75) - series.quantile(0.25)
        })

    st.dataframe(pd.DataFrame(stats))



# ============================================================
# TAB 3 — VISUALIZATIONS
# ============================================================
# with tabs[2]:
#     st.header("📈 Visualizations")

#     min_trip, max_trip = st.slider(
#         "Trip Count Range",
#         int(aggregated_df["trip_count"].min()),
#         int(aggregated_df["trip_count"].max()),
#         (int(aggregated_df["trip_count"].min()), int(aggregated_df["trip_count"].max()))
#     )

#     filtered_df = aggregated_df[
#         (aggregated_df["trip_count"] >= min_trip) &
#         (aggregated_df["trip_count"] <= max_trip)
#     ]

#     # Boxplot
#     st.subheader("Boxplot")
#     fig, ax = plt.subplots()
#     sns.boxplot(data=filtered_df[["trip_count", "route_count"]], ax=ax)
#     st.pyplot(fig)

#     # Histogram
#     st.subheader("Histogram")
#     fig, ax = plt.subplots()
#     sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
#     st.pyplot(fig)

#     # Density Plot
#     st.subheader("Density Plot")
#     fig, ax = plt.subplots()
#     sns.kdeplot(filtered_df["trip_count"], fill=True, ax=ax)
#     st.pyplot(fig)

#     # Scatter Plot
#     st.subheader("Scatter Plot")
#     fig, ax = plt.subplots()
#     sns.scatterplot(data=filtered_df, x="route_count", y="trip_count", ax=ax)
#     st.pyplot(fig)

#     # Hexbin Plot
#     st.subheader("Hexagonal Binning Plot")
#     fig, ax = plt.subplots()
#     hb = ax.hexbin(filtered_df["route_count"], filtered_df["trip_count"], gridsize=30, cmap="inferno")
#     plt.colorbar(hb)
#     st.pyplot(fig)

#     # Contour Plot
#     st.subheader("Contour Plot")
#     fig, ax = plt.subplots()
#     sns.kdeplot(
#         x=filtered_df["route_count"],
#         y=filtered_df["trip_count"],
#         cmap="coolwarm",
#         levels=10,
#         ax=ax
#     )
#     st.pyplot(fig)

#     # Violin Plot
#     st.subheader("Violin Plot")
#     fig, ax = plt.subplots()
#     sns.violinplot(data=filtered_df["trip_count"], ax=ax)
#     st.pyplot(fig)

#     # Correlation Matrix
#     st.subheader("Correlation Matrix")
#     corr = filtered_df.select_dtypes(include=np.number).corr()
#     fig, ax = plt.subplots(figsize=(6,4))
#     sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
#     st.pyplot(fig)



# ============================================================
# TAB 3 — VISUALIZATIONS
# ============================================================
# with tabs[2]:
#     st.header("📈 Visualizations")

#     # --------------------------------------------------------
#     # FILTER USING SLIDER
#     # --------------------------------------------------------
#     min_trip, max_trip = st.slider(
#         "Trip Count Range",
#         int(aggregated_df["trip_count"].min()),
#         int(aggregated_df["trip_count"].max()),
#         (
#             int(aggregated_df["trip_count"].min()),
#             int(aggregated_df["trip_count"].max())
#         )
#     )

#     filtered_df = aggregated_df[
#         (aggregated_df["trip_count"] >= min_trip) &
#         (aggregated_df["trip_count"] <= max_trip)
#     ]

#     # --------------------------------------------------------
#     # NORMALIZATION (Z-SCORE STANDARDIZATION)
#     # --------------------------------------------------------
#     normalized_df = filtered_df.copy()

#     for col in ["trip_count", "route_count"]:
#         normalized_df[col + "_z"] = (
#             normalized_df[col] - normalized_df[col].mean()
#         ) / normalized_df[col].std()

#     # --------------------------------------------------------
#     # BOXPLOT
#     # --------------------------------------------------------
#     st.subheader("Boxplot")
#     fig, ax = plt.subplots()
#     sns.boxplot(data=filtered_df[["trip_count", "route_count"]], ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # HISTOGRAM
#     # --------------------------------------------------------
#     st.subheader("Histogram")
#     fig, ax = plt.subplots()
#     sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # DENSITY PLOT
#     # --------------------------------------------------------
#     st.subheader("Density Plot")
#     fig, ax = plt.subplots()
#     sns.kdeplot(filtered_df["trip_count"], fill=True, ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # SCATTER PLOT (RAW DATA)
#     # --------------------------------------------------------
#     st.subheader("Scatter Plot (Raw Data)")
#     fig, ax = plt.subplots()
#     sns.scatterplot(
#         data=filtered_df,
#         x="route_count",
#         y="trip_count",
#         ax=ax
#     )
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # SCATTER PLOT (NORMALIZED DATA)
#     # --------------------------------------------------------
#     st.subheader("Scatter Plot (Normalized Data)")
#     fig, ax = plt.subplots()
#     sns.scatterplot(
#         data=normalized_df,
#         x="route_count_z",
#         y="trip_count_z",
#         ax=ax
#     )
#     ax.set_xlabel("Route Count (Z-score)")
#     ax.set_ylabel("Trip Count (Z-score)")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # HEXAGONAL BINNING
#     # --------------------------------------------------------
#     st.subheader("Hexagonal Binning Plot")
#     fig, ax = plt.subplots()
#     hb = ax.hexbin(
#         filtered_df["route_count"],
#         filtered_df["trip_count"],
#         gridsize=30,
#         cmap="inferno"
#     )
#     plt.colorbar(hb)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # CONTOUR PLOT
#     # --------------------------------------------------------
#     st.subheader("Contour Plot")
#     fig, ax = plt.subplots()
#     sns.kdeplot(
#         x=filtered_df["route_count"],
#         y=filtered_df["trip_count"],
#         cmap="coolwarm",
#         levels=10,
#         ax=ax
#     )
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # VIOLIN PLOT
#     # --------------------------------------------------------
#     st.subheader("Violin Plot")
#     fig, ax = plt.subplots()
#     sns.violinplot(
#         data=filtered_df["trip_count"],
#         ax=ax
#     )
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # CORRELATION MATRIX (RAW DATA)
#     # --------------------------------------------------------
#     st.subheader("Correlation Matrix (Raw Data)")
#     corr = filtered_df.select_dtypes(include=np.number).corr()

#     fig, ax = plt.subplots(figsize=(6, 4))
#     sns.heatmap(
#         corr,
#         annot=True,
#         cmap="viridis",
#         ax=ax
#     )
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # PEARSON CORRELATION (EXPLICIT)
#     # --------------------------------------------------------
#     st.subheader("Pearson Correlation")

#     pearson_corr = normalized_df["trip_count_z"].corr(
#         normalized_df["route_count_z"],
#         method="pearson"
#     )

#     st.write(
#         f"**Pearson Correlation between Trip Count and Route Count:** `{pearson_corr:.4f}`"
#     )

#     # --------------------------------------------------------
#     # CORRELATION MATRIX (NORMALIZED DATA)
#     # --------------------------------------------------------
#     st.subheader("Correlation Matrix (Normalized Data)")

#     norm_corr = normalized_df[
#         ["trip_count_z", "route_count_z"]
#     ].corr()

#     fig, ax = plt.subplots(figsize=(5, 3))
#     sns.heatmap(
#         norm_corr,
#         annot=True,
#         cmap="coolwarm",
#         ax=ax
#     )
#     st.pyplot(fig)




# with tabs[2]:
#     st.header("📈 Visualizations")

#     # --------------------------------------------------------
#     # FILTER USING SLIDER
#     # --------------------------------------------------------
#     min_trip, max_trip = st.slider(
#         "Trip Count Range",
#         int(aggregated_df["trip_count"].min()),
#         int(aggregated_df["trip_count"].max()),
#         (
#             int(aggregated_df["trip_count"].min()),
#             int(aggregated_df["trip_count"].max())
#         )
#     )

#     filtered_df = aggregated_df[
#         (aggregated_df["trip_count"] >= min_trip) &
#         (aggregated_df["trip_count"] <= max_trip)
#     ].copy()

#     # --------------------------------------------------------
#     # NORMALIZATION (Z-SCORE)
#     # --------------------------------------------------------
#     normalized_df = filtered_df.copy()

#     for col in ["trip_count", "route_count"]:
#         normalized_df[col + "_z"] = (
#             normalized_df[col] - normalized_df[col].mean()
#         ) / normalized_df[col].std()

#     # --------------------------------------------------------
#     # BOXPLOT
#     # --------------------------------------------------------
#     st.subheader("Boxplot")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.boxplot(data=filtered_df[["trip_count", "route_count"]], ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # HISTOGRAM
#     # --------------------------------------------------------
#     st.subheader("Histogram")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
#     ax.set_xlabel("Trip Count")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # DENSITY PLOT
#     # --------------------------------------------------------
#     st.subheader("Density Plot")
#     fig, ax = plt.subplots(figsize=(8, 4))
#     sns.kdeplot(filtered_df["trip_count"], fill=True, ax=ax)
#     ax.set_xlabel("Trip Count")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # LOG TRANSFORMATION (BEFORE vs AFTER)
#     # --------------------------------------------------------
#     st.subheader("Log Transformation on Trip Count (Before vs After)")

#     filtered_df["trip_count_log"] = np.log1p(filtered_df["trip_count"])

#     col1, col2 = st.columns(2)

#     with col1:
#         st.write("### Before Log Transformation")
#         fig, ax = plt.subplots()
#         sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
#         ax.set_xlabel("Trip Count")
#         st.pyplot(fig)

#     with col2:
#         st.write("### After Log Transformation (log(1 + x))")
#         fig, ax = plt.subplots()
#         sns.histplot(filtered_df["trip_count_log"], kde=True, ax=ax)
#         ax.set_xlabel("Log Trip Count")
#         st.pyplot(fig)

#     # --------------------------------------------------------
#     # SCATTER PLOT (RAW)
#     # --------------------------------------------------------
#     st.subheader("Scatter Plot (Raw Data)")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.scatterplot(
#         data=filtered_df,
#         x="route_count",
#         y="trip_count",
#         ax=ax
#     )
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # SCATTER PLOT (NORMALIZED)
#     # --------------------------------------------------------
#     st.subheader("Scatter Plot (Normalized Data)")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.scatterplot(
#         data=normalized_df,
#         x="route_count_z",
#         y="trip_count_z",
#         ax=ax
#     )
#     ax.set_xlabel("Route Count (Z-score)")
#     ax.set_ylabel("Trip Count (Z-score)")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # HEXBIN PLOT
#     # --------------------------------------------------------
#     st.subheader("Hexagonal Binning Plot")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     hb = ax.hexbin(
#         filtered_df["route_count"],
#         filtered_df["trip_count"],
#         gridsize=30,
#         cmap="inferno"
#     )
#     plt.colorbar(hb, ax=ax)
#     ax.set_xlabel("Route Count")
#     ax.set_ylabel("Trip Count")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # CONTOUR PLOT
#     # --------------------------------------------------------
#     st.subheader("Contour Plot")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.kdeplot(
#         x=filtered_df["route_count"],
#         y=filtered_df["trip_count"],
#         cmap="coolwarm",
#         levels=10,
#         ax=ax
#     )
#     ax.set_xlabel("Route Count")
#     ax.set_ylabel("Trip Count")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # VIOLIN PLOT
#     # --------------------------------------------------------
#     st.subheader("Violin Plot (Trip Count)")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.violinplot(y=filtered_df["trip_count"], ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # CORRELATION MATRIX (RAW DATA)
#     # --------------------------------------------------------
#     st.subheader("Correlation Matrix (Raw Data)")
#     corr = filtered_df.select_dtypes(include=np.number).corr()

#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # PEARSON CORRELATION
#     # --------------------------------------------------------
#     st.subheader("Pearson Correlation")

#     pearson_corr = normalized_df["trip_count_z"].corr(
#         normalized_df["route_count_z"],
#         method="pearson"
#     )

#     st.write(
#         f"**Pearson Correlation between Trip Count and Route Count:** `{pearson_corr:.4f}`"
#     )

#     # --------------------------------------------------------
#     # CORRELATION MATRIX (NORMALIZED DATA)
#     # --------------------------------------------------------
#     st.subheader("Correlation Matrix (Normalized Data)")
#     norm_corr = normalized_df[
#         ["trip_count_z", "route_count_z"]
#     ].corr()

#     fig, ax = plt.subplots(figsize=(5, 3))
#     sns.heatmap(norm_corr, annot=True, cmap="coolwarm", ax=ax)
#     st.pyplot(fig)


# with tabs[2]:
#     st.header("📈 Visualizations")

#     # --------------------------------------------------------
#     # FILTER USING SLIDER
#     # --------------------------------------------------------
#     min_trip, max_trip = st.slider(
#         "Trip Count Range",
#         int(aggregated_df["trip_count"].min()),
#         int(aggregated_df["trip_count"].max()),
#         (
#             int(aggregated_df["trip_count"].min()),
#             int(aggregated_df["trip_count"].max())
#         )
#     )

#     filtered_df = aggregated_df[
#         (aggregated_df["trip_count"] >= min_trip) &
#         (aggregated_df["trip_count"] <= max_trip)
#     ].copy()

#     # --------------------------------------------------------
#     # NORMALIZATION (Z-SCORE)
#     # --------------------------------------------------------
#     normalized_df = filtered_df.copy()

#     for col in ["trip_count", "route_count"]:
#         normalized_df[col + "_z"] = (
#             normalized_df[col] - normalized_df[col].mean()
#         ) / normalized_df[col].std()

#     # --------------------------------------------------------
#     # BOXPLOT
#     # --------------------------------------------------------
#     st.subheader("Boxplot")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.boxplot(data=filtered_df[["trip_count", "route_count"]], ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # HISTOGRAM
#     # --------------------------------------------------------
#     st.subheader("Histogram")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
#     ax.set_xlabel("Trip Count")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # DENSITY PLOT
#     # --------------------------------------------------------
#     st.subheader("Density Plot")
#     fig, ax = plt.subplots(figsize=(8, 4))
#     sns.kdeplot(filtered_df["trip_count"], fill=True, ax=ax)
#     ax.set_xlabel("Trip Count")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # LOG TRANSFORMATION (BEFORE vs AFTER)
#     # --------------------------------------------------------
#     st.subheader("Log Transformation on Trip Count (Before vs After)")

#     filtered_df["trip_count_log"] = np.log1p(filtered_df["trip_count"])

#     col1, col2 = st.columns(2)

#     with col1:
#         st.write("### Before Log Transformation")
#         fig, ax = plt.subplots()
#         sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
#         ax.set_xlabel("Trip Count")
#         st.pyplot(fig)

#     with col2:
#         st.write("### After Log Transformation (log(1 + x))")
#         fig, ax = plt.subplots()
#         sns.histplot(filtered_df["trip_count_log"], kde=True, ax=ax)
#         ax.set_xlabel("Log Trip Count")
#         st.pyplot(fig)

#     # --------------------------------------------------------
#     # HEXBIN PLOT (RAW)
#     # --------------------------------------------------------
#     st.subheader("Hexagonal Binning Plot")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     hb = ax.hexbin(
#         filtered_df["route_count"],
#         filtered_df["trip_count"],
#         gridsize=30,
#         cmap="inferno"
#     )
#     plt.colorbar(hb, ax=ax)
#     ax.set_xlabel("Route Count")
#     ax.set_ylabel("Trip Count")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # HEXBIN PLOT (LOG-TRANSFORMED TRIP COUNT)
#     # --------------------------------------------------------
#     st.subheader("Hexagonal Binning Plot (Log-Transformed Trip Count)")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     hb = ax.hexbin(
#         filtered_df["route_count"],
#         filtered_df["trip_count_log"],
#         gridsize=30,
#         cmap="magma"
#     )
#     plt.colorbar(hb, ax=ax)
#     ax.set_xlabel("Route Count")
#     ax.set_ylabel("Log Trip Count")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # SCATTER PLOT (RAW)
#     # --------------------------------------------------------
#     st.subheader("Scatter Plot (Raw Data)")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.scatterplot(
#         data=filtered_df,
#         x="route_count",
#         y="trip_count",
#         ax=ax
#     )
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # SCATTER PLOT (NORMALIZED)
#     # --------------------------------------------------------
#     st.subheader("Scatter Plot (Normalized Data)")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.scatterplot(
#         data=normalized_df,
#         x="route_count_z",
#         y="trip_count_z",
#         ax=ax
#     )
#     ax.set_xlabel("Route Count (Z-score)")
#     ax.set_ylabel("Trip Count (Z-score)")
#     st.pyplot(fig)


#     # --------------------------------------------------------
#     # CONTOUR PLOT
#     # --------------------------------------------------------
#     st.subheader("Contour Plot")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.kdeplot(
#         x=filtered_df["route_count"],
#         y=filtered_df["trip_count"],
#         cmap="coolwarm",
#         levels=10,
#         ax=ax
#     )
#     ax.set_xlabel("Route Count")
#     ax.set_ylabel("Trip Count")
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # VIOLIN PLOT
#     # --------------------------------------------------------
#     st.subheader("Violin Plot (Trip Count)")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.violinplot(y=filtered_df["trip_count"], ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # CORRELATION MATRIX (RAW DATA)
#     # --------------------------------------------------------
#     st.subheader("Correlation Matrix (Raw Data)")
#     corr = filtered_df.select_dtypes(include=np.number).corr()
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # PEARSON CORRELATION WITH P-VALUE
#     # --------------------------------------------------------
#     st.subheader("Pearson Correlation")
#     from scipy.stats import pearsonr
#     pearson_r, pearson_p = pearsonr(
#         normalized_df["trip_count_z"],
#         normalized_df["route_count_z"]
#     )
#     st.write(
#         f"**Pearson Correlation:** `{pearson_r:.4f}`  \n"
#         f"**p-value:** `{pearson_p:.4e}`"
#     )

#     # --------------------------------------------------------
#     # CORRELATION MATRIX (NORMALIZED DATA)
#     # --------------------------------------------------------
#     st.subheader("Correlation Matrix (Normalized Data)")
#     norm_corr = normalized_df[
#         ["trip_count_z", "route_count_z"]
#     ].corr()
#     fig, ax = plt.subplots(figsize=(5, 3))
#     sns.heatmap(norm_corr, annot=True, cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

#     # --------------------------------------------------------
#     # SPEARMAN CORRELATION PLOT
#     # --------------------------------------------------------
#     st.subheader("Spearman Correlation (Rank-Based)")
#     from scipy.stats import spearmanr
#     spearman_corr, spearman_p = spearmanr(
#         filtered_df["route_count"],
#         filtered_df["trip_count"]
#     )
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.scatterplot(
#         x=filtered_df["route_count"].rank(),
#         y=filtered_df["trip_count"].rank(),
#         ax=ax
#     )
#     ax.set_xlabel("Route Count (Rank)")
#     ax.set_ylabel("Trip Count (Rank)")
#     ax.set_title(
#         f"Spearman ρ = {spearman_corr:.4f}, p-value = {spearman_p:.4e}"
#     )
#     st.pyplot(fig)

with tabs[2]:
    st.header("📈 Visualizations")

    # --------------------------------------------------------
    # FILTER USING SLIDER
    # --------------------------------------------------------
    min_trip, max_trip = st.slider(
        "Trip Count Range",
        int(aggregated_df["trip_count"].min()),
        int(aggregated_df["trip_count"].max()),
        (
            int(aggregated_df["trip_count"].min()),
            int(aggregated_df["trip_count"].max())
        )
    )

    filtered_df = aggregated_df[
        (aggregated_df["trip_count"] >= min_trip) &
        (aggregated_df["trip_count"] <= max_trip)
    ].copy()

    # --------------------------------------------------------
    # NORMALIZATION (Z-SCORE)
    # --------------------------------------------------------
    normalized_df = filtered_df.copy()
    for col in ["trip_count", "route_count"]:
        normalized_df[col + "_z"] = (
            normalized_df[col] - normalized_df[col].mean()
        ) / normalized_df[col].std()

    # --------------------------------------------------------
    # BOXPLOT
    # --------------------------------------------------------
    st.subheader("Boxplot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=filtered_df[["trip_count", "route_count"]], ax=ax)
    st.pyplot(fig)

    # # --------------------------------------------------------
    # # HISTOGRAM WITH CUSTOM BINS
    # # --------------------------------------------------------
    # st.subheader("Histogram (Custom Bins to Reduce Outliers)")
    # bins = [0, 100, 200, 400, 600, 1000, 2000, 5000, 9000]
    # fig, ax = plt.subplots(figsize=(10, 5))
    # sns.histplot(filtered_df["trip_count"], bins=bins, kde=True, ax=ax)
    # ax.set_xlabel("Trip Count")
    # ax.set_ylabel("Frequency")
    # st.pyplot(fig)
    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------
    st.subheader("Histogram")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
    ax.set_xlabel("Trip Count")
    st.pyplot(fig)


    # --------------------------------------------------------
    # DENSITY PLOT
    # --------------------------------------------------------
    st.subheader("Density Plot")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(filtered_df["trip_count"], fill=True, ax=ax)
    ax.set_xlabel("Trip Count")
    st.pyplot(fig)
    
    # --------------------------------------------------------
    # DYNAMIC HISTOGRAM WITH RADIO MENU & LOG TRANSFORM
    # --------------------------------------------------------
    st.subheader("Dynamic Histogram with Trip Count Range & Log Transform")

    # Define trip count ranges for the radio menu
    range_options = {
        "0–200": (0, 200),
        "200–400": (200, 400),
        "400–600": (400, 600),
        "600–1000": (600, 1000),
        "1000–2000": (1000, 2000),
        "2000–5000": (2000, 5000),
        "5000–9416": (5000, 9416)
    }

    # Horizontal radio menu
    selected_range = st.radio(
        "Select Trip Count Range:",
        options=list(range_options.keys()),
        horizontal=True
    )

    # Checkbox to toggle log transformation
    log_checkbox = st.checkbox("Apply Log Transformation (log(Trip Count + 1))")

    # Filter dataset dynamically based on selected range
    min_val, max_val = range_options[selected_range]
    filtered_hist_df = filtered_df[
        (filtered_df["trip_count"] >= min_val) &
        (filtered_df["trip_count"] <= max_val)
    ].copy()

    # Apply log transformation if checkbox is checked
    if log_checkbox:
        filtered_hist_df["trip_count_plot"] = np.log1p(filtered_hist_df["trip_count"])
        xlabel = "Log(Trip Count + 1)"
        title = f"Histogram of Log-Transformed Trip Count ({selected_range})"
    else:
        filtered_hist_df["trip_count_plot"] = filtered_hist_df["trip_count"]
        xlabel = "Trip Count"
        title = f"Histogram of Trip Count ({selected_range})"

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(filtered_hist_df["trip_count_plot"], kde=True, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    st.pyplot(fig)



    # --------------------------------------------------------
    # LOG TRANSFORMATION (BEFORE vs AFTER)
    # --------------------------------------------------------
    st.subheader("Log Transformation on Trip Count (Before vs After)")
    filtered_df["trip_count_log"] = np.log1p(filtered_df["trip_count"])

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Before Log Transformation")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
        ax.set_xlabel("Trip Count")
        st.pyplot(fig)

    with col2:
        st.write("### After Log Transformation (log(1 + x))")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(filtered_df["trip_count_log"], kde=True, ax=ax)
        ax.set_xlabel("Log Trip Count")
        st.pyplot(fig)

    # --------------------------------------------------------
    # HEXBIN PLOT (RAW)
    # --------------------------------------------------------
    st.subheader("Hexagonal Binning Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    hb = ax.hexbin(
        filtered_df["route_count"],
        filtered_df["trip_count"],
        gridsize=30,
        cmap="inferno"
    )
    plt.colorbar(hb, ax=ax)
    ax.set_xlabel("Route Count")
    ax.set_ylabel("Trip Count")
    st.pyplot(fig)

    # --------------------------------------------------------
    # HEXBIN PLOT (LOG-TRANSFORMED TRIP COUNT)
    # --------------------------------------------------------
    st.subheader("Hexagonal Binning Plot (Log-Transformed Trip Count)")
    fig, ax = plt.subplots(figsize=(10, 5))
    hb = ax.hexbin(
        filtered_df["route_count"],
        filtered_df["trip_count_log"],
        gridsize=30,
        cmap="magma"
    )
    plt.colorbar(hb, ax=ax)
    ax.set_xlabel("Route Count")
    ax.set_ylabel("Log Trip Count")
    st.pyplot(fig)

    # --------------------------------------------------------
    # SCATTER PLOT (RAW)
    # --------------------------------------------------------
    st.subheader("Scatter Plot (Raw Data)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=filtered_df,
        x="route_count",
        y="trip_count",
        ax=ax
    )
    st.pyplot(fig)

    # --------------------------------------------------------
    # SCATTER PLOT (NORMALIZED)
    # --------------------------------------------------------
    st.subheader("Scatter Plot (Normalized Data)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=normalized_df,
        x="route_count_z",
        y="trip_count_z",
        ax=ax
    )
    ax.set_xlabel("Route Count (Z-score)")
    ax.set_ylabel("Trip Count (Z-score)")
    st.pyplot(fig)


    # --------------------------------------------------------
    # CONTOUR PLOT
    # --------------------------------------------------------
    st.subheader("Contour Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(
        x=filtered_df["route_count"],
        y=filtered_df["trip_count"],
        cmap="coolwarm",
        levels=10,
        ax=ax
    )
    ax.set_xlabel("Route Count")
    ax.set_ylabel("Trip Count")
    st.pyplot(fig)

    # --------------------------------------------------------
    # VIOLIN PLOT
    # --------------------------------------------------------
    st.subheader("Violin Plot (Trip Count)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(y=filtered_df["trip_count"], ax=ax)
    st.pyplot(fig)

    # --------------------------------------------------------
    # CORRELATION MATRIX (RAW DATA)
    # --------------------------------------------------------
    st.subheader("Correlation Matrix (Raw Data)")
    corr = filtered_df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
    st.pyplot(fig)

    # --------------------------------------------------------
    # PEARSON CORRELATION WITH P-VALUE
    # --------------------------------------------------------
    st.subheader("Pearson Correlation")
    from scipy.stats import pearsonr
    pearson_r, pearson_p = pearsonr(
        normalized_df["trip_count_z"],
        normalized_df["route_count_z"]
    )
    st.write(
        f"**Pearson Correlation:** `{pearson_r:.4f}`  \n"
        f"**p-value:** `{pearson_p:.4e}`"
    )

    # --------------------------------------------------------
    # CORRELATION MATRIX (NORMALIZED DATA)
    # --------------------------------------------------------
    st.subheader("Correlation Matrix (Normalized Data)")
    norm_corr = normalized_df[
        ["trip_count_z", "route_count_z"]
    ].corr()
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(norm_corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --------------------------------------------------------
    # SPEARMAN CORRELATION PLOT
    # --------------------------------------------------------
    st.subheader("Spearman Correlation (Rank-Based)")
    from scipy.stats import spearmanr
    spearman_corr, spearman_p = spearmanr(
        filtered_df["route_count"],
        filtered_df["trip_count"]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        x=filtered_df["route_count"].rank(),
        y=filtered_df["trip_count"].rank(),
        ax=ax
    )
    ax.set_xlabel("Route Count (Rank)")
    ax.set_ylabel("Trip Count (Rank)")
    ax.set_title(
        f"Spearman ρ = {spearman_corr:.4f}, p-value = {spearman_p:.4e}"
    )
    st.pyplot(fig)




# ============================================================
# TAB 4 — BUS STOP PROFILE
# ============================================================
with tabs[3]:
    st.header("🚌 Bus Stop Profiles")

    stop_name = st.selectbox("Select a Bus Stop", stops_df["name"].tolist())
    selected_stop = stops_df[stops_df["name"] == stop_name].iloc[0]

    st.write(f"**Stop Name:** {selected_stop['name']}")
    st.write(f"**Trip Count:** {selected_stop['trip_count']}")
    st.write(f"**Route Count:** {selected_stop['route_count']}")

    st.map(pd.DataFrame({"lat": [selected_stop["lat"]], "lon": [selected_stop["lon"]]}))

# ============================================================
# TAB 5 — ROUTE EXPLORER
# ============================================================
with tabs[4]:
    st.header("🛣 Route Explorer")

    route_name = st.selectbox("Select a Route", routes_df["name"].tolist())
    route_data = routes_df[routes_df["name"] == route_name].iloc[0]

    coords = route_data["coords"]
    path_coords = [[lon, lat] for lon, lat in coords]

    st.write(f"### {route_name}")

    if "full_name" in route_data:
        st.write(f"**Full Name:** {route_data['full_name']}")
    if "trip_count" in route_data:
        st.write(f"**Trip Count:** {route_data['trip_count']}")
    if "stop_count" in route_data:
        st.write(f"**Stop Count:** {route_data['stop_count']}")

    path_df = pd.DataFrame([{"path": path_coords, "name": route_name}])

    route_layer = pdk.Layer(
        "PathLayer",
        data=path_df,
        get_path="path",
        get_width=5,
        get_color=[0, 128, 255],
        pickable=True
    )

    view = pdk.ViewState(
        latitude=np.mean([lat for _, lat in coords]),
        longitude=np.mean([lon for lon, _ in coords]),
        zoom=12,
        pitch=45
    )

    st.pydeck_chart(pdk.Deck(
        layers=[route_layer],
        initial_view_state=view,
        tooltip={"text": "{name}"}
    ))

# ============================================================
# TAB 6 — MAPS (OPENSTREETMAP)
# ============================================================
with tabs[5]:
    st.header("🗺 Spatial Visualizations (OpenStreetMap)")

    osm_layer = pdk.Layer(
        "TileLayer",
        data=None,
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        get_tile_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    )

    # Bus Stops Map
    st.subheader("Bus Stops Map")
    stops_map_df = downsample_df(stops_df[['name','lon','lat','trip_count','route_count']])

    stop_layer = pdk.Layer(
        "ScatterplotLayer",
        data=stops_map_df,
        get_position='[lon, lat]',
        get_radius=50,
        get_fill_color='[0, 128, 255, 140]',
        pickable=True
    )

    view = pdk.ViewState(
        latitude=stops_map_df['lat'].mean(),
        longitude=stops_map_df['lon'].mean(),
        zoom=11,
        pitch=30
    )

    st.pydeck_chart(pdk.Deck(
        layers=[osm_layer, stop_layer],
        initial_view_state=view,
        tooltip={"text": "{name}\nTrips: {trip_count}\nRoutes: {route_count}"}
    ))

    st.markdown("---")

    # Routes Map
    st.subheader("Routes Map")

    paths = []
    for _, row in routes_df.iterrows():
        coords = row["coords"]
        if len(coords) > 200:
            idx = np.round(np.linspace(0, len(coords)-1, 200)).astype(int)
            coords = [coords[i] for i in idx]

        paths.append({"name": row["name"], "path": [[lon, lat] for lon, lat in coords]})

    route_layer = pdk.Layer(
        "PathLayer",
        data=paths,
        get_path="path",
        get_width=4,
        get_color=[255, 0, 0],
        pickable=True
    )

    all_coords = [pt for p in paths for pt in p["path"]]
    center_lat = np.mean([c[1] for c in all_coords])
    center_lon = np.mean([c[0] for c in all_coords])

    view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=30)

    st.pydeck_chart(pdk.Deck(
        layers=[osm_layer, route_layer],
        initial_view_state=view,
        tooltip={"text": "{name}"}
    ))
    # Conclusion
    st.markdown("""
    ### 📝 Conclusion (Key Takeaways)

    - **Trip count and route count are positively correlated**, meaning bus stops with more routes tend to have more trips.  
    - **Trip count distribution is skewed**, showing that a few stops handle very high activity.  
    - **High variability (Std Dev, MAD, IQR)** shows diverse usage patterns across stops.  
    - **Hexbin and Contour plots show hotspots**, indicating concentrated transport activity.  
    - **Violin plot reveals multiple distribution peaks**, suggesting different stop categories.

    Overall, Bengaluru bus operations show **high clustering**, **strong route-trip relationships**, and **several high-traffic hotspots**.
    """)






















































# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pydeck as pdk
# import re

# # -------------------------------
# # PAGE CONFIG
# # -------------------------------
# st.set_page_config(
#     page_title="Bengaluru Metropolitan Transportation Data Analysis",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
#     <style>
#     body {
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         background-color: #f7f7f7;
#     }
#     .css-18ni7ap { font-size:20px; font-weight:bold; }
#     </style>
# """, unsafe_allow_html=True)

# # -------------------------------
# # CONSTANTS
# # -------------------------------
# MAX_MAP_POINTS = 5000
# SAMPLE_SEED = 42

# # -------------------------------
# # UTILITY FUNCTIONS
# # -------------------------------
# def extract_point(point):
#     match = re.findall(r"POINT\s*\(([^)]+)\)", point)
#     if match:
#         lon, lat = match[0].split()
#         return float(lon), float(lat)
#     return np.nan, np.nan

# def extract_linestring(ls):
#     coords = re.findall(r"LINESTRING\s*\(([^)]+)\)", ls)
#     if coords:
#         pairs = coords[0].split(",")
#         return [(float(p.split()[0]), float(p.split()[1])) for p in pairs]
#     return []

# def downsample_df(df: pd.DataFrame, n: int = MAX_MAP_POINTS, seed: int = SAMPLE_SEED) -> pd.DataFrame:
#     if df is None or len(df) <= n:
#         return df
#     return df.sample(n=n, random_state=seed).reset_index(drop=True)

# # FIX — custom MAD function
# def calc_mad(series):
#     return (series - series.mean()).abs().mean()

# # -------------------------------
# # LOAD DATA
# # -------------------------------
# AGGREGATED_PATH = "https://raw.githubusercontent.com/HemanthGowdaaa/BMTC_DataAnalysis/main/aggregated.csv"
# ROUTES_PATH = "https://raw.githubusercontent.com/HemanthGowdaaa/BMTC_DataAnalysis/main/routes.csv"
# STOPS_PATH = "https://raw.githubusercontent.com/HemanthGowdaaa/BMTC_DataAnalysis/main/stops.csv"

# @st.cache_data
# def load_stops(path):
#     df = pd.read_csv(path)
#     df["lon"], df["lat"] = zip(*df["geometry"].apply(extract_point))
#     return df

# @st.cache_data
# def load_aggregated(path):
#     df = pd.read_csv(path)
#     df["lon"], df["lat"] = zip(*df["geometry"].apply(extract_point))
#     return df

# @st.cache_data
# def load_routes(path):
#     df = pd.read_csv(path)
#     df["coords"] = df["geometry"].apply(extract_linestring)
#     return df

# stops_df = load_stops(STOPS_PATH)
# aggregated_df = load_aggregated(AGGREGATED_PATH)
# routes_df = load_routes(ROUTES_PATH)

# # -------------------------------
# # TITLE
# # -------------------------------
# st.title("🚍 Bengaluru Metropolitan Transportation Data Analysis")
# st.write("Interactive analytics for bus stops, aggregated summaries, and route geometries.")

# # -------------------------------
# # TABS
# # -------------------------------
# tabs = st.tabs([
#     "📌 Overview", 
#     "📊 Statistics", 
#     "📈 Visualizations", 
#     "🚌 Bus Stop Profiles", 
#     "🛣 Route Explorer",
#     "🗺 Maps"
# ])

# # ============================================================
# # TAB 1 — OVERVIEW
# # ============================================================
# with tabs[0]:
#     st.header("📌 Dataset Overview")
#     st.subheader("Stops Dataset")
#     st.dataframe(stops_df.head())
#     st.subheader("Aggregated Dataset")
#     st.dataframe(aggregated_df.head())
#     st.subheader("Routes Dataset")
#     st.dataframe(routes_df.head())

# # ============================================================
# # TAB 2 — STATISTICS
# # ============================================================
# with tabs[1]:
#     st.header("📊 Statistical Summary")
#     st.subheader("Summary Statistics")
#     st.write(aggregated_df[["trip_count", "route_count"]].describe())

#     st.subheader("Variability Metrics")
#     col1, col2 = st.columns(2)

#     # Trip Count
#     with col1:
#         tc = aggregated_df["trip_count"]
#         st.write("### Trip Count")
#         st.write(f"Std Dev: {tc.std():.2f}")
#         st.write(f"MAD: {calc_mad(tc):.2f}")
#         st.write(f"IQR: {tc.quantile(0.75) - tc.quantile(0.25):.2f}")

#     # Route Count
#     with col2:
#         rc = aggregated_df["route_count"]
#         st.write("### Route Count")
#         st.write(f"Std Dev: {rc.std():.2f}")
#         st.write(f"MAD: {calc_mad(rc):.2f}")
#         st.write(f"IQR: {rc.quantile(0.75) - rc.quantile(0.25):.2f}")

# # ============================================================
# # TAB 3 — VISUALIZATIONS
# # ============================================================
# with tabs[2]:
#     st.header("📈 Visualizations")

#     min_trip, max_trip = st.slider(
#         "Trip Count Range",
#         int(aggregated_df["trip_count"].min()),
#         int(aggregated_df["trip_count"].max()),
#         (int(aggregated_df["trip_count"].min()), int(aggregated_df["trip_count"].max()))
#     )

#     filtered_df = aggregated_df[
#         (aggregated_df["trip_count"] >= min_trip) &
#         (aggregated_df["trip_count"] <= max_trip)
#     ]

#     st.subheader("Boxplot")
#     fig, ax = plt.subplots()
#     sns.boxplot(data=filtered_df[["trip_count", "route_count"]], ax=ax)
#     st.pyplot(fig)

#     st.subheader("Histogram")
#     fig, ax = plt.subplots()
#     sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
#     st.pyplot(fig)

#     st.subheader("Scatter Plot")
#     fig, ax = plt.subplots()
#     sns.scatterplot(data=filtered_df, x="route_count", y="trip_count", ax=ax)
#     st.pyplot(fig)

#     st.subheader("Correlation Heatmap")
#     fig, ax = plt.subplots()
#     sns.heatmap(filtered_df[["trip_count","route_count"]].corr(), annot=True, cmap="coolwarm", ax=ax)
#     st.pyplot(fig)

# # ============================================================
# # TAB 5 — BUS STOP PROFILE
# # ============================================================
# with tabs[3]:
#     st.header("🚌 Bus Stop Profiles")

#     stop_name = st.selectbox("Select a Bus Stop", stops_df["name"].tolist())
#     selected_stop = stops_df[stops_df["name"] == stop_name].iloc[0]

#     st.write(f"**Stop Name:** {selected_stop['name']}")
#     st.write(f"**Trip Count:** {selected_stop['trip_count']}")
#     st.write(f"**Route Count:** {selected_stop['route_count']}")

#     st.map(pd.DataFrame({"lat": [selected_stop["lat"]], "lon": [selected_stop["lon"]]}))

# # ============================================================
# # TAB 6 — ROUTE EXPLORER
# # ============================================================
# with tabs[4]:
#     st.header("🛣 Route Explorer")

#     route_name = st.selectbox("Select a Route", routes_df["name"].tolist())
#     route_data = routes_df[routes_df["name"] == route_name].iloc[0]

#     coords = route_data["coords"]
#     path_coords = [[lon, lat] for lon, lat in coords]

#     st.write(f"### {route_name}")

#     if "full_name" in route_data:
#         st.write(f"**Full Name:** {route_data['full_name']}")
#     if "trip_count" in route_data:
#         st.write(f"**Trip Count:** {route_data['trip_count']}")
#     if "stop_count" in route_data:
#         st.write(f"**Stop Count:** {route_data['stop_count']}")

#     # MAP
#     path_df = pd.DataFrame([{"path": path_coords, "name": route_name}])

#     route_layer = pdk.Layer(
#         "PathLayer",
#         data=path_df,
#         get_path="path",
#         get_width=5,
#         get_color=[0, 128, 255],
#         pickable=True
#     )

#     view = pdk.ViewState(
#         latitude=np.mean([lat for lon, lat in path_coords]),
#         longitude=np.mean([lon for lon, lat in path_coords]),
#         zoom=12,
#         pitch=45
#     )

#     st.pydeck_chart(pdk.Deck(
#         layers=[osm_layer, route_layer],
#         initial_view_state=view,
#         tooltip={"text": "{name}"}
#     ))





# # ============================================================
# # TAB 4 — MAPS (OPENSTREETMAP)
# # ============================================================
# with tabs[5]:
#     st.header("🗺 Spatial Visualizations (OpenStreetMap)")

#     osm_layer = pdk.Layer(
#         "TileLayer",
#         data=None,
#         min_zoom=0,
#         max_zoom=19,
#         tile_size=256,
#         get_tile_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
#     )

#     # ---- Stops Map ----
#     st.subheader("Bus Stops Map")
#     stops_map_df = downsample_df(stops_df[['name','lon','lat','trip_count','route_count']])

#     stop_layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=stops_map_df,
#         get_position='[lon, lat]',
#         get_radius=50,
#         get_fill_color='[0, 128, 255, 140]',
#         pickable=True
#     )

#     view = pdk.ViewState(
#         latitude=stops_map_df['lat'].mean(),
#         longitude=stops_map_df['lon'].mean(),
#         zoom=11,
#         pitch=30
#     )

#     st.pydeck_chart(pdk.Deck(
#         layers=[osm_layer, stop_layer],
#         initial_view_state=view,
#         tooltip={"text": "{name}\nTrips: {trip_count}\nRoutes: {route_count}"}
#     ))

#     st.markdown("---")

#     # ---- Routes Map ----
#     st.subheader("Routes Map")

#     paths = []
#     for _, row in routes_df.iterrows():
#         coords = row["coords"]
#         if len(coords) > 200:
#             idx = np.round(np.linspace(0, len(coords)-1, 200)).astype(int)
#             coords = [coords[i] for i in idx]

#         paths.append({"name": row["name"], "path": [[lon, lat] for lon, lat in coords]})

#     route_layer = pdk.Layer(
#         "PathLayer",
#         data=paths,
#         get_path="path",
#         get_width=4,
#         get_color=[255, 0, 0],
#         pickable=True
#     )

#     all_coords = [pt for p in paths for pt in p["path"]]
#     center_lat = np.mean([c[1] for c in all_coords])
#     center_lon = np.mean([c[0] for c in all_coords])

#     view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=30)

#     st.pydeck_chart(pdk.Deck(
#         layers=[osm_layer, route_layer],
#         initial_view_state=view,
#         tooltip={"text": "{name}"}
#     ))

