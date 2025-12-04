
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

    # Variability Metrics for each column
    st.subheader("📌 Variability Metrics")
    col1, col2 = st.columns(2)

    with col1:
        tc = aggregated_df["trip_count"]
        st.write("### Trip Count")
        st.write(f"Std Dev: {tc.std():.2f}")
        st.write(f"MAD: {calc_mad(tc):.2f}")
        st.write(f"IQR: {(tc.quantile(0.75) - tc.quantile(0.25)):.2f}")

    with col2:
        rc = aggregated_df["route_count"]
        st.write("### Route Count")
        st.write(f"Std Dev: {rc.std():.2f}")
        st.write(f"MAD: {calc_mad(rc):.2f}")
        st.write(f"IQR: {(rc.quantile(0.75) - rc.quantile(0.25)):.2f}")

    # Generalized Variability Table
    st.subheader("📌 Variability Summary (All Numeric Columns)")
    num_cols = aggregated_df.select_dtypes(include=np.number).columns
    variability = []

    for col in num_cols:
        series = aggregated_df[col]
        variability.append({
            "Column": col,
            "Std Dev": series.std(),
            "MAD": calc_mad(series),
            "IQR": series.quantile(0.75) - series.quantile(0.25)
        })

    st.dataframe(pd.DataFrame(variability))

# ============================================================
# TAB 3 — VISUALIZATIONS
# ============================================================
with tabs[2]:
    st.header("📈 Visualizations")

    min_trip, max_trip = st.slider(
        "Trip Count Range",
        int(aggregated_df["trip_count"].min()),
        int(aggregated_df["trip_count"].max()),
        (int(aggregated_df["trip_count"].min()), int(aggregated_df["trip_count"].max()))
    )

    filtered_df = aggregated_df[
        (aggregated_df["trip_count"] >= min_trip) &
        (aggregated_df["trip_count"] <= max_trip)
    ]

    # Boxplot
    st.subheader("Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df[["trip_count", "route_count"]], ax=ax)
    st.pyplot(fig)

    # Histogram
    st.subheader("Histogram")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
    st.pyplot(fig)

    # Density Plot
    st.subheader("Density Plot")
    fig, ax = plt.subplots()
    sns.kdeplot(filtered_df["trip_count"], fill=True, ax=ax)
    st.pyplot(fig)

    # Scatter Plot
    st.subheader("Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="route_count", y="trip_count", ax=ax)
    st.pyplot(fig)

    # Hexbin Plot
    st.subheader("Hexagonal Binning Plot")
    fig, ax = plt.subplots()
    hb = ax.hexbin(filtered_df["route_count"], filtered_df["trip_count"], gridsize=30, cmap="inferno")
    plt.colorbar(hb)
    st.pyplot(fig)

    # Contour Plot
    st.subheader("Contour Plot")
    fig, ax = plt.subplots()
    sns.kdeplot(
        x=filtered_df["route_count"],
        y=filtered_df["trip_count"],
        cmap="coolwarm",
        levels=10,
        ax=ax
    )
    st.pyplot(fig)

    # Violin Plot
    st.subheader("Violin Plot")
    fig, ax = plt.subplots()
    sns.violinplot(data=filtered_df["trip_count"], ax=ax)
    st.pyplot(fig)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr = filtered_df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap="viridis", ax=ax)
    st.pyplot(fig)

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

