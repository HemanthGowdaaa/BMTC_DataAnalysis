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
#     page_title="BMTC Analytics Dashboard",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
#     <style>
#     /* General font and background */
#     body {
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         background-color: #f7f7f7;
#     }
#     /* Tab headers */
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

# # -------------------------------
# # LOAD DATA FROM PATHS
# # -------------------------------
# STOPS_PATH = "/Users/hemanth/Desktop/DataSets/routes/backend/analytics/bmtc_dashboard/stops.csv"
# AGGREGATED_PATH = "/Users/hemanth/Desktop/DataSets/routes/backend/analytics/bmtc_dashboard/aggregated.csv"
# ROUTES_PATH = "/Users/hemanth/Desktop/DataSets/routes/backend/analytics/bmtc_dashboard/routes.csv"

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
# # PAGE TITLE
# # -------------------------------
# st.title("🚍 BMTC Multi-Dataset Analytics Dashboard")
# st.write("Interactive analytics for bus stops, aggregated summaries, and route geometries.")

# # -------------------------------
# # TABS
# # -------------------------------
# tabs = st.tabs([
#     "📌 Overview", 
#     "📊 Statistics", 
#     "📈 Visualizations", 
#     "🗺 Maps", 
#     "🚌 Bus Stop Profiles", 
#     "🛣 Route Explorer"
# ])

# # ============================================================
# # TAB 1: OVERVIEW
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
# # TAB 2: STATISTICS
# # ============================================================
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

# # ============================================================
# # TAB 3: VISUALIZATIONS
# # ============================================================
# with tabs[2]:
#     st.header("📈 Visualizations")
#     min_trip, max_trip = st.slider(
#         "Trip Count Range",
#         int(aggregated_df["trip_count"].min()),
#         int(aggregated_df["trip_count"].max()),
#         (int(aggregated_df["trip_count"].min()), int(aggregated_df["trip_count"].max()))
#     )
#     filtered_df = aggregated_df[(aggregated_df["trip_count"]>=min_trip) & (aggregated_df["trip_count"]<=max_trip)]

#     st.subheader("Boxplot")
#     fig, ax = plt.subplots()
#     sns.boxplot(data=filtered_df[["trip_count","route_count"]], ax=ax)
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
# # TAB 4: MAPS
# # ============================================================
# with tabs[3]:
#     st.header("🗺 Spatial Visualizations")

#     # Stops Map
#     st.subheader("Bus Stops Map")
#     stops_map_df = downsample_df(stops_df[['name','lon','lat','trip_count','route_count']])
#     layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=stops_map_df,
#         get_position='[lon, lat]',
#         get_radius=50,
#         radius_scale=1,
#         get_fill_color='[0, 128, 255, 140]',
#         pickable=True
#     )
#     view = pdk.ViewState(
#         latitude=stops_map_df['lat'].mean(),
#         longitude=stops_map_df['lon'].mean(),
#         zoom=11,
#         pitch=30
#     )
#     st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style='mapbox://styles/mapbox/light-v9'))

#     st.markdown("---")

#     # Routes Map 
#     st.subheader("Routes Map")
#     paths = []
#     for _, row in routes_df.iterrows():
#         coords = row['coords']
#         if len(coords) > 200: # downsample long routes
#             idx = np.round(np.linspace(0, len(coords)-1, 200)).astype(int)
#             coords = [coords[i] for i in idx]
#         paths.append({"name": row["name"], "path": [[lon,lat] for lon,lat in coords]})
#     layer = pdk.Layer(
#         "PathLayer",
#         data=paths,
#         get_path="path",
#         get_color=[255,0,0],
#         get_width=4,
#         pickable=True
#     )
#     all_coords = [pt for p in paths for pt in p["path"]]
#     center_lat = np.mean([c[1] for c in all_coords])
#     center_lon = np.mean([c[0] for c in all_coords])
#     view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=30)
#     st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style='mapbox://styles/mapbox/light-v9'))

# # ============================================================
# # TAB 5: BUS STOP PROFILE
# # ============================================================
# with tabs[4]:
#     st.header("🚌 Bus Stop Profiles")
#     stop_name = st.selectbox("Select a Bus Stop", stops_df["name"].tolist())
#     selected_stop = stops_df[stops_df["name"]==stop_name].iloc[0]
#     st.write(f"**Stop Name:** {selected_stop['name']}")
#     st.write(f"**Trip Count:** {selected_stop['trip_count']}")
#     st.write(f"**Route Count:** {selected_stop['route_count']}")
#     st.map(pd.DataFrame({'lat':[selected_stop['lat']], 'lon':[selected_stop['lon']]}))

# # ============================================================
# # TAB 6: ROUTE EXPLORER
# # ============================================================
# # with tabs[5]:
# #     st.header("🛣 Route Explorer")
# #     route_name = st.selectbox("Select a Route", routes_df["name"].tolist())
# #     route_data = routes_df[routes_df["name"]==route_name].iloc[0]
# #     path_coords = [[lon, lat] for lon, lat in route_data["coords"]]

# #     st.write(f"**Route Name:** {route_data['name']}")
# #     st.write(f"**Full Name:** {route_data['full_name']}")
# #     st.write(f"**Trip Count:** {route_data['trip_count']}")
# #     st.write(f"**Stop Count:** {route_data['stop_count']}")

# #     path_df = pd.DataFrame([{"path": path_coords}])
# #     st.pydeck_chart(
# #         pdk.Deck(
# #             map_style="mapbox://styles/mapbox/light-v9",
# #             initial_view_state=pdk.ViewState(
# #                 latitude=np.mean([lat for lon, lat in path_coords]),
# #                 longitude=np.mean([lon for lon, lat in path_coords]),
# #                 zoom=12,
# #                 pitch=45
# #             ),
# #             layers=[
# #                 pdk.Layer(
# #                     "PathLayer",
# #                     data=path_df,
# #                     get_path="path",
# #                     get_width=5,
# #                     get_color=[0,128,255]
# #                 )
# #             ]
# #         )
# #     )
# with tabs[4]:
#     st.header("🗺 Spatial Visualizations")

#     # Stops Map
#     st.subheader("Bus Stops Map")
#     stops_map_df = downsample_df(stops_df[['name','lon','lat','trip_count','route_count']])
#     layer = pdk.Layer(
#         "ScatterplotLayer",
#         data=stops_map_df,
#         get_position='[lon, lat]',
#         get_radius=50,
#         radius_scale=1,
#         get_fill_color='[0, 128, 255, 140]',
#         pickable=True
#     )
#     view = pdk.ViewState(
#         latitude=stops_map_df['lat'].mean(),
#         longitude=stops_map_df['lon'].mean(),
#         zoom=11,
#         pitch=30
#     )
#     st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style='mapbox://styles/mapbox/light-v9'))

#     st.markdown("---")

#     # Routes Map
#     st.subheader("Routes Map")
#     paths = []
#     for _, row in routes_df.iterrows():
#         coords = row['coords']
#         if len(coords) > 200: # downsample long routes
#             idx = np.round(np.linspace(0, len(coords)-1, 200)).astype(int)
#             coords = [coords[i] for i in idx]
#         paths.append({"name": row["name"], "path": [[lon,lat] for lon,lat in coords]})
#     layer = pdk.Layer(
#         "PathLayer",
#         data=paths,
#         get_path="path",
#         get_color=[255,0,0],
#         get_width=4,
#         pickable=True
#     )
#     all_coords = [pt for p in paths for pt in p["path"]]
#     center_lat = np.mean([c[1] for c in all_coords])
#     center_lon = np.mean([c[0] for c in all_coords])
#     view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=30)
#     st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style='mapbox://styles/mapbox/dark-v10'))



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

# -------------------------------
# LOAD DATA
# -------------------------------
STOPS_PATH = "/Users/hemanth/Desktop/DataSets/routes/backend/analytics/bmtc_dashboard/stops.csv"
AGGREGATED_PATH = "/Users/hemanth/Desktop/DataSets/routes/backend/analytics/bmtc_dashboard/aggregated.csv"
ROUTES_PATH = "/Users/hemanth/Desktop/DataSets/routes/backend/analytics/bmtc_dashboard/routes.csv"

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
st.title("🚍 Bengaluru Metropolitan Transportation Data Analysis")
st.write("Interactive analytics for bus stops, aggregated summaries, and route geometries.")

# -------------------------------
# TABS
# -------------------------------
tabs = st.tabs([
    "📌 Overview", 
    "📊 Statistics", 
    "📈 Visualizations", 
    "🗺 Maps", 
    "🚌 Bus Stop Profiles", 
    "🛣 Route Explorer"
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
with tabs[1]:
    st.header("📊 Statistical Summary")
    st.subheader("Summary Statistics")
    st.write(aggregated_df[["trip_count","route_count"]].describe())

    st.subheader("Variability Metrics")
    col1, col2 = st.columns(2)
    with col1:
        tc = aggregated_df["trip_count"]
        st.write("### Trip Count")
        st.write(f"Std Dev: {tc.std():.2f}")
        st.write(f"MAD: {tc.mad():.2f}")
        st.write(f"IQR: {tc.quantile(0.75) - tc.quantile(0.25):.2f}")

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
# TAB 4: MAPS (OPENSTREETMAP VERSION)
# ============================================================
with tabs[3]:
    st.header("🗺 Spatial Visualizations (OpenStreetMap)")

    # -------------------------
    # OSM TileLayer
    # -------------------------
    osm_layer = pdk.Layer(
        "TileLayer",
        data=None,
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        get_tile_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    )

    # -------------------------
    # Stops Map
    # -------------------------
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

    # -------------------------
    # Routes Map
    # -------------------------
    st.subheader("Routes Map")

    paths = []
    for _, row in routes_df.iterrows():
        coords = row["coords"]
        if len(coords) > 200:
            idx = np.round(np.linspace(0, len(coords) - 1, 200)).astype(int)
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

# ============================================================
# TAB 5: BUS STOP PROFILE
# ============================================================
with tabs[4]:
    st.header("🚌 Bus Stop Profiles")

    stop_name = st.selectbox("Select a Bus Stop", stops_df["name"].tolist())
    selected_stop = stops_df[stops_df["name"] == stop_name].iloc[0]

    st.write(f"**Stop Name:** {selected_stop['name']}")
    st.write(f"**Trip Count:** {selected_stop['trip_count']}")
    st.write(f"**Route Count:** {selected_stop['route_count']}")

    st.map(pd.DataFrame({"lat": [selected_stop["lat"]], "lon": [selected_stop["lon"]]}))

# ============================================================
# TAB 6: ROUTE EXPLORER (OSM VERSION)
# ============================================================
with tabs[5]:
    st.header("🛣 Route Explorer")

    route_name = st.selectbox("Select a Route", routes_df["name"].tolist())
    route_data = routes_df[routes_df["name"] == route_name].iloc[0]

    coords = route_data["coords"]
    path_coords = [[lon, lat] for lon, lat in coords]

    # -------------------------
    # INFO
    # -------------------------
    st.write(f"### {route_name}")
    if "full_name" in route_data:
        st.write(f"**Full Name:** {route_data['full_name']}")
    if "trip_count" in route_data:
        st.write(f"**Trip Count:** {route_data['trip_count']}")
    if "stop_count" in route_data:
        st.write(f"**Stop Count:** {route_data['stop_count']}")

    # -------------------------
    # MAP RENDER
    # -------------------------
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
        latitude=np.mean([lat for lon, lat in path_coords]),
        longitude=np.mean([lon for lon, lat in path_coords]),
        zoom=12,
        pitch=45
    )

    st.pydeck_chart(pdk.Deck(
        layers=[osm_layer, route_layer],
        initial_view_state=view,
        tooltip={"text": "{name}"}
    ))

