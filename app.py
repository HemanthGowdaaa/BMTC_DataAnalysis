

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
#     page_title="Bengaluru Metropolitan Transportation Analysis",
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
#     match = re.findall(r"POINT\s*\(([^)]+)\)", str(point))
#     if match:
#         parts = match[0].split()
#         if len(parts) >= 2:
#             lon, lat = parts[0], parts[1]
#             return float(lon), float(lat)
#     return np.nan, np.nan

# def extract_linestring(ls):
#     coords = re.findall(r"LINESTRING\s*\(([^)]+)\)", str(ls))
#     if coords:
#         pairs = coords[0].split(",")
#         out = []
#         for p in pairs:
#             parts = p.strip().split()
#             if len(parts) >= 2:
#                 out.append((float(parts[0]), float(parts[1])))
#         return out
#     return []

# def downsample_df(df: pd.DataFrame, n: int = MAX_MAP_POINTS, seed: int = SAMPLE_SEED) -> pd.DataFrame:
#     if df is None:
#         return pd.DataFrame()
#     if len(df) <= n:
#         return df.reset_index(drop=True)
#     return df.sample(n=n, random_state=seed).reset_index(drop=True)

# def mad(series: pd.Series) -> float:
#     """
#     Mean Absolute Deviation to mimic pandas.Series.mad()
#     (pandas.mad() = mean(abs(x - mean(x)))).
#     """
#     series = series.dropna()
#     if series.empty:
#         return float("nan")
#     return series.sub(series.mean()).abs().mean()

# # -------------------------------
# # LOAD DATA FROM PATHS
# # -------------------------------
# AGGREGATED_PATH = "https://raw.githubusercontent.com/HemanthGowdaaa/BMTC_DataAnalysis/main/aggregated.csv"
# ROUTES_PATH = "https://raw.githubusercontent.com/HemanthGowdaaa/BMTC_DataAnalysis/main/routes.csv"
# STOPS_PATH = "https://raw.githubusercontent.com/HemanthGowdaaa/BMTC_DataAnalysis/main/stops.csv"

# @st.cache_data
# def load_stops(path):
#     df = pd.read_csv(path)
#     # safe fallback if geometry missing
#     if "geometry" in df.columns:
#         df["lon"], df["lat"] = zip(*df["geometry"].apply(extract_point))
#     else:
#         # try common lon/lat names
#         if "lon" not in df.columns and "longitude" in df.columns:
#             df = df.rename(columns={"longitude": "lon"})
#         if "lat" not in df.columns and "latitude" in df.columns:
#             df = df.rename(columns={"latitude": "lat"})
#         if "lon" not in df.columns or "lat" not in df.columns:
#             df["lon"] = np.nan
#             df["lat"] = np.nan
#     return df

# @st.cache_data
# def load_aggregated(path):
#     df = pd.read_csv(path)
#     if "geometry" in df.columns:
#         df["lon"], df["lat"] = zip(*df["geometry"].apply(extract_point))
#     else:
#         if "lon" not in df.columns and "longitude" in df.columns:
#             df = df.rename(columns={"longitude": "lon"})
#         if "lat" not in df.columns and "latitude" in df.columns:
#             df = df.rename(columns={"latitude": "lat"})
#         if "lon" not in df.columns or "lat" not in df.columns:
#             df["lon"] = np.nan
#             df["lat"] = np.nan
#     return df

# @st.cache_data
# def load_routes(path):
#     df = pd.read_csv(path)
#     if "geometry" in df.columns:
#         df["coords"] = df["geometry"].apply(extract_linestring)
#     else:
#         df["coords"] = [[] for _ in range(len(df))]
#     return df

# # load
# stops_df = load_stops(STOPS_PATH)
# aggregated_df = load_aggregated(AGGREGATED_PATH)
# routes_df = load_routes(ROUTES_PATH)

# # -------------------------------
# # PAGE TITLE
# # -------------------------------
# st.title("🚍 Bengaluru Metropolitan Transportation Analysis")
# st.write("Interactive analytics for bus stops, aggregated summaries, and route geometries.")

# # -------------------------------
# # TABS
# # -------------------------------
# tabs = st.tabs([
#     "📌 Overview",
#     "📊 Statistics",
#     "📈 Visualizations",
#     "🚌 Bus Stop Profiles",
#     "🗺 Maps",
# ])

# # ============================================================
# # TAB 1: OVERVIEW
# # ============================================================
# with tabs[0]:
#     st.header("📌 Dataset Overview")
#     st.subheader("Stops Dataset (first rows)")
#     st.dataframe(stops_df.head(), use_container_width=True)
#     st.subheader("Aggregated Dataset (first rows)")
#     st.dataframe(aggregated_df.head(), use_container_width=True)
#     st.subheader("Routes Dataset (first rows)")
#     st.dataframe(routes_df.head(), use_container_width=True)

# # ============================================================
# # TAB 2: STATISTICS
# # ============================================================
# with tabs[1]:
#     st.header("📊 Statistical Summary")

#     # ensure required cols exist
#     needed_cols = ["trip_count", "route_count"]
#     missing = [c for c in needed_cols if c not in aggregated_df.columns]
#     if missing:
#         st.error(f"aggregated_df missing columns: {missing}")
#     else:
#         # Summary Table
#         st.subheader("Summary Statistics")
#         st.write(aggregated_df[needed_cols].describe())

#         # ----------------------------
#         # Additional Metrics
#         # ----------------------------
#         st.subheader("📌 Additional Metrics")
#         colA, colB = st.columns(2)

#         with colA:
#             total_stops = stops_df.shape[0]
#             st.metric("Total Number of Bus Stops", total_stops)

#         with colB:
#             st.write("### Top 5 Routes With Most Routes")
#             possible_cols = ["name", "route_name", "route", "route_no", "route_id"]
#             route_col = next((c for c in possible_cols if c in aggregated_df.columns), None)

#             if route_col:
#                 top5_routes = aggregated_df.nlargest(5, "route_count")[[route_col, "route_count"]]
#                 st.dataframe(top5_routes)
#             else:
#                 st.write("No route name column found; showing top5 by index")
#                 st.dataframe(aggregated_df.nlargest(5, "route_count")[["route_count"]])

#         # ----------------------------
#         # Variability Metrics
#         # ----------------------------
#         st.subheader("Variability Metrics")
#         col1, col2 = st.columns(2)

#         # Trip Count
#         with col1:
#             tc = aggregated_df["trip_count"].dropna()
#             st.write("### Trip Count")
#             st.write(f"Std Dev: {tc.std():.2f}")
#             st.write(f"MAD: {mad(tc):.2f}")
#             st.write(f"IQR: {(tc.quantile(0.75) - tc.quantile(0.25)):.2f}")

#         # Route Count
#         with col2:
#             rc = aggregated_df["route_count"].dropna()
#             st.write("### Route Count")
#             st.write(f"Std Dev: {rc.std():.2f}")
#             st.write(f"MAD: {mad(rc):.2f}")
#             st.write(f"IQR: {(rc.quantile(0.75) - rc.quantile(0.25)):.2f}")

# # ============================================================
# # TAB 3: VISUALIZATIONS
# # ============================================================
# with tabs[2]:
#     st.header("📈 Visualizations")

#     if "trip_count" not in aggregated_df.columns:
#         st.error("Column 'trip_count' not found in aggregated_df.")
#     else:
#         min_trip = int(aggregated_df["trip_count"].min())
#         max_trip = int(aggregated_df["trip_count"].max())
#         min_selected, max_selected = st.slider(
#             "Trip Count Range",
#             min_trip, max_trip, (min_trip, max_trip)
#         )
#         filtered_df = aggregated_df[(aggregated_df["trip_count"] >= min_selected) & (aggregated_df["trip_count"] <= max_selected)]

#         st.subheader("Boxplot")
#         fig, ax = plt.subplots()
#         sns.boxplot(data=filtered_df[["trip_count", "route_count"]], ax=ax)
#         st.pyplot(fig)

#         st.subheader("Histogram")
#         fig, ax = plt.subplots()
#         sns.histplot(filtered_df["trip_count"], kde=True, ax=ax)
#         st.pyplot(fig)

#         st.subheader("Scatter Plot")
#         fig, ax = plt.subplots()
#         sns.scatterplot(data=filtered_df, x="route_count", y="trip_count", ax=ax)
#         st.pyplot(fig)

#         st.subheader("Correlation Heatmap")
#         fig, ax = plt.subplots()
#         sns.heatmap(filtered_df[["trip_count", "route_count"]].corr(), annot=True, cmap="coolwarm", ax=ax)
#         st.pyplot(fig)

# # ============================================================
# # TAB 4: BUS STOP PROFILES
# # ============================================================
# with tabs[3]:
#     st.header("🚌 Bus Stop Profiles")
#     # auto-detect stop-name column
#     stop_name_cols = ["stop_name", "name", "stop", "stopid"]
#     stop_name_col = next((c for c in stop_name_cols if c in stops_df.columns), None)
#     if stop_name_col is None:
#         st.error("No stop name column found in stops_df.")
#     else:
#         stop_list = stops_df[stop_name_col].fillna("").unique().tolist()
#         selected = st.selectbox("Select a Bus Stop", stop_list)
#         stop_info = stops_df[stops_df[stop_name_col] == selected]
#         if stop_info.empty:
#             st.write("No data for selected stop.")
#         else:
#             row = stop_info.iloc[0]
#             st.write(f"**Stop Name:** {row.get(stop_name_col, '')}")
#             # safe getters for trip_count/route_count
#             st.write(f"**Trip Count:** {row.get('trip_count', 'N/A')}")
#             st.write(f"**Route Count:** {row.get('route_count', 'N/A')}")
#             # map
#             lat = row.get("lat", None)
#             lon = row.get("lon", None)
#             if pd.notna(lat) and pd.notna(lon):
#                 st.map(pd.DataFrame({'lat':[lat], 'lon':[lon]}))
#             else:
#                 st.write("No location coordinates available for this stop.")

# # ============================================================
# # TAB 5: MAPS
# # ============================================================
# # with tabs[4]:
# #     st.header("🗺 Spatial Visualizations")

# #     # Stops Map
# #     st.subheader("Bus Stops Map")
# #     if "lat" in stops_df.columns and "lon" in stops_df.columns:
# #         stops_map_df = downsample_df(stops_df[['name' if 'name' in stops_df.columns else stop_name_col, 'lon', 'lat', 'trip_count' if 'trip_count' in stops_df.columns else None, 'route_count' if 'route_count' in stops_df.columns else None]].dropna(how='all', axis=1))
# #         if stops_map_df.empty:
# #             st.write("No stops to display on map.")
# #         else:
# #             # ensure numeric lat/lon
# #             stops_map_df = stops_map_df.dropna(subset=['lat', 'lon'])
# #             if stops_map_df.empty:
# #                 st.write("Stops do not have valid lat/lon.")
# #             else:
# #                 layer = pdk.Layer(
# #                     "ScatterplotLayer",
# #                     data=stops_map_df,
# #                     get_position='[lon, lat]',
# #                     get_radius=50,
# #                     radius_scale=1,
# #                     get_fill_color='[0, 128, 255, 140]',
# #                     pickable=True
# #                 )
# #                 view = pdk.ViewState(
# #                     latitude=float(stops_map_df['lat'].mean()),
# #                     longitude=float(stops_map_df['lon'].mean()),
# #                     zoom=11,
# #                     pitch=30
# #                 )
# #                 st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style='mapbox://styles/mapbox/light-v9'))
# #     else:
# #         st.write("Stops dataset does not contain lat/lon columns.")

# #     st.markdown("---")

# #     # # Routes Map
# #     # st.subheader("Routes Map")
# #     # # build paths list for pydeck; guard empty coords
# #     # paths = []
# #     # for _, row in routes_df.iterrows():
# #     #     coords = row.get('coords', []) or []
# #     #     if not coords:
# #     #         continue
# #     #     # downsample very long routes
# #     #     if len(coords) > 200:
# #     #         idx = np.round(np.linspace(0, len(coords)-1, 200)).astype(int)
# #     #         coords = [coords[i] for i in idx]
# #     #     path_pts = [[float(lon), float(lat)] for lon, lat in coords]
# #     #     # name fallback
# #     #     name = row.get("name") or row.get("route_name") or row.get("id") or ""
# #     #     paths.append({"name": name, "path": path_pts})

# #     # if not paths:
# #     #     st.write("No route geometries available to display.")
# #     # else:
# #     #     layer = pdk.Layer(
# #     #         "PathLayer",
# #     #         data=paths,
# #     #         get_path="path",
# #     #         get_color=[255, 0, 0],
# #     #         get_width=4,
# #     #         pickable=True
# #     #     )
# #     #     all_coords = [pt for p in paths for pt in p["path"]]
# #     #     center_lat = np.mean([c[1] for c in all_coords])
# #     #     center_lon = np.mean([c[0] for c in all_coords])
# #     #     view = pdk.ViewState(latitude=float(center_lat), longitude=float(center_lon), zoom=11, pitch=30)
# #     #     st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, map_style='mapbox://styles/mapbox/dark-v10'))


# # # Routes Map
# # st.subheader("Routes Map")

# # paths = []

# # for _, row in routes_df.iterrows():
# #     coords = row.get("coords", []) or []
# #     if not coords:
# #         continue

# #     # downsample long routes
# #     if len(coords) > 200:
# #         idx = np.round(np.linspace(0, len(coords)-1, 200)).astype(int)
# #         coords = [coords[i] for i in idx]

# #     # ensure valid float values
# #     cleaned = []
# #     for lon, lat in coords:
# #         if not (pd.isna(lon) or pd.isna(lat)):
# #             cleaned.append([float(lon), float(lat)])

# #     if not cleaned:
# #         continue

# #     route_name = row.get("name") or row.get("route_name") or "Route"
# #     paths.append({"name": route_name, "path": cleaned})

# # if not paths:
# #     st.write("No route geometries available to display.")
# # else:
# #     # Flatten to compute center safely
# #     all_points = [pt for p in paths for pt in p["path"]]

# #     center_lon = float(np.mean([pt[0] for pt in all_points]))
# #     center_lat = float(np.mean([pt[1] for pt in all_points]))

# #     layer = pdk.Layer(
# #         "PathLayer",
# #         paths,
# #         get_path="path",
# #         get_width=5,
# #         get_color=[255, 0, 0],
# #         pickable=True
# #     )

# #     view_state = pdk.ViewState(
# #         longitude=center_lon,
# #         latitude=center_lat,
# #         zoom=11,
# #         pitch=30
# #     )

# #     st.pydeck_chart(pdk.Deck(
# #         layers=[layer],
# #         initial_view_state=view_state,
# #         map_style="mapbox://styles/mapbox/dark-v10"
# #     ))



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

