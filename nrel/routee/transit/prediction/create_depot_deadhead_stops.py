import pandas as pd
import numpy as np
import geopandas as gpd
from geopy.distance import geodesic

def create_depot_deadhead_stops(first_stops_gdf: gpd.GeoDataFrame, 
                                    last_stops_gdf: gpd.GeoDataFrame,
                                    deadhead_trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creat stop_times and stops for deadhead trips from and to depots for each block to generate the feed object for depot deadhead trips.
    Parameters
    ----------
    first_stops_gdf: gpd.GeoDataFrame
        GeoDataFrame of first stops for each block id with depot geometry (result from add_depot_to_blocks.py).
    last_stops_gdf: gpd.GeoDataFrame
        GeoDataFrame of last stops for each block id with depot geometry (result from add_depot_to_blocks.py).
    deadhead_trips: pd.DataFrame
        deadhead trip results from create_depot_deadhead_trips.py.
    Returns
    -------
    pd.DataFrame
        DataFrame of stop_times and stops for the deadhead trips.
    """

    from_depot = first_stops_gdf.copy()
    to_depot = last_stops_gdf.copy()

    # Calculate distance from depot to first stop
    from_depot['distance_m'] = from_depot.apply(
    lambda row: geodesic(
        (row.geometry_origin.y, row.geometry_origin.x),
        (row.geometry_destination.y, row.geometry_destination.x)
    ).meters,
    axis=1)
    # Calculate distance from last stop to depot
    to_depot['distance_m'] = to_depot.apply(
    lambda row: geodesic(
        (row.geometry_origin.y, row.geometry_origin.x),
        (row.geometry_destination.y, row.geometry_destination.x)
    ).meters,
    axis=1)
    # Assume average speed of 30 km/h (to be consistant with the number adopted in gtfs_feature_processing.py) 
    # to estimate travel time
    from_depot['travel_time_sec'] = (from_depot['distance_m'] / 30000) * 3600
    to_depot['travel_time_sec'] = (to_depot['distance_m'] / 30000) * 3600
    # Calculate departure time from depot for deadhead trip to first stop
    from_depot['departure_time'] = from_depot['arrival_time'] - pd.to_timedelta(from_depot['travel_time_sec'], unit='s')
    # Calculate arrival time at depot for deadhead trip from last stop
    to_depot['arrival_time'] = to_depot['departure_time'] + pd.to_timedelta(to_depot['travel_time_sec'], unit='s')

    # Create stop_times df for deadhead trips
    deadhead_trips_df = deadhead_trips.copy()
    stop_times_df = pd.DataFrame(columns=['trip_id', 'stop_sequence', 'arrival_time', 'stop_id', 'departure_time', 'shape_dist_traveled'])
    stop_times_df['trip_id'] =  deadhead_trips_df['trip_id'].repeat(2).values
    stop_times_df['stop_sequence'] = [1, 2] * len(deadhead_trips_df)
    stop_times_df['arrival_time'] = [x for pair in zip(from_depot['departure_time'].to_list(), from_depot['arrival_time'].to_list()) for x in pair] \
                                    + [x for pair in zip(to_depot['departure_time'].to_list(), to_depot['arrival_time'].to_list()) for x in pair]
    stop_times_df['stop_id'] = range(1, len(stop_times_df) + 1)
    stop_times_df['stop_id'] = stop_times_df['stop_id'].apply(lambda x: f"depot_deadhead_{x}")
    stop_times_df['departure_time'] = stop_times_df['arrival_time']
    stop_times_df['shape_dist_traveled'] = 0.0

    # Create stops df for deadhead trips
    stops_df = pd.DataFrame(columns=['stop_id', 'stop_lat', 'stop_lon'])
    stops_df['stop_id'] = stop_times_df['stop_id']

    x_start_from_depot = from_depot.geometry_origin.apply(lambda p: p.x).to_numpy()
    x_end_from_depot = from_depot.geometry_destination.apply(lambda p: p.x).to_numpy()
    x_start_to_depot = to_depot.geometry_origin.apply(lambda p: p.x).to_numpy()
    x_end_to_depot = to_depot.geometry_destination.apply(lambda p: p.x).to_numpy()
    stop_lon_from_depot = np.ravel(np.column_stack((x_start_from_depot, x_end_from_depot)))
    stop_lon_to_depot = np.ravel(np.column_stack((x_start_to_depot, x_end_to_depot)))
                        

    y_start_from_depot = from_depot.geometry_origin.apply(lambda p: p.y).to_numpy()
    y_end_from_depot = from_depot.geometry_destination.apply(lambda p: p.y).to_numpy()
    y_start_to_depot = to_depot.geometry_origin.apply(lambda p: p.y).to_numpy()
    y_end_to_depot = to_depot.geometry_destination.apply(lambda p: p.y).to_numpy()
    stop_lat_from_depot = np.ravel(np.column_stack((y_start_from_depot, y_end_from_depot)))
    stop_lat_to_depot = np.ravel(np.column_stack((y_start_to_depot, y_end_to_depot)))

    stops_df['stop_lat'] = list(stop_lat_from_depot) + list(stop_lat_to_depot)
    stops_df['stop_lon'] = list(stop_lon_from_depot) + list(stop_lon_to_depot)

    return stop_times_df, stops_df