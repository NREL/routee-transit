import pandas as pd
import numpy as np
import geopandas as gpd
from geopy.distance import geodesic

def create_betweenTrip_deadhead_stops(feed, deadhead_trips: pd.DataFrame) -> pd.DataFrame:
    """Create stop_times and stops for deadhead trips between consecutive trips to generate the feed object for between trip deadhead trips.
    Parameters
    ----------
    feed: Any
        GTFS feed object (e.g. result from read_in_gtfs).
    deadhead_trips: pd.DataFrame
        deadhead trip results from create_betweenTrip_deadhead_trips.py.
    Returns
    -------
    pd.DataFrame
        DataFrame of stop_times and stops for the deadhead trips.
    """
    # Calculate distance from end stop of first trip to start stop of second trip
    deadhead_trips['from_trip'] = deadhead_trips['trip_id'].apply(lambda x: x.split('_to_')[0])
    deadhead_trips['to_trip'] = deadhead_trips['trip_id'].apply(lambda x: x.split('_to_')[1])
    
    # First stops of deadhead trips
    first_stops = feed.stop_times[feed.stop_times['trip_id'].isin(deadhead_trips['from_trip'])].copy()
    first_stops = first_stops.sort_values(by=['trip_id', 'stop_sequence'])
    first_stops = first_stops.groupby('trip_id').last().reset_index()
    first_stops = first_stops.rename(columns={'stop_id': 'from_stop_id',
                                              'trip_id': 'from_trip'})

    # Last stops of deadhead trips
    last_stops = feed.stop_times[feed.stop_times['trip_id'].isin(deadhead_trips['to_trip'])].copy()
    last_stops = last_stops.sort_values(by=['trip_id', 'stop_sequence'])
    last_stops = last_stops.groupby('trip_id').first().reset_index()
    last_stops = last_stops.rename(columns={'stop_id': 'to_stop_id',
                                            'trip_id': 'to_trip'})
    # Merge to get stop ids and stop lat/lon
    deadhead_trips = deadhead_trips.merge(first_stops[['from_trip', 'from_stop_id', 'departure_time']], on='from_trip', how='left')
    deadhead_trips = deadhead_trips.merge(last_stops[['to_trip', 'to_stop_id', 'arrival_time']], on='to_trip', how='left')
    deadhead_trips = deadhead_trips.merge(feed.stops[['stop_id', 'stop_lat', 'stop_lon']], left_on='from_stop_id', right_on='stop_id', how='left')
    deadhead_trips = deadhead_trips.rename(columns={'stop_lat': 'from_stop_lat', 'stop_lon': 'from_stop_lon'})
    deadhead_trips = deadhead_trips.merge(feed.stops[['stop_id', 'stop_lat', 'stop_lon']], left_on='to_stop_id', right_on='stop_id', how='left')
    deadhead_trips = deadhead_trips.rename(columns={'stop_lat': 'to_stop_lat', 'stop_lon': 'to_stop_lon'})
    deadhead_trips = deadhead_trips.drop(columns=['stop_id_x', 'stop_id_y'])
    # Create geometry columns for geospatial calculations
    deadhead_trips['geometry_origin'] = gpd.points_from_xy(deadhead_trips['from_stop_lon'], deadhead_trips['from_stop_lat'])
    deadhead_trips['geometry_destination'] = gpd.points_from_xy(deadhead_trips['to_stop_lon'], deadhead_trips['to_stop_lat'])
    # Calculate distance from origin to destination for deadhead trips
    deadhead_trips['distance_m'] = deadhead_trips.apply(
    lambda row: geodesic(
        (row.geometry_origin.y, row.geometry_origin.x),
        (row.geometry_destination.y, row.geometry_destination.x)
    ).meters,
    axis=1)
    # Assume average speed of 30 km/h (to be consistant with the number adopted in gtfs_feature_processing.py) 
    # to estimate travel time
    deadhead_trips['travel_time_sec'] = (deadhead_trips['distance_m'] / 30000) * 3600
    # Calculate arrival time at to_stop for deadhead trip
    deadhead_trips['arrival_time_cal'] = deadhead_trips['departure_time'] + pd.to_timedelta(deadhead_trips['travel_time_sec'], unit='s')
    # Use the minimum of scheduled arrival time and calculated arrival time
    deadhead_trips['arrival_time'] = deadhead_trips[['arrival_time', 'arrival_time_cal']].min(axis=1)
    
    # Create stop_times df for deadhead trips
    stop_times_df = pd.DataFrame(columns=['trip_id', 'stop_sequence', 'arrival_time', 'stop_id', 'departure_time', 'shape_dist_traveled'])
    stop_times_df['trip_id'] =  deadhead_trips['trip_id'].repeat(2).values
    stop_times_df['stop_sequence'] = [1, 2] * len(deadhead_trips)
    stop_times_df['arrival_time'] = [x for pair in zip(deadhead_trips['departure_time'].to_list(), deadhead_trips['arrival_time'].to_list()) for x in pair]
    stop_times_df['stop_id'] = range(1, len(stop_times_df) + 1)
    stop_times_df['stop_id'] = stop_times_df['stop_id'].apply(lambda x: f"betweenTrip_deadhead_{x}")
    stop_times_df['departure_time'] = stop_times_df['arrival_time']
    stop_times_df['shape_dist_traveled'] = 0.0
    
    # Create stops df for deadhead trips
    stops_df = pd.DataFrame(columns=['stop_id', 'stop_lat', 'stop_lon'])
    stops_df['stop_id'] = stop_times_df['stop_id']  
    x_start = deadhead_trips.geometry_origin.apply(lambda p: p.x).to_numpy()
    x_end = deadhead_trips.geometry_destination.apply(lambda p: p.x).to_numpy()
    stop_lon = np.ravel(np.column_stack((x_start, x_end)))
    y_start = deadhead_trips.geometry_origin.apply(lambda p: p.y).to_numpy()
    y_end = deadhead_trips.geometry_destination.apply(lambda p: p.y).to_numpy()
    stop_lat = np.ravel(np.column_stack((y_start, y_end)))
    stops_df['stop_lat'] = stop_lat
    stops_df['stop_lon'] = stop_lon

    deadhead_trips['block_id'] = deadhead_trips['trip_id'] # Use trip_id as block_id for deadhead trips for trace generation purpose in generate_deadhead_traces.py
    return stop_times_df, stops_df, deadhead_trips[['geometry_origin', 'geometry_destination','block_id']]