from typing import Any
from pathlib import Path
import os

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def add_depot_to_blocks(trips_df: pd.DataFrame, feed: Any, path_to_depots: str | Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Add origin/destination depot geometry to each block id.

    Parameters
    ----------
    trips_df: pd.DataFrame
        trips_df of selected date and route (result from read_in_gtfs).
    feed : Any
        GTFS feed object (e.g. result from read_in_gtfs).
    path_to_depots : str | Path
        Path to a vector file (GeoJSON/Shapefile) containing depot point geometries.

    Returns
    -------
    tuple[GeoDataFrame, GeoDataFrame]
        (first_stops_gdf, last_stops_gdf). Each GeoDataFrame contains the stop
        geometry (column 'stop_geometry') and the matched depot geometry
        (column 'depot_geometry').
    """

    # Process trips and stops dataframes in feed to get first and last stops of each block id
    trips_df = trips_df.copy()
    stop_times_df = feed.stop_times
    stops_df = feed.stops   
    blocks_trips_stops = stop_times_df.merge(trips_df[['trip_id', 'block_id']], on='trip_id', how='right')
    blocks_trips_stops = blocks_trips_stops.merge(stops_df, on='stop_id', how='left')

    blocks_trips_stops = blocks_trips_stops.sort_values(by=['block_id', 'arrival_time'])
    first_stops = blocks_trips_stops.groupby('block_id').first().reset_index()
    last_stops = blocks_trips_stops.groupby('block_id').last().reset_index()

    first_stops = first_stops[['block_id', 'arrival_time', 'stop_lat', 'stop_lon']]
    last_stops = last_stops[['block_id', 'arrival_time', 'stop_lat', 'stop_lon']]

    first_stops['geometry'] = first_stops.apply(lambda row: Point(row['stop_lon'], row['stop_lat']), axis=1)
    last_stops['geometry'] = last_stops.apply(lambda row: Point(row['stop_lon'], row['stop_lat']), axis=1)
    first_stops_gdf = gpd.GeoDataFrame(first_stops, geometry='geometry', crs='EPSG:4326')
    last_stops_gdf = gpd.GeoDataFrame(last_stops, geometry='geometry', crs='EPSG:4326')

    # Read depot locations; ensure file exists
    if not os.path.exists(path_to_depots):
        raise FileNotFoundError(f"Depot file not found: {path_to_depots}")
    depots_df = gpd.read_file(path_to_depots)
    # Ensure depot geometries are points and in WGS84
    if depots_df.crs is None:
        depots_df = depots_df.set_crs(epsg=4326)
    else:
        depots_df = depots_df.to_crs(epsg=4326)

    # Create a simple mapping from depot index to geometry for fast lookup
    depots_geom_map = depots_df['geometry'].to_dict()

    # Project to Web Mercator (EPSG:3857) for distance computations
    proj_crs = "EPSG:3857"
    first_proj = first_stops_gdf.to_crs(proj_crs).reset_index(drop=True)
    last_proj = last_stops_gdf.to_crs(proj_crs).reset_index(drop=True)
    depots_proj = depots_df.to_crs(proj_crs).copy()

    # Use spatial join nearest to find the closest depot for each left row.
    # sjoin_nearest returns rows indexed by the left GeoDataFrame's index; if
    # multiple matches exist for the same left index we keep the closest (by
    # depot_dist_m). We then align back to the original stop GeoDataFrames by
    # resetting their indices to RangeIndex.
    first_nn = first_proj.sjoin_nearest(depots_proj[['geometry']], how='left', distance_col='depot_dist_m')
    last_nn = last_proj.sjoin_nearest(depots_proj[['geometry']], how='left', distance_col='depot_dist_m')

    # Drop duplicates 
    first_nn = first_nn.drop_duplicates(subset = ['block_id'])
    last_nn = last_nn.drop_duplicates(subset = ['block_id'])

    # Align and assign nearest depot index to the original stop GeoDataFrames
    first_stops_gdf = first_stops_gdf.copy().reset_index(drop=True)
    last_stops_gdf = last_stops_gdf.copy().reset_index(drop=True)
    first_stops_gdf['nearest_depot_idx'] = first_nn['index_right'].reindex(first_stops_gdf.index).astype('Int64')
    last_stops_gdf['nearest_depot_idx'] = last_nn['index_right'].reindex(last_stops_gdf.index).astype('Int64')

    # Set stop geometry and map matched depot geometry from the original
    # depots DataFrame (index_right references that index)
    first_stops_gdf['geometry_destination'] = first_stops_gdf.geometry
    last_stops_gdf['geometry_origin'] = last_stops_gdf.geometry
    first_stops_gdf['geometry_origin'] = first_stops_gdf['nearest_depot_idx'].map(depots_geom_map)
    last_stops_gdf['geometry_destination'] = last_stops_gdf['nearest_depot_idx'].map(depots_geom_map)

    # Set the arrival time as departure time for deadhead trip to depot for the last_stop_gdf
    last_stops_gdf['departure_time'] = last_stops_gdf['arrival_time']
    # Drop the arrival_time column for the last_stop_gdf
    last_stops_gdf = last_stops_gdf.drop(columns=['arrival_time'])

    # Keep only relevant columns and set stop_geometry as the active geometry
    first_stops_gdf = first_stops_gdf.drop(columns=['geometry'])
    first_stops_gdf = gpd.GeoDataFrame(first_stops_gdf, geometry='geometry_destination', crs='EPSG:4326')

    last_stops_gdf = last_stops_gdf.drop(columns=['geometry'])
    last_stops_gdf = gpd.GeoDataFrame(last_stops_gdf, geometry='geometry_origin', crs='EPSG:4326')

    return first_stops_gdf, last_stops_gdf