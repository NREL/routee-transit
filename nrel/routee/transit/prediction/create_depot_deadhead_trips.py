import pandas as pd
def create_depot_deadhead_trips(trips_df: pd.DataFrame) -> pd.DataFrame:
    """Create deadhead trips from and to depots for each block.
    Parameters
    ----------
    trips_df : pd.DataFrame
        trips_df of selected date route (e.g. result from read_in_gtfs).

    Returns
    -------
    pd.DataFrame: DataFrame with created deadhead trips.
    """

    existing_trips_df = trips_df
    block_ids = existing_trips_df['block_id'].dropna().unique().tolist()
    # For each block id, create two deadhead trips: one from depot to first stop,
    # and one from last stop to depot.
    to_depot_trips = pd.DataFrame({'trip_id': [], 'route_id': [], 'service_id': [], 'block_id': [], 
                                   'shape_id': [], 'route_short_name': [], 'route_type': [], 'route_desc': [],
                                   'agency_id': []})
    from_depot_trips = pd.DataFrame({'trip_id': [], 'route_id': [], 'service_id': [], 'block_id': [], 
                                     'shape_id': [], 'route_short_name': [], 'route_type': [], 'route_desc': [],
                                     'agency_id': []})
    for block_id in block_ids:
        block_trips = existing_trips_df[existing_trips_df['block_id'] == block_id]
        if block_trips.empty:
            continue
        first_trip = block_trips.iloc[0]
        last_trip = block_trips.iloc[-1]
        # Create trip from depot to first stop
        from_depot_trip_id = f"depot_to_{first_trip['trip_id']}"
        from_depot_trip = {
            'trip_id': from_depot_trip_id,
            'route_id': first_trip['route_id'],
            'service_id': first_trip['service_id'],
            'block_id': block_id,
            'shape_id': f"from_depot_{block_id}",
            'route_short_name': first_trip.get('route_short_name', ''),
            'route_type': first_trip.get('route_type', 3),  # Default to bus
            'route_desc': f"Deadhead from depot to {first_trip['trip_id']}",
            'agency_id': first_trip.get('agency_id', None)
        }
        from_depot_trips = pd.concat([from_depot_trips, pd.DataFrame([from_depot_trip])], ignore_index=True)
        # Create trip from last stop to depot
        to_depot_trip_id = f"{last_trip['trip_id']}_to_depot"
        to_depot_trip = {
            'trip_id': to_depot_trip_id,
            'route_id': last_trip['route_id'],
            'service_id': last_trip['service_id'],
            'block_id': block_id,
            'shape_id': f"to_depot_{block_id}",
            'route_short_name': last_trip.get('route_short_name', ''),
            'route_type': last_trip.get('route_type', 3),  # Default to bus
            'route_desc': f"Deadhead from {last_trip['trip_id']} to depot",
            'agency_id': last_trip.get('agency_id', None)
        }
        to_depot_trips = pd.concat([to_depot_trips, pd.DataFrame([to_depot_trip])], ignore_index=True)

    deadhead_trips_df = pd.concat([from_depot_trips, to_depot_trips], ignore_index=True)
    return deadhead_trips_df

    