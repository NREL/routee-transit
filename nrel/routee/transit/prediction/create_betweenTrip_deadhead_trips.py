import pandas as pd


def create_betweenTrip_deadhead_trips(
    trips_df: pd.DataFrame, stop_times_df: pd.DataFrame
) -> pd.DataFrame:
    """Create deadhead trips between consecutive trips for each block.
    Parameters
    ----------
    trips_df : pd.DataFrame
        GTFS trips_df (e.g. result from read_in_gtfs).

    stop_times_df: pd.DataFrame
        stop_times df in feed resulted from read_in_gtfs.

    Returns
    -------
    pd.DataFrame: DataFrame with created deadhead trips.
    """

    # For each block id, create one deadhead trip between consecutive trips.
    deadhead_trips = pd.DataFrame(
        {
            "trip_id": [],
            "route_id": [],
            "service_id": [],
            "block_id": [],
            "shape_id": [],
            "route_short_name": [],
            "route_type": [],
            "route_desc": [],
            "agency_id": [],
        }
    )
    trip_start = (
        stop_times_df.groupby("trip_id")["arrival_time"].min().reset_index()
    )  # trip start time of each trip
    trips_df = trips_df.merge(
        trip_start, on="trip_id", how="left"
    )  # only look at trips on selected date and route
    trips_df = trips_df.sort_values(by=["block_id", "arrival_time"])
    block_gb = trips_df.groupby("block_id")
    dh_dfs = list()
    for _, block_df in block_gb:
        block_df["to_trip"] = block_df["trip_id"].shift(-1)
        block_df["deadhead_trip"] = block_df["trip_id"] + "_to_" + block_df["to_trip"]
        block_df = block_df.dropna(subset=["to_trip"])
        block_df = block_df[
            ["deadhead_trip", "route_id", "service_id", "block_id", "shape_id"]
        ]
        block_df = block_df.rename(columns=({"deadhead_trip": "trip_id"}))
        dh_dfs.append(block_df)
    deadhead_trips = pd.concat(dh_dfs).reset_index(drop=True)

    deadhead_trips["route_short_name"] = None
    deadhead_trips["route_type"] = 3
    deadhead_trips["route_desc"] = "Deadhead_from_" + deadhead_trips["trip_id"]
    deadhead_trips["agency_id"] = None
    deadhead_trips["shape_id"] = deadhead_trips["trip_id"]

    return deadhead_trips
