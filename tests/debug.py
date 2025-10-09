import multiprocessing as mp
from functools import partial
from pathlib import Path

import pandas as pd

import nrel.routee.powertrain as pt

# Set constants
MI_PER_KM = 0.6213712


def predict_trip_energy(
    t_df: pd.DataFrame,
    routee_model_str: str | Path,
) -> pd.DataFrame:
    """Predict energy consumption using a provided RouteE model and trip data.

    Args:
        t_df (pd.DataFrame): DataFrame containing trip link data, including distance,
            travel time, grade, and elevation.
        routee_model_str (str): String specifying a nrel.routee.powertrain model for energy
            estimation. This could be the name of a model package with RouteE Powertrain
            or the path to a file hosting one.

    Returns:
        pd.DataFrame: DataFrame with predicted energy consumption for each trip link.
    """
    required_columns = ["kilometers", "travel_time_minutes", "grade"]
    missing = [col for col in required_columns if col not in t_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in t_df: {', '.join(missing)}. "
            "Required columns are 'kilometers', 'travel_time_minutes', and 'grade'."
        )

    routee_model = pt.load_model(routee_model_str)

    # Calculate speed and convert to mph
    t_df["miles"] = MI_PER_KM * t_df["kilometers"]
    t_df["gpsspeed"] = t_df["miles"] / (t_df["travel_time_minutes"] / 60)

    # Perform prediction
    result = routee_model.predict(links_df=t_df)    
    return result.copy()


def predict_for_all_trips(
    routee_input_df: pd.DataFrame,
    routee_vehicle_model: str | Path,
    n_processes: int,
) -> pd.DataFrame:
    """Predict energy consumption for a set of trips in parallel."""
    links_df_by_trip = [
        routee_input_df[routee_input_df["trip_id"] == trip_id].copy()
        for trip_id in routee_input_df["trip_id"].unique()
    ]
    # Run RouteE energy prediction in parallel
    predict_partial = partial(
        predict_trip_energy,
        routee_model_str=routee_vehicle_model,
    )
    with mp.Pool(n_processes) as pool:
        predictions_by_trip = pool.map(predict_partial, links_df_by_trip)

import pandas as pd
import numpy as np

# # Create fake trip link data
# routee_input_df = pd.DataFrame({
#     "trip_id": np.repeat(["trip_1", "trip_2"], 3),  # two trips, 3 links each
#     "kilometers": [0.5, 0.8, 1.0, 1.2, 0.7, 1.5],
#     "travel_time_minutes": [1.0, 2.0, 2.5, 3.0, 1.5, 4.0],
#     "grade": [0.01, -0.02, 0.0, 0.03, -0.01, 0.02],
# })

routee_input_df = pd.read_csv('/Users/yhe/github_repo/routee-transit/scripts/debug_routee_input.csv')

if __name__ == "__main__":
    predict_for_all_trips(
        routee_input_df,
        "Transit_Bus_Diesel",
        4)