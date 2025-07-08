# RouteE Transit

Welcome to the RouteE Transit documentation! This project provides tools for predicting energy consumption of transit bus trips based on NREL's [RouteE-Powertrain](https://github.com/NREL/routee-powertrain) package. Core capabilities include:

- Training new RouteE models for different models of transit buses based on historic energy consumption data from vehicle telematics, or outputs from simulation tools such as NREL's [FASTSim](https://github.com/NREL/fastsim).

- Predicting energy consumption for individual bus trips, complete bus blocks including passenger and deadhead trips, or entire bus fleets based on GTFS data.

RouteE Transit is intended to be powertrain-agnostic and to provide accurate energy consumption predictions for various vehicle types including diesel, hybrid, and battery-electric buses.

## Repository Structure
- `data/`: data used in routee-transit
- `docs/`: project documentation used to build docs site with `mkdocs`
- `reports/`: destination folder for model outputs
- `routee_transit/`: core routee-transit Python packge
    - `gtfs_processing/`: code related to aggregating US GTFS feeds using the Mobility Database. Likely to be moved to a separate repo in the future.
    - `prediction/`: code for making predictions of transit bus energy consumption based on GTFS feeds and pre-trained RouteE models.
    - `training/`: code for training new RouteE transit bus models based on telematics data or FASTSim outputs
    - `utils/`: shared utilities across routee_transit submodules
    - `vehicle_models/`: where trained RouteE models are stored so they can be accessed when making predictions
- `scripts/`: scripts and examples of training and predicting with RouteE Transit
- `tests/`: tests
