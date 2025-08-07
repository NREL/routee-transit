# RouteE Transit

RouteE Transit is a Python package that provides comprehensive tools for predicting energy consumption of transit bus systems. Built on top of NREL's [RouteE Powertrain](https://github.com/NREL/routee-powertrain) package, RouteE Transit focuses on transit bus applications, predicting energy consumption for buses based on GTFS data.

The package enables users to work with pre-trained transit bus energy models or their own RouteE Powertrain models based on real-world telematics data or simulation outputs. RouteE Transit models predict vehicle energy consumption for transit trips based on factors such as road grade, estimated vehicle speed, and distance.

## Key Features

- **GTFS Integration**: Seamlessly work with General Transit Feed Specification (GTFS) data to analyze entire transit networks
- **Powertrain Agnostic**: Support for various vehicle types including diesel, hybrid, and battery-electric buses
- **Fleet-wide Analysis**: Predict energy consumption for individual trips, complete bus blocks, or entire bus fleets


<!-- ## Quickstart -->



## Available Models
Pretrained RouteE transit bus models are included in the RouteE Powertrain package. You can list all available models (including transit buses and other vehicles) with:
```python
import nrel.routee.powertrain as pt

# List all available pre-trained models
print(pt.list_available_models())
```

Each model includes multiple estimators that account for different combinations of features such as speed, road grade, and stop frequency.


