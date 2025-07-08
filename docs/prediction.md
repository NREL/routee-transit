# RouteE-Transit Prediction Pipeline
A core function of `routee_transit` is to predict the energy consumption of bus trips given a static GTFS feed. Predictions are made using NREL's [RouteE-PowerTrain](https://github.com/NREL/routee-powertrain) package. To prepare to run a RouteE model, GTFS features need to be adapted into RouteE features (such as vehicle speed, road grade, and distance along road links). In RouteE-Transit, this take place across the following steps:

## 1) Specify GTFS Context
First, users need to specify the scope of predictions by supplying a static GTFS feed. In the future, RouteE-Transit will have additional flexibility to set prediction scope based on a subset of trips specified by date range, route, etc. Currently, we will generate predictions for every trip in the feed.

## 2) Refine Shapes
In this step, we take the shapes provided in `shapes.txt`, upsample them to approximately 1 Hz resolution for better map matching accuracy, and match the shapes to map links on a base map using NREL's `mappymatch` package. We then calculate the distance between successive points in the shape.

Finally, we use `gradeit` to add road grade information to each point.

## 3) Estimate Speeds
This step pulls together the estimated distances from Step 2 with the time intervals between stops specified in `stop_times.txt` to obtain speed estimates. Speed profiles can be further enhanced with data from the map-matched links, including posted speed limits from OpenStreetMap or additional information such as time-dependent speed profiles and intersection signals from NREL's internal TomTom network.

## 4) Predict Energy Consumption Using RouteE
In the last step, we read in a trained RouteE model for the transit bus model we'd like to evaluate and use it to predict energy consumption for each trip.