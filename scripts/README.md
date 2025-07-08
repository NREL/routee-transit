# RouteE-Transit Scripts
This directory contains scripts for running RouteE-Transit. Files included:

- `design_blocks.py`: Bus block design script left over from original routee-transit effort. Still included for now as a reference for continued development efforts, as this functionality hasn't been refactored yet.
- `predict_with_routee.py`: Energy prediction script left over from original routee-transit effort. Still included for now as a reference for continued development efforts, as this functionality hasn't been refactored yet.
- `single_agency_full_analysis.py`: Script to run full OSM-based RouteE-Transit energy prediction pipeline for a single agency (by default, Utah Transit Authority in Salt Lake City). Note that to factor in road grade in energy predictions, USGS elevation raster files must be available (e.g., in `data/usgs_elevation`)