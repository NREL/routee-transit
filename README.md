# routee-transit
Application of RouteE Powertrain to transit bus energy prediction using GFTS data as inputs.

## Setup
We use `poetry` for dependency management and packaging. To install Poetry, see [the documentation](https://python-poetry.org/docs/).

Once you have `poetry` installed, from the root directory (`routee-transit`), run:

`poetry install`

This will create a virtual environment based on the dependencies described in `pyproject.toml`. To execute code in this virtual environment, use `poetry run <my_file.py>`, or `poetry shell` to run all subsequent commands in that environment. You can also configure your IDE's Python interpreter to use the environment created by Poetry (you can find the path to it with `poetry env info --path`).

If you add new packages while development, add them with `poetry add <new-package-name>`.

## Running the current pipeline
`scripts/single_agency_full_analysis.py` provides an example of running the full pipeline to predict energy consumption for some or all trips in an agency's GTFS feed.

