# Visualizer

## Usage

1. Export the `DATA_ROOT` environment variable to point at a directory with an image collection containing labels: `export DATA_ROOT="/path/to/collection/standard"`
   - in `DATA_ROOT`, there should be a sub-directory called `labels`.
2. Install the plugin with `dev` features: `pip install -e ".[dev]"`
3. Use `streamlit` to run the visualizer: `streamlit run visualizer.py`
4. A new `streamlit` window will open in your default web browser.
