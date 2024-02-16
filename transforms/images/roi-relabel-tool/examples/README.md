# Visualizer

## Usage

1. Export the `DATA_ROOT` environment variable to point at a directory with an image collection containing labels: `export DATA_ROOT="/path/to/data"`
   - in `DATA_ROOT`, there should be a subdirectory called `input` which contains the labeled images that need to be relabeled.
2. Install the plugin with `examples` features: `pip install ".[examples]"`
3. Use `streamlit` to run the visualizer: `streamlit run visualizer.py`
4. A new `streamlit` window will open in your default web browser.
