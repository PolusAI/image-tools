from pathlib import Path
import sys

ij_converter_path = Path(__file__).parents[1].joinpath('src')
sys.path.append(str(ij_converter_path))