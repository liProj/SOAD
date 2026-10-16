"""Compatibility entry point for the vector SOAD architecture figure."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('fig_pipeline_vector.py')), run_name='__main__')
