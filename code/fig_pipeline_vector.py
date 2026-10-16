"""Build the user-selected vector architecture figure."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('fig_pipeline_selected.py')), run_name='__main__')
