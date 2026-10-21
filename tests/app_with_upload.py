# Test harness: run main.py with the file uploader returning tests/fixtures/prices.csv.
import os
import runpy
import sys

import streamlit as st
import streamlit.delta_generator as dg

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "fixtures", os.environ.get("PO_FIXTURE", "prices.csv"))
ROOT = os.path.dirname(HERE)

dg.DeltaGenerator.file_uploader = lambda self, *a, **k: open(CSV, "rb")
st.file_uploader = lambda *a, **k: open(CSV, "rb")

os.chdir(ROOT)
sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, "main.py"), run_name="__main__")
