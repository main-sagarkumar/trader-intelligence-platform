# Project Startup Guide

## Activate Environment

.\venv\Scripts\Activate.ps1

## Verify Python

where python
## Run Tests

pytest -v

## Start API

uvicorn src.api.main:app --reload

## Start Streamlit

streamlit run app/streamlit_app.py

## Open

API:
http://localhost:8000/docs

Dashboard:
http://localhost:8501