#!/bin/sh
# Entrypoint for the app container. Runs the full pipeline (train once if
# no model artifact exists, then populate the analytical DB) before
# launching the Streamlit UI -- so `docker-compose up` really is the
# single command the assignment asks for, with no manual follow-up steps.
set -e

echo "== Credit Risk Platform: startup =="

echo "-> Waiting for database..."
python -m src.utils.docker_utils

if [ ! -f "/app/models/lgbm_credit_risk.joblib" ]; then
    echo "-> No trained model found. Training now (this runs once; the artifact"
    echo "   persists in the models/ volume for subsequent restarts)..."
    python -m src.ml.train
    python -m src.ml.rules
else
    echo "-> Existing trained model found in models/ -- skipping training."
fi

echo "-> Loading scored applicants into the analytical database..."
python -m src.data.db_loader

echo "-> Launching Streamlit UI on :8501"
exec streamlit run app/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true
