#!/bin/bash

# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Run the application
chainlit run interventions/interv.py --host 0.0.0.0 --port $PORT 