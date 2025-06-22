#!/bin/bash

# Exit on error
set -o errexit
# Set default port if not provided
export PORT=${PORT:-10000}

# Debug: Print the port being used
echo "Starting Chainlit on port: $PORT"
# Install dependencies
pip install -r requirements.txt

# Run the application
chainlit run interv.py --host 0.0.0.0 --port $PORT 
