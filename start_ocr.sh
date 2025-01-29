#!/bin/bash

# Navigate to the ocr directory
cd
if [ ! -d "license_plate_ocr" ]; then
    echo "Directory license_plate_ocr does not exist."
    exit 1
fi
cd license_plate_ocr || exit

# Activate the virtual environment
source ./venv/bin/activate

# Check if port 5000 is in use and kill the process if it is
if fuser 5000/tcp > /dev/null 2>&1; then
    echo "Port 5000 is in use. Killing the process..."
    fuser -k 5000/tcp
fi

# Run the Python application in the background and redirect output to a log file
nohup python3 -u app.py > license_plate_ocr.log 2>&1 &

# Optional: Sleep for a few seconds to allow the app to start
sleep 2