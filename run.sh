#!/bin/bash

# Run the application
docker run docker-filler --gpus all --shm-size 8g -v "${pwd}":/Workspace python ./src/main.py