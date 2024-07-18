#!/bin/bash

# Run the application
docker run --rm --gpus all --shm-size 8g -v $(pwd):/Workspace fractal-filler/python:1.0 python ./src/main.py
