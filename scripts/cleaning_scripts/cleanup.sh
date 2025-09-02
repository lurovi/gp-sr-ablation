#!/usr/bin/env bash
# Usage: ./cleanup.sh

ROOT_DIR="results/"
echo $ROOT_DIR

# Find and delete all equations*.json and estimator*.pkl files
find "$ROOT_DIR" -type f -name "equations*.json" -delete
find "$ROOT_DIR" -type f -name "estimator*.pkl" -delete

