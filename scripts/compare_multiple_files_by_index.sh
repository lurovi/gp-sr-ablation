#!/bin/bash

# Usage check
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <prefix1> <prefix2> <start_index> <end_index>"
    exit 1
fi

prefix1="$1"
prefix2="$2"
start="$3"
end="$4"

# Track failing indexes
failed_indexes=()
missing_indexes=()

# Loop over index range
for ((i=start; i<=end; i++)); do
    file1="${prefix1}_${i}.txt"
    file2="${prefix2}_${i}.txt"

    # Check if both files exist before comparing
    if [[ ! -f "$file1" || ! -f "$file2" ]]; then
        echo "[Index $i] One or both files are missing: $file1, $file2"
        missing_indexes+=("$i")
        continue
    fi

    # Call compare_files.sh and capture output
    output=$(./scripts/compare_files.sh "$file1" "$file2")

    # If content differs, record the index
    if [[ "$output" == *"different content"* ]]; then
        failed_indexes+=("$i")
    fi
done

# Final summary
if [ "${#missing_indexes[@]}" -ne 0 ]; then
    echo "Missing files at indexes: ${missing_indexes[*]}"
fi

if [ "${#failed_indexes[@]}" -eq 0 ] && [ "${#missing_indexes[@]}" -eq 0 ]; then
    echo "All files have the same content across indexes ${start}-${end}."
else
    if [ "${#failed_indexes[@]}" -ne 0 ]; then
        echo "Files differ at the following indexes: ${failed_indexes[*]}"
    fi
    exit 1
fi
