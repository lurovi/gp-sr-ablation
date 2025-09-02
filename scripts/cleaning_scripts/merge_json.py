#!/usr/bin/env python3
import os
import json

def process_folder(folder):
    # Collect all result and equation files
    result_files = [f for f in os.listdir(folder) if f.startswith("result") and f.endswith(".json")]
    for result_file in result_files:
        # Extract ID from filename
        try:
            file_id = result_file[len("result"):-len(".json")]
        except Exception:
            continue
        eq_file = f"equations{file_id}.json"
        result_path = os.path.join(folder, result_file)
        eq_path = os.path.join(folder, eq_file)

        if os.path.exists(eq_path):
            # Load result and equation JSONs
            with open(result_path, "r") as f:
                result_data = json.load(f)
            with open(eq_path, "r") as f:
                eq_data = json.load(f)

            # Merge keys (excluding "progress")
            for key, value in eq_data.items():
                if key != "progress":
                    result_data[key] = value

            # Overwrite result file
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)

def traverse(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # If the folder contains only files (terminal node)
        if dirnames == [] and filenames:
            process_folder(dirpath)

if __name__ == "__main__":
    root_dir = "results"  # change if needed
    traverse(root_dir)

