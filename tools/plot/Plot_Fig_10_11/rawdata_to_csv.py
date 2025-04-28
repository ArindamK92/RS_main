import re
import pandas as pd
#import os

def parse_log_file_to_csv(input_txt_file, output_csv_file):
    with open(input_txt_file, 'r') as f:
        txt_content = f.read()

    records = []
    current_record = {}

    for line in txt_content.splitlines():
        line = line.strip()

        if line.startswith("Input Graph File:"):
            graph_path = line.split(":")[1].strip()
            graph_name = graph_path.split("/")[-1].split(".")[0]
            current_record['Graph'] = graph_name
        elif line.startswith("Communities Used:"):
            current_record['k'] = int(line.split(":")[1].strip())
        elif line.startswith("Step"):
            match = re.match(r"Step (\d+[A-Z]?): .*: (\d+) ms", line)
            if match:
                step_label = f"Step {match.group(1)}"
                current_record[step_label] = int(match.group(2))
        elif line.startswith("Total Execution Time::"):
            current_record['Total'] = int(line.split("::")[1].strip().split()[0])
            records.append(current_record)
            current_record = {}  # Reset for next block

    all_columns = ['Graph', 'k', 'Step 1', 'Step 2A', 'Step 2B', 'Step 2C',
                   'Step 2D', 'Step 2E', 'Step 2F', 'Step 3', 'Total']
    df = pd.DataFrame(records)
    df = df.reindex(columns=all_columns)

    df.to_csv(output_csv_file, index=False)
    print(f"Saved parsed results to {output_csv_file}")

parse_log_file_to_csv("raw_results.txt", "timing_results.csv")
