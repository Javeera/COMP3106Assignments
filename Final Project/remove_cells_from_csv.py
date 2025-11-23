import pandas as pd

def split_csv(file_path, rows_per_file=30000):
    df = pd.read_csv(file_path)
    total_rows = len(df)
    file_count = (total_rows // rows_per_file) + 1

    for i in range(file_count):
        start = i * rows_per_file
        end = start + rows_per_file
        chunk = df[start:end]

        chunk.tocsv(f"part{i+1}.csv", index=False)

print(f"Done! Split into {33} files.")

split_csv("C:\Users\javee\Desktop\School\3106\COMP3106Assignments\Final Project\sbdb_query_results_12columns.csv", rows_per_file=30000)