import csv
import glob
import os

csv_paths = glob.glob("/home/marlon/edu/mestrado/arthur/rearing/project_folder/csv/targets_inserted/*.csv")
save_dir = "./data/rearing_csv"
os.makedirs(save_dir, exist_ok=True)

for csv_file in csv_paths:
    filename = os.path.basename(csv_file)
    save_path = os.path.join(save_dir, filename)
    
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        columns = next(reader)
        rearing_index = columns.index("rearing")
        
        rearing_data = []
        for row in reader:
            if len(row) > rearing_index:
                rearing_data.append([row[rearing_index]])
    
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rearing"])
        writer.writerows(rearing_data)
    
    print(f"Processed {filename}: {len(rearing_data)} rows saved to {save_path}")
