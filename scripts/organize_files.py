import os
import shutil
import re

base_dir = "/home/marlon/edu/mestrado/comp_neuroetho/keypoint_comp_neuroetho/projects/elm_ms/data/vids"
video_dirs = ['G0', 'G39', 'G41', 'G42', 'G43', 'G45', 'G64', 'G65', 'G66', 'G67', 'G68']

# Pattern to match video names: R<number>G<number><type>.mp4
pattern = re.compile(r'R\d+G\d+([A-Z]\d*)\.mp4')

for video_dir in video_dirs:
    video_dir_path = os.path.join(base_dir, video_dir)
    
    if not os.path.exists(video_dir_path):
        continue
        
    for filename in os.listdir(video_dir_path):
        if not filename.endswith('.mp4'):
            continue

        match = pattern.match(filename)
        if match:
            file_type = match.group(1)
            
            # Skip type 'T'
            if file_type == 'T':
                continue
            
            type_dir = os.path.join(base_dir, file_type)
            os.makedirs(type_dir, exist_ok=True)
            
            # Get the base name (without .mp4)
            base_name = filename[:-4]
            
            # Copy mp4 file
            src_mp4 = os.path.join(video_dir_path, filename)
            dst_mp4 = os.path.join(type_dir, filename)
            shutil.copy2(src_mp4, dst_mp4)
            print(f"Copied {filename} to {file_type}/")
            
            # Find and copy associated .csv and .h5 files
            for other_file in os.listdir(video_dir_path):
                if other_file.startswith(base_name) and (other_file.endswith('.csv') or other_file.endswith('.h5')):
                    src_other = os.path.join(video_dir_path, other_file)
                    dst_other = os.path.join(type_dir, other_file)
                    shutil.copy2(src_other, dst_other)
                    print(f"Copied {other_file} to {file_type}/")

print("Organization complete!")