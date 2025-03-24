import os
import shutil

data_path = "/tree"

if os.path.exists(data_path):
    print("Contents of /tree:")
    print(os.listdir(data_path))  # Lists files and folders in /data
else:
    print(f"Directory {data_path} does not exist.")
    
source_file = "/tree/xmrtree.pkl"
destination_folder = "/home/jvedor/test"  # Destination should be a directory

# Construct the full path inside the destination folder
destination_path = os.path.join(destination_folder, os.path.basename(source_file))

# Copy the file
if os.path.exists(source_file):
    shutil.copy(source_file, destination_path)
    print(f"File copied to {destination_path}")
else:
    print(f"File {source_file} does NOT exist!")