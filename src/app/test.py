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

# Copy the file
if os.path.exists(source_file):
    shutil.copy(source_file, destination_folder)
    print(f"File copied to {destination_folder}")
else:
    print(f"File {source_file} does NOT exist!")