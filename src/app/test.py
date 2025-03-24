import os
import shutil

data_path = "/tree"

if os.path.exists(data_path):
    print("Contents of /tree:")
    print(os.listdir(data_path))  # Lists files and folders in /data
else:
    print(f"Directory {data_path} does not exist.")
    
source_file = "/tree/xmrtree.pkl"

destination_file = "/home/jvedor/test/xmrtree.pkl"

shutil.copy(source_file, destination_file)