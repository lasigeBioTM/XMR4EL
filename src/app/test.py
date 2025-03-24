import os

data_path = "/tree"

if os.path.exists(data_path):
    print("Contents of /tree:")
    print(os.listdir(data_path))  # Lists files and folders in /data
else:
    print(f"Directory {data_path} does not exist.")