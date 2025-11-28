import os 
import shutil

# a tiny function to remove intermediate files
def remove_unnecessary_files(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)