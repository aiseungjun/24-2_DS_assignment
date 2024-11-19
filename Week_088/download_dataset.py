import os

drive_folder_link = "https://drive.google.com/drive/folders/15ybznqG9liiRfCJ7BWGy-LYjgqsIV98X"
download_path = "./dataset"

os.system("pip install gdown")
os.system(f"gdown --folder {drive_folder_link} -O {download_path}")
