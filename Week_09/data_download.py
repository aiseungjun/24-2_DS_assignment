import os
import requests
from tqdm import tqdm

def download_file(url, dest_path):
    """
    Downloads a file from the given URL to the specified path with resume support.
    """
    # Get the size of the existing file, if any
    temp_file_size = 0
    if os.path.exists(dest_path):
        temp_file_size = os.path.getsize(dest_path)

    headers = {"Range": f"bytes={temp_file_size}-"}  # Resume from where the download stopped
    response = requests.get(url, stream=True, headers=headers, verify=False)

    total_size = int(response.headers.get("content-length", 0)) + temp_file_size

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"Starting download: {os.path.basename(dest_path)}")
    print(f"File size: {total_size / (1024 * 1024):.2f} MB")
    print(f"Already downloaded: {temp_file_size / (1024 * 1024):.2f} MB\n")

    with open(dest_path, "ab") as f, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total_size,
        initial=temp_file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print(f"\nDownload complete: {os.path.basename(dest_path)}")



def setup_ucf101(data_dir, annot_dir):
    """
    Downloads UCF101 dataset and annotation files, and prepares the directory structure.
    """
    # URLs for UCF101 dataset and annotations
    video_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
    annot_url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

    # Paths for the downloaded files
    video_path = os.path.join(data_dir, "UCF101.rar")
    annot_path = os.path.join(annot_dir, "UCF101TrainTestSplits.zip")

    # Download the files
    download_file(video_url, video_path)
    download_file(annot_url, annot_path)

    print("\nDownload complete!")
    print("Please extract the files manually to the appropriate directories:")
    print(f"  - Extract UCF101.rar into: {data_dir}")
    print(f"  - Extract UCF101TrainTestSplits.zip into: {annot_dir}")

# Set up directories
DATA_DIR = "./data"
ANNOT_DIR = "./data"

# Run the setup
setup_ucf101(DATA_DIR, ANNOT_DIR)
