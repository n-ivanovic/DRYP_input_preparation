# Import modules:
import os
import tarfile
import requests
from tqdm import tqdm
from components.configuration import config


# ------------------------------------------------------------------------------------------------
# Data download functions:
# ------------------------------------------------------------------------------------------------


# Define a function to download and extract the file:
def download_data(output_path):

    """
    This function downloads and unzips G³M 1.0 steady-state model output.

    Parameters:
    ----------
    output_path : str
        The path to the output directory.
    
    Returns:
    -------
    None
    """

    # Get the URL settings:
    url = config.url['wte']

    # Remove query parameters from the URL to clean the filename
    clean_url = url.split('?')[0]
    # Set the filename
    filename = os.path.join(output_path, clean_url.rsplit('/', 1)[-1])
    # Download the file if it doesn't exist
    if not os.path.exists(filename):
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            # Open the file to write the chunks and display the tqdm progress bar
            with open(filename, 'wb') as f, tqdm(
                desc=f"Downloading {filename}...",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024) as bar:
                    for chunk in r.iter_content(chunk_size=8192*1000):
                        size = f.write(chunk)
                        bar.update(size)
        # If the download fails, print the error:
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}. Error: {e}")
    # Extract the tar.gz file and flatten the folder structure
    if filename.endswith(".tar.gz"):
        with tarfile.open(filename, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isreg():  # Skip if the TarInfo is not files
                    member.name = os.path.basename(member.name)  # Remove folder names
                    tar.extract(member, path=output_path)
        # Remove the tar.gz file
        os.remove(filename)