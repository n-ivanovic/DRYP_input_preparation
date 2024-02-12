# Import modules:
import os
import subprocess
from tqdm import tqdm
from components.configuration import config


# ------------------------------------------------------------------------------------------------
# Data download functions:
# ------------------------------------------------------------------------------------------------


# Define a function to download the file:
def download_file(url, output_path):

    """
    This function downloads a file from a given url.
    
    Parameters:
    ----------
    url : str
        The url of the file to be downloaded.
    output_path : str
        The path to the output directory.
            
    Returns:
    -------
    download_success : bool
        The boolean value indicating whether the download was successful.
    """

    # Downloading the zip file:
    download_result = subprocess.run(f"wget -O {output_path} {url}", shell=True)
    # Check if the download was successful:
    if download_result.returncode != 0:
        print(f"Download failed! Error: {download_result.stderr}")
        return False
    return True


# Define a function to unzip the file:
def unzip_file(zip_path, output_path, folder_name):

    """
    This function unzips a file.

    Parameters:
    ----------
    zip_path : str
        The path to the zip file.
    output_path : str
        The path to the output directory.
    folder_name : str
        The name of the folder to be created.

    Returns:
    -------
    unzip_success : bool
        The boolean value indicating whether the unzip was successful.
    """

    # Adjust the output_path to include the folder_name:
    final_output_path = os.path.join(output_path, folder_name)
    # Create the folder if it doesn't exist:
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    # Unzipping:
    unzip_result = subprocess.run(f"unzip -o {zip_path} -d {final_output_path} >/dev/null 2>&1", shell=True)
    # Check if the unzip was successful:
    if unzip_result.returncode != 0:
        print(f"Unzip failed! Error: {unzip_result.stderr}")
        return False
    return True


# Defining a function to download soil data:
def download_data(output_path, download_type='both'):
    
    """
    This function downloads the soil data.

    Parameters:
    ----------
    output_path : str
        The path to the output directory.
    download_type : str
        'soil', 'depth', or 'both'. Determines the type of data to download.

    Returns:
    -------
    None.
    """


    # Check if the output directory exists, if not, create it:
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Downloading and unzipping soil data:
    if download_type in ['soil', 'both']:
        # Define the url for soil data:
        urls = config.url['layers', 'top_subsoil']
        # Loop through the urls:
        for key, url in tqdm(urls.items()):
            # Define the zip folder:
            zip_folder = f"{key}.zip"
            # Define the zip folder path:
            zip_folder_path = os.path.join(output_path, zip_folder)
            # Check if the zip folder exists:
            if not os.path.exists(zip_folder_path):
                # Downloading the file:
                download_success = download_file(url, zip_folder_path)
                # Check if the download was successful:
                if not download_success:
                    print(f"Skipping unzipping for {zip_folder} due to failed download.")
                    continue
            # Unzipping the file:
            unzip_success = unzip_file(zip_folder_path, output_path, key)
            # Check if the unzip was successful:
            if unzip_success:
                print(f"Unzipping {zip_folder} successful!")
                # Remove the zip folder:
                os.remove(zip_folder_path)

    # Downloading and unzipping soil depth data:
    if download_type in ['depth', 'both']:
        # Define the url for soil depth:
        url = config.url['soil_depth']
        # Downloading soil depth data:
        download_soil_depth = subprocess.run(f"wget -O {output_path}/SOIL-DEPTH_M_250m.tif {url}", shell=True)
        # Check if the download was successful:
        if download_soil_depth.returncode != 0:
            print(f"Download failed! Error: {download_soil_depth.stderr}")