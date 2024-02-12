# Import modules:
import os
import tarfile
import requests
from tqdm import tqdm
from termcolor import colored
from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth
from concurrent.futures import ThreadPoolExecutor
from components.configuration import config


# ------------------------------------------------------------------------------------------------
# Data download functions:
# ------------------------------------------------------------------------------------------------


# Define a function to extract individual files from the tar file:
def extract_tar(tar_path, output_path):
    
    """
    Extracts individual files from the tar file to the specified output path.

    Parameters:
    ----------
    tar_path : str
        The path to the tar file.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    None
    """

    # Extract individual files from the tar file:
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            # Check if the member is a regular file:
            if member.isreg():
                # Extract the member to the specified output path:
                member.name = os.path.basename(member.name)
                tar.extract(member, output_path)
        # Remove the tar file:
        os.remove(tar_path)


# Define a function to download and extract the file:
def download_extract(url, auth, link, output_path):

    """
    Downloads and extracts the file.

    Parameters:
    ----------
    url : str
        URL of the data.
    auth : tuple
        Username and password for the authorization.
    link : str
        URL of the file.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    None
    """

    # Set the filename:
    filename = os.path.join(output_path, link.rsplit('/', 1)[-1])
    # Download the file if it doesn't exist:
    if not os.path.exists(filename):
        try:
            # Send GET request to the source URL with authorization:
            r = requests.get(url + link, stream=True, auth=HTTPBasicAuth(*auth))
            r.raise_for_status()
            # Write the file to the specified output path:
            with open(filename, 'wb') as f:
                # Write the file in chunks:
                for chunk in r.iter_content(chunk_size=8192*1000):
                    f.write(chunk)
        # If the download fails, print the error:
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {link}. Error: {e}")
    # Apply the extract_tar function:
    if filename.endswith(".tar"):
        extract_tar(filename, output_path)


# Define a function to download data:
def download_data(output_path, data_type):

    """
    Downloads data from the official sources.

    Parameters:
    ----------
    output_path : str
        The path to the output directory.
    data_type : str
        Type of the data.

    Returns:
    -------
    None
    """

    # Get the URL and authorization settings:
    url = config.url[data_type]
    auth = config.auth[data_type]
    
    # Send GET request to the source URL with authorization:
    response = requests.get(url, auth=HTTPBasicAuth(*auth))
    soup = BeautifulSoup(response.text, 'html.parser')
    # DEM data:
    if data_type == 'DEM':
        # Get the links to the tiff files:
        links = [link['href'] for link in soup.find_all('a', href=True) 
                 if link['href'].startswith('./distribute/v1.0.2/dem') 
                 and link['href'].endswith('.tar')]
        # Print a message:
        print('The number links found:', len(links))
    # Upstream data:
    elif data_type == 'UPSTREAM':
        # Get the links to the tar files ending with 'upa' and '.tar':
        links = [link['href'] for link in soup.find_all('a', href=True) 
                 if link['href'].startswith('./distribute/v1.0/upa') 
                 and link['href'].endswith('.tar')]
    # Download and extract the files:
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(download_extract, [url]*len(links), [auth]*len(links), links, [output_path]*len(links)), 
                  total=len(links), desc="Downloading and extracting files..."))
    # Print a message:
    print('Download and extraction complete!')
    print(colored('==========================================================================================', 'blue'))