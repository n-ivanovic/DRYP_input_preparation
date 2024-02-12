##################################################################################################
###################################### Author details ############################################
##################################################################################################

__date__        = "September 2023"
__author__      = "Nikola Ivanović"
__email__       = "nikola.ivanovic@kaust.edu.sa"
__department__  = "Earth Science and Engineering"
__institution__ = "King Abdullah University of Science and Technology (KAUST), SA"

##################################################################################################
######################################### Headline ###############################################
##################################################################################################

"""
Aim:
---
This script prepares groundwater and boundary conditions data for the DRYP 1.0 model. The script
downloads, processes, and saves the data outputs.

Input:
-----
1. Aquifer saturated hydraulic conductivity (downloaded from GLHYMPS website).
2. Water table depth (G³M 1.0 steady-state model output).

Operations:
----------
1. Download the data.
2. Process the downloaded data, and save the outputs.

Outputs:
-------
1. Aquifer Saturated Hydraulic Conductivity (Ksat_aq)	->		Ksat_aq
2. Specific Yield (Sy)									->		Sy
3. Initial Conditions Water Table Elevation (WTE)		->		wte
4. Flux Boundary Conditions (FBC)						->		None
5. Head Boundary Conditions (HBC)						->		hbc
"""

##################################################################################################
###################################  Main body of the script  ####################################
##################################################################################################

# Import modules:
import os
from components.timing import script_timer
from components.configuration import config
from components.downloading import download_data
from components.parameters import process_data

##################################################################################################
# Script timer:
##################################################################################################

# Display the runtime of the script at the end:
runtime = script_timer()

##################################################################################################
# Set up directories:
##################################################################################################

# Get the directories:
input_dir_ksat_aq = config.dir['input_ksat_aq']
input_dir_wte = config.dir['input_wte']
input_dir_sy = config.dir['input_sy']
output_dir = config.dir['output']
temp_dir = config.dir['temp']

##################################################################################################
# Download the data:
##################################################################################################

# Download the GLHYMPS data manually.
# Download the G³M 1.0 data:
# download_data(input_dir_wte)

##################################################################################################
# Process the data and save the outputs:
##################################################################################################

# Calculate the parameters:
res_crop_ksat_aq, res_crop_wte = process_data(input_dir_ksat_aq, input_dir_wte,  
                                              temp_dir, output_dir)
# Calculate the specific yield (Sy):
# sy = calculate_sy(input_dir_sy, temp_dir, output_dir)

##################################################################################################
# Remove temporary files:
##################################################################################################

# If the user wants to delete the temporary directory:
if config.intermediate_step['delete']:
    os.rmdir(temp_dir)

##################################################################################################
######################################  End of the script  #######################################
##################################################################################################