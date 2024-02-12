##################################################################################################
###################################### Author details ############################################
##################################################################################################

__date__        = "September 2023"
__author__      = "Nikola Ivanović"
__email__       = "nikola.ivanovic@kaust.edu.sa"
__department__  = "Earth Science and Engineering"
__institution__ = "King Abdullah University of Science and Technology, SA"

##################################################################################################
####################################### Headline #################################################
##################################################################################################

"""
Aim:
---
This script prepares soil and subsurface data for the DRYP 1.0 model. The script downloads,
processes, and saves the data outputs.
Input:
-----
1. Soil data (downloaded from ISRIC website).

Operations:
----------
1. Download the data.
2. Process the downloaded data and save the outputs.

Outputs:
-------
1. Soil porosity (porosity)		                ->	    	      WCsat
2. Theta residual			                    -> 	           	  WCres
3. Available water content (AWC)  	            ->	              WCavail
4. Wilting point (wp)		                   	->	    	      CRIT-WILT
5. Soil depth (D)			                    -> 	   	          SOIL-DEPTH
6. Soil particle distribution parameter (b)     ->	    	      N
7. Soil suction head (psi)	                	->                Alfa & N
8. Saturated hydraulic conductivity (Ksat)      -> 	   	          Ksat
9. Sigma Ksat				                    ->	              std_Ksat
10. Initial soil water content                  ->	              None
"""

##################################################################################################
###################################  Main body of the script  ####################################
##################################################################################################

# Import modules:
import os
from components.timing import script_timer
from components.configuration import config
from components.downloading import download_data
from components.parameters import results

##################################################################################################
# Script timer:
##################################################################################################

# Display the runtime of the script at the end:
runtime = script_timer()

##################################################################################################
# Set up directories:
##################################################################################################

# Get the directories:
input_dir = config.dir['input']
output_dir = config.dir['output']
temp_dir = config.dir['temp']

##################################################################################################
# Download the data:
##################################################################################################

# Download and unzip the data (if not already downloaded):
# download_data(input_dir)

##################################################################################################
# Process the data and save the outputs:
##################################################################################################

# Process the data:
output = results(input_dir, temp_dir, output_dir)

##################################################################################################
# Remove temporary files:
##################################################################################################

# If the user wants to delete the temporary directory:
if config.intermediate_step['delete']:
    os.rmdir(temp_dir)

##################################################################################################
######################################  End of the script  #######################################
##################################################################################################