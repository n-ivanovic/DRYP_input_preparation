# DRYP Input Data Preparation

![GitHub last commit](https://img.shields.io/github/last-commit/n-ivanovic/DRYP_input_preparation?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/n-ivanovic/DRYP_input_preparation?style=flat-square)

## 📝 Overview

This repository contains the code and instructions for preparing the input data for the *Parsimonious Dryland Water Partition* model - [**`DRYP`**](https://github.com/n-ivanovic/DRYP). The DRYP model is a hydrological model that simulates the water balance of a catchment, including the processes of precipitation, evapotranspiration, runoff, and groundwater flow. The model requires a range of input data, including terrain, surface, soil, and subsurface data, as well as groundwater and boundary conditions.

## 📁 Repository Structure

The repository is structured as follows:

- [`/0_terrain_data`](https://github.com/n-ivanovic/DRYP_input_preparation/tree/Main/0_terrain_data): Contains the code and instructions for preparing terrain data.
- [`/1_surface_data`](https://github.com/n-ivanovic/DRYP_input_preparation/tree/Main/1_surface_data): Contains the code and instructions for preparing surface data.
- [`/2_soil_and_subsurface_data`](https://github.com/n-ivanovic/DRYP_input_preparation/tree/Main/2_soil_and_subsurface_data): Contains the code and instructions for preparing soil and subsurface data.
- [`/3_groundwater_and_boundary_conditions_data`](https://github.com/n-ivanovic/DRYP_input_preparation/tree/Main/3_groundwater_and_boundary_conditions_data): Contains the code and instructions for preparing groundwater and boundary conditions data.
- `README.md`: This file with information about the dataset, how to use it, and repository guidelines.
- `LICENSE`: The license file detailing the terms under which the dataset is made available.
- `requirements.txt`: A file containing the required Python packages to run the code in this repository.

### Terrain Data Preparation

- **Input:** DEM and UPSTREAM tiles from [MERIT Hydro](https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/).
- **Operations:** Download, merge, reproject/resample, and calculate terrain parameters.
- **Outputs:** DEM, cell area, flow direction, (CHB) boundary conditions, terrain mask, river lengths.

### Surface Data Preparation

- In development.

### Soil and Subsurface Data Preparation

- **Input:** Soil data from [HiHydroSoil_v2.0](https://www.futurewater.eu/projects/hihydrosoil/) with following [layers](https://gee-community-catalog.org/projects/hihydro_soil/), and soil depth from [ISRIC](https://www.isric.org/explore/soilgrids).
- **Operations:** Download and process soil data.
- **Outputs:** Soil porosity, residual theta, AWC, wilting point, soil depth, particle distribution parameter, soil suction head, Ksat, sigma Ksat.

### Groundwater and Boundary Conditions Data Preparation

- **Input:** Aquifer Ksat from [GLHYMPS](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/TTJNIU), water table depth from [G³M 1.0](https://zenodo.org/records/1315471, ) [DOI](https://gmd.copernicus.org/preprints/gmd-2018-120/gmd-2018-120.pdf).
- **Operations:** Download and process groundwater data.
- **Outputs:** Aquifer Ksat, specific yield, WTE, (CHB) boundary conditions.

## ⚙️ Environment Setup

To run the code in this repository, you will need the following:

- Ensure that Python 3 and pip (Python package installer) are installed on your system.
- Clone the repository to your local machine.
- Navigate to the repository directory and install the required Python packages with:

  ```bash
  pip install -r requirements.txt
  ```

## 📊 How to Use the Repository

To use this repository:

- After successfully cloning the repository and setting up the environment, navigate to the desired directory (e.g., `/0_terrain_data`) and run the Python script.
- The download functions have been commented out to avoid unnecessary downloads. `Uncomment` the download functions to download the data.
- Follow the `instructions` in the script to download, process, and prepare the input data for the DRYP model.
- Refer to this README.md for metadata and context.

## 🤝 How to Contribute

Contributions to this dataset are welcome! If you have updated information, corrections, or suggestions, please feel free to:

- Open an issue to discuss your proposed changes.
- Submit a pull request with your updates.
- **`Please ensure that any contributed data follows the existing structure for consistency!`**

## 📜 License

This dataset is provided under MIT license. Please refer to the [LICENSE](https://github.com/n-ivanovic/DRYP_input_preparation/blob/Main/LICENSE) file for more details.

## ✉️ Contact

For questions or support regarding this dataset, please open an issue in this repository.
