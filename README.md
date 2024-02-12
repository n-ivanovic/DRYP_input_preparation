# DRYP 1.0 Model Input Data Preparation

## 📝 Overview

[Provide a brief intro about the DRYP model and the purpose of these scripts.]

## 📦 Data Sources

- **DEM and UPSTREAM tiles:** MERIT Hydro
- 

## 📁 Repository Structure

[has to be properly sorted]

### Terrain Data Preparation

- **Aim:** Prepare terrain data for the DRYP model, including topography, flow direction, and basin mask.
- **Input:** DEM and UPSTREAM tiles from MERIT Hydro.
- **Operations:** Download, merge, resample, and calculate terrain parameters.
- **Outputs:** DEM, cell area, flow direction, boundary conditions, terrain mask, river lengths.

### Soil and Subsurface Data Preparation

- **Aim:** Prepare soil and subsurface data for the DRYP model.
- **Input:** Soil data from ISRIC.
- **Operations:** Download and process soil data.
- **Outputs:** Soil porosity, residual theta, AWC, wilting point, soil depth, particle distribution parameter, soil suction head, Ksat, sigma Ksat.

### Groundwater and Boundary Conditions Data Preparation

- **Aim:** Prepare groundwater and boundary conditions data for the DRYP model.
- **Input:** Aquifer Ksat from GLHYMPS, water table depth from G³M 1.0.
- **Operations:** Download and process data.
- **Outputs:** Aquifer Ksat, specific yield, WTE, head boundary conditions.

## 📊 How to Use the Dataset

[]

## ⚙️ Environment Setup

To run the code in this repository, you will need the following:

- Ensure that Python 3 and pip (Python package installer) are installed on your system.
- Clone the repository to your local machine.
- Navigate to the repository directory and install the required Python packages with:

  ```bash
  pip install -r requirements.txt
  ```

- Once the dependencies are installed, you can run the scripts located in the /code directory.

## 🤝 How to Contribute

Contributions to this dataset are welcome! If you have updated information, corrections, or suggestions, please feel free to:

- Open an issue to discuss your proposed changes.
- Submit a pull request with your updates.
- **`Please ensure that any contributed data follows the existing structure for consistency!`**

## 📜 License

This dataset is provided under MIT license. Please refer to the [LICENSE](https://github.com/n-ivanovic/DRYP_input_preparation/blob/Main/LICENSE) file for more details.

## ✉️ Contact

For questions or support regarding this dataset, please open an issue in this repository.