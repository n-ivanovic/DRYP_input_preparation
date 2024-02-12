import os
import numpy as np
from termcolor import colored
from scipy.spatial import KDTree
from components.timing import timer
from components.saving import save_data
from components.plotting import plot_data
from components.cropping import crop_data
from components.configuration import config
from components.reprojecting import reproject_data


# ------------------------------------------------------------------------------------------------
# Soil parameters functions:
# ------------------------------------------------------------------------------------------------


# Define a function to adjust the data range:
def parameter_range(data):

    """
    This function adjusts the data range, and caps the data within the range. The 
    range is determined by the 5th and 95th percentiles of the data, which are 
    calculated and used as the lowerand upper bounds, respectively. The main purpose 
    of this function is to remove outliers, and prevent potential mathematical errors 
    due to division by zero.

    Parameters:
    ----------
    data : numpy array
        The data array to be adjusted.

    Returns:
    -------
    data : numpy array
        The adjusted data array.
    """
    
    # Calculate the 5th and 95th percentiles of the data:
    lower_bound = np.nanpercentile(data, 5)
    upper_bound = np.nanpercentile(data, 95)
    # Cap the data within the bounds (i.e. remove outliers):
    data[data < lower_bound] = lower_bound
    data[data > upper_bound] = upper_bound
    
    # Return the capped data:
    return data


# Define a function to fill in all NaNs:
def fill_all_nans(data, k=3):

    """
    This function fills in all NaNs in a data array.

    Parameters:
    ----------
    data : numpy array
        The data array to be filled.
    k : int
        The number of nearest neighbors to use for filling in the NaNs.

    Returns:
    -------
    filled_array : numpy array
        The filled data array.
    """

    # Create a copy of the original array:
    filled_array = data.copy()
    # Find coordinates where data exists (not NaN):
    existing_data_coords = np.column_stack(np.where(~np.isnan(filled_array)))
    # Find coordinates where data is missing (NaN):
    missing_data_coords = np.column_stack(np.where(np.isnan(filled_array)))
    # Build KDTree using existing data coordinates:
    tree = KDTree(existing_data_coords)
    # Query KDTree to find indices of k nearest existing data points for each missing data point:
    _, ind = tree.query(missing_data_coords, k=k)
    # Extract k nearest existing values:
    k_nearest_values = filled_array[tuple(existing_data_coords[ind].reshape(-1, 2).T)].reshape(len(missing_data_coords), k)
    # Fill missing values using the average of k nearest existing values:
    nearest_values = np.nanmean(k_nearest_values, axis=1)
    filled_array[missing_data_coords[:, 0], missing_data_coords[:, 1]] = nearest_values

    # Return the filled array:
    return filled_array


# Define a function to process the data:
def process_data(parameter, input_path, temp_path, output_path, 
                 data_name, title, cbar_label, cmap, save=True):

    """
    This function processes the data by cropping, resampling, and saving the data.

    Parameters:
    ----------
    parameter : str
        The parameter to process.
    input_path : str
        The path to the input directory.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.
    data_name : str
        Name of the data to be processed.
    title : str
        The title of the data.
    cbar_label : str
        The colorbar label of the data.
    cmap : str
        The colormap of the data.
    save : bool
        Whether to save the data or not. Default is True.

    Returns:
    -------
    res_crop_data : numpy array
        The cropped and resampled data.
    res_crop_metadata : dict
        The metadata of the resampled cropped data.
    """

    # Get the intermediate step for plotting:
    intermediate_step = config.intermediate_step['plot']
    
    # Construct the file path:
    if parameter == 'SOIL-DEPTH':
        data_path = os.path.join(input_path, f"{parameter}_M_250m.tif")
    else:
        data_path = os.path.join(input_path, 'top_subsoil', parameter, 
                                 f"{parameter}_M_250m_TOPSOIL.tif")
    # If the parameter is 'ALFA', or 'N', don't save the data output:
    if parameter == 'ALFA' or parameter == 'N':
        save = False
    # Crop the dataset:
    crop_dataset, crop_metadata = crop_data(data_path, parameter)
    # Reproject and resample the data:
    res_crop_data, res_crop_metadata = reproject_data(crop_dataset, crop_metadata)
    # Correct the data range to remove outliers, and division by zero:
    with timer('Correcting the data range...'):
        res_crop_data = parameter_range(res_crop_data)
        print(colored(' ✔ Done!', 'green'))
    # Fill in the NaNs:
    with timer('Filling in the NaNs...'):
        res_crop_data = fill_all_nans(res_crop_data)
        print(colored(' ✔ Done!', 'green'))
    # Correct the units of the following parameters:
    if parameter == 'ALFA':
        res_crop_data = res_crop_data / 10                  # [cm^-1] to [mm^-1]
    elif parameter == 'Ksat':
        res_crop_data = res_crop_data * 10 / 24             # [cm d^-1] to [mm h^-1]
    elif parameter == 'SOIL-DEPTH':
        res_crop_data = res_crop_data * 1000                # [m] to [mm]
    # Save the resampled data:
    if save:
        save_data(res_crop_data, res_crop_metadata, output_path, 
                  data_name=f"resampled_{data_name}")
    # Plot the resampled data:
    if intermediate_step:
        plot_data(res_crop_data, res_crop_metadata, temp_path,
                  data_name=f"resampled_{data_name}", title=f"Resampled {title}",
                  cbar_label=cbar_label, cmap=cmap)
    # Print a message:
    print(f"{colored(parameter, 'blue', attrs=['bold', 'underline'])}", colored('is processed!', 'green')) 
    print(colored('==========================================================================================', 'blue'))

    # Return the dictionary parameters:
    return {'parameter': parameter, 'data': res_crop_data, 'metadata': res_crop_metadata}


# Define a function to calculate the soil particle distribution parameter (b):
def calculate_b(n, metadata, temp_path, output_path):
        
    """
    This function calculates the soil particle distribution parameter (b).

    Parameters:
    ----------
    n : numpy array
        The soil pore size distribution parameter (n).
    metadata : dict
        The dictionary of metadata.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    b : numpy array
        The soil particle distribution parameter (b).
    """

    # Calculate parameter b:
    with timer('Calculating b parameter...'):
        # Calculate:
        b = 1 / (n - 1)
        print(colored(' ✔ Done!', 'green'))
    # Save the data:
    save_data(b, metadata, output_path, 
                data_name='parameter_b')                  
    # Plot the data:
    plot_data(b, metadata, temp_path,
                data_name='parameter_b', title='Parameter (b)',
                cbar_label='Soil Particle Distribution []', cmap='viridis')
    print(colored('==========================================================================================', 'blue'))
    
    # Return soil particle distribution parameter (b):
    return b


# Define a function to calculate the soil suction head parameter (psi):
def calculate_psi(alfa, n, theta_s, theta, metadata, temp_path, output_path):
            
    """
    This function calculates the soil suction head parameter (psi).

    Parameters:
    ----------
    alfa : numpy array
        The soil pore size distribution parameter (alfa).
    n : numpy array
        The soil pore size distribution parameter (n).
    theta_s : numpy array
        The saturated water content (WCsat).
    theta : numpy array
        The current water content (WCavail).
    metadata : dict
        The dictionary of metadata.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    psi : numpy array
        The soil suction head parameter (psi).
    """

    # Calculate parameter psi:
    with timer('Calculating psi parameter...'):
        # Calculate:
        psi = (1 / alfa) * (((theta_s / theta) ** ((n - 1) / n)) - 1) ** (1 / n)
        print(colored(' ✔ Done!', 'green'))
    # Save the data:
    save_data(psi, metadata, output_path,
              data_name='parameter_psi')
    # Plot the data:
    plot_data(psi, metadata, temp_path,
              data_name='parameter_psi', title='Parameter (psi)', 
              cbar_label='Soil Suction Head [mm]', cmap='viridis')
    print(colored('==========================================================================================', 'blue'))
    
    # Return soil suction head parameter (psi):
    return psi


# Define a function to calculate sigma Ksat:
def calculate_sigma_ksat(ksat, metadata, temp_path, output_path):
    
    """
    This function creates an ones array as the sigma Ksat.

    Parameters:
    ----------
    ksat : numpy array
        The saturated hydraulic conductivity (Ksat).
    metadata : dict
        The dictionary of metadata.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    sigma_ksat : numpy array
        The standard deviation of Ksat.
    """

    # Calculate sigma Ksat:
    with timer('Calculating sigma Ksat...'):
        # Calculate:
        sigma_ksat_value = np.nanstd(ksat)
        # Create an array of the same shape as Ksat, and fill it with the std. dev. of Ksat:
        sigma_ksat = (np.ones(ksat.shape) * sigma_ksat_value).astype(np.float32)
        print(colored(' ✔ Done!', 'green'))
    # Save the data:
    save_data(sigma_ksat, metadata, output_path, 
                data_name='sigma_ksat')
    print(colored('==========================================================================================', 'blue'))

    # Return the standard deviation of Ksat:
    return sigma_ksat


# Define a function to summarize the results:
def summarize_results(results):

    """
    This function prints out the NaN count, maximum value, mean value, minimum value, and 
    Inf count for each parameter in the results.

    Parameters:
    ----------
    results : dict
        The dictionary of soil data processing results.
    
    Returns:
    -------
    None.
    """

    # Print the header of the table:
    print(f"{'Parameter':<15}{'NaN':<10}{'Max':<10}{'Mean':<10}{'Min':<10}{'Inf':<10}")
    # Loop through the parameters in results:
    for key in results.keys():
        if 'data' in results[key]:
            # Check if the data is a numeric numpy array:
            if isinstance(results[key]['data'], np.ndarray) and np.issubdtype(results[key]['data'].dtype, np.number):
                # Compute the NaN count, maximum value, mean value, minimum value, and Inf count:
                nan_count = f"{np.count_nonzero(np.isnan(results[key]['data']))}"
                max_value = f"{np.nanmax(results[key]['data']):.3f}"
                mean_value = f"{np.nanmean(results[key]['data']):.3f}"
                min_value = f"{np.nanmin(results[key]['data']):.3f}"
                inf_count = f"{np.count_nonzero(np.isinf(results[key]['data']))}"
                # Print the results in a row following column positions:
                print(f"{key:<15}{nan_count:<10}{max_value:<10}{mean_value:<10}{min_value:<10}{inf_count:<10}")
                # Save the computed values back to results
                results[key].update({
                    'nan_count': nan_count, 
                    'max_value': max_value,
                    'mean_value': mean_value,
                    'min_value': min_value,
                    'inf_count': inf_count})
            else:
                print(f"{key}:", colored(' data is not a numeric NumPy array!', 'red'))


# Define a function to create a dictionary of soil data processing results:
def results(input_path, temp_path, output_path):

    """
    This function creates a dictionary of soil data processing results.

    Parameters:
    ----------
    input_path : str
        The path to the input directory.
    temp_path : str
        The path to the temporary directory.
    output_path : str
        The path to the output directory.

    Returns:
    -------
    results : dict
        The dictionary of soil data processing results.
    """

    # Define the data parameters:
    parameters = ['ALFA', 'CRIT-WILT', 'Ksat', 'N', 'WCavail', 'WCres', 'WCsat', 'SOIL-DEPTH']
    # Define the data names:
    data_names = ['alfa', 'crit_wilt', 'ksat', 'n', 'wc_avail', 'wc_res', 'wc_sat', 'soil_depth']
    # Define the data titles:
    titles = ['Alfa Parameter',   
              'Wilting Point', 
              'Saturated Hydraulic Conductivity', 
              'N Parameter', 
              'Available Water Content', 
              'Residual Water Content', 
              'Saturater Water Content',
              'Soil Depth']
    # Define the data colorbar labels:
    cbar_labels = [r'Alfa Parameter of the van Genuchten model [mm$^{-1}$]', 
                   r'Wilting point [m$^3$ m$^{-3}$]', 
                   r'Saturated hydraulic conductivity [mm h$^{-1}$]', 
                   r'N Parameter of the van Genuchten model []', 
                   r'Available water content [m$^3$ m$^{-3}$]', 
                   r'Residual Water Content [m$^3$ m$^{-3}$]', 
                   r'Saturater Water Content [m$^3$ m$^{-3}$]',
                   r'Soil Depth [mm]']
    # Define the data colormaps:
    cmaps = ['viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis', 'viridis']
    # Create a dictionary to store the results:
    results = {}
    # Loop through the parameters:
    for parameter, data_name, title, cbar_label, cmap in zip(parameters, data_names, titles, cbar_labels, cmaps):
        # Process the data:
        result = process_data(parameter, input_path, temp_path, output_path, 
                              data_name, title, cbar_label, cmap)
        # Save the parameter and its corresponding data in the results dictionary:
        results[result['parameter']] = {'data': result['data'], 
                                        'metadata': result['metadata']}
    # Calculate parameter b:
    b = calculate_b(results['N']['data'], results['N']['metadata'], temp_path, output_path)
    # Calculate parameter psi:
    psi = calculate_psi(results['ALFA']['data'], results['N']['data'], results['WCsat']['data'], 
                        results['WCavail']['data'], results['N']['metadata'], temp_path, output_path)
    # Calculate the standard deviation of Ksat:
    sigma_Ksat = calculate_sigma_ksat(results['Ksat']['data'], results['Ksat']['metadata'], temp_path, output_path)
    # Save the b parameter and its corresponding data in the results dictionary:
    results['b'] = {'data': b, 'metadata': results['N']['metadata']}
    # Save the psi parameter and its corresponding data in the results dictionary:
    results['psi'] = {'data': psi, 'metadata': results['N']['metadata']}
    # Save the sigma_Ksat parameter and its corresponding data in the results dictionary:
    results['sigma_Ksat'] = {'data': sigma_Ksat, 'metadata': results['Ksat']['metadata']}
    # Summarize the results:
    summarize_results(results)

    # Return the results dictionary:
    return results