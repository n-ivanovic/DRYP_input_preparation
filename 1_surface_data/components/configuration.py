# Import modules:
import sys
import importlib


# ------------------------------------------------------------------------------------------------
# Configuration functions:
# ------------------------------------------------------------------------------------------------


# Define the configuration singleton class:
class ConfigSingleton:

    """
    This class is used to create a singleton object that holds the configuration 
    settings. The configuration settings are loaded from the configuration module
    specified in the argument, or from the configuration module specified in the
    command line.

    Attributes:
    ----------
    _instance : class
        The single instance of this class.
    config : module
        The configuration module.

    Methods:
    -------
    __new__(cls, config_name="config_AP")
        This function is used to control the creation of a single instance of this
        class, and to load the configuration settings.
    """
    
    # Define the class attribute to hold the single instance of this class:
    _instance = None
    # Define the function to load the configuration settings:
    def __new__(cls, config_name="config_AP"):

        """
        This function is used to control the creation of a single instance of this
        class. If an instance already exists, it is returned. If not, a new instance
        is created and returned. This function is also used to load the configuration
        settings. The configuration settings are loaded from the configuration module
        specified in the argument, or from the configuration module specified in the
        command line.
        
        Parameters:
        ----------
        cls : class
            The class to instantiate.
        config_name : str
            The name of the configuration module to import.
        
        Returns:
        -------
        cls._instance : class
            The single instance of this class.
        """

        # Check if an instance already exists:
        if cls._instance is None:
            # Create and assign a new instance:
            cls._instance = super(ConfigSingleton, cls).__new__(cls)
            try:
                # Check if the script is being run in a Jupyter Notebook environment:
                if 'ipykernel' in sys.modules:
                    # If so, import the configuration module specified in the argument:
                    cls._instance.config = importlib.import_module(config_name)
                else:
                    # If not, check if the user specified the configuration file in the command line:
                    config_name_from_arg = sys.argv[1] if len(sys.argv) > 1 else config_name
                    # Import the configuration module specified in the command line:
                    cls._instance.config = importlib.import_module(config_name_from_arg)
            except ImportError:
                    print(f"Error: The configuration module {config_name} could not be imported.")
                    sys.exit(1)
                     
        # Return the single instance:
        return cls._instance

# Create a singleton object that holds the configuration settings:
config = ConfigSingleton().config