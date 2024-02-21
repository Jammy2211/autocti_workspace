from os import path

input(
    "\n"
    "############################################\n"
    "### AUTOCTI WORKSPACE WORKING DIRECTORY ###\n"
    "############################################\n\n"
    """
    PyAutoCTI scripts assume that the `autocti_workspace` directory is the Python working directory. 
    This means that, when you run an example script, you should run it from the `autocti_workspace` 
    as follows:


    cd path/to/autocti_workspace (if you are not already in the autocti_workspace).
    python3 scripts/overview/overview_1_what_is_cti.py

    The reasons for this are so that PyAutoCTI can:

    - Load configuration settings from config files in the `autocti_workspace/config` folder.
    - Load example data from the `autocti_workspace/dataset` folder.
    - Output the results of models ctis to your hard-disk to the `autocti/output` folder. 

    Jupyter notebooks update the current working directory to the `autocti_workspace` directory via a magicmethod.

    If you have any errors relating to importing modules, loading data or outputting results it is likely because you
    are not running the script with the `autocti_workspace` as the working directory!

    [Press Enter to continue]
    """
)

input(
    "\n"
    "###############################\n"
    "##### MATPLOTLIB BACKEND ######\n"
    "###############################\n\n"
    """
    We`re now going to plot an image in PyAutoCTI using Matplotlib, using the backend specified in the following
    config file (the backend tells Matplotlib where to render the plot)"


    autocti_workspace/config/visualize/general.yaml -> [general] -> `backend`


    The default entry for this is `default` (check the config file now). This uses the default Matplotlib backend
    on your computer. For most users, pushing Enter now will show the figure without error.

    However, we have had reports that if the backend is set up incorrectly on your system this plot can either
    raise an error or cause the `welcome.py` script to crash without a message. If this occurs after you
    push Enter, the error is because the Matplotlib backend on your computer is set up incorrectly.

    To fix this in PyAutoCTI, try changing the backend entry in the config file to one of the following values:"

    backend=TKAgg
    backend=Qt5Agg
    backeknd=Qt4Agg

    NOTE: If a matplotlib figure window appears, you may need to close it via the X button and then press 
    enter to continue the script.

    [Press Enter to continue]
    """
)

try:
    import numba
except ModuleNotFoundError:
    input(
        "##################\n"
        "##### NUMBA ######\n"
        "##################\n\n"
        """
        Numba is not currently installed.
        
        Numba is a library which makes PyAutoCTI run a lot faster. Certain functionality is disabled without numba
        and will raise an exception if it is used.
        
        If you have not tried installing numba, I recommend you try and do so now by running the following 
        commands in your command line / bash terminal now:
        
        pip install --upgrade pip
        pip install numba
        
        If your numba installation raises an error and fails, you should go ahead and use PyAutoCTI without numba to 
        decide if it is the right software for you. If it is, you should then commit time to bug-fixing the numba
        installation. Feel free to raise an issue on GitHub for support with installing numba.

        A warning will crop up throughout your *PyAutoCTI** use until you install numba, to remind you to do so.
        
        [Press Enter to continue]
        """
    )

import autocti as ac
import autocti.plot as aplt
import matplotlib.pyplot as plt

dataset_path = path.join("dataset", "dataset_1d", "simple")
shape_native = (200,)
norm = 100

region_list = [(10, 20)]

layout = ac.Layout1D(
    shape_1d=shape_native,
    region_list=region_list,
)

dataset = ac.Dataset1D.from_fits(
    data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
    noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
    pre_cti_data_path=path.join(dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"),
    layout=layout,
    pixel_scales=0.1,
)

dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

input(
    "\n"
    "###############################\n"
    "## EXAMPLE CTI DATASET ###\n"
    "###############################\n\n"
    """
    The image displayed on your screen shows a 1D CTI calibration dataset, which will be fitted in example tutorials.
        
    [Press Enter to continue]
    """
)

input(
    ""
    "\n"
    "###############################\n"
    "######## WORKSPACE TOUR #######\n"
    "###############################\n\n"
    """
    PyAutoCTI is now set up and you can begin exploring the workspace. We recommend new users begin by following the
    'introduction.ipynb' notebook, which gives an overview of PyAutoCTI and the workspace.

    Examples are provided as both Jupyter notebooks in the 'notebooks' folder and Python scripts in the 'scripts'
    folder. It is up to you how you would prefer to use PyAutoCTI. With these folders, you can find the following
    packages:
        
    - overview: An overview of what CTI is and how to use PyAutoCTI to correct for it in imaging data. 
    
    - dataset_1d: Examples for analysing and simulating 1D CTI calibration data.
    
    - imaging_ci: Examples for analysing and simulating charge injection imaging CTI calibration data.
     
     - plot: An API reference guide of all of PyAutoCTI's plotting and visualization tools.
          
     - misc: Miscellaneous scripts for specific lens analysis.
    
    Once you a familiar with the PyAutoCTI API you should be ready to use it to compose and cti models for your
    model-fitting problem!
    """
)
