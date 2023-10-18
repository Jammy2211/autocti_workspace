"""
Simulator: Start Here
=====================

This script is the starting point for simulating a 1D CTI dataset and it provides an overview of
the simulation API.

This script simulates the simplest 1D CTI dataset in the workspace, where the CTI model has just two trap species,
the volume filling behaviour is simple and the injected charge signal (e.g. the FPR) is uniform across the image.

After reading this script, the `examples` folder provide examples for simulating more complex CTI datasets in different
ways.

__Model__

This script simulates a 1D dataset with CTI, where:

 - CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the direction uses the `CCD` class.

 __Plotters__

To output images of the simulated data, `Plotter` objects are used, which are high-level wrappers of matplotlib
code which produce high quality visualization of strong lenses.

The `Plotter` API is described in the `autocti_workspace/*/plot/start_here.py` script.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
from os import path
import autocti as ac
import autocti.plot as aplt

"""
__Dataset Paths__

The 'dataset_name' describes the type of data being simulated (in this case, 1D CTI data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The data will be output to '/autocti_workspace/dataset/dataset_type/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_type/dataset_name/noise_map.fits'.
 - The pre-cti data will be output to '/autocti_workspace/dataset/dataset_type/dataset_name/pre_cti_data.fits'.
"""
dataset_type = "dataset_1d"
dataset_name = "simple"

"""
The path where the dataset will be output, which in this case is:

`/autocti_workspace/dataset/dataset_1d/simple`
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Shape__

The 1D shape of each 1D dataset, where the dataset we simulate is 200 pixels long.
"""
shape_native = (200,)

"""
__Regions__

Use `Region1D` objects to define the locations of the prescan and overscan on the 1D data. 

1D regions are defined as a tuple of the form (x0, x1) = (left-pixel, right-pixel), where the integer values of the
tuple are used to perform NumPy array indexing of the 1D data.

For example, if the overscan of 1D data is between pixels 40 and 50, its region is `region=(40, 50)`.

These define where the prescan and overscan are located when simulating the 1D data.

For this 1D dataset the prescan spans the first 10 pixels and overscan the last 10 pixels.
"""
prescan = ac.Region1D((0, 10))
overscan = ac.Region1D((190, 200))

"""
__FPR / EPER__

Specify the charge regions on the 1D CTI Dataset, corresponding to where an injected signal is present that has its 
electrons captured and trailed by CTI.

This is referred to as a the "First Pixel Response" (FPR), with the trails of electrons which appear after it 
referred to as the "Extended Pixel Edge Response" (EPER).

When simulating the 1D dataset, charge will be added to this region of the 1D data, which will then be trailed by CTI.

For the fiducial 1D dataset this region is 10 pixels after the prescan, meaning the EPER trails span pixels 20 -> 200.
"""
region_list = [(10, 20)]

"""
The dataset consists of multiple charge lines at different normalizations. 

Below, we specify the normalization of every 1D dataset, where the size of this list determines how many datasets 
are simulated in total.
"""
norm_list = [100, 5000, 25000, 200000]

"""
__Layout__

We now create a `Layout1D` object for every 1D dataset fitted in this script.

This object contains all functionality associated with the layout of the data (e.g. where the FPR is, where the
EPERs are, where the overscans are, etc.). 

The simulation procedure uses this object to create the initial electrons in the 1D dataset (e.g. the FPR), which are 
then trailed by CTI to form the EPERs.
"""
layout_list = [
    ac.Layout1D(
        shape_1d=shape_native,
        region_list=region_list,
        prescan=prescan,
        overscan=overscan,
    )
    for norm in norm_list
]

"""
__Clocker / arCTIc__

To model the CCD clocking process, including CTI, we use  arCTIc, or the "algorithm for Charge Transfer Inefficiency 
clocking".

arCTIc is written in c++ can be used standalone outside of **PyAutoCTI** as described on its GitHub 
page (https://github.com/jkeger/arctic). **PyAutoCTI** uses arCTIc's built-in Python wrapper.

In **PyAutoCTI** we call arCTIc via a `Clocker` object, which is a Python class that wraps arCTIc. This class has 
many optional inputs that customize how clocking is performed, but we'll omit these for now to keep things simple.

For this example, we only input the `express` parameter, which determines how many electrons are clocked per cycle
and trades off speed for accuracy. For this example we use `express=5`, which is a good balance.
"""
clocker = ac.Clocker1D(express=5)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the simulated data, which in this example contains: 

 - 2 `TrapInstantCapture` species, which captures electrons during clocking instantly and release them according to 
   an exponential probability distribution defined by a single release times.
 
 - A simple CCDPhase volume filling parametrization.
"""
trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)
trap_list = [trap_0, trap_1]

ccd = ac.CCDPhase(well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0)

cti = ac.CTI1D(trap_list=trap_list, ccd=ccd)

"""
__Simulate__

To simulate the 1D CTI dataset we first create a simulator, which includes:

 - The read noise of the data, which increasing means the data is lower signal-to-noise.
 
 - The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are using 
   is 0.1" / pixel. This is used for visualization only, specifically to convert axis labels from pixels to arc-seconds.
   
 - The normalizaiton, `norm` of the dataset, which is the total number of electrons in the dataset before CTI (e.g.
   the FPR).
"""
simulator_list = [
    ac.SimulatorDataset1D(read_noise=0.01, pixel_scales=0.1, norm=norm)
    for norm in norm_list
]

"""
We now pass each layout to each simulator, which: 

 - Creates each 1D dataset using its `norm` value.
 
 - Adds CTI to the data using the `cti` model.
 
 - Adds read noise to the data.
 
This creates a list of `Dataset1D` instances, which include the data (with CTI), noise-maps and the pre-cti data.
"""
dataset_list = [
    simulator.via_layout_from(clocker=clocker, layout=layout, cti=cti)
    for layout, simulator in zip(layout_list, simulator_list)
]

"""
We plot the first dataset in the list, which is the dataset with the lowest normalization.
"""
dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0])
dataset_plotter.subplot_dataset()

"""
__Output__

Output the simulated dataset to the dataset path as .fits files.

If you are unfamiliar with .fits files, this is the standard file format of astronomical data and you can open 
them using the software ds9 (https://sites.google.com/cfa.harvard.edu/saoimageds9/home).
"""
[
    dataset.output_to_fits(
        data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
        noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
        ),
        overwrite=True,
    )
    for dataset, norm in zip(dataset_list, norm_list)
]

"""
__Visualize__

In the same folder as the .fits files, we also output plots of the simulated dataset in .png format.

Having .png files like this is useful, as they can be opened quickly and easily by the user to check the dataset.
"""
for dataset, norm in zip(dataset_list, norm_list):
    output = aplt.Output(
        path=path.join(dataset_path, f"norm_{int(norm)}"),
        filename="dataset_1d",
        format="png",
    )

    mat_plot = aplt.MatPlot1D(output=output)

    dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset, mat_plot_1d=mat_plot)
    dataset_plotter.subplot_dataset()

"""
We also output subplots of the simulated dataset in .png format, as well as other images which summarize the dataset.

These plots include 1D binned up images of the FPR and EPER, so that electron capture and trailing can be seen clearly.
"""
for dataset, norm in zip(dataset_list, norm_list):
    output = aplt.Output(
        path=path.join(dataset_path, f"norm_{int(norm)}", "binned_1d"), format="png"
    )

    mat_plot = aplt.MatPlot1D(output=output)

    dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset, mat_plot_1d=mat_plot)
    dataset_plotter.figures_1d(region="fpr", data=True, data_logy=True)
    dataset_plotter.figures_1d(region="eper", data=True, data_logy=True)


"""
__CTI json__

Save the `Clocker1D` and `CTI1D` in the dataset folder as a .json file, ensuring the true traps and CCD settings 
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the methods `cti = ac.from_json()` and `clocker = ac.from_json()`.
"""
ac.output_to_json(
    obj=cti,
    file_path=path.join(dataset_path, "cti.json"),
)
ac.output_to_json(
    obj=clocker,
    file_path=path.join(dataset_path, "clocker.json"),
)

"""
__True Likelihood__

Fit the true model to the data and output the true `log_likelihood`, which can act as a verification of the quality of
model fits.
"""
fit_list = []

for dataset in dataset_list:
    post_cti_data = clocker.add_cti(data=dataset.pre_cti_data, cti=cti)

    fit_list.append(ac.FitDataset1D(dataset=dataset, post_cti_data=post_cti_data))

true_log_likelihood_list = [fit.log_likelihood for fit in fit_list]
true_log_likelihood = sum(true_log_likelihood_list)

true_log_likelihood_file = path.join(dataset_path, "true_log_likelihood.json")

with open(true_log_likelihood_file, "w+") as f:
    json.dump(true_log_likelihood_list, f, indent=4)
    json.dump(true_log_likelihood, f, indent=4)

"""
Finish.
"""
