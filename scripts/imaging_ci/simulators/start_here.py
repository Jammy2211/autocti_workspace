"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script is the starting point for simulating a 2D charge injection CTI dataset and it provides 
an overview of the simulation API.

This script simulates the simplest 2D charge injection CTI dataset in the workspace, where the CTI model has just two
trap species, the volume filling behaviour is simple and the injected charge signal (e.g. the FPR) is uniform across
the image.

After reading this script, the `examples` folder provide examples for simulating more complex CTI datasets in different
ways.

__Model__

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallel direction using the `CCD` class.
 
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

from os import path
import autocti as ac
import autocti.plot as aplt

"""
__Dataset Paths__

The 'dataset_name' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_name/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_name/dataset_name/noise_map.fits'.
 - The pre_cti_data will be output to '/autocti_workspace/dataset/dataset_name/dataset_name/pre_cti_data.fits'.
"""
dataset_type = "imaging_ci"
dataset_name = "simple"

"""
Returns the path where the dataset will be output, which in this case is:

'/autocti_workspace/dataset/imaging_ci/simple'
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Shape__

The 2D shape of each charge injection image and other quantities in the dataset. 

The dataset we simulate has 2000 rows (over which parallel CTI trailling occurs) and is 100 columns across.
"""
shape_native = (2000, 100)

"""
__Regions__

We next define the locations of the prescan and overscan on the 2D data. 

2D regions are defined as a tuple of the form (y0, y1, x0, x1) = (top-row, bottom-row, left-column, right-column), 
where the integer values of the tuple are used to perform NumPy array indexing of the 2D data.

For example, if the serial overscan of 2D data is 100 columns from the read-out electronics and spans a total of
150 rows, its region is `region=(0, 150, 0, 100)`.

These are used to visualize these regions of the 2D CTI dataset during the model-fit and customize aspects of the 
model-fit.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

"""
Specify the charge regions on the 2D CTI Dataset, corresponding to where a signal is contained that has its electrons 
captured and trailed by CTI (e.g. the FPR).

This dataset has five charge regions, which are spaced in on / off blocks of 200 pixels.

Note that the charge injections do not extend to inside of the serial prescan or serial overscan regions.
"""
region_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

"""
Specify the normalization of the charge in every individual 2D CTI charge injection dataset. 

When simulated, the normalization of the charge in every charge injection image is set by these values, meaning this
is the total number of electrons in the FPR of every charge injection image.
"""
norm_list = [100, 5000, 25000, 200000]

"""
The total number of charge injection images that are simulated, which is simply the number of normalizations 
specified above.
"""
total_datasets = len(norm_list)

"""
__Layout__

We now create a `Layout2D` object for every 2D charge injection dataset fitted in this script.

This object contains all functionality associated with the layout of the data (e.g. where the FPR is, where the
EPERs are, where the overscans are, etc.). 

The simulation procedure uses this object to create the initial electrons in the 2D dataset (e.g. the FPR), which are 
then trailed by CTI to form the EPERs.
"""
layout_list = [
    ac.Layout2DCI(
        shape_2d=shape_native,
        region_list=region_list,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )
    for i in range(total_datasets)
]

"""
__Clocker / arCTIc__

To model the CCD clocking process, including CTI, we use  arCTIc, or the "algorithm for Charge Transfer Inefficiency 
clocking".

arCTIc is written in c++ can be used standalone outside of **PyAutoCTI** as described on its GitHub 
page (https://github.com/jkeger/arctic). **PyAutoCTI** uses arCTIc's built-in Python wrapper.

In **PyAutoCTI** we call arCTIc via a `Clocker` object, which is a Python class that wraps arCTIc. This class has 
many optional inputs that customize how clocking is performed, but we'll omit these for now to keep things simple.

For clocking, we use: 

 - `parallel_express`: determines how many electrons are clocked per cycle and trades off speed for accuracy, where 
   `parallel_express=5` is a good balance.

 - 'ROEChargeInjection': which transfers the charge of every pixel over the full CCD.
 
 - `parallel_fast_mode`: which speeds up the analysis by only passing to arCTIc unique columns (for uniform charge
 injection data all columsn are identical, thus only one arCTIc call is required).
"""
clocker = ac.Clocker2D(
    parallel_express=5, parallel_roe=ac.ROEChargeInjection(), parallel_fast_mode=True
)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 2 `TrapInstantCapture` species in the parallel direction, which captures electrons during clocking instantly and 
   releases them according to an exponential probability distribution defined by a single release times.
 
 - A simple CCDPhase volume filling parametrization.
"""
parallel_trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
parallel_trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)

parallel_trap_list = [parallel_trap_0, parallel_trap_1]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)

cti = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

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
    ac.SimulatorImagingCI(read_noise=4.0, pixel_scales=0.1, norm=norm)
    for norm in norm_list
]

"""
We now pass each charge injection pattern to the simulator, which: 

 - Generates the charge injection image of each exposure using its input `norm`.

 - Adds CTI to the data using the `cti` model.
 
 - Adds read noise to the data.

This creates a list of `ImagingCI` instances, which include the data (with CTI), noise-maps and the pre-cti data.
"""
dataset_list = [
    simulator.via_layout_from(clocker=clocker, layout=layout, cti=cti)
    for layout, simulator in zip(layout_list, simulator_list)
]

"""
We plot the first dataset in the list, which is the dataset with the lowest normalization.
"""
dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset_list[0])
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
        filename="imaging_ci",
        format="png",
    )

    mat_plot = aplt.MatPlot2D(output=output)

    dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset, mat_plot_2d=mat_plot)
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

    dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset, mat_plot_1d=mat_plot)
    dataset_plotter.figures_1d(region="parallel_fpr", data=True, data_logy=True)
    dataset_plotter.figures_1d(region="parallel_eper", data=True, data_logy=True)


"""
__CTI json__

Save the `Clocker2D` and `CTI2D` in the dataset folder as a .json file, ensuring the true traps and CCD settings are 
safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `CTI2D.from_json`.
"""
cti.output_to_json(file_path=path.join(dataset_path, "cti.json"))
clocker.output_to_json(file_path=path.join(dataset_path, "clocker.json"))

"""
Finished.
"""
