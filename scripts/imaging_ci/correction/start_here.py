"""
Correction: Start Here
======================

In this script, we correct CTI from charge injection imaging using a known CTI model.

Whilst correcting CTI calibration data is not something one would commonly do, this script is here to illustrate
the API for performing CTI correction, which can easily be applied to science data.

The correction of CTI calibration data can also be used as a diagnostic for the quality of the CTI model that is
calibrated.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Dataset__

The paths pointing to the dataset we will use for CTI modeling.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging_ci", dataset_name)

"""
__Layout__

The 2D shape of the images.
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

This is not used internally by **PyAutoCTI**, and only required for loading the dataset because the dataset file
names use the normalizations.
"""
norm_list = [100, 5000, 25000, 200000]

"""
The total number of charge injection images that are fitted.
"""
total_datasets = len(norm_list)

"""
__Layout__

We now create a `Layout2D` object for every 1D dataset fitted in this script.

This object contains all functionality associated with the layout of the data (e.g. where the FPR is, where the
EPERs are, where the overscans are, etc.). 

This is used for performing tasks like extracting a small region of the data for visualization.
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
__Dataset__

We now use a `ImagingCI` object to load every 2D CTI charge injection dataset, including a noise-map and pre-cti data 
containing the data before read-out and therefore without CTI. 

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are using 
is 0.1" / pixel.
"""
dataset_list = [
    ac.ImagingCI.from_fits(
        data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
        noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
        ),
        layout=layout,
        pixel_scales=0.1,
    )
    for layout, norm in zip(layout_list, norm_list)
]

"""
Use a `ImagingCIPlotter` to the plot the data, including: 

 - `data`: The 1D CTI data.
 - `noise_map`: The noise-map of the data, which quantifies the noise in every pixel as their RMS values.
 - `pre_cti_data`: The data before CTI, which has CTI added to it for every CTI model, which is compared to the data. 
 - `signal_to_noise_map`: Quantifies the signal-to-noise in every pixel.
"""
dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset_list[0])
dataset_plotter.subplot_dataset()

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
__Model__

We now compose the CTI model we will use to correct CTI from the data.

In this example, the true CTI model used to simulate the data is specified below. The `results` and `database` 
packages have tutorials showing how to directly use the results of a CTI calibration.
"""
parallel_trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
parallel_trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)

parallel_trap_list = [parallel_trap_0, parallel_trap_1]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)

cti = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

"""
__Correction__

We use the CTI model and clocker to perform the CTI correction, by calling the function `remove_cti` which is 
a wrapper to arCTIc.
"""
data_corrected_list = [
    clocker.remove_cti(data=dataset.data, cti=cti) for dataset in dataset_list
]

"""
__Output__

Output the corrected image to the dataset path as a .png file.
"""
for data_corrected, norm in zip(data_corrected_list, norm_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(
            path=path.join(dataset_path, f"norm_{int(norm)}", "correction"),
            filename=f"data_corrected",
            format="png",
        )
    )

    array_2d_plotter = aplt.Array2DPlotter(array=data_corrected, mat_plot_2d=mat_plot)
    array_2d_plotter.figure_2d()

"""
This is a hack so we can use an `ImagingCIPlotter` to plot the corrected binned regions.
"""
for dataset, data_corrected in zip(dataset_list, data_corrected_list):
    dataset.data = data_corrected


"""
Output plots of the corrected EPER and FPR's binned up in 1D, so that correction due to electron capture and trailing 
can be seen clearly.
"""
for dataset, norm in zip(dataset_list, norm_list):
    output = aplt.Output(
        path=path.join(dataset_path, f"norm_{int(norm)}", "correction", "binned_1d"),
        format="png",
    )

    mat_plot = aplt.MatPlot1D(output=output)

    dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset, mat_plot_1d=mat_plot)
    dataset_plotter.figures_1d(region="parallel_fpr", data=True)
    dataset_plotter.figures_1d(region="parallel_eper", data=True)

"""
Output the simulated dataset to the dataset path as .fits files.

If you are unfamiliar with .fits files, this is the standard file format of astronomical data and you can open 
them using the software ds9 (https://sites.google.com/cfa.harvard.edu/saoimageds9/home).
"""
[
    data_corrected.output_to_fits(
        file_path=path.join(dataset_path, f"norm_{int(norm)}", "data_corrected.fits"),
        overwrite=True,
    )
    for data_corrected, norm in zip(data_corrected_list, norm_list)
]

"""
__CTI json__

Save the `Clocker2D` and `CTI2D` in the dataset folder as a .json file, ensuring the traps and CCD settings used to
perform the correction are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `CTI2D.from_json`.
"""
ac.output_to_json(
    obj=cti,
    file_path=path.join(dataset_path, "cti_correction.json"),
)
ac.output_to_json(
    obj=clocker,
    file_path=path.join(dataset_path, "clocker_correction.json"),
)

"""
Finished.
"""
