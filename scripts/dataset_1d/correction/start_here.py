"""
Correction: Start Here
======================

In this script, we correct CTI from a 1D CTI calibration dataset using a known CTI model.

Whilst correcting CTI calibration data is not something one would commonly do, this script is here to illustrate
the API for performing CTI correction.

The correction of CTI calibration data can also be used as a diagnostic for the quality of the CTI model that is
calibrated.
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
__Dataset__

Load the CTI dataset 'dataset_1d/simple' 'from .fits files, which is the dataset we will use to perform CTI modeling.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)

"""
__Shape__

The 1D shape of each 1D dataset.
"""
shape_native = (200,)

"""
__Regions__

The locations of the prescan and overscan on the 1D data, which is used to visualize the 1D CTI dataset during the 
model-fit and customize aspects of the model-fit.
"""
prescan = ac.Region1D((0, 10))
overscan = ac.Region1D((190, 200))

"""
Specify the charge regions on the 1D CTI Dataset, corresponding to where a signal is contained that has its electrons 
captured and trailed by CTI.
"""
region_list = [(10, 20)]

"""
__Normalizations__

We require the normalization of the charge in every CTI dataset, as the names of the files are tagged with this.
"""
norm_list = [100, 5000, 25000, 200000]

"""
__Layout__

We use the regions and norm_list above to create the `Layout1D` of every 1D CTI dataset we fit. This is used 
for visualizing the model-fit.
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
__Dataset__

We now load every cti-dataset, including a noise-map and pre-cti data containing the data before read-out and
therefore without CTI.
"""
dataset_list = [
    ac.Dataset1D.from_fits(
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
__Clocker__

The `Clocker1D` object models the read-out process of every 1D dataset as if it were clocked out on a real CCD. This 
includes the addition of CTI. 
"""
clocker = ac.Clocker1D(express=5)

"""
__CTI Model__

We now compose the CTI model we will use to correct CTI from the data.

In this example, the true CTI model used to simulate the data is specified below. The `results` and `database` 
packages have tutorials showing how to directly use the results of a CTI calibration.
"""
trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)
trap_list = [trap_0, trap_1]

ccd = ac.CCDPhase(well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0)

cti = ac.CTI1D(trap_list=trap_list, ccd=ccd)

"""
__Correction__

We use the CTI model and clocker to perform the CTI correction.
"""
data_corrected_1d_list = [
    clocker.remove_cti(data=dataset.data, cti=cti) for dataset in dataset_list
]

"""
__Output__

Output the corrected image to the dataset path as a .png file.
"""
for data_corrected_1d, norm in zip(data_corrected_1d_list, norm_list):
    mat_plot = aplt.MatPlot1D(
        output=aplt.Output(
            path=path.join(dataset_path, "correction", f"norm_{int(norm)}"),
            filename=f"data_corrected",
            format="png",
        )
    )

    array_1d_plotter = aplt.Array1DPlotter(y=data_corrected_1d, mat_plot_1d=mat_plot)
    array_1d_plotter.figure_1d()

"""
Output the image, noise-map and pre CTI image of the dataset to .fits files.
"""
[
    data_corrected_1d.output_to_fits(
        file_path=path.join(
            dataset_path, "correction", f"norm_{int(norm)}", f"data_corrected.fits"
        ),
        overwrite=True,
    )
    for data_corrected_1d, norm in zip(data_corrected_1d_list, norm_list)
]

"""
__CTI json__

Save the `Clocker2D` and `CTI2D` in the dataset folder as a .json file, ensuring the traps and CCD settings used to
perform the correction are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `CTI2D.from_json`.
"""
cti.output_to_json(file_path=path.join(dataset_path, "cti_correction.json"))
clocker.output_to_json(file_path=path.join(dataset_path, "clocker_correction.json"))

"""
Finished.
"""
