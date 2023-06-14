"""
Plots FPA: Fit
==============

CTI calibration is typically performed independently on many CCD's over a telescope's focal plane array (FPA).
For example, for Euclid, the FPA is a 6x6 grid of CCDs.

Visualizing the results of a CTI calibration in a way that shows the results across all CCDs in the FPA is
challenging, as there is a lot of information to convey.

Fits to each CCD are also performed independently, meaning that the model-fit of a given CCD does not have any
information on the CTI best-fit of its neighboring CCDs, meaning that visualizing the results of a CTI calibration
across the FPA is not a trivial task.

The `autocti_workspace/*/plot/fpa` package provides tools for simulating an FPA of CTI calibration data, fitting it in
a realistic calibration setting and plotting the results of the fit on a single figure showing the whole FPA via
the database.

__Fit__

This script fits the simulated CTI calibration data simulated in the `plot/fpa/imaging_ci/simulator.py` script. It
outputs visuals which summarize the results of the fit over the full FPA concisely in a single matplotlib figure,
in particular:

 - An image of the datasets and fits to all 6x6 CCDs, so that CTI in each CCD compared to one another can be inspected.

 - The same figure above but for the FPRs and EPERs only.

These images are not output to hard-disk as .png files, as each CCD of the FPA is fitted independently and thus
the information to do this is not available until the end of the script.

These figures are therefore instead produced in the `database.py` example script, which loads the results of the fit
from the hard-disk and uses the database to produce the figures.

__Database__

The visuals output in this script are created be rerunning the model-fit from the results on the hard-disk. This can
make replotting visuals and customizing the appearance of plots straight forward cumbersome and slow.

The script `plot/ccd/imaging_ci/database.py` shows how to load the results of the fit performed here via an .sqlite
database, which is a convenient and efficient way to produce these visuals.

__Model__

This script fits a 1D dataset with CTI, where:

 - CTI is added to the image using a 1 `Trap` species model.
 - The volume filling behaviour in the direction uses the `CCD` class.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import copy
import os

import numpy as np
from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Dataset__

Load the CTI dataset 'imaging_ci/simple' 'from .fits files, which is the dataset we will use to perform CTI modeling.
"""
dataset_type = "imaging_ci"

shape_native = (100, 100)

parallel_overscan = ac.Region2D((95, 100, 5, 95))
serial_prescan = ac.Region2D((0, 100, 0, 5))
serial_overscan = ac.Region2D((0, 95, 95, 100))

region_list = [
    (5, 25, serial_prescan[3], serial_overscan[2]),
]

norm_list = [1000, 10000]

settings_dict = {
    "CCD": f"1-1.E",
    "IJON": 214,
    "IJOFF": 200,
    "IDDLY": np.round(0.001536, 5),
    "IG1": np.round(4.2500439453125, 3),
    "IG2": np.round(6.0028662109375, 3),
}

workspace_path = os.getcwd()

output = aplt.Output(path=workspace_path, filename="fpa_quad_grid", format="png")

multi_plotter_list = []

fpa_i_range = 6
fpa_j_range = 6

fpa_quad_array_list = []

for norm in norm_list:
    fpa_quad_array = np.zeros(shape=(fpa_i_range * 2, fpa_j_range * 2))

    for fpa_i in range(fpa_i_range):
        for fpa_j in range(fpa_j_range):
            plotter_list = []

            for quad_k in range(4):
                fpa_y = (2 * fpa_i) + quad_k % 2
                fpa_x = (2 * fpa_j) + quad_k % 2

                dataset_name = f"data_fpa_{fpa_i}_{fpa_j}_quad_{quad_k}"
                dataset_path = path.join(
                    "dataset", dataset_type, "diagnostic_plot", dataset_name
                )

                layout = ac.Layout2DCI(
                    shape_2d=shape_native,
                    region_list=region_list,
                    parallel_overscan=parallel_overscan,
                    serial_prescan=serial_prescan,
                    serial_overscan=serial_overscan,
                )

                dataset_quad = ac.ImagingCI.from_fits(
                    data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
                    noise_map_path=path.join(
                        dataset_path, f"norm_{int(norm)}", "noise_map.fits"
                    ),
                    pre_cti_data_path=path.join(
                        dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
                    ),
                    layout=layout,
                    pixel_scales=0.1,
                    settings_dict=settings_dict,
                )

                eper = np.sum(
                    layout.extract.parallel_eper.binned_array_1d_from(
                        array=dataset_quad.data,
                        settings=ac.SettingsExtract(pixels=(0, 1)),
                    )
                )

                fpa_quad_array[fpa_y, fpa_x] = eper

    fpa_quad_array = ac.Array2D.no_mask(values=fpa_quad_array, pixel_scales=1.0)

    fpa_quad_array_list.append(fpa_quad_array)

mat_plot = aplt.MatPlot2D(output=output)

array_plotter_list = [
    aplt.Array2DPlotter(array=fpa_quad_array, mat_plot_2d=mat_plot)
    for fpa_quad_array in fpa_quad_array_list
]

multi_plotter = aplt.MultiFigurePlotter(
    plotter_list=array_plotter_list, subplot_shape=(2, 2)
)

multi_plotter.subplot_of_figure(func_name="figure_2d", figure_name=None)
