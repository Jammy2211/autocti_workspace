"""
Plots: ImagingCIPlotter
=======================

This example illustrates how to plot a `ImagingCI` dataset using an `ImagingCIPlotter`.
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
Load the charge injection dataset 'imaging_ci/uniform/parallel_x2' 'from .fits files, which is the dataset we will
use to illustrate plotting charge injection line data.
"""
dataset_name = "parallel_x2"
dataset_path = path.join("dataset", "imaging_ci", "uniform", dataset_name)

"""
The locations of the serial prescan and serial overscan on the image, which is used to visualize the `ImagingCI` during the 
model-fit.
"""
layout_list = ac.Scans(
    parallel_overscan=ac.Region2D((1980, 2000, 5, 95)),
    serial_prescan=ac.Region2D((0, 2000, 0, 5)),
    serial_overscan=ac.Region2D((0, 1980, 95, 100)),
)

"""
Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks.
"""
regions_ci = [
    (0, 200, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
    (400, 600, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
    (800, 1000, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
    (1200, 1400, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
    (1600, 1800, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
]


"""
We require the normalization of every charge injection image, as the names of the files are tagged with the charge
injection normalization level.
"""
normalization_list = [100, 5000, 25000, 84700]

"""
Use the charge injection normalization_list and regions to create `PatternCIUniform` of every image we'll fit. The
`PatternCI` is used for visualizing the model-fit.
"""
pattern_cis = [
    ac.ci.PatternCIUniform(normalization=normalization, regions=regions_ci)
    for normalization in normalization_list
]

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `ImagingCI` object.
"""
imaging_ci_list = [
    ac.ci.ImagingCI.from_fits(
        image_path=path.join(dataset_path, f"image_{pattern.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{pattern.normalization}.fits"
        ),
        pre_cti_image_path=path.join(
            dataset_path, f"pre_cti_image_{pattern.normalization}.fits"
        ),
        pixel_scales=0.1,
        pattern_ci=pattern,
        roe_corner=(1, 0),
    )
    for pattern in pattern_cis
]


"""
We now pass the imaging to a `ImagingCIPlotter` and call various `figure_*` methods to plot different attributes.
"""
imaging_ci_plotter = aplt.ImagingCIPlotter(imaging=imaging_ci_list[0])
imaging_ci_plotter.figures_2d(
    image=True,
    noise_map=True,
    pre_cti_image=True,
    inverse_noise_map=True,
    potential_chi_squared_map=True,
    absolute_signal_to_noise_map=True,
)

"""
The `ImagingCIPlotter` may also plot a subplot of all of these attributes.
"""
imaging_ci_plotter.subplot_imaging_ci()

"""
We can also call `figures_1d_*` methods which create 1D plots of regions of the image binned over the parallel or
serial direction.

The regions available are:

 `parallel_front_edge`: The charge injection region binned up over all columns (e.g. across serial).
 `parallel_trails`: The parallel CTI trails behind the charge injection region binned up over all columns (e.g. 
  across serial).
 `serial_front_edge`: The charge injection region binned up over all rows (e.g. across parallel).
 `serial_trails`: The serial CTI trails behind the charge injection region binned up over all rows (e.g. across serial).
"""
imaging_ci_plotter.figures_1d_ci_line_region(
    line_region="parallel_front_edge", image=True, pre_cti_image=True
)
imaging_ci_plotter.figures_1d_ci_line_region(
    line_region="parallel_trails", image=True, pre_cti_image=True
)

"""
There is also a subplot of these 1D plots.
"""
imaging_ci_plotter.subplot_1d_ci_line_region(line_region="parallel_front_edge")

"""`
Imaging` contains the following attributes which can be plotted automatically via the `Include2D` object.

(By default, an `Array2D` does not contain a `Mask2D`, we therefore manually created an `Array2D` with a mask to 
illustrate the plotted of a mask and its border below).
"""
include_2d = aplt.Include2D(
    parallel_overscan=True, serial_prescan=True, serial_overscan=True
)
imaging_ci_plotter = aplt.ImagingCIPlotter(
    imaging=imaging_ci_list[0], include_2d=include_2d
)
imaging_ci_plotter.figures_2d(image=True)

"""
Our `ImagingCI` dataset consists of many images taken at different charge injection levels. We may wish to plot
all images on the same subplot, which can be performed using the method `subplot_of_figure`.
"""
imaging_ci_plotter_list = [
    aplt.ImagingCIPlotter(imaging=imaging_ci) for imaging_ci in imaging_ci_list
]
multi_plotter = aplt.MultiPlotter(plotter_list=imaging_ci_plotter_list)
multi_plotter.subplot_of_figure(func_name="figures", figure_name="image")
multi_plotter.subplot_of_figure(func_name="figures", figure_name="pre_cti_image")

"""
This method can also plot all of the 1D figures that we plotted above.
"""
multi_plotter.subplot_of_figure(
    func_name="figures_1d_ci_line_region",
    figure_name="image",
    line_region="parallel_front_edge",
)

"""
Finish.
"""
