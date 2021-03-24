"""
Plots: CIImagingPlotter
=======================

This example illustrates how to plot a `CIImaging` dataset using an `CIImagingPlotter`.
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
Load the charge injection dataset 'ci_imaging/uniform/parallel_x2' 'from .fits files, which is the dataset we will
use to illustrate plotting charge injection line data.
"""
dataset_name = "parallel_x2"
dataset_path = path.join("dataset", "ci_imaging", "uniform", dataset_name)

"""
The locations of the prescans and overscans on the image, which is used to visualize the `CIImaging` during the 
model-fit.
"""
scans = ac.Scans(
    parallel_overscan=ac.Region2D((1980, 2000, 5, 95)),
    serial_prescan=ac.Region2D((0, 2000, 0, 5)),
    serial_overscan=ac.Region2D((0, 1980, 95, 100)),
)

"""
Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks.
"""
ci_regions = [
    (0, 200, scans.serial_prescan[3], scans.serial_overscan[2]),
    (400, 600, scans.serial_prescan[3], scans.serial_overscan[2]),
    (800, 1000, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1200, 1400, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1600, 1800, scans.serial_prescan[3], scans.serial_overscan[2]),
]


"""
We require the normalization of every charge injection image, as the names of the files are tagged with the charge
injection normalization level.
"""
normalizations = [100, 5000, 25000, 84700]

"""
Use the charge injection normalizations and regions to create `CIPatternUniform` of every image we'll fit. The
`CIPattern` is used for visualizing the model-fit.
"""
ci_patterns = [
    ac.ci.CIPatternUniform(normalization=normalization, regions=ci_regions)
    for normalization in normalizations
]

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `CIImaging` object.
"""
ci_imaging_list = [
    ac.ci.CIImaging.from_fits(
        image_path=path.join(dataset_path, f"image_{pattern.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{pattern.normalization}.fits"
        ),
        ci_pre_cti_path=path.join(
            dataset_path, f"ci_pre_cti_{pattern.normalization}.fits"
        ),
        pixel_scales=0.1,
        ci_pattern=pattern,
        roe_corner=(1, 0),
    )
    for pattern in ci_patterns
]


"""
We now pass the imaging to a `CIImagingPlotter` and call various `figure_*` methods to plot different attributes.
"""
ci_imaging_plotter = aplt.CIImagingPlotter(imaging=ci_imaging_list[0])
ci_imaging_plotter.figures(
    image=True,
    noise_map=True,
    ci_pre_cti=True,
    inverse_noise_map=True,
    potential_chi_squared_map=True,
    absolute_signal_to_noise_map=True,
)

"""
The `CIImagingPlotter` may also plot a subplot of all of these attributes.
"""
ci_imaging_plotter.subplot_ci_imaging()

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
ci_imaging_plotter.figures_1d_ci_line_region(
    line_region="parallel_front_edge", image=True, ci_pre_cti=True
)
ci_imaging_plotter.figures_1d_ci_line_region(
    line_region="parallel_trails", image=True, ci_pre_cti=True
)

"""
There is also a subplot of these 1D plots.
"""
ci_imaging_plotter.subplot_1d_ci_line_region(line_region="parallel_front_edge")

"""`
Imaging` contains the following attributes which can be plotted automatically via the `Include2D` object.

(By default, an `Array2D` does not contain a `Mask2D`, we therefore manually created an `Array2D` with a mask to 
illustrate the plotted of a mask and its border below).
"""
include_2d = aplt.Include2D(
    parallel_overscan=True, serial_prescan=True, serial_overscan=True
)
ci_imaging_plotter = aplt.CIImagingPlotter(
    imaging=ci_imaging_list[0], include_2d=include_2d
)
ci_imaging_plotter.figures(image=True)

"""
Our `CIImaging` dataset consists of many images taken at different charge injection levels. We may wish to plot
all images on the same subplot, which can be performed using the method `subplot_of_figure`.
"""
ci_imaging_plotter_list = [
    aplt.CIImagingPlotter(imaging=ci_imaging) for ci_imaging in ci_imaging_list
]
multi_plotter = aplt.MultiPlotter(plotter_list=ci_imaging_plotter_list)
multi_plotter.subplot_of_figure(func_name="figures", figure_name="image")
multi_plotter.subplot_of_figure(func_name="figures", figure_name="ci_pre_cti")

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
