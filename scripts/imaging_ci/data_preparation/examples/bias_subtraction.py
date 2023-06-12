"""
Data Preparation: Bias Subtraction
----------------------------------

Throughout the `autocti_workspace/imaging_ci` package, the charge injection imaging is assumed to already be
bias subtracted, which may not be the case for real data.

This script demonstrates **PyAutoCTI** methods which estimate the bias using each row of the serial prescan and use
this to subtract it from the data, therefore performing bias correction.

This uses the fact that in charge injection data the serial prescan pixels should not have signal from any other source
(e.g. CTI cannot trail electrons into this region). There may be cosmic rays, but these can be flagged and masked.

By taking the median of each row of the serial prescan one can therefore estimate the bias level in that row, which is
subtracted from the fata row-by-row.

This script first demonstrates a simple example, where there are no cosmic rays in the charge injection imaging. All
functionality can use masking and cosmic rays, as illustrated in other proprocessing scripts.
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
__Dataset + Layout__

We begin by loading a charge injection image which has not been bias corrected, where the bias creates a constant
value of 2000e- in every pixel. Every pixel in the data therefore has a signal of 2000e- or above.

You should be familiar with the **PyAutoCTI** API below, if not check out other scripts throughout the workspace.
"""
dataset_name = "bias_uncorrected"
dataset_path = path.join("dataset", "imaging_ci", dataset_name)

shape_native = (2000, 100)

"""
The overscans input here are not used to perform pre-CTI data estimation.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

"""
These charge injection regions are used to estimate the charge injection normalization in each column.
"""
region_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

norm = 5000

"""
The layout object contains all the charge injection normalization functionality used in this example.
"""
layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=region_list,
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

"""
We will demonstrate injection estimation on a charge injection image which we load via a .fits file below.
"""
dataset = ac.ImagingCI.from_fits(
    data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
    noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
    pre_cti_data_path=path.join(dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"),
    layout=layout,
    pixel_scales=0.1,
)

"""
A plot of the data shows it has a minimum signal of 2000e- in all pixels, including those with no charge injection.
"""
array_2d_plotter = aplt.Array2DPlotter(array=dataset.data)
array_2d_plotter.figure_2d()

"""
__Bias Level List__

Every row of the serial prescan should have no signal from any other source (e.g. no CTI or charge injection).

The median of the serial prescan therefore provides an estimate of the bias, which if subtracted from the data
provides the bias corrected data.

For more realistic CCD data, it is common for the bias to vary row-to-row. Ttaking the median of the prescan row-by-row 
is therefore a more accurate bias estimate, with each value subtracted from the data.

The 2D region of the serial prescan is contained in the layout's extract object.

To estimate the normalization of prescan row, we use the `median_list_from` of the `Layout2DCI` object with the 
input `pixels=(0, 5)`, which:

 - Extracts rows of the serial prescan of the charge injection data between the 0th and 4th pixels (noting that
 the `serial_prescan` above which defines where the serial prescan is spans 5 pixels in total).

 - Takes the median of each row.

The normalizations are returned as a list:
"""
bias_estimate_list = layout.extract.serial_prescan.median_list_from(
    array=dataset.data, settings=ac.SettingsExtract(pixels=(0, 5))
)

print(bias_estimate_list)

"""
We now subtract every bias value estimated row-by-row from the charge injection image.
"""
for row_index in range(dataset.data.shape_native[0]):
    dataset.data.native[row_index, :] -= bias_estimate_list[row_index]
