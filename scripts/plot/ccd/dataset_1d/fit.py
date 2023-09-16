"""
Plots CCD: Fit
==============

CTI calibration is typically performed on a CCD using many images (8-32 or more), where the images vary in the level of
charge injected into the CCD (the charge injection level).

Visualizing the results of a CTI calibration in a way that shows the results across all injection levels is
challenging, as there is a lot of information to convey.

The `autocti_workspace/*/plot/ccd` package provides tools for simulating CTI calibration data, fitting it in a realistic
calibration setting and plotting the results of the fit.

__Fit__

This script fits the simulated CTI calibration data simulated in the `plot/ccd/dataset_1d/simulator.py` script. It
outputs visuals which summarize the results of the fit concise in a single matplotlib figure, in particular:

 - An image of the datasets and fits to all 32 simulated datasets, where these are group in columns of the 4 different
   quadrants across the 8 different charge injection levels.

 - The same figure above but for the FPRs and EPERs only.

These images are both output to hard-disk as .png files during the model-fit and shown how to output via a
`Plotter` object at the end of the script.

__Database__

The visuals output in this script are created be rerunning the model-fit from the results on the hard-disk. This can
make replotting visuals and customizing the appearance of plots straight forward cumbersome and slow.

The script `plot/ccd/dataset_1d/database.py` shows how to load the results of the fit performed here via an .sqlite
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
from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Dataset__

Load the CTI dataset 'dataset_1d/simple' 'from .fits files, which is the dataset we will use to perform CTI modeling.

You should be familiar with how we load datasets in this way, if not checkout the `overview` 
and `modeling/start_here.py` examples.
"""
dataset_type = "dataset_1d"

shape_native = (21,)

prescan = ac.Region1D((0, 1))
overscan = ac.Region1D((20, 21))

region_list = [(1, 10)]

norm_list = [100, 500, 1000, 5000, 10000, 25000, 100000, 200000]

dataset_list = []
dataset_full_list = []

for norm in norm_list:
    for quadrant_id in range(4):
        dataset_name = f"data_quad_{quadrant_id}"
        dataset_path = path.join("dataset", dataset_type, "ccd_plot", dataset_name)

        layout = ac.Layout1D(
            shape_1d=shape_native,
            region_list=region_list,
            prescan=prescan,
            overscan=overscan,
        )

        dataset_quad = ac.Dataset1D.from_fits(
            data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
            noise_map_path=path.join(
                dataset_path, f"norm_{int(norm)}", "noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
            ),
            layout=layout,
            pixel_scales=0.1,
        )

        # dataset_full_list = copy.deepcopy(dataset_quad_list)

        dataset_full_quad = ac.Dataset1D.from_fits(
            data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
            noise_map_path=path.join(
                dataset_path, f"norm_{int(norm)}", "noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
            ),
            layout=layout,
            pixel_scales=0.1,
        )

        mask = ac.Mask1D.all_false(
            shape_slim=dataset_quad.shape_slim,
            pixel_scales=dataset_quad.pixel_scales,
        )

        mask = ac.Mask1D.masked_fpr_and_eper_from(
            mask=mask,
            layout=dataset_quad.layout,
            settings=ac.SettingsMask1D(fpr_pixels=(0, 9)),
            pixel_scales=dataset_quad.pixel_scales,
        )

        dataset_quad = dataset_quad.apply_mask(mask=mask)

        dataset_list += [dataset_quad]
        dataset_full_list += [dataset_full_quad]

clocker = ac.Clocker1D(express=5)

trap_0 = af.Model(ac.TrapInstantCapture)

trap_list = [trap_0]

ccd = af.Model(ac.CCDPhase)
ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

search = af.Nautilus(
    path_prefix=path.join("plot_ccd", "dataset_1d"), name="ccd_8x4", n_live=100
)

analysis_list = [
    ac.AnalysisDataset1D(dataset=dataset, clocker=clocker, dataset_full=dataset_1d_full)
    for dataset, dataset_1d_full in zip(dataset_list, dataset_full_list)
]

analysis = sum(analysis_list)

analysis.n_cores = 1

result_list = search.fit(model=model, analysis=analysis)

"""
__Plotting__

The model-fit above creates images of summarizing the fit over the 32 CCD / quadrant images in the `image` folder.
This includes fits of all 32 images, residuals and zoom ins on the FPR and EPERs regions.

The results above return a result_list, which consists of the model-fit to the 32 (8 charge injection regions x 4
quadrants) individual datasets. This can be used to separately reproduce these visuals.

The example below shows how we can create a plot of the EPER trails of all 32 datasets, using the `FitDataset1DPlotter`
and a `MultiFigurePlotter`.
"""
fit_list = [result.max_log_likelihood_fit for result in result_list]

mat_plot = aplt.MatPlot1D(
    output=aplt.Output(path=path.join("scripts", "plot", "images"), format="png")
)

fit_plotter_list = [
    aplt.FitDataset1DPlotter(
        fit=fit,
        mat_plot_1d=mat_plot,
    )
    for fit in fit_list
]
multi_plotter = aplt.MultiFigurePlotter(plotter_list=fit_plotter_list)

multi_plotter.subplot_of_figure(
    func_name="figures_1d", figure_name="data", region="eper"
)

"""
__Database__

Creating custom figures as above is somewhat cumbersome, as it requires us to rerun the model-fit, load the results,
and create the plotter and figure for each dataset.

The example `plots/dataset_1d/database.py` shows how we can use the database to load the results of the model-fit
and create the figure above in a single line of code.
"""
