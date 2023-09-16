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

This script fits the simulated CTI calibration data simulated in the `plot/fpa/dataset_1d/simulator.py` script. It
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
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)

dataset_type = "dataset_1d"

shape_native = (21,)

prescan = ac.Region1D((0, 1))
overscan = ac.Region1D((20, 21))

region_list = [(1, 10)]

norm_list = [1000]
norm = 1000

clocker = ac.Clocker1D(express=5)

trap_0 = af.Model(ac.TrapInstantCapture)

trap_list = [trap_0]

ccd = af.Model(ac.CCDPhase)
ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

for fpa_i in range(6):
    for fpa_j in range(6):
        dataset_list = []
        dataset_full_list = []

        dataset_name = f"data_fpa_{fpa_i}_{fpa_j}"
        dataset_path = path.join("dataset", dataset_type, "fpa_plot", dataset_name)

        layout = ac.Layout1D(
            shape_1d=shape_native,
            region_list=region_list,
            prescan=prescan,
            overscan=overscan,
        )

        dataset_1d_ccd = ac.Dataset1D.from_fits(
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

        # dataset_full_list = copy.deepcopy(dataset_1d_quad_list)

        dataset_1d_full_ccd = ac.Dataset1D.from_fits(
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
            shape_slim=dataset_1d_ccd.shape_slim,
            pixel_scales=dataset_1d_ccd.pixel_scales,
        )

        mask = ac.Mask1D.masked_fpr_and_eper_from(
            mask=mask,
            layout=dataset_1d_ccd.layout,
            settings=ac.SettingsMask1D(fpr_pixels=(0, 9)),
            pixel_scales=dataset_1d_ccd.pixel_scales,
        )

        dataset_1d_ccd = dataset_1d_ccd.apply_mask(mask=mask)

        dataset_list += [dataset_1d_ccd]
        dataset_full_list += [dataset_1d_full_ccd]

        search = af.Nautilus(
            path_prefix=path.join("plot_fpa", "dataset_1d"),
            name=f"fpa_{fpa_i}_{fpa_j}",
            unique_tag=f"fpa_{fpa_i}_{fpa_j}",
            n_live=100,
        )

        analysis_list = [
            ac.AnalysisDataset1D(
                dataset=dataset, clocker=clocker, dataset_full=dataset_1d_full
            )
            for dataset, dataset_1d_full in zip(dataset_list, dataset_full_list)
        ]

        analysis = sum(analysis_list)

        analysis.n_cores = 1

        result_list = search.fit(model=model, analysis=analysis)

"""
__Database__

Creating custom figures as above is somewhat cumbersome, as it requires us to rerun the model-fit, load the results,
and create the plotter and figure for each dataset.

The example `plots/dataset_1d/database.py` shows how we can use the database to load the results of the model-fit
and create the figure above in a single line of code.
"""
