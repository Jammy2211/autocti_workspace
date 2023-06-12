"""
Temporal: Individual Fits
=========================

This script fits multiple 1D CTI calibration datasets, representative of data taken over the course of a space
mission where radiation damage increases therefore also increasing the level of CTI.

The model-fitting aims to determine the increase in the density of traps with time. It fits each dataset one-by-one
and uses the results post-analysis to determine the density evolution parameters by interpolating the results as a
function of time.

__Database__

The results and interpolated CTI models computed in this script are created be rerunning the model-fit from the
results on the hard-disk. This can make recomputing results cumbersome and slow.

The script `advanced/temporal/database.py` shows how to load the results of the fit performed here via an .sqlite
database, which is a convenient and efficient way to quickly analyse the temporal evolution of CTI.

__Model__

In this script, we will fit multiple charge injection imaging to calibrate CTI, where:

 - The CTI model consists of one `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autofit as af
import autocti as ac

"""
__Dataset__

Load the CTI dataset 'dataset_1d/temporal' 'from .fits files, which is the dataset we will use to perform CTI 
modeling.
"""
dataset_type = "dataset_1d"
dataset_label = "temporal"
dataset_path = path.join("dataset", dataset_type, dataset_label)

"""
__Layout__

The 1D shape of each data.
"""
shape_native = (200,)

"""
The locations (using NumPy array indexes) of the prescan and overscan on the data.

For the fiducial 1D dataset the prescan spans the first 10 pixels and overscan the last 10 pixels.
"""
prescan = ac.Region1D((0, 10))
overscan = ac.Region1D((190, 200))

"""
Specify the regions of the dataset where charge was present before CTI, called the First Pixel Response (FPR). 

For the fiducial 1D dataset this is 10 pixels after the prescan.
"""
region_list = [(10, 20)]

"""
The normalization of the charge region (e.g. the FPR) of every dataset.
"""
norm_list = [100, 5000, 25000, 200000]

"""
The total number of charge injection datas that are simulated.
"""
total_datasets = len(norm_list)

"""
Create the layout of the charge injection pattern for every charge injection normalization.
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
__Clocker__

The `Clocker1D` object models the read-out process of every 1D dataset as if it were clocked out on a real CCD. This 
includes the addition of CTI. 
"""
clocker = ac.Clocker1D(express=5)


"""
We now load every data, noise-map and pre-CTI data as instances of the `Dataset1D` object.

We load and fit each dataset, accquried at different times, one-by-one. We do this in a for loop to avoid loading 
everything into memory.
"""
ml_instances_list = []

time_list = range(0, 5)

for time in time_list:
    dataset_time = f"time_{time}"
    dataset_time_path = path.join(dataset_path, dataset_time)

    dataset_list = [
        ac.Dataset1D.from_fits(
            data_path=path.join(dataset_time_path, f"norm_{int(norm)}", "data.fits"),
            noise_map_path=path.join(
                dataset_time_path, f"norm_{int(norm)}", "noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_time_path, f"norm_{int(norm)}", "pre_cti_data.fits"
            ),
            layout=layout,
            pixel_scales=0.1,
        )
        for layout, norm in zip(layout_list, norm_list)
    ]

    """
    __Mask__
    
    We apply a 1D mask which removes the FPR (e.g. all 5 pixels where the charge injection is performed).
    """
    mask = ac.Mask1D.all_false(
        shape_slim=dataset_list[0].shape_slim,
        pixel_scales=dataset_list[0].pixel_scales,
    )

    mask = ac.Mask1D.masked_fpr_and_eper_from(
        mask=mask,
        layout=dataset_list[0].layout,
        settings=ac.SettingsMask1D(fpr_pixels=(0, 10)),
        pixel_scales=dataset_list[0].pixel_scales,
    )

    dataset_list = [dataset.apply_mask(mask=mask) for dataset in dataset_list]

    """
    __Clocking__
    
    The `Clocker` models the CCD read-out, including CTI. 
    """
    clocker = ac.Clocker1D(express=5, roe=ac.ROEChargeInjection())

    """
    __Time__
    
    The CTI model composed below has an input not seen in other scripts, `time`.
    
    This is the time that the CTI calibration data was acquired, and is not a free parameter in the fit. 
    
    For interpolation it plays a crucial role, as the CTI model is interpolated to the time of every dataset as input
    into the model below. If the `time` input were missing, interpolation could not be performed.

    __Model__
    
    We now compose our CTI model, which represents the trap species and CCD volume filling behaviour used to fit the 
    charge  injection data. In this example we fit a CTI model with:
    
     - One `TrapInstantCapture`'s [2 parameters].
    
     - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].
    
    The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.
    """
    trap_0 = af.Model(ac.TrapInstantCapture)
    traps = [trap_0]
    ccd = af.Model(ac.CCDPhase)
    ccd.well_notch_depth = 0.0
    ccd.full_well_depth = 200000.0

    model = af.Collection(
        cti=af.Model(
            ac.CTI1D,
            trap_list=[trap_0],
            ccd=ccd,
        ),
        time=time,
    )

    """
    __Search__
    
    The model is fitted to the data using the nested sampling algorithm 
    Dynesty (https://dynesty.readthedocs.io/en/latest/).
    """
    search = af.DynestyStatic(
        path_prefix=path.join(dataset_label, dataset_time),
        name="species[x1]",
        nlive=50,
    )

    """
    __Analysis__
    
    The `AnalysisDataset1D` object defines the `log_likelihood_function` used by the non-linear search to fit the 
    model to the `Dataset1D` dataset.
    """
    analysis_list = [
        ac.AnalysisDataset1D(dataset=dataset, clocker=clocker)
        for dataset in dataset_list
    ]
    analysis = sum(analysis_list)
    analysis.n_cores = 1

    """
    __Model-Fit__
    
    We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
    search to find which models fit the data with the highest likelihood.
    """
    result_list = search.fit(model=model, analysis=analysis)

    """
    __Instances__
    
    Interpolation uses the maximum log likelihood model of each fit to build an interpolation model of the CTI as a
    function of time. 
    
    We therefore store the maximum log likelihood model of every fit in a list, which is used below.
    """
    ml_instances_list.append(result_list[0].instance)

"""
__Interpolation__

Now all fits are complete, we use the `ml_instances_list` to build an interpolation model of the CTI as a function of time.

This is performed using the `LinearInterpolator` object, which interpolates the CTI model parameters as a function of
time linearly between the values computed by the model-fits above.

More advanced interpolation schemes are available and described in the `interpolation.py` example.
"""
interpolator = af.LinearInterpolator(instances=ml_instances_list)

"""
The model can be interpolated to any time, for example time=1.5.

This returns a new `instance` of the CTI model, as an instance of the `CTI1D` object, where the parameters are computed 
by interpolating between the values computed above.
"""
instance = interpolator[interpolator.time == 1.5]

"""
The `density` of the `TrapInstantCapture` at time 1.5 is between the value inferred for the first and second fits taken
at times 1.0 and 2.0.
"""
print(instance.cti.trap_list[0].density)
print(ml_instances_list[0].cti.trap_list[0].density)
print(ml_instances_list[1].cti.trap_list[0].density)

"""
__Serialization__

The interpolator and model can be serialized to a .json file, so they can easily be loaded into other scripts.
"""
json_file = path.join(dataset_path, "interpolator.json")

interpolator.output_to_json(file_path=json_file)

interpolator = af.LinearInterpolator.from_json(file_path=json_file)
