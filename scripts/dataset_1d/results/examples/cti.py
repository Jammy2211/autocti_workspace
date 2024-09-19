"""
Results: CTI
============
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
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)

shape_native = (200,)

prescan = ac.Region1D(region=(0, 10))
overscan = ac.Region1D(region=(190, 200))

region_list = [(10, 20)]

norm_list = [100, 5000, 25000, 200000]

total_datasets = len(norm_list)

layout_list = [
    ac.Layout1D(
        shape_1d=shape_native,
        region_list=region_list,
        prescan=prescan,
        overscan=overscan,
    )
    for i in range(total_datasets)
]

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

clocker = ac.Clocker1D(express=5)

trap_0 = af.Model(ac.TrapInstantCapture)
trap_1 = af.Model(ac.TrapInstantCapture)

trap_0.add_assertion(trap_0.release_timescale < trap_1.release_timescale)

trap_list = [trap_0, trap_1]

ccd = af.Model(ac.CCDPhase)
ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

search = af.Nautilus(
    path_prefix=path.join("dataset_1d", dataset_name), name="species[x2]", n_live=100
)

analysis_list = [
    ac.AnalysisDataset1D(dataset=dataset, clocker=clocker) for dataset in dataset_list
]

analysis = sum(analysis_list)

result_list = search.fit(model=model, analysis=analysis)


samples = result_list[0].samples


delta_ellipticity_list = []

print(samples.model.info)

for sample in samples.sample_list:
    instance = sample.instance_for_model(model=samples.model, ignore_assertions=True)
    delta_ellipticity_list.append(instance.cti.delta_ellipticity)

(
    median_delta_ellipticity,
    upper_delta_ellipticity,
    lower_delta_ellipticity,
) = af.marginalize(
    parameter_list=delta_ellipticity_list,
    sigma=2.0,
    weight_list=samples.weight_list,
)

print(median_delta_ellipticity)
