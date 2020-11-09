import autofit as af
import autocti as ac

"""
In this pipeline, we fit `CIImaging` data using only a parallel CTI model with an input number of trap species. 

The pipeline is three phases:

Phase 1: 

    Fit a small section (the left-most 60 columns) of the charge injection images using a parallel CTI model
    with 1 `Trap` species and a model for the parallel CCD volume filling parameters.

    Traps : Parallel (x1)
    CCD Volume : Parallel 
    Prior Passing  None
    Notes : Fits a small section of the CTI calibration data for efficiency.

Phase 2: 

    Fit a small section (the left-most 60 columns) of the charge injection images using a parallel CTI model
    with the input number of `Trap` species and a model for the parallel CCD volume filling parameters. Priors are 
    initialized using the results of phase 1.

    Traps : Parallel (x input)
    CCD Volume : Parallel 
    Prior Passing  Parallel (model -> phase 1)
    Notes : Fits a small section of the CTI calibration data for efficiency.

Phase 3:

    Refit the phase 2 model, using priors initialized from the results of phase 1, the full dataset and the
    noise-scaled noise-map from a hyper phase is the feature is turned on.

    Traps : Parallel (x input)
    CCD Volume : Parallel 
    Prior Passing  Parallel (model -> phase 2).
    Notes : Fits the full CTI calibration data.
"""


def make_pipeline(setup, settings, total_parallel_traps, parallel_ccd):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_parallel"

    """
    This pipeline is tagged according to whether:
    """

    #    path_prefix = f"{setup.path_prefix}/{pipeline_name}/{setup.tag}"
    path_prefix = pipeline_name

    """
    Phase 1: Fit the dataset with a one species parallel CTI model and parallel CCD filling model, where we:

    1) Extract and fit the 60 columns of the `CIImaging` dataset closest to the read-out register (and
       therefore least affected by serial CTI).
    """

    phase1 = ac.PhaseCIImaging(
        search=af.MultiNest(name="phase[1]_parallel[x1]", n_live_points=30),
        parallel_traps=[af.PriorModel(ac.TrapInstantCapture)],
        parallel_ccd=parallel_ccd,
        settings=settings,
    )

    """
    Phase 2: Fit the dataset with an input number of parallel traps and parallel CCD filling model, where we:

    1) Extract and fit the 60 columns of the `CIImaging` dataset closest to the read-out register (and
       therefore least affected by serial CTI)..
    2) Use priors on the trap density and ccd volume filling parameters based on the results of phase 1.
    """

    #    previous_total_density = phase1.result.instance.parallel_traps[0].density

    parallel_trap = af.PriorModel(ac.TrapInstantCapture)
    #  parallel_trap.density = af.UniformPrior(lower_limit=0.0, upper_limit=previous_total_density)
    parallel_traps = [parallel_trap for i in range(total_parallel_traps)]

    phase2 = ac.PhaseCIImaging(
        search=af.MultiNest(
            name="phase[2]_parallel[multi]", n_live_points=75, sampling_efficiency=0.2
        ),
        parallel_traps=parallel_traps,
        parallel_ccd=phase1.result.model.parallel_ccd,
        settings=settings,
    )

    """
    Phase 3: Fit the full `CIImaging` dataset with the input number of parallel traps and parallel CCD filling model, 
    where we:

    1) Initialize priors from the results of phase 2.
    2) Use the scaled noise-map from the phase 2 hyper-phase, if the feature is turned on.
    """

    phase3 = ac.PhaseCIImaging(
        search=af.MultiNest(
            name="phase[3]_parallel[multi]", n_live_points=75, sampling_efficiency=0.2
        ),
        parallel_traps=phase2.result.model.parallel_traps,
        parallel_ccd=phase2.result.model.parallel_ccd,
        settings=settings,
    )

    return ac.Pipeline(pipeline_name, path_prefix, phase1, phase2, phase3)
