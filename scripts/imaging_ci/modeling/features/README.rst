The ``features`` folder contains scripts for **PyAutoCTI** modeling features that analyse charge
injection imaging data with more complex CTI models or additional complexity, for example non-uniform charge
injection lines or cosmic rays.

Files (Beginner)
----------------

- ``serial_cti.py``: Fits a serial CTI model to charge injection calibration data.
- ``visualize_full.py``: Visualize the full charge injeciton dataset, even for fits where a subset if masked or extracted.

Files (Advanced)
----------------

- ``cosmic_rays.py``: Fitting charge injection data with cosmic rays, which are first flagged and masked.
- ``non_uniform.py``: Fitting charge injection data with column-by-column non-uniform charge injection .



