The ``examples`` folder contains example scripts for simulating 2D charge injection CTI datasets using functionality not
described in ``start_here.ipynb``

Files (Beginner)
----------------

These scripts simulate 2D charge injection CTI datasets where:

- ``serial_cti.py``: There is serial CTI in the data as opposed to parallel CTI.
- ``parallel_and_serial_cti.py``: There is both serial and parallel CTI in the data.
- ``non_uniform.py``: The level of injected charge in every column varies.
- ``cosmic_rays.py``: The data includes cosmic rays which hit the CCD during data acquisition.
- ``poisson.py``: The density of traps for parallel CTI in each column varies according to a Poisson distribution.
- ``bias_uncorrected.py``: The bias subtraction is not applied to the data.