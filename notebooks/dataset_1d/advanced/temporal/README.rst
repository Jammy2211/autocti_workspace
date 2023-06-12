The ``temporal`` folder contains tutorials explaining how to fit CTI calibration datasets taken over time, with
gradually increasing radiation damage, and interpolate the results to get a CTI model for correction at any point in
time.

Files (Beginner)
----------------

- ``fit.py``: Fit a CTI dataset taken over time and interpolate the results.
- ``database.py``: Loading results via the sqlite3 database tools in order to perform temporal interpolation.
- ``plot.py``: Plotting the results of the temporal fits and interpolation.