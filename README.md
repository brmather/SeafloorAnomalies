# Seafloor Anomalies

Workflows to interrogate the relationships between subducting seafloor anomalies and porphyry copper deposits. These Python workflows accompany the results presented in:

> Mather, B., Müller, R.D., Alfonso, C.P., Wright, N.M., Seton, M. (In prep.) Subducting seafloor anomalies promote porphyry copper formation.

## Jupyter notebooks

1. __01-Identify-Shear-Zones.ipynb__: workflow to identify age discontinuities in seafloor age grids.
2. __02-SZ-Subduction.ipynb__: tracks where seafloor age discontinuities intersect subduction zones.
3. __03-Synthetic-Seamounts.ipynb__: tracks where seamount chains associated with a mantle plume intersect subduction zones.
4. __04-Conjugate-LIPs.ipynb__: tracks where LIPs and their conjugates intersect subduction zones.
5. __05-Relationship-to-mineral-deposits.ipynb__: examines statistical relationships to mineral deposits from the Hoggard _et al._ (2020), Nat. Geo. dataset.
6. __06-Plot-Timeseries.ipynb__: notebook for generating maps of porphyry copper deposits, seafloor anomalies and their intersections with subduction zones through time (associated Python files plot these timeseries in parallel using multiple CPU processors).
7. __07-Input-to_PU-learn.ipynb__: compiles data to be used in the positive-unlabelled machine learning classifier developed by Alfonso _et al._, (In prep.) These files can be copied into the "prepared_data" folder of the [PU-learn workflow](https://github.com/cpalfonso/stellar-data-mining).
8. __Fig-bathymetry.ipynb__: creates a map of present-day seafloor anomalies.
9. __Fig-plot-timeseries.py__: creates maps of seafloor anomalies at timesteps specified in the Python script.

## Data

The `data` folder contains input GPlates data on large igneous provinces (LIPs), seamount chains, and the Hoggard _et al._ (2020), Nat. Geo. dataset of porphyry copper deposits.

#### References

Hoggard, M. J., Czarnota, K., Richards, F. D., Huston, D. L., Jaques, A. L., & Ghelichkhan, S. (2020). Global distribution of sediment-hosted metals controlled by craton edge stability. Nature Geoscience, (May). https://doi.org/10.31223/osf.io/2kjvc
