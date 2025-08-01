This repository contains the code developed for the following paper:

Fernando Coello*, Thomas Decorte*, Iris Janssens, Steven Mortier, Jordi Sardans, Josep Peñuelas and Tim Verdonck. *Global Crop-Specific Fertilization Dataset from 1961-2019.* Sci Data 12, 40 (2025). https://doi.org/10.1038/s41597-024-04215-x

\* *Fernando Coello and Thomas Decorte contributed equally to this work*

___

The repository is organized into the following folders:

* **source**: Contains all the Python code used for the machine learning (ML) models, including dataset preprocessing. It also includes the calculation of SHAP values for model interpretability.
* **prediction_corrected_byTotals**: Contains the R code used to adjust the ML predictions based on the total fertilizer consumption in each country.
* **Validation**: Includes Python code for comparing the adjusted ML predictions with national databases.
* **Map_Creation**: Contains both Python and R code used to allocate the ML results into spatial resolutions.
