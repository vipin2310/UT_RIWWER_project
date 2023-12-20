# Urban Technology Project on RIWWER data

This project is part of the Urban Technology course at the BHT Berlin.
For this I am using the RIWWER dataset on sewage water overflow from WBD.
It is provided for the research project RIWWER.

## What is done so far:
- Load the dataset from Fraunhofer OwnCloud: Vierlinden_2021_All.csv (merged from multiple CSV files by Teo)
- Load the dataset on the target variables: "Kreuzweg_outflow [l/s]", "Kaiserstr_outflow [l/s]"
- Merged the two datasets on Datetime column
- Analyzed missing values:
    - For "Niederschlag":
        - Filled missing values with 0, because it probably meant no rainfall (Don't know exactly what Teo did there)
    - For highly missing values (measured at Franz Lenze Platz): 
        - Removed the respective columns (same as Teo did)
    - For the first 3 days (01-01-2021 00:00:00 to 01-04-2021 06:00:00) some sensory data was missing: 
        - Removed the rows on the first 3 days, since it will probably not make a big difference and imputing might be worse (?)
    - Else there were only a few missing values (maybe 2 days in total):
        - Imputed them with linear interpolation (Different then Teo, he used Backfill then Forwardfill)
- Created profiling report using ydata
- Some feature importance analysis:
    - Coefficients of linear regression
- Adjusted Teo's notebook on NHits since data preparation was already done by my notebook
- Ran Teo's notebook on NHits (with training) for target variable:
    - "Kreuzweg_outflow [l/s]"

## What I still want to do:
- Understand the results
- Run training on the other target variable:
    - "Kaiserstr_outflow [l/s]"
- Better understand Teo's notebook
- Train the other time series models
- Maybe some more feature engineering (PCA, etc.)
- Understand what Teo did for perturbation analysis (I think in his "dropout" notebooks)
- Prepare a presentation

## Not sure If I should do:
- Understand the Bellinge dataset
- Run everything on the Bellinge dataset