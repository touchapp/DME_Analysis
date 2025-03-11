# Medicare DME Data Analysis

This repository contains scripts for analyzing Medicare Durable Medical Equipment (DME) data from 2017-2022.

## Directory Structure

```
.
├── dme_analysis/              # Main package
│   ├── __init__.py            # Package initialization
│   ├── utils/                 # Utility modules
│   │   ├── __init__.py        # Subpackage initialization
│   │   ├── data_dictionary.py # Data dictionary and column categorization
│   │   └── data_import.py     # Data import functions
│   └── ...                    # Analysis modules
├── dme_data_analysis.py       # Main analysis script
├── fraud_detector.py          # Fraud detection script
└── ...                        # Data files and other scripts
```

## Usage

### Importing Data

You can import the DME data using the utility functions:

```python
from dme_analysis.utils import import_data_for_years

# Import data for years 2017-2022
df_by_year = import_data_for_years(range(2017, 2023))
```

### Data Dictionary

The data dictionary contains descriptions for all columns in the dataset:

```python
from dme_analysis.utils import DATA_DICTIONARY

# Get the description of a column
print(DATA_DICTIONARY['DME_Tot_Suplr_Benes'])
```

### Analyzing the Data

The main analysis script can be run directly:

```bash
python dme_data_analysis.py
```

Or imported in a Jupyter notebook:

```python
import dme_data_analysis as dme
%matplotlib inline

# Run the analysis
df_by_year, visualizations = dme.main()

# Display visualizations
visualizations['spending_trends']
```

### Fraud Detection

The fraud detection script can be used to identify potential fraud patterns:

```python
import fraud_detector as fd
%matplotlib inline

# Run the fraud detection analysis
df_by_year, visualizations, data = fd.main()

# Display fraud indicators
visualizations['high_growth_suppliers']
```

## Column Descriptions

The DME dataset contains the following types of columns:

1. Supplier Information (e.g., `Suplr_NPI`, `Suplr_Prvdr_Last_Name_Org`)
2. DME-specific fields (e.g., `DME_Tot_Suplr_Benes`, `DME_Suplr_Mdcr_Pymt_Amt`)
3. Prosthetic and Orthotic fields (e.g., `POS_Tot_Suplr_Benes`, `POS_Suplr_Mdcr_Pymt_Amt`)
4. Drug and Nutritional fields (e.g., `Drug_Tot_Suplr_Benes`, `Drug_Suplr_Mdcr_Pymt_Amt`)
5. Beneficiary Demographics (e.g., `Bene_Avg_Age`, `Bene_Feml_Cnt`)
6. Health Conditions (e.g., `Bene_CC_PH_Hypertension_V2_Pct`, `Bene_CC_BH_Mood_V2_Pct`)
