import pandas as pd
import os
from pathlib import Path
import glob

# Define a data dictionary to explain columns
DATA_DICTIONARY = {
    # Supplier identification
    'Suplr_NPI': 'National Provider Identifier - unique ID for healthcare providers',
    'Suplr_Prvdr_Last_Name_Org': 'Last name or organization name of the supplier',
    'Suplr_Prvdr_First_Name': 'First name of the supplier (if individual)',
    'Suplr_Prvdr_MI': 'Middle initial of the supplier (if individual)',
    'Suplr_Prvdr_Crdntls': 'Credentials of the supplier',
    'Suplr_Prvdr_Gndr': 'Gender of the supplier (if individual)',
    'Suplr_Prvdr_Ent_Cd': 'Entity code of the supplier',

    # Location information
    'Suplr_Prvdr_St1': 'Street address line 1',
    'Suplr_Prvdr_St2': 'Street address line 2',
    'Suplr_Prvdr_City': 'City of the supplier',
    'Suplr_Prvdr_State_Abrvtn': 'State abbreviation',
    'Suplr_Prvdr_State_FIPS': 'State FIPS code',
    'Suplr_Prvdr_Zip5': 'ZIP code (5 digits)',
    'Suplr_Prvdr_RUCA': 'Rural-Urban Commuting Area (RUCA) code',
    'Suplr_Prvdr_RUCA_Desc': 'Description of the RUCA code',
    'Suplr_Prvdr_Cntry': 'Country of the supplier',

    # Specialty information
    'Suplr_Prvdr_Spclty_Desc': 'Description of supplier specialty',
    'Suplr_Prvdr_Spclty_Srce': 'Source of specialty information',

    # Claims and beneficiary information
    'Tot_Suplr_HCPCS_Cds': 'Total number of unique HCPCS codes billed',
    'Tot_Suplr_Benes': 'Total number of Medicare beneficiaries',
    'Tot_Suplr_Clms': 'Total number of claims submitted',
    'Tot_Suplr_Srvcs': 'Total number of services provided',

    # Financial information
    'Suplr_Sbmtd_Chrgs': 'Total charges submitted to Medicare',
    'Suplr_Mdcr_Alowd_Amt': 'Total Medicare allowed amount',
    'Suplr_Mdcr_Pymt_Amt': 'Total Medicare payment amount',
    'Suplr_Mdcr_Stdzd_Pymt_Amt': 'Total Medicare standardized payment amount',

    # DME specific information
    'DME_Sprsn_Ind': 'DME suppression indicator',
    'DME_Tot_Suplr_HCPCS_Cds': 'Total DME HCPCS codes',
    'DME_Tot_Suplr_Benes': 'Total DME beneficiaries',
    'DME_Tot_Suplr_Clms': 'Total DME claims',
    'DME_Tot_Suplr_Srvcs': 'Total DME services',
    'DME_Suplr_Sbmtd_Chrgs': 'Total DME submitted charges',
    'DME_Suplr_Mdcr_Alowd_Amt': 'Total DME Medicare allowed amount',
    'DME_Suplr_Mdcr_Pymt_Amt': 'Total DME Medicare payment amount',
    'DME_Suplr_Mdcr_Stdzd_Pymt_Amt': 'Total DME Medicare standardized payment amount',

    # Year identifier
    'year': 'Calendar year of the data',

    # Fraud data
    'has_existing_fraud_case': 'Indicates if supplier has an existing fraud case (true/false)',
    'organization_name': 'Name of organization from fraud database',
    'entity_type': 'Type of entity in fraud case',
    'role': 'Role in fraud case (e.g., Defendant)',
    'name': 'Name of the fraud case',
    'url': 'URL to fraud case information',
    'component': 'Component of justice department handling the case',
    'topics': 'Topics related to the fraud case',
    'entity_match_count': 'Number of matches for this entity in fraud database'
}


def load_supplier_data(start_year=2018, end_year=2022):
    """Load supplier data from CSV files"""
    # Initialize an empty list to store dataframes
    dfs = []

    # Iterate through the years
    for year in range(start_year, end_year + 1):
        # Get path to year directory
        year_dir = Path(f"data/{year}")

        # Find all CSV files in the year directory
        csv_files = list(year_dir.glob("*.csv"))

        if not csv_files:
            print(f"Warning: No CSV files found for year {year}")
            continue

        # Load each CSV file in the directory
        for csv_file in csv_files:
            print(f"Loading {csv_file}...")
            try:
                # Read the CSV file with low_memory=False to prevent mixed types warning
                df = pd.read_csv(csv_file, low_memory=False)

                # Add a 'year' column to track the source
                df['year'] = year

                # Append to our list of dataframes
                dfs.append(df)

                print(f"  Loaded {len(df)} rows from {csv_file.name}")
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")

    # Combine all dataframes
    if not dfs:
        raise ValueError(
            "No data was loaded. Check if CSV files exist in the specified directories.")

    print("Combining all dataframes...")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined supplier DataFrame shape: {combined_df.shape}")

    return combined_df


def load_fraud_data(file_path="data/fraud_data/fraud-export.csv"):
    """Load fraud data from CSV file"""
    print(f"Loading fraud data from {file_path}...")
    fraud_df = pd.read_csv(file_path, low_memory=False)
    print(f"  Loaded {len(fraud_df)} fraud records")

    # Convert NPI to integer for consistent joining
    fraud_df['npi'] = pd.to_numeric(fraud_df['npi'], errors='coerce')
    fraud_df = fraud_df.dropna(subset=['npi'])  # Drop rows with NaN NPIs
    fraud_df['npi'] = fraud_df['npi'].astype(
        'int64')  # Ensure NPIs are integers

    return fraud_df


def merge_data():
    """Merge supplier and fraud data on NPI, only keeping matching records"""
    # Load both datasets
    supplier_df = load_supplier_data()
    fraud_df = load_fraud_data()

    # Rename 'npi' column in fraud data to match supplier data
    fraud_df = fraud_df.rename(columns={'npi': 'Suplr_NPI'})

    # Remove duplicate fraud records for the same NPI (keep first occurrence)
    fraud_df_unique = fraud_df.drop_duplicates(
        subset=['Suplr_NPI'], keep='first')
    print(f"Found {len(fraud_df_unique)} unique suppliers in fraud database")

    # First, perform an inner merge to get all suppliers with fraud records
    print("Merging supplier and fraud data on NPI (inner join)...")
    matched_df = supplier_df.merge(
        fraud_df_unique,
        on='Suplr_NPI',
        how='inner'
    )

    print(f"Matched DataFrame shape: {matched_df.shape}")
    print(f"Found {len(matched_df)} matching records across all years")

    # Create a summary of matches by year
    matches_by_year = matched_df.groupby('year').size()
    print("\nMatches by year:")
    for year, count in matches_by_year.items():
        print(f"  Year {year}: {count} matches")

    # Get unique NPIs that have fraud records
    unique_fraud_npis = matched_df['Suplr_NPI'].nunique()
    print(f"\nFound {unique_fraud_npis} unique suppliers with fraud records")

    # Create a time series summary for each supplier with fraud
    print("\nGenerating time series data for each fraud supplier...")

    # Get key metrics for analysis
    time_series_metrics = [
        'Suplr_NPI',
        'Suplr_Prvdr_Last_Name_Org',
        'year',
        'Tot_Suplr_Benes',
        'Tot_Suplr_Clms',
        'Tot_Suplr_Srvcs',
        'Suplr_Sbmtd_Chrgs',
        'Suplr_Mdcr_Alowd_Amt',
        'Suplr_Mdcr_Pymt_Amt',
        'has_existing_fraud_case'
    ]

    # Create a time series dataset with just the key metrics
    time_series_df = matched_df[time_series_metrics]

    # Calculate the trend for each supplier over time
    supplier_trends = {}
    for npi in time_series_df['Suplr_NPI'].unique():
        supplier_data = time_series_df[time_series_df['Suplr_NPI'] == npi].sort_values(
            'year')

        if len(supplier_data) > 1:  # Only calculate trend if supplier appears in multiple years
            first_year = supplier_data['year'].min()
            last_year = supplier_data['year'].max()

            # Get first and last year values for key financial metrics
            first_claims = supplier_data[supplier_data['year']
                                         == first_year]['Tot_Suplr_Clms'].values[0]
            last_claims = supplier_data[supplier_data['year']
                                        == last_year]['Tot_Suplr_Clms'].values[0]

            first_payment = supplier_data[supplier_data['year']
                                          == first_year]['Suplr_Mdcr_Pymt_Amt'].values[0]
            last_payment = supplier_data[supplier_data['year']
                                         == last_year]['Suplr_Mdcr_Pymt_Amt'].values[0]

            # Calculate percent change
            claims_change = ((last_claims - first_claims) /
                             first_claims * 100) if first_claims > 0 else 0
            payment_change = ((last_payment - first_payment) /
                              first_payment * 100) if first_payment > 0 else 0

            supplier_trends[npi] = {
                'years_present': supplier_data['year'].tolist(),
                'name': supplier_data['Suplr_Prvdr_Last_Name_Org'].iloc[0],
                'total_years': len(supplier_data),
                'claims_change_pct': claims_change,
                'payment_change_pct': payment_change,
                'first_year': first_year,
                'last_year': last_year,
                'has_fraud_case': supplier_data['has_existing_fraud_case'].iloc[0]
            }

    # Print summary of the time series analysis
    print(f"\nTime series analysis for {len(supplier_trends)} suppliers:")
    print(
        f"  Average years present: {sum(s['total_years'] for s in supplier_trends.values()) / len(supplier_trends):.1f}")

    return matched_df, supplier_trends, time_series_df


def print_data_dictionary(selected_columns=None):
    """Print data dictionary information for selected columns"""
    if selected_columns:
        print("\nData Dictionary (selected columns):")
        for col in selected_columns:
            if col in DATA_DICTIONARY:
                print(f"  {col}: {DATA_DICTIONARY[col]}")
            else:
                print(f"  {col}: Description not available")
    else:
        print("\nData Dictionary:")
        for col, desc in DATA_DICTIONARY.items():
            print(f"  {col}: {desc}")


if __name__ == "__main__":
    # Merge data and create fraud_matches_df
    matched_df, supplier_trends, time_series_df = merge_data()

    # Display a sample of the matched data
    print("\nSample of matched data (first 5 rows):")
    sample_cols = ['Suplr_NPI', 'Suplr_Prvdr_Last_Name_Org', 'year',
                   'has_existing_fraud_case', 'Suplr_Mdcr_Pymt_Amt', 'Tot_Suplr_Clms']
    print(matched_df[sample_cols].head())

    # Print the data dictionary for these columns
    print_data_dictionary(sample_cols)

    # Save the matched dataset
    matched_df.to_csv("supplier_fraud_matches.csv", index=False)
    print("Saved matched dataset to supplier_fraud_matches.csv")

    # Save the time series dataset
    time_series_df.to_csv("supplier_fraud_time_series.csv", index=False)
    print("Saved time series dataset to supplier_fraud_time_series.csv")

    # Print examples of suppliers with multiple years of data
    multi_year_suppliers = [
        npi for npi, data in supplier_trends.items() if data['total_years'] > 1]
    if multi_year_suppliers:
        print("\nExamples of suppliers with data across multiple years:")
        # Show up to 5 examples
        for i, npi in enumerate(multi_year_suppliers[:5]):
            trend = supplier_trends[npi]
            print(f"{i+1}. {trend['name']} (NPI: {npi})")
            print(
                f"   Years present: {', '.join(map(str, trend['years_present']))}")
            print(
                f"   Claims change from {trend['first_year']} to {trend['last_year']}: {trend['claims_change_pct']:.1f}%")
            print(
                f"   Payment change from {trend['first_year']} to {trend['last_year']}: {trend['payment_change_pct']:.1f}%")

    # Make the DataFrame available globally
    global time_series_fraud_df
    time_series_fraud_df = time_series_df
