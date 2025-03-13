import pandas as pd
import os
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define key metrics we want to analyze
KEY_METRICS = [
    'Tot_Suplr_Benes',             # Number of beneficiaries
    'Tot_Suplr_Clms',              # Number of claims
    'Tot_Suplr_Srvcs',             # Number of services
    'Suplr_Sbmtd_Chrgs',           # Submitted charges
    'Suplr_Mdcr_Alowd_Amt',        # Medicare allowed amount
    'Suplr_Mdcr_Pymt_Amt',         # Medicare payment amount
    'Suplr_Mdcr_Stdzd_Pymt_Amt'    # Standardized payment amount
]

# Define derived metrics we'll calculate
DERIVED_METRICS = [
    'avg_charge_per_claim',        # Average charge per claim
    'avg_payment_per_claim',       # Average payment per claim
    'avg_services_per_claim',      # Average services per claim
    'payment_to_charge_ratio',     # Ratio of payment to submitted charges
    'yoy_growth_claims',           # Year-over-year growth in claims
    'yoy_growth_charges',          # Year-over-year growth in charges
    'yoy_growth_payments',         # Year-over-year growth in payments
    'yoy_growth_beneficiaries'     # Year-over-year growth in beneficiaries
]

# Define the columns to keep (relevant for fraud analysis)
RELEVANT_COLUMNS = [
    # Identification columns
    'Suplr_NPI',                    # Provider identifier
    'Suplr_Prvdr_Last_Name_Org',    # Provider/organization name
    'Suplr_Prvdr_Spclty_Desc',      # Specialty description
    'Suplr_Prvdr_State_Abrvtn',     # State (for geographic analysis)
    'year',                         # Year of the data

    # Key metrics for analysis
    'Tot_Suplr_Benes',              # Number of beneficiaries
    'Tot_Suplr_Clms',               # Number of claims
    'Tot_Suplr_Srvcs',              # Number of services
    'Suplr_Sbmtd_Chrgs',            # Submitted charges
    'Suplr_Mdcr_Alowd_Amt',         # Medicare allowed amount
    'Suplr_Mdcr_Pymt_Amt',          # Medicare payment amount
    'Suplr_Mdcr_Stdzd_Pymt_Amt',    # Standardized payment amount

    # Fraud information
    'has_existing_fraud_case',      # Whether the supplier has a fraud case
    'topics',                       # Topics related to the fraud case
    'url'                           # URL to the fraud case details
]


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

    # Clean up the has_existing_fraud_case column to ensure proper boolean values
    if 'has_existing_fraud_case' in fraud_df.columns:
        if fraud_df['has_existing_fraud_case'].dtype == 'object':
            fraud_df['has_existing_fraud_case'] = fraud_df['has_existing_fraud_case'].map(
                {'true': True, 'false': False}
            )

    return fraud_df


def merge_data_and_filter_fraud_cases():
    """
    Merge supplier and fraud data, filtering to only include suppliers 
    with confirmed fraud cases (has_existing_fraud_case = True) and
    keeping only relevant columns
    """
    # Load both datasets
    supplier_df = load_supplier_data()
    fraud_df = load_fraud_data()

    # Rename 'npi' column in fraud data to match supplier data
    fraud_df = fraud_df.rename(columns={'npi': 'Suplr_NPI'})

    # Filter fraud data to only include confirmed fraud cases
    confirmed_fraud_df = fraud_df[fraud_df['has_existing_fraud_case'] == True]
    confirmed_fraud_df = confirmed_fraud_df.drop_duplicates(
        subset=['Suplr_NPI'], keep='first')

    print(
        f"Found {len(confirmed_fraud_df)} unique suppliers with confirmed fraud cases")

    # Merge with supplier data
    print("Merging supplier and confirmed fraud data on NPI (inner join)...")
    fraud_suppliers_df = supplier_df.merge(
        confirmed_fraud_df,
        on='Suplr_NPI',
        how='inner'
    )

    print(
        f"Confirmed fraud suppliers DataFrame shape: {fraud_suppliers_df.shape}")
    print(f"Found {len(fraud_suppliers_df)} matching records across all years")

    # Filter to keep only relevant columns
    # First, check which relevant columns actually exist in the dataframe
    available_columns = [
        col for col in RELEVANT_COLUMNS if col in fraud_suppliers_df.columns]

    # Filter the dataframe to only include available relevant columns
    filtered_df = fraud_suppliers_df[available_columns]
    print(
        f"Filtered to {len(available_columns)} relevant columns for analysis")

    # Create a summary of matches by year
    matches_by_year = filtered_df.groupby('year').size()
    print("\nConfirmed fraud cases by year:")
    for year, count in matches_by_year.items():
        print(f"  Year {year}: {count} suppliers")

    # Get unique NPIs with confirmed fraud
    unique_fraud_npis = filtered_df['Suplr_NPI'].nunique()
    print(
        f"\nFound {unique_fraud_npis} unique suppliers with confirmed fraud cases")

    return filtered_df


def calculate_derived_metrics(df):
    """Calculate derived metrics for each supplier-year combination"""
    # Group by supplier and year
    metrics_df = df.copy()

    # Calculate average metrics
    metrics_df['avg_charge_per_claim'] = metrics_df['Suplr_Sbmtd_Chrgs'] / \
        metrics_df['Tot_Suplr_Clms'].replace(0, np.nan)
    metrics_df['avg_payment_per_claim'] = metrics_df['Suplr_Mdcr_Pymt_Amt'] / \
        metrics_df['Tot_Suplr_Clms'].replace(0, np.nan)
    metrics_df['avg_services_per_claim'] = metrics_df['Tot_Suplr_Srvcs'] / \
        metrics_df['Tot_Suplr_Clms'].replace(0, np.nan)
    metrics_df['payment_to_charge_ratio'] = metrics_df['Suplr_Mdcr_Pymt_Amt'] / \
        metrics_df['Suplr_Sbmtd_Chrgs'].replace(0, np.nan)

    return metrics_df


def calculate_year_over_year_growth(df):
    """Calculate year-over-year growth rates for each supplier"""
    # Create a DataFrame to store the results
    growth_df = df.copy()

    # Group by supplier and sort by year
    for npi in growth_df['Suplr_NPI'].unique():
        supplier_data = growth_df[growth_df['Suplr_NPI']
                                  == npi].sort_values('year')

        if len(supplier_data) > 1:  # Only calculate YoY if supplier appears in multiple years
            for i in range(1, len(supplier_data)):
                prev_idx = supplier_data.index[i-1]
                curr_idx = supplier_data.index[i]

                # Calculate YoY growth for key metrics
                for metric in ['Tot_Suplr_Clms', 'Suplr_Sbmtd_Chrgs', 'Suplr_Mdcr_Pymt_Amt', 'Tot_Suplr_Benes']:
                    if metric in supplier_data.columns:  # Make sure the column exists
                        prev_value = supplier_data.loc[prev_idx, metric]
                        curr_value = supplier_data.loc[curr_idx, metric]

                        if prev_value > 0:
                            growth = (curr_value - prev_value) / \
                                prev_value * 100
                        else:
                            growth = None  # Can't calculate growth if previous value is 0

                        growth_col = f'yoy_growth_{metric.lower().replace("suplr_", "").replace("tot_suplr_", "")}'
                        growth_df.loc[curr_idx, growth_col] = growth

    return growth_df


def analyze_fraud_patterns(fraud_df):
    """Analyze patterns in confirmed fraud cases"""
    # Calculate derived metrics
    metrics_df = calculate_derived_metrics(fraud_df)

    # Calculate year-over-year growth
    full_metrics_df = calculate_year_over_year_growth(metrics_df)

    # Get summary statistics for multi-year suppliers
    suppliers_with_multiple_years = full_metrics_df.groupby(
        'Suplr_NPI').filter(lambda x: len(x) > 1)

    # Calculate supplier-level statistics
    supplier_stats = []
    for npi in suppliers_with_multiple_years['Suplr_NPI'].unique():
        supplier_data = full_metrics_df[full_metrics_df['Suplr_NPI'] == npi].sort_values(
            'year')

        # Basic info
        name = supplier_data['Suplr_Prvdr_Last_Name_Org'].iloc[0]
        years = supplier_data['year'].tolist()
        specialty = supplier_data['Suplr_Prvdr_Spclty_Desc'].iloc[
            0] if 'Suplr_Prvdr_Spclty_Desc' in supplier_data.columns else "Unknown"
        state = supplier_data['Suplr_Prvdr_State_Abrvtn'].iloc[0] if 'Suplr_Prvdr_State_Abrvtn' in supplier_data.columns else "Unknown"
        first_year = min(years)
        last_year = max(years)

        # Calculate total growth
        first_year_data = supplier_data[supplier_data['year'] == first_year]
        last_year_data = supplier_data[supplier_data['year'] == last_year]

        # Calculate percentage change for key metrics
        changes = {}
        for metric in KEY_METRICS:
            if metric in first_year_data.columns and metric in last_year_data.columns:
                first_val = first_year_data[metric].values[0]
                last_val = last_year_data[metric].values[0]

                if first_val > 0:
                    pct_change = (last_val - first_val) / first_val * 100
                else:
                    pct_change = np.nan

                changes[f'{metric}_pct_change'] = pct_change

        # Get max year-over-year growth values
        yoy_metrics = [
            col for col in full_metrics_df.columns if col.startswith('yoy_growth_')]
        max_growths = {}
        for metric in yoy_metrics:
            valid_values = supplier_data[metric].dropna()
            if not valid_values.empty:
                max_growths[f'max_{metric}'] = valid_values.max()
                max_growths[f'max_{metric}_year'] = supplier_data.loc[valid_values.idxmax(
                ), 'year']

        # Combine into a single record
        supplier_record = {
            'Suplr_NPI': npi,
            'name': name,
            'specialty': specialty,
            'state': state,
            'years_present': years,
            'first_year': first_year,
            'last_year': last_year,
            'num_years': len(years)
        }
        supplier_record.update(changes)
        supplier_record.update(max_growths)

        supplier_stats.append(supplier_record)

    supplier_stats_df = pd.DataFrame(supplier_stats)

    # Get extremes for each metric
    extreme_changes = {}
    change_metrics = [
        col for col in supplier_stats_df.columns if col.endswith('_pct_change')]
    for metric in change_metrics:
        base_metric = metric.replace('_pct_change', '')
        if not supplier_stats_df[metric].dropna().empty:
            # Get largest increases
            largest_increase_idx = supplier_stats_df[metric].idxmax()
            extreme_changes[f'largest_{base_metric}_increase'] = {
                'npi': supplier_stats_df.loc[largest_increase_idx, 'Suplr_NPI'],
                'name': supplier_stats_df.loc[largest_increase_idx, 'name'],
                'pct_change': supplier_stats_df.loc[largest_increase_idx, metric],
                'years': f"{supplier_stats_df.loc[largest_increase_idx, 'first_year']} to {supplier_stats_df.loc[largest_increase_idx, 'last_year']}"
            }

            # Get largest decreases
            largest_decrease_idx = supplier_stats_df[metric].idxmin()
            extreme_changes[f'largest_{base_metric}_decrease'] = {
                'npi': supplier_stats_df.loc[largest_decrease_idx, 'Suplr_NPI'],
                'name': supplier_stats_df.loc[largest_decrease_idx, 'name'],
                'pct_change': supplier_stats_df.loc[largest_decrease_idx, metric],
                'years': f"{supplier_stats_df.loc[largest_decrease_idx, 'first_year']} to {supplier_stats_df.loc[largest_decrease_idx, 'last_year']}"
            }

    return full_metrics_df, supplier_stats_df, extreme_changes


def create_time_series_dataset(fraud_df):
    """
    Create a time series dataset with only the most relevant information
    for analyzing changes over time
    """
    # Sort by supplier and year
    time_series_df = fraud_df.sort_values(['Suplr_NPI', 'year'])

    # Add derived metrics
    time_series_df = calculate_derived_metrics(time_series_df)

    # Calculate growth metrics
    time_series_df = calculate_year_over_year_growth(time_series_df)

    return time_series_df


def print_fraud_patterns_summary(supplier_stats_df, extreme_changes):
    """Print a summary of the fraud patterns analysis"""
    # Print overall stats
    print("\n===== FRAUD PATTERNS ANALYSIS =====")
    print(
        f"Analyzed {len(supplier_stats_df)} suppliers with confirmed fraud cases across multiple years")

    avg_years = supplier_stats_df['num_years'].mean()
    print(f"Average years of data per supplier: {avg_years:.1f}")

    # Print most common specialties
    if 'specialty' in supplier_stats_df.columns:
        print("\nMost common specialties among fraud suppliers:")
        specialty_counts = supplier_stats_df['specialty'].value_counts().head(
            5)
        for specialty, count in specialty_counts.items():
            print(f"  {specialty}: {count} suppliers")

    # Print extreme changes
    print("\nLargest changes in key metrics from first to last year:")
    for metric_name, data in extreme_changes.items():
        if 'increase' in metric_name:
            change_type = "increase"
        else:
            change_type = "decrease"

        print(f"\n  {metric_name.replace('_', ' ').title()}:")
        print(f"    Supplier: {data['name']} (NPI: {data['npi']})")
        print(f"    {change_type.title()}: {data['pct_change']:.1f}%")
        print(f"    Years: {data['years']}")

    # Print common fraud patterns
    print("\nCommon patterns in confirmed fraud cases:")

    # Check for large increases in submitted charges
    if 'Suplr_Sbmtd_Chrgs_pct_change' in supplier_stats_df.columns:
        large_charge_increases = supplier_stats_df[supplier_stats_df['Suplr_Sbmtd_Chrgs_pct_change'] > 50]
        if not large_charge_increases.empty:
            print(
                f"  - {len(large_charge_increases)} suppliers showed >50% increase in submitted charges")

    # Check for large increases in claims
    if 'Tot_Suplr_Clms_pct_change' in supplier_stats_df.columns:
        large_claim_increases = supplier_stats_df[supplier_stats_df['Tot_Suplr_Clms_pct_change'] > 50]
        if not large_claim_increases.empty:
            print(
                f"  - {len(large_claim_increases)} suppliers showed >50% increase in claims")

    # Check for large increases in payment amounts
    if 'Suplr_Mdcr_Pymt_Amt_pct_change' in supplier_stats_df.columns:
        large_payment_increases = supplier_stats_df[supplier_stats_df['Suplr_Mdcr_Pymt_Amt_pct_change'] > 50]
        if not large_payment_increases.empty:
            print(
                f"  - {len(large_payment_increases)} suppliers showed >50% increase in Medicare payments")

        # Check for large decreases after potential investigation/charges
        large_payment_decreases = supplier_stats_df[supplier_stats_df['Suplr_Mdcr_Pymt_Amt_pct_change'] < -50]
        if not large_payment_decreases.empty:
            print(
                f"  - {len(large_payment_decreases)} suppliers showed >50% decrease in Medicare payments")


def main():
    # Get confirmed fraud cases with only relevant columns
    fraud_df = merge_data_and_filter_fraud_cases()

    # Create time series dataset for analysis
    time_series_df = create_time_series_dataset(fraud_df)

    # Analyze patterns
    full_metrics_df, supplier_stats_df, extreme_changes = analyze_fraud_patterns(
        fraud_df)

    # Print summary of fraud patterns
    print_fraud_patterns_summary(supplier_stats_df, extreme_changes)

    # Save time series data to CSV
    time_series_df.to_csv("fraud_time_series.csv", index=False)
    print("\nSaved time series data to fraud_time_series.csv")

    # Save supplier-level summary to CSV
    supplier_stats_df.to_csv("fraud_supplier_summary.csv", index=False)
    print("Saved supplier summary to fraud_supplier_summary.csv")

    # Create a simple Excel report that's easier to read
    print("Creating Excel report with time series data...")
    with pd.ExcelWriter("fraud_pattern_analysis.xlsx") as writer:
        # First sheet: Supplier summary
        supplier_stats_df.to_excel(
            writer, sheet_name="Supplier Summary", index=False)

        # Second sheet: Year-by-year metrics for all suppliers
        time_series_df.to_excel(
            writer, sheet_name="Time Series Data", index=False)

    print("Excel report saved to fraud_pattern_analysis.xlsx")

    return time_series_df, supplier_stats_df


if __name__ == "__main__":
    time_series_df, supplier_stats_df = main()
