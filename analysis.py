import pandas as pd
import os
import glob
from dme_dictionary import DATA_DICTIONARY
import numpy as np
import locale

# Set locale for currency formatting
locale.setlocale(locale.LC_ALL, '')

# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Don't wrap the output
# Don't add new lines in wide DataFrames
pd.set_option('display.expand_frame_repr', False)

# Path to data directories
data_dir = 'data'
years = range(2018, 2023)  # 2018 to 2022

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each year
for year in years:
    # Get the CSV file path
    csv_files = glob.glob(f"{data_dir}/{year}/*.csv")

    if not csv_files:
        print(f"No CSV files found for year {year}")
        continue

    # Get the first CSV file (there should be only one per year based on our observation)
    csv_file = csv_files[0]
    print(f"Loading data from {csv_file}")

    # Read the CSV into a DataFrame with mixed type handling
    df = pd.read_csv(csv_file, low_memory=False)

    # Add a 'year' column
    df['year'] = year

    # Append to our list of DataFrames
    dfs.append(df)

# Function to format dollar amounts (K or M based on size)


def format_dollar_amount(amount):
    if amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    else:
        return f"${amount/1000:.1f}K"


# Combine all DataFrames into one
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")

    # Display the first few rows with all columns
    # print("\nFirst few rows of the combined DataFrame:")
    # print(combined_df.head())

    # Create a dictionary to store column information
    column_info = {}

    # Check which columns from our data are in the data dictionary
    for column in combined_df.columns:
        if column in DATA_DICTIONARY:
            column_info[column] = DATA_DICTIONARY[column]
        else:
            column_info[column] = "Description not available"

    # Display column information
    print("\nColumn Information:")
    # for column, count in zip(combined_df.columns, combined_df.count()):
    #     description = column_info.get(column, "Description not available")
    #     print(f"Column: {column}")
    #     print(f"  Description: {description}")
    #     print(f"  Non-null count: {count}/{len(combined_df)} entries")
    #     print(f"  Data type: {combined_df[column].dtype}")
    #     print()

    # Add data dictionary descriptions as attributes
    combined_df.attrs['column_descriptions'] = column_info

    # Summary statistics for numerical columns
    # print("\nSummary statistics for numerical columns:")
    # print(combined_df.describe())

    print("\n" + "="*100)
    print("Year-over-Year Growth Rate Analysis")
    print("="*100)

    # Create a DataFrame to analyze suppliers by year-over-year growth rate
    # First, group by supplier and year to get annual totals
    supplier_yearly = combined_df.groupby(['Suplr_NPI', 'Suplr_Prvdr_Last_Name_Org', 'year']).agg({
        'Suplr_Sbmtd_Chrgs': 'sum',
        'Suplr_Mdcr_Pymt_Amt': 'sum',
        'Tot_Suplr_Benes': 'mean',  # Average number of beneficiaries
        'Tot_Suplr_Clms': 'sum'     # Total claims
    }).reset_index()

    # Create a pivot table to have years as columns
    pivot_charges = supplier_yearly.pivot_table(
        index=['Suplr_NPI', 'Suplr_Prvdr_Last_Name_Org'],
        columns='year',
        values='Suplr_Mdcr_Pymt_Amt',
        fill_value=0
    )

    # Calculate year-over-year growth rates
    growth_rates = pd.DataFrame(index=pivot_charges.index)

    # Calculate growth rate for each year pair (2019/2018, 2020/2019, etc.)
    for year_pair in [(2019, 2018), (2020, 2019), (2021, 2020), (2022, 2021)]:
        current, previous = year_pair
        growth_rates[f'growth_{current}'] = (
            (pivot_charges[current] - pivot_charges[previous]) /
            pivot_charges[previous].replace(0, float('nan'))
        ) * 100  # Convert to percentage

    # Calculate average growth rate across all years
    growth_cols = [
        col for col in growth_rates.columns if col.startswith('growth_')]
    growth_rates['avg_growth'] = growth_rates[growth_cols].mean(axis=1)

    # Filter out suppliers that weren't present in all years
    valid_suppliers = pivot_charges[(pivot_charges[2018] > 0) &
                                    (pivot_charges[2019] > 0) &
                                    (pivot_charges[2020] > 0) &
                                    (pivot_charges[2021] > 0) &
                                    (pivot_charges[2022] > 0)]

    # Filter suppliers with significant payment amounts (at least $100K in the last year)
    significant_suppliers = valid_suppliers[valid_suppliers[2022] >= 100000]
    print(
        f"Filtering to {len(significant_suppliers)} suppliers with at least $100,000 in payments in 2022")

    # Merge growth rates with valid and significant suppliers
    valid_growth = growth_rates.loc[significant_suppliers.index].reset_index()

    # Sort by average growth rate in descending order
    top_growth = valid_growth.sort_values('avg_growth', ascending=False)

    # Merge with additional data for reporting
    supplier_totals = supplier_yearly.groupby(['Suplr_NPI', 'Suplr_Prvdr_Last_Name_Org']).agg({
        'Suplr_Sbmtd_Chrgs': 'sum',
        'Suplr_Mdcr_Pymt_Amt': 'sum',
        'Tot_Suplr_Benes': 'mean',
        'Tot_Suplr_Clms': 'sum'
    }).reset_index()

    top_growth_with_data = pd.merge(
        top_growth,
        supplier_totals,
        on=['Suplr_NPI', 'Suplr_Prvdr_Last_Name_Org']
    )

    # Format the output for the top 10 suppliers
    print("The analysis identified suppliers with the highest growth rates based on Medicare payment amounts from 2018 to 2022.")
    print("Here are the top 10 suppliers with extraordinary growth (minimum $100K in 2022 payments):\n")

    # Get top 10 suppliers
    top_10_suppliers = top_growth_with_data.head(10)
    top_10_npi = top_10_suppliers['Suplr_NPI'].tolist()

    # Filter the original data for just these suppliers
    top_supplier_data = supplier_yearly[supplier_yearly['Suplr_NPI'].isin(
        top_10_npi)]

    # Format and display each supplier's information
    for i, (_, supplier) in enumerate(top_10_suppliers.iterrows(), 1):
        npi = supplier['Suplr_NPI']
        name = supplier['Suplr_Prvdr_Last_Name_Org']
        avg_growth = supplier['avg_growth']
        total_payments = supplier['Suplr_Mdcr_Pymt_Amt']

        # Get yearly data for this supplier
        yearly_data = top_supplier_data[top_supplier_data['Suplr_NPI'] == npi].sort_values(
            'year')

        print(f"{i}. **{name}** (NPI: {npi})")
        print(f"   - Average growth rate: {avg_growth:.2f}%")
        print(
            f"   - Total Medicare payments: ${total_payments/1000000:.2f} million")

        # Show yearly payment amounts
        yearly_payments = []
        for year in range(2018, 2023):
            year_data = yearly_data[yearly_data['year'] == year]
            if not year_data.empty:
                payment = year_data['Suplr_Mdcr_Pymt_Amt'].values[0]
                yearly_payments.append(format_dollar_amount(payment))
            else:
                yearly_payments.append("$0")

        print(
            f"   - Yearly payments: 2018: {yearly_payments[0]}, 2019: {yearly_payments[1]}, 2020: {yearly_payments[2]}, 2021: {yearly_payments[3]}, 2022: {yearly_payments[4]}")

        # Analyze growth pattern
        payment_pattern = yearly_data['Suplr_Mdcr_Pymt_Amt'].tolist()
        years_list = yearly_data['year'].tolist()
        benes_pattern = yearly_data['Tot_Suplr_Benes'].tolist()

        # Identify the largest year-over-year jump
        max_jump = 0
        max_jump_year_idx = 0
        for j in range(1, len(payment_pattern)):
            if payment_pattern[j-1] > 0:
                jump_pct = (
                    payment_pattern[j] - payment_pattern[j-1]) / payment_pattern[j-1] * 100
                if jump_pct > max_jump:
                    max_jump = jump_pct
                    max_jump_year_idx = j

        if max_jump_year_idx > 0:
            from_year = years_list[max_jump_year_idx-1]
            to_year = years_list[max_jump_year_idx]
            from_amount = payment_pattern[max_jump_year_idx-1]
            to_amount = payment_pattern[max_jump_year_idx]

            # Format amounts with K or M suffix based on size
            from_amount_str = format_dollar_amount(from_amount)
            to_amount_str = format_dollar_amount(to_amount)

            print(
                f"   - Growth pattern: Major increase from {from_year} to {to_year} ({from_amount_str} to {to_amount_str})")

        # Check for consistent growth
        growth_consistent = True
        for j in range(1, len(payment_pattern)):
            if payment_pattern[j] <= payment_pattern[j-1]:
                growth_consistent = False
                break

        if growth_consistent and len(payment_pattern) > 2:
            print("   - Pattern shows consistent year-over-year growth")

        # Check for beneficiary growth
        if not pd.isna(benes_pattern).all() and len(benes_pattern) >= 2:
            first_valid_idx = next((i for i, x in enumerate(
                benes_pattern) if not pd.isna(x)), None)
            last_valid_idx = next((i for i, x in enumerate(
                reversed(benes_pattern)) if not pd.isna(x)), None)
            if first_valid_idx is not None and last_valid_idx is not None:
                last_valid_idx = len(benes_pattern) - 1 - last_valid_idx
                first_benes = benes_pattern[first_valid_idx]
                last_benes = benes_pattern[last_valid_idx]
                if not pd.isna(first_benes) and not pd.isna(last_benes) and first_benes > 0:
                    bene_growth = (last_benes - first_benes) / \
                        first_benes * 100
                    print(
                        f"   - Beneficiary growth: {bene_growth:.1f}% increase (from {first_benes:.0f} to {last_benes:.0f})")

        print("")  # Add a blank line between suppliers

    # =====================================
    # Analysis of High Submitted Charges vs Low Allowed/Paid Amounts
    # =====================================
    print("\n" + "="*100)
    print("Analysis of High Submitted Charges with Low Allowed/Paid Amounts")
    print("="*100)

    # Aggregate data by supplier across all years
    supplier_totals_with_allowed = combined_df.groupby(['Suplr_NPI', 'Suplr_Prvdr_Last_Name_Org']).agg({
        'Suplr_Sbmtd_Chrgs': 'sum',
        'Suplr_Mdcr_Alowd_Amt': 'sum',
        'Suplr_Mdcr_Pymt_Amt': 'sum',
        'Tot_Suplr_Benes': 'mean',
        'Tot_Suplr_Clms': 'sum'
    }).reset_index()

    # Calculate ratios
    supplier_totals_with_allowed['submitted_allowed_ratio'] = supplier_totals_with_allowed['Suplr_Sbmtd_Chrgs'] / \
        supplier_totals_with_allowed['Suplr_Mdcr_Alowd_Amt']
    supplier_totals_with_allowed['submitted_paid_ratio'] = supplier_totals_with_allowed['Suplr_Sbmtd_Chrgs'] / \
        supplier_totals_with_allowed['Suplr_Mdcr_Pymt_Amt']

    # Filter for suppliers with substantial submitted charges (at least $100,000) to focus on meaningful outliers
    significant_suppliers = supplier_totals_with_allowed[
        supplier_totals_with_allowed['Suplr_Sbmtd_Chrgs'] >= 100000]

    # Find outliers with highest submitted-to-allowed ratio
    top_submitted_allowed_outliers = significant_suppliers.sort_values(
        'submitted_allowed_ratio', ascending=False).head(10)

    print("Top 10 Suppliers with Highest Submitted Charges to Allowed Amount Ratio:\n")

    for i, (_, supplier) in enumerate(top_submitted_allowed_outliers.iterrows(), 1):
        npi = supplier['Suplr_NPI']
        name = supplier['Suplr_Prvdr_Last_Name_Org']
        submitted = supplier['Suplr_Sbmtd_Chrgs']
        allowed = supplier['Suplr_Mdcr_Alowd_Amt']
        paid = supplier['Suplr_Mdcr_Pymt_Amt']
        ratio = supplier['submitted_allowed_ratio']

        # Format amounts with K or M suffix based on size
        submitted_str = format_dollar_amount(submitted)
        allowed_str = format_dollar_amount(allowed)
        paid_str = format_dollar_amount(paid)

        print(f"{i}. **{name}** (NPI: {npi})")
        print(f"   - Submitted charges: {submitted_str}")
        print(f"   - Allowed amount: {allowed_str}")
        print(f"   - Paid amount: {paid_str}")
        print(f"   - Submitted to allowed ratio: {ratio:.2f}x")
        print(
            f"   - Allowed amount is {(allowed/submitted)*100:.1f}% of submitted charges")
        print(
            f"   - Paid amount is {(paid/submitted)*100:.1f}% of submitted charges")
        print("")  # Add a blank line between suppliers

    # Find outliers with highest submitted-to-paid ratio
    top_submitted_paid_outliers = significant_suppliers.sort_values(
        'submitted_paid_ratio', ascending=False).head(10)

    print("\nTop 10 Suppliers with Highest Submitted Charges to Paid Amount Ratio:\n")

    for i, (_, supplier) in enumerate(top_submitted_paid_outliers.iterrows(), 1):
        npi = supplier['Suplr_NPI']
        name = supplier['Suplr_Prvdr_Last_Name_Org']
        submitted = supplier['Suplr_Sbmtd_Chrgs']
        allowed = supplier['Suplr_Mdcr_Alowd_Amt']
        paid = supplier['Suplr_Mdcr_Pymt_Amt']
        ratio = supplier['submitted_paid_ratio']

        # Format amounts with K or M suffix based on size
        submitted_str = format_dollar_amount(submitted)
        allowed_str = format_dollar_amount(allowed)
        paid_str = format_dollar_amount(paid)

        print(f"{i}. **{name}** (NPI: {npi})")
        print(f"   - Submitted charges: {submitted_str}")
        print(f"   - Allowed amount: {allowed_str}")
        print(f"   - Paid amount: {paid_str}")
        print(f"   - Submitted to paid ratio: {ratio:.2f}x")
        print(
            f"   - Paid amount is {(paid/submitted)*100:.1f}% of submitted charges")
        print("")  # Add a blank line between suppliers

    # =====================================
    # Peer Group Analysis for Fraud Detection
    # =====================================
    print("\n" + "="*100)
    print("Peer Group Analysis for Fraud Detection")
    print("="*100)

    if dfs:
        # Ensure we have the required columns for analysis
        required_columns = ['Suplr_NPI', 'Suplr_Prvdr_Last_Name_Org', 'Suplr_Prvdr_Spclty_Desc',
                            'Suplr_Prvdr_State_Abrvtn', 'Suplr_Sbmtd_Chrgs', 'Suplr_Mdcr_Pymt_Amt',
                            'Tot_Suplr_Clms', 'Tot_Suplr_Srvcs']

        # Check if all required columns exist in the combined dataframe
        missing_columns = [
            col for col in required_columns if col not in combined_df.columns]
        if missing_columns:
            print(
                f"Warning: Missing columns needed for peer group analysis: {missing_columns}")
            print("Skipping peer group analysis.")
        else:
            # Calculate aggregated metrics by supplier for analysis
            supplier_metrics = combined_df.groupby(['Suplr_NPI', 'Suplr_Prvdr_Last_Name_Org',
                                                    'Suplr_Prvdr_Spclty_Desc', 'Suplr_Prvdr_State_Abrvtn']).agg({
                                                        'Suplr_Sbmtd_Chrgs': 'sum',
                                                        'Suplr_Mdcr_Pymt_Amt': 'sum',
                                                        'Tot_Suplr_Clms': 'sum',
                                                        'Tot_Suplr_Srvcs': 'sum'
                                                    }).reset_index()

            # Add derived metrics
            supplier_metrics['Avg_Chrg_Per_Clm'] = supplier_metrics['Suplr_Sbmtd_Chrgs'] / \
                supplier_metrics['Tot_Suplr_Clms']
            supplier_metrics['Avg_Pymt_Per_Clm'] = supplier_metrics['Suplr_Mdcr_Pymt_Amt'] / \
                supplier_metrics['Tot_Suplr_Clms']
            supplier_metrics['Avg_Srvcs_Per_Clm'] = supplier_metrics['Tot_Suplr_Srvcs'] / \
                supplier_metrics['Tot_Suplr_Clms']

            # 1. Analysis by Specialty
            print("\nAnalysis by Specialty:")
            print("-" * 50)

            # Get the specialties with at least 5 suppliers for meaningful comparison
            specialty_counts = supplier_metrics['Suplr_Prvdr_Spclty_Desc'].value_counts(
            )
            valid_specialties = specialty_counts[specialty_counts >= 5].index.tolist(
            )

            if valid_specialties:
                print(
                    f"Found {len(valid_specialties)} specialties with at least 5 suppliers for peer comparison.")

                # Calculate peer group metrics for each specialty
                peer_specialty_metrics = supplier_metrics[supplier_metrics['Suplr_Prvdr_Spclty_Desc'].isin(valid_specialties)].groupby(
                    'Suplr_Prvdr_Spclty_Desc').agg({
                        'Suplr_Sbmtd_Chrgs': ['median', 'mean', 'std'],
                        'Suplr_Mdcr_Pymt_Amt': ['median', 'mean', 'std'],
                        'Tot_Suplr_Clms': ['median', 'mean', 'std'],
                        'Tot_Suplr_Srvcs': ['median', 'mean', 'std'],
                        'Avg_Chrg_Per_Clm': ['median', 'mean', 'std'],
                        'Avg_Pymt_Per_Clm': ['median', 'mean', 'std'],
                        'Avg_Srvcs_Per_Clm': ['median', 'mean', 'std']
                    })

                # Find outliers within each specialty (suppliers with metrics > 3x the median)
                outliers_by_specialty = []

                for specialty in valid_specialties:
                    specialty_group = supplier_metrics[supplier_metrics['Suplr_Prvdr_Spclty_Desc'] == specialty]
                    specialty_medians = peer_specialty_metrics.loc[specialty]

                    # Check for outliers in claims, charges, and payments
                    claim_outliers = specialty_group[specialty_group['Tot_Suplr_Clms']
                                                     > 3 * specialty_medians[('Tot_Suplr_Clms', 'median')]]
                    charge_outliers = specialty_group[specialty_group['Suplr_Sbmtd_Chrgs']
                                                      > 3 * specialty_medians[('Suplr_Sbmtd_Chrgs', 'median')]]
                    payment_outliers = specialty_group[specialty_group['Suplr_Mdcr_Pymt_Amt']
                                                       > 3 * specialty_medians[('Suplr_Mdcr_Pymt_Amt', 'median')]]

                    # Find suppliers that are outliers in at least two categories
                    all_outliers = pd.concat([
                        claim_outliers[['Suplr_NPI']].assign(metric='claims'),
                        charge_outliers[['Suplr_NPI']].assign(
                            metric='charges'),
                        payment_outliers[['Suplr_NPI']].assign(
                            metric='payments')
                    ])

                    outlier_counts = all_outliers.groupby('Suplr_NPI').size()
                    multiple_outliers = outlier_counts[outlier_counts >= 2].index.tolist(
                    )

                    if multiple_outliers:
                        for npi in multiple_outliers:
                            supplier = specialty_group[specialty_group['Suplr_NPI']
                                                       == npi].iloc[0]
                            outliers_by_specialty.append({
                                'NPI': npi,
                                'Name': supplier['Suplr_Prvdr_Last_Name_Org'],
                                'Specialty': specialty,
                                'State': supplier['Suplr_Prvdr_State_Abrvtn'],
                                'Total_Claims': supplier['Tot_Suplr_Clms'],
                                'Claim_Ratio': supplier['Tot_Suplr_Clms'] / specialty_medians[('Tot_Suplr_Clms', 'median')],
                                'Total_Charges': supplier['Suplr_Sbmtd_Chrgs'],
                                'Charge_Ratio': supplier['Suplr_Sbmtd_Chrgs'] / specialty_medians[('Suplr_Sbmtd_Chrgs', 'median')],
                                'Total_Payments': supplier['Suplr_Mdcr_Pymt_Amt'],
                                'Payment_Ratio': supplier['Suplr_Mdcr_Pymt_Amt'] / specialty_medians[('Suplr_Mdcr_Pymt_Amt', 'median')]
                            })

                # Display the top outliers by specialty
                if outliers_by_specialty:
                    # Sort by highest combined ratio (sum of all ratios)
                    for outlier in sorted(outliers_by_specialty,
                                          key=lambda x: (
                                              x['Claim_Ratio'] + x['Charge_Ratio'] + x['Payment_Ratio']),
                                          reverse=True)[:10]:
                        print(
                            f"\n**{outlier['Name']}** (NPI: {outlier['NPI']})")
                        print(
                            f"  Specialty: {outlier['Specialty']} | State: {outlier['State']}")
                        print(
                            f"  Total Claims: {outlier['Total_Claims']:.0f} ({outlier['Claim_Ratio']:.1f}x specialty median)")

                        # Format monetary values
                        charges_str = format_dollar_amount(
                            outlier['Total_Charges'])
                        payments_str = format_dollar_amount(
                            outlier['Total_Payments'])

                        print(
                            f"  Total Charges: {charges_str} ({outlier['Charge_Ratio']:.1f}x specialty median)")
                        print(
                            f"  Total Payments: {payments_str} ({outlier['Payment_Ratio']:.1f}x specialty median)")
                else:
                    print("No significant specialty outliers found.")
            else:
                print(
                    "No specialties with enough suppliers for meaningful peer comparison.")

            # 2. Analysis by State
            print("\nAnalysis by State:")
            print("-" * 50)

            # Get the states with at least 5 suppliers for meaningful comparison
            state_counts = supplier_metrics['Suplr_Prvdr_State_Abrvtn'].value_counts(
            )
            valid_states = state_counts[state_counts >= 5].index.tolist()

            if valid_states:
                print(
                    f"Found {len(valid_states)} states with at least 5 suppliers for peer comparison.")

                # Calculate peer group metrics for each state
                peer_state_metrics = supplier_metrics[supplier_metrics['Suplr_Prvdr_State_Abrvtn'].isin(valid_states)].groupby(
                    'Suplr_Prvdr_State_Abrvtn').agg({
                        'Suplr_Sbmtd_Chrgs': ['median', 'mean', 'std'],
                        'Suplr_Mdcr_Pymt_Amt': ['median', 'mean', 'std'],
                        'Tot_Suplr_Clms': ['median', 'mean', 'std'],
                        'Tot_Suplr_Srvcs': ['median', 'mean', 'std'],
                        'Avg_Chrg_Per_Clm': ['median', 'mean', 'std'],
                        'Avg_Pymt_Per_Clm': ['median', 'mean', 'std'],
                        'Avg_Srvcs_Per_Clm': ['median', 'mean', 'std']
                    })

                # Find outliers within each state (suppliers with metrics > 3x the median)
                outliers_by_state = []

                for state in valid_states:
                    state_group = supplier_metrics[supplier_metrics['Suplr_Prvdr_State_Abrvtn'] == state]
                    state_medians = peer_state_metrics.loc[state]

                    # Check for outliers in claims, charges, and payments
                    claim_outliers = state_group[state_group['Tot_Suplr_Clms']
                                                 > 3 * state_medians[('Tot_Suplr_Clms', 'median')]]
                    charge_outliers = state_group[state_group['Suplr_Sbmtd_Chrgs']
                                                  > 3 * state_medians[('Suplr_Sbmtd_Chrgs', 'median')]]
                    payment_outliers = state_group[state_group['Suplr_Mdcr_Pymt_Amt']
                                                   > 3 * state_medians[('Suplr_Mdcr_Pymt_Amt', 'median')]]

                    # Find suppliers that are outliers in at least two categories
                    all_outliers = pd.concat([
                        claim_outliers[['Suplr_NPI']].assign(metric='claims'),
                        charge_outliers[['Suplr_NPI']].assign(
                            metric='charges'),
                        payment_outliers[['Suplr_NPI']].assign(
                            metric='payments')
                    ])

                    outlier_counts = all_outliers.groupby('Suplr_NPI').size()
                    multiple_outliers = outlier_counts[outlier_counts >= 2].index.tolist(
                    )

                    if multiple_outliers:
                        for npi in multiple_outliers:
                            supplier = state_group[state_group['Suplr_NPI']
                                                   == npi].iloc[0]
                            outliers_by_state.append({
                                'NPI': npi,
                                'Name': supplier['Suplr_Prvdr_Last_Name_Org'],
                                'Specialty': supplier['Suplr_Prvdr_Spclty_Desc'],
                                'State': state,
                                'Total_Claims': supplier['Tot_Suplr_Clms'],
                                'Claim_Ratio': supplier['Tot_Suplr_Clms'] / state_medians[('Tot_Suplr_Clms', 'median')],
                                'Total_Charges': supplier['Suplr_Sbmtd_Chrgs'],
                                'Charge_Ratio': supplier['Suplr_Sbmtd_Chrgs'] / state_medians[('Suplr_Sbmtd_Chrgs', 'median')],
                                'Total_Payments': supplier['Suplr_Mdcr_Pymt_Amt'],
                                'Payment_Ratio': supplier['Suplr_Mdcr_Pymt_Amt'] / state_medians[('Suplr_Mdcr_Pymt_Amt', 'median')]
                            })

                # Display the top outliers by state
                if outliers_by_state:
                    # Sort by highest combined ratio (sum of all ratios)
                    for outlier in sorted(outliers_by_state,
                                          key=lambda x: (
                                              x['Claim_Ratio'] + x['Charge_Ratio'] + x['Payment_Ratio']),
                                          reverse=True)[:10]:
                        print(
                            f"\n**{outlier['Name']}** (NPI: {outlier['NPI']})")
                        print(
                            f"  State: {outlier['State']} | Specialty: {outlier['Specialty']}")
                        print(
                            f"  Total Claims: {outlier['Total_Claims']:.0f} ({outlier['Claim_Ratio']:.1f}x state median)")

                        # Format monetary values
                        charges_str = format_dollar_amount(
                            outlier['Total_Charges'])
                        payments_str = format_dollar_amount(
                            outlier['Total_Payments'])

                        print(
                            f"  Total Charges: {charges_str} ({outlier['Charge_Ratio']:.1f}x state median)")
                        print(
                            f"  Total Payments: {payments_str} ({outlier['Payment_Ratio']:.1f}x state median)")
                else:
                    print("No significant state outliers found.")
            else:
                print("No states with enough suppliers for meaningful peer comparison.")

            # 3. Combined specialty-state analysis for the most precise peer grouping
            print("\nAnalysis by Combined Specialty-State Groups:")
            print("-" * 50)

            # Create specialty-state combination for more precise peer groups
            supplier_metrics['Specialty_State'] = supplier_metrics['Suplr_Prvdr_Spclty_Desc'] + \
                ' - ' + supplier_metrics['Suplr_Prvdr_State_Abrvtn']

            # Get specialty-state combinations with at least 5 suppliers
            specialty_state_counts = supplier_metrics['Specialty_State'].value_counts(
            )
            valid_specialty_states = specialty_state_counts[specialty_state_counts >= 5].index.tolist(
            )

            if valid_specialty_states:
                print(
                    f"Found {len(valid_specialty_states)} specialty-state combinations with at least 5 suppliers.")

                # Calculate metrics for each specialty-state combination
                peer_combined_metrics = supplier_metrics[supplier_metrics['Specialty_State'].isin(valid_specialty_states)].groupby(
                    'Specialty_State').agg({
                        'Suplr_Sbmtd_Chrgs': ['median', 'mean', 'std'],
                        'Suplr_Mdcr_Pymt_Amt': ['median', 'mean', 'std'],
                        'Tot_Suplr_Clms': ['median', 'mean', 'std'],
                        'Tot_Suplr_Srvcs': ['median', 'mean', 'std'],
                        'Avg_Chrg_Per_Clm': ['median', 'mean', 'std'],
                        'Avg_Pymt_Per_Clm': ['median', 'mean', 'std'],
                        'Avg_Srvcs_Per_Clm': ['median', 'mean', 'std']
                    })

                # Find outliers within each specialty-state group
                outliers_combined = []

                for group in valid_specialty_states:
                    combined_group = supplier_metrics[supplier_metrics['Specialty_State'] == group]
                    combined_medians = peer_combined_metrics.loc[group]

                    # Check for outliers in claims, charges, and payments
                    claim_outliers = combined_group[combined_group['Tot_Suplr_Clms']
                                                    > 3 * combined_medians[('Tot_Suplr_Clms', 'median')]]
                    charge_outliers = combined_group[combined_group['Suplr_Sbmtd_Chrgs']
                                                     > 3 * combined_medians[('Suplr_Sbmtd_Chrgs', 'median')]]
                    payment_outliers = combined_group[combined_group['Suplr_Mdcr_Pymt_Amt']
                                                      > 3 * combined_medians[('Suplr_Mdcr_Pymt_Amt', 'median')]]

                    # Find suppliers that are outliers in at least two categories
                    all_outliers = pd.concat([
                        claim_outliers[['Suplr_NPI']].assign(metric='claims'),
                        charge_outliers[['Suplr_NPI']].assign(
                            metric='charges'),
                        payment_outliers[['Suplr_NPI']].assign(
                            metric='payments')
                    ])

                    outlier_counts = all_outliers.groupby('Suplr_NPI').size()
                    multiple_outliers = outlier_counts[outlier_counts >= 2].index.tolist(
                    )

                    if multiple_outliers:
                        for npi in multiple_outliers:
                            supplier = combined_group[combined_group['Suplr_NPI']
                                                      == npi].iloc[0]
                            outliers_combined.append({
                                'NPI': npi,
                                'Name': supplier['Suplr_Prvdr_Last_Name_Org'],
                                'Specialty': supplier['Suplr_Prvdr_Spclty_Desc'],
                                'State': supplier['Suplr_Prvdr_State_Abrvtn'],
                                'Group': group,
                                'Total_Claims': supplier['Tot_Suplr_Clms'],
                                'Claim_Ratio': supplier['Tot_Suplr_Clms'] / combined_medians[('Tot_Suplr_Clms', 'median')],
                                'Total_Charges': supplier['Suplr_Sbmtd_Chrgs'],
                                'Charge_Ratio': supplier['Suplr_Sbmtd_Chrgs'] / combined_medians[('Suplr_Sbmtd_Chrgs', 'median')],
                                'Total_Payments': supplier['Suplr_Mdcr_Pymt_Amt'],
                                'Payment_Ratio': supplier['Suplr_Mdcr_Pymt_Amt'] / combined_medians[('Suplr_Mdcr_Pymt_Amt', 'median')]
                            })

                # Display the top outliers by combined group
                if outliers_combined:
                    print(
                        "\nMost Significant Outliers by Combined Specialty-State Group:")
                    # Sort by highest combined ratio (sum of all ratios)
                    for outlier in sorted(outliers_combined,
                                          key=lambda x: (
                                              x['Claim_Ratio'] + x['Charge_Ratio'] + x['Payment_Ratio']),
                                          reverse=True)[:10]:
                        print(
                            f"\n**{outlier['Name']}** (NPI: {outlier['NPI']})")
                        print(
                            f"  Specialty: {outlier['Specialty']} | State: {outlier['State']}")
                        print(
                            f"  Total Claims: {outlier['Total_Claims']:.0f} ({outlier['Claim_Ratio']:.1f}x peer group median)")

                        # Format monetary values
                        charges_str = format_dollar_amount(
                            outlier['Total_Charges'])
                        payments_str = format_dollar_amount(
                            outlier['Total_Payments'])

                        print(
                            f"  Total Charges: {charges_str} ({outlier['Charge_Ratio']:.1f}x peer group median)")
                        print(
                            f"  Total Payments: {payments_str} ({outlier['Payment_Ratio']:.1f}x peer group median)")
                else:
                    print("No significant combined specialty-state outliers found.")
            else:
                print(
                    "No specialty-state combinations with enough suppliers for meaningful peer comparison.")
else:
    print("No data was loaded. Please check if the CSV files exist.")


print("Stopping here")
