#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DME Fraud Detection Script
This script analyzes Medicare DME supplier data to identify potential fraud indicators,
with a focus on suspicious growth patterns similar to credit card fraud detection techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import sys

# Import from the new module structure
from dme_analysis.utils import (
    DATA_DICTIONARY,
    import_dme_data,
    get_column_mapping,
    get_column_category,
    import_data_for_years
)


def detect_high_growth_suppliers(df_by_year, metric='DME_Suplr_Mdcr_Pymt_Amt', top_n=50):
    """
    Identify suppliers with abnormally high growth rates year over year.

    Parameters:
    -----------
    df_by_year : dict
        Dictionary containing DataFrames by year
    metric : str
        The metric to analyze for growth (default: Medicare payments)
    top_n : int
        Number of top growth suppliers to identify

    Returns:
    --------
    growth_df : DataFrame
        DataFrame containing suppliers with their growth rates
    """
    print(f"Identifying suppliers with highest year-over-year growth rates...")

    # Check if metric exists in all dataframes
    for year, df in df_by_year.items():
        if metric not in df.columns:
            available_metrics = [
                col for col in df.columns if 'Pymt' in col or 'Amt' in col]
            if not available_metrics:
                print(
                    f"Error: No payment metrics found in data for year {year}.")
                return pd.DataFrame()

            # Use the first available payment metric
            metric = available_metrics[0]
            print(f"Using alternate metric: {metric}")
            break

    # Get all available years
    years = sorted(df_by_year.keys())

    if len(years) < 2:
        print("Error: Need at least two years of data to calculate growth rates")
        return pd.DataFrame()

    # Get column mappings from the most recent year's data
    recent_year = max(years)
    column_map = get_column_mapping(df_by_year[recent_year])

    # Create a dictionary to store supplier data across years
    supplier_data = {}
    supplier_info = {}

    # Process each supplier's data for each year
    for year in years:
        df = df_by_year[year]

        # Get NPI column name
        npi_col = column_map['supplier_npi']
        if npi_col is None:
            # Create a synthetic NPI using index
            df['synthetic_npi'] = 'NPI' + df.index.astype(str)
            npi_col = 'synthetic_npi'

        # Group by supplier NPI and sum the metric
        supplier_metric = df.groupby(npi_col)[metric].sum().reset_index()

        # Store in dictionary
        for _, row in supplier_metric.iterrows():
            npi = row[npi_col]
            value = row[metric]

            if npi not in supplier_data:
                supplier_data[npi] = {}

                # Store supplier info for later use
                supplier_row = df[df[npi_col] == npi].iloc[0] if len(
                    df[df[npi_col] == npi]) > 0 else None
                if supplier_row is not None:
                    supplier_info[npi] = {
                        'name': supplier_row[column_map['supplier_name']] if column_map['supplier_name'] is not None else f"Supplier {npi}",
                        'state': supplier_row[column_map['supplier_state']] if column_map['supplier_state'] is not None else 'Unknown'
                    }
                else:
                    supplier_info[npi] = {
                        'name': f"Supplier {npi}",
                        'state': 'Unknown'
                    }

            supplier_data[npi][year] = value

    # Calculate year-over-year growth rates
    growth_data = []

    for npi, year_values in supplier_data.items():
        # Need at least two years of data for this supplier
        if len(year_values) < 2:
            continue

        for i in range(len(years) - 1):
            current_year = years[i]
            next_year = years[i + 1]

            # Skip if supplier doesn't have data for both years
            if current_year not in year_values or next_year not in year_values:
                continue

            current_value = year_values[current_year]
            next_value = year_values[next_year]

            # Skip if current value is zero (would result in infinity growth)
            if current_value == 0:
                continue

            # Calculate growth rate
            growth_rate = ((next_value - current_value) / current_value) * 100

            growth_data.append({
                'Supplier NPI': npi,
                'Supplier Name': supplier_info[npi]['name'],
                'Supplier State': supplier_info[npi]['state'],
                'Year Period': f"{current_year}-{next_year}",
                'Start Year Value': current_value,
                'End Year Value': next_value,
                'Growth Rate (%)': growth_rate,
                'Absolute Growth': next_value - current_value
            })

    # Convert to DataFrame
    growth_df = pd.DataFrame(growth_data)

    # Sort by growth rate (descending)
    growth_df = growth_df.sort_values('Growth Rate (%)', ascending=False)

    return growth_df.head(top_n)


def plot_high_growth_suppliers(growth_df, top_n=20):
    """
    Create a visualization of suppliers with highest growth rates.

    Parameters:
    -----------
    growth_df : DataFrame
        DataFrame from detect_high_growth_suppliers function
    top_n : int
        Number of top suppliers to visualize

    Returns:
    --------
    fig : Figure
        Matplotlib figure object containing the visualization
    """
    # Take top N suppliers
    plot_df = growth_df.head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot horizontal bar chart
    bars = sns.barplot(
        x='Growth Rate (%)',
        y='Supplier Name',
        data=plot_df,
        palette='viridis',
        ax=ax
    )

    # Add value labels
    for i, bar in enumerate(bars.patches):
        value = plot_df.iloc[i]['Growth Rate (%)']
        ax.text(
            bar.get_width() + 10,
            bar.get_y() + bar.get_height()/2,
            f"{value:,.1f}%",
            ha='left',
            va='center',
            fontweight='bold'
        )

    # Add a second x-axis for absolute growth
    ax2 = ax.twiny()
    ax2.set_xlabel('Absolute Growth ($)', color='red')
    ax2.tick_params(axis='x', colors='red')

    # Plot absolute growth as scatter points
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax2.scatter(row['Absolute Growth'], i, color='red', s=100, alpha=0.7)

    # Format the x-axis for absolute growth with dollar amounts
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f'${x:,.0f}'))

    # Set labels and title
    ax.set_xlabel('Growth Rate (%)', fontsize=14)
    ax.set_ylabel('Supplier', fontsize=14)
    ax.set_title(f'Top {top_n} Suppliers by Growth Rate',
                 fontsize=16, fontweight='bold')

    # Add year period information
    if not plot_df.empty:
        year_periods = plot_df['Year Period'].unique()
        period_str = ', '.join(year_periods)
        ax.text(
            0.5, 1.05, f"Year Period(s): {period_str}", transform=ax.transAxes, ha='center')

    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig


def main():
    """Main function to import and analyze DME data for fraud detection."""
    print("DME Fraud Detection Analysis")
    print("===========================\n")

    # Import data for years 2017-2022 using the utility function
    df_by_year = import_data_for_years(range(2017, 2023))

    if not df_by_year:
        print("\nError: No data files were successfully imported. Cannot proceed with analysis.")
        return {}, {}

    print(f"\n{len(df_by_year)} year(s) of data imported.")

    # ----- FRAUD DETECTION ANALYSIS -----
    print("\n1. High Growth Rate Analysis")
    print("--------------------------\n")

    # Detect suppliers with abnormally high growth rates
    growth_df = detect_high_growth_suppliers(df_by_year, top_n=50)

    if growth_df.empty:
        print("No suppliers with high growth rates detected.")
        return df_by_year, {}, {}

    # Print summary of top 15 high-growth suppliers
    print("Top 15 suppliers with highest growth rates:")

    # Format the output for display
    formatted_growth_df = growth_df.head(15).copy()
    formatted_growth_df['Growth Rate (%)'] = formatted_growth_df['Growth Rate (%)'].apply(
        lambda x: f"{x:.2f}%")
    formatted_growth_df['Start Year Value'] = formatted_growth_df['Start Year Value'].apply(
        lambda x: f"${x:,.2f}")
    formatted_growth_df['End Year Value'] = formatted_growth_df['End Year Value'].apply(
        lambda x: f"${x:,.2f}")
    formatted_growth_df['Absolute Growth'] = formatted_growth_df['Absolute Growth'].apply(
        lambda x: f"${x:,.2f}")

    print(formatted_growth_df.to_string(index=False))

    # ----- VISUALIZATIONS -----
    print("\n2. Generating Fraud Detection Visualizations")
    print("------------------------------------------\n")

    # Setting plot style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = [14, 9]

    # Generate visualization
    growth_fig = plot_high_growth_suppliers(growth_df, top_n=20)
    visualizations = {'high_growth_suppliers': growth_fig}
    data = {'high_growth_suppliers': growth_df}

    # Save visualizations to files if not in a notebook environment
    try:
        # Check if we're in a notebook environment
        if 'ipykernel' not in sys.modules:
            print("\nSaving visualizations to files...")
            os.makedirs('fraud_visualizations', exist_ok=True)
            for name, fig in visualizations.items():
                fig.savefig(
                    f'fraud_visualizations/{name}.png', dpi=300, bbox_inches='tight')
                print(f"Saved: fraud_visualizations/{name}.png")
    except Exception as e:
        print(f"Error saving visualizations: {str(e)}")
        print("Note: Visualizations will be displayed if run in a Jupyter notebook")

    # When run in Jupyter, the figures will be displayed inline
    return df_by_year, visualizations, data


if __name__ == "__main__":
    main()
