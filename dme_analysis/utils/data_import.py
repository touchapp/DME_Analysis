#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DME Data Import Utilities
This module contains functions for importing DME data.
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict


def import_dme_data(file_path):
    """
    Import and preprocess DME data from a CSV file.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing DME data

    Returns:
    --------
    df : DataFrame
        Processed DataFrame containing DME data
    """
    print(f"Importing data from {file_path}...")

    try:
        # Import data with appropriate dtypes to handle monetary values correctly
        df = pd.read_csv(file_path, low_memory=False)

        # Convert monetary columns to numeric
        money_columns = [col for col in df.columns if any(
            x in col for x in ['Pymt', 'Amt', 'Chrgs'])]
        for col in money_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"Successfully imported data with shape: {df.shape}")
        return df

    except Exception as e:
        print(f"Error importing data: {str(e)}")
        return None


def get_column_mapping(df):
    """
    Get a mapping of expected column names to actual column names in the DataFrame.
    This helps handle variations in column names across different datasets.

    Parameters:
    -----------
    df : DataFrame
        DataFrame to inspect for column names

    Returns:
    --------
    column_map : dict
        Dictionary mapping expected column names to actual column names
    """
    column_map = {}

    # Map for supplier organization name
    if 'Suplr_Prvdr_Last_Name_Org' in df.columns:
        column_map['supplier_name'] = 'Suplr_Prvdr_Last_Name_Org'
    elif 'Suplr_Prvdr_Org_Name' in df.columns:
        column_map['supplier_name'] = 'Suplr_Prvdr_Org_Name'
    elif 'Suplr_Name' in df.columns:
        column_map['supplier_name'] = 'Suplr_Name'
    elif 'Supplier_Name' in df.columns:
        column_map['supplier_name'] = 'Supplier_Name'
    else:
        # If no suitable column exists, create a placeholder
        print("Warning: No supplier name column found. Using placeholder names.")
        column_map['supplier_name'] = None

    # Map for supplier state
    if 'Suplr_Prvdr_State_Abrvtn' in df.columns:
        column_map['supplier_state'] = 'Suplr_Prvdr_State_Abrvtn'
    elif 'Suplr_State' in df.columns:
        column_map['supplier_state'] = 'Suplr_State'
    elif 'State' in df.columns:
        column_map['supplier_state'] = 'State'
    else:
        print("Warning: No supplier state column found. Using placeholder.")
        column_map['supplier_state'] = None

    # Map for supplier NPI
    if 'Suplr_NPI' in df.columns:
        column_map['supplier_npi'] = 'Suplr_NPI'
    elif 'NPI' in df.columns:
        column_map['supplier_npi'] = 'NPI'
    elif 'Provider_NPI' in df.columns:
        column_map['supplier_npi'] = 'Provider_NPI'
    else:
        print("Warning: No NPI column found. Using index as placeholder.")
        column_map['supplier_npi'] = None

    # Check if key columns are missing
    if None in column_map.values():
        print("\nAvailable columns in the dataset:")
        # Show first 20 columns
        for i, col in enumerate(sorted(df.columns)[:20]):
            print(f"  {i+1}. {col}")

        if len(df.columns) > 20:
            print(f"  ... and {len(df.columns) - 20} more columns")

    return column_map


def import_data_for_years(years_range, base_path="data"):
    """
    Import data for multiple years.

    Parameters:
    -----------
    years_range : range or list
        Range or list of years to import (e.g., range(2017, 2023))
    base_path : str
        Base path where data files are stored

    Returns:
    --------
    df_by_year : dict
        Dictionary with years as keys and DataFrames as values
    """
    df_by_year = {}

    for year in years_range:
        # Try different file name patterns
        file_patterns = [
            f"{base_path}/{year}/mup_dme_ry24_p05_v10_dy{str(year)[-2:]}_supr.csv",
            f"{base_path}/dme_data_{year}.csv",
            f"{base_path}/{year}/dme_data_{year}.csv",
            f"dme_data_{year}.csv"
        ]

        file_found = False
        for file_path in file_patterns:
            if os.path.exists(file_path):
                df = import_dme_data(file_path)
                if df is not None:
                    df_by_year[year] = df
                    file_found = True
                    break

        if not file_found:
            print(f"Warning: No data file found for {year}")

    return df_by_year
