#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DME Data Dictionary
This module contains the data dictionary for the DME dataset.
"""

# Data dictionary mapping variable names to their descriptions
DATA_DICTIONARY = {
    # Supplier Information
    'Suplr_NPI': "National Provider Identifier for the DME supplier",
    'Suplr_Prvdr_Last_Name_Org': "Organization name of the DME supplier",
    'Suplr_Prvdr_First_Name': "First name of the DME supplier (if individual)",
    'Suplr_Prvdr_MI': "Middle initial of the DME supplier (if individual)",
    'Suplr_Prvdr_Crdntls': "Credentials of the DME supplier",
    'Suplr_Prvdr_Gndr': "Gender of the DME supplier (if individual)",
    'Suplr_Prvdr_Ent_Cd': "Entity code of the DME supplier",
    'Suplr_Prvdr_St1': "Street address line 1 of the DME supplier",
    'Suplr_Prvdr_St2': "Street address line 2 of the DME supplier",
    'Suplr_Prvdr_City': "City where the DME supplier is located",
    'Suplr_Prvdr_State_Abrvtn': "State abbreviation where the DME supplier is located",
    'Suplr_Prvdr_State_FIPS': "FIPS code for the state where the DME supplier is located",
    'Suplr_Prvdr_Zip5': "5-digit ZIP code where the DME supplier is located",
    'Suplr_Prvdr_Cntry': "Country where the DME supplier is located",
    'Suplr_Prvdr_RUCA': "Rural-Urban Commuting Area code for the DME supplier",
    'Suplr_Prvdr_RUCA_Desc': "Description of the Rural-Urban Commuting Area for the DME supplier",
    'Suplr_Prvdr_Spclty_Desc': "Specialty description of the DME supplier",
    'Suplr_Prvdr_Spclty_Srce': "Source of the specialty information",

    # DME-specific fields
    'DME_Sprsn_Ind': "Indicator for suppression of DME data (Y/N)",
    'DME_Tot_Suplr_Benes': "Total number of beneficiaries served by the supplier for DME",
    'DME_Tot_Suplr_Clms': "Total number of claims submitted by the supplier for DME",
    'DME_Tot_Suplr_Srvcs': "Total number of services provided by the supplier for DME",
    'DME_Tot_Suplr_HCPCS_Cds': "Total number of unique HCPCS codes billed by the supplier for DME",
    'DME_Suplr_Sbmtd_Chrgs': "Total submitted charges by the supplier for DME",
    'DME_Suplr_Mdcr_Alowd_Amt': "Total Medicare allowed amount for the supplier for DME",
    'DME_Suplr_Mdcr_Pymt_Amt': "Total Medicare payment amount to the supplier for DME",
    'DME_Suplr_Mdcr_Stdzd_Pymt_Amt': "Total Medicare standardized payment amount to the supplier for DME",

    # Prosthetic and Orthotic fields
    'POS_Sprsn_Ind': "Indicator for suppression of POS data (Y/N)",
    'POS_Tot_Suplr_Benes': "Total number of beneficiaries served by the supplier for POS",
    'POS_Tot_Suplr_Clms': "Total number of claims submitted by the supplier for POS",
    'POS_Tot_Suplr_Srvcs': "Total number of services provided by the supplier for POS",
    'POS_Tot_Suplr_HCPCS_Cds': "Total number of unique HCPCS codes billed by the supplier for POS",
    'POS_Suplr_Sbmtd_Chrgs': "Total submitted charges by the supplier for POS",
    'POS_Suplr_Mdcr_Alowd_Amt': "Total Medicare allowed amount for the supplier for POS",
    'POS_Suplr_Mdcr_Pymt_Amt': "Total Medicare payment amount to the supplier for POS",
    'POS_Suplr_Mdcr_Stdzd_Pymt_Amt': "Total Medicare standardized payment amount to the supplier for POS",

    # Drug and Nutritional fields
    'Drug_Sprsn_Ind': "Indicator for suppression of Drug data (Y/N)",
    'Drug_Tot_Suplr_Benes': "Total number of beneficiaries served by the supplier for Drug",
    'Drug_Tot_Suplr_Clms': "Total number of claims submitted by the supplier for Drug",
    'Drug_Tot_Suplr_Srvcs': "Total number of services provided by the supplier for Drug",
    'Drug_Tot_Suplr_HCPCS_Cds': "Total number of unique HCPCS codes billed by the supplier for Drug",
    'Drug_Suplr_Sbmtd_Chrgs': "Total submitted charges by the supplier for Drug",
    'Drug_Suplr_Mdcr_Alowd_Amt': "Total Medicare allowed amount for the supplier for Drug",
    'Drug_Suplr_Mdcr_Pymt_Amt': "Total Medicare payment amount to the supplier for Drug",
    'Drug_Suplr_Mdcr_Stdzd_Pymt_Amt': "Total Medicare standardized payment amount to the supplier for Drug",

    # Overall supplier fields
    'Tot_Suplr_Benes': "Total number of beneficiaries served by the supplier overall",
    'Tot_Suplr_Clms': "Total number of claims submitted by the supplier overall",
    'Tot_Suplr_Srvcs': "Total number of services provided by the supplier overall",
    'Tot_Suplr_HCPCS_Cds': "Total number of unique HCPCS codes billed by the supplier overall",
    'Suplr_Sbmtd_Chrgs': "Total submitted charges by the supplier overall",
    'Suplr_Mdcr_Alowd_Amt': "Total Medicare allowed amount for the supplier overall",
    'Suplr_Mdcr_Pymt_Amt': "Total Medicare payment amount to the supplier overall",
    'Suplr_Mdcr_Stdzd_Pymt_Amt': "Total Medicare standardized payment amount to the supplier overall",

    # Beneficiary Demographics
    'Bene_Avg_Age': "Average age of beneficiaries served by this supplier",
    'Bene_Age_LT_65_Cnt': "Count of beneficiaries under 65 years of age",
    'Bene_Age_65_74_Cnt': "Count of beneficiaries 65-74 years of age",
    'Bene_Age_75_84_Cnt': "Count of beneficiaries 75-84 years of age",
    'Bene_Age_GT_84_Cnt': "Count of beneficiaries greater than 84 years of age",
    'Bene_Male_Cnt': "Count of male beneficiaries",
    'Bene_Feml_Cnt': "Count of female beneficiaries",
    'Bene_Race_Wht_Cnt': "Count of white beneficiaries",
    'Bene_Race_Black_Cnt': "Count of Black or African American beneficiaries",
    'Bene_Race_Api_Cnt': "Count of Asian/Pacific Islander beneficiaries",
    'Bene_Race_Hspnc_Cnt': "Count of Hispanic beneficiaries",
    'Bene_Race_Natind_Cnt': "Count of Native American/Alaska Native beneficiaries",
    'Bene_Race_Othr_Cnt': "Count of beneficiaries of other races",
    'Bene_Dual_Cnt': "Count of dual-eligible beneficiaries (Medicare and Medicaid)",
    'Bene_Ndual_Cnt': "Count of non-dual-eligible beneficiaries",
    'Bene_Avg_Risk_Scre': "Average risk score of beneficiaries",

    # Health Conditions
    'Bene_CC_PH_Hypertension_V2_Pct': "Percentage of beneficiaries with hypertension",
    'Bene_CC_PH_Hyperlipidemia_V2_Pct': "Percentage of beneficiaries with hyperlipidemia",
    'Bene_CC_PH_Diabetes_V2_Pct': "Percentage of beneficiaries with diabetes",
    'Bene_CC_PH_Arthritis_V2_Pct': "Percentage of beneficiaries with arthritis",
    'Bene_CC_PH_IschemicHeart_V2_Pct': "Percentage of beneficiaries with ischemic heart disease",
    'Bene_CC_PH_COPD_V2_Pct': "Percentage of beneficiaries with COPD",
    'Bene_CC_PH_CKD_V2_Pct': "Percentage of beneficiaries with chronic kidney disease",
    'Bene_CC_PH_Cancer6_V2_Pct': "Percentage of beneficiaries with cancer",
    'Bene_CC_PH_Asthma_V2_Pct': "Percentage of beneficiaries with asthma",
    'Bene_CC_PH_Afib_V2_Pct': "Percentage of beneficiaries with atrial fibrillation",
    'Bene_CC_PH_HF_NonIHD_V2_Pct': "Percentage of beneficiaries with heart failure",
    'Bene_CC_PH_Stroke_TIA_V2_Pct': "Percentage of beneficiaries with stroke/TIA",
    'Bene_CC_PH_Osteoporosis_V2_Pct': "Percentage of beneficiaries with osteoporosis",
    'Bene_CC_PH_Parkinson_V2_Pct': "Percentage of beneficiaries with Parkinson's disease",
    'Bene_CC_BH_Mood_V2_Pct': "Percentage of beneficiaries with mood disorders",
    'Bene_CC_BH_Depress_V1_Pct': "Percentage of beneficiaries with depression",
    'Bene_CC_BH_Anxiety_V1_Pct': "Percentage of beneficiaries with anxiety",
    'Bene_CC_BH_Tobacco_V1_Pct': "Percentage of beneficiaries with tobacco use disorder",
    'Bene_CC_BH_Alz_NonAlzdem_V2_Pct': "Percentage of beneficiaries with Alzheimer's/dementia",
    'Bene_CC_BH_Schizo_OthPsy_V1_Pct': "Percentage of beneficiaries with schizophrenia or other psychotic disorders",
    'Bene_CC_BH_Alcohol_Drug_V1_Pct': "Percentage of beneficiaries with alcohol/drug use disorders",
    'Bene_CC_BH_ADHD_OthCD_V1_Pct': "Percentage of beneficiaries with ADHD",
    'Bene_CC_BH_Bipolar_V1_Pct': "Percentage of beneficiaries with bipolar disorder",
    'Bene_CC_BH_PD_V1_Pct': "Percentage of beneficiaries with personality disorders",
    'Bene_CC_BH_PTSD_V1_Pct': "Percentage of beneficiaries with PTSD"
}


def get_column_category(column_name):
    """
    Determine the category of a column based on its name.

    Parameters:
    -----------
    column_name : str
        The name of the column

    Returns:
    --------
    category : str
        The category of the column
    """
    if column_name.startswith('Suplr_Prvdr_'):
        return 'Supplier Information'
    elif column_name.startswith('Suplr_') and not column_name.startswith('Suplr_Prvdr_'):
        return 'Overall Supplier Metrics'
    elif column_name.startswith('DME_'):
        return 'Durable Medical Equipment'
    elif column_name.startswith('POS_'):
        return 'Prosthetics and Orthotics'
    elif column_name.startswith('Drug_'):
        return 'Drug and Nutritional Products'
    elif column_name.startswith('Bene_'):
        if any(x in column_name for x in ['_CC_', 'Risk']):
            return 'Health Conditions'
        else:
            return 'Beneficiary Demographics'
    elif column_name.startswith('Tot_'):
        return 'Overall Provider Metrics'
    else:
        return 'Other'
