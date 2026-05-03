"""
ALY 6140 Module 2 Assignment
Copyright (c) 2021 -- This is the 2021 Spring B version of the Template
Licensed
Written by Jeff Hackmeister

# you can also rely on the docstring documentation from pandas on how to format docstrings:
# https://pandas.pydata.org/pandas-docs/stable/development/contributing_docstring.html
"""

import pandas as pd
import json


def import_csv(path):
    """
    :param path: direct file path to the CSV file
    :return: pandas DataFrame containing the CSV data
    """
    csv_data = pd.read_csv(filepath_or_buffer=path, sep=',', header=0, encoding='utf-8')
    return csv_data


def import_text(path):
    """
    :param path: direct file path to the pipe-delimited text file
    :return: pandas DataFrame containing the text file data
    """
    text_data = pd.read_csv(filepath_or_buffer=path, sep='|', header=0, encoding='utf-8')
    return text_data


def import_json(path):
    """
    :param path: direct file path to the JSON file
    :return: pandas DataFrame parsed from a JSON list of records
    """
    with open(path, mode='r', encoding='utf-8') as file:
        raw = json.load(file)
    json_data = pd.json_normalize(data=raw)
    return json_data


def import_excel(path, skiprows=24):
    """
    :param path: direct file path to the Excel (.xlsx) file
    :param skiprows: number of rows to skip before the header row (default 24)
    :return: pandas DataFrame containing the Excel data, columns M through AR
    """
    excel_data = pd.read_excel(
        io=path,
        sheet_name='financials',
        header=0,
        skiprows=skiprows,
        usecols='M:AR',
        engine='openpyxl'
    )
    # Drop columns that are entirely empty
    excel_data = excel_data.dropna(axis=1, how='all')
    return excel_data


if __name__ == '__main__':
    csv_data   = import_csv(r'C:\Users\jhack\OneDrive\Northeastern\ALY6140 - Analytics Systems Technology (Python)\Module2\Neural_data.csv')
    text_data  = import_text(r'C:\Users\jhack\OneDrive\Northeastern\ALY6140 - Analytics Systems Technology (Python)\Module2\network_data.txt')
    json_data  = import_json(r'C:\Users\jhack\OneDrive\Northeastern\ALY6140 - Analytics Systems Technology (Python)\Module2\nested_data.json')
    excel_data = import_excel(r'C:\Users\jhack\OneDrive\Northeastern\ALY6140 - Analytics Systems Technology (Python)\Module2\Excel_report.xlsx')
