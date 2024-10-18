"""
Data saving for tabular data
"""

from datetime import datetime



def saveCsvWithDateTime(data, path):
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%SZ')
    filename = f"{path}{timestamp}.csv"
    data.to_csv(filename, index=False)
