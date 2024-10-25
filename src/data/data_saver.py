"""
Data saving for tabular data
"""

import os
from datetime import datetime


def saveCsvWithDateTime(data, path):
    if not os.path.exists(path):
        os.makedirs(path)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    filename = os.path.join(path, timestamp + ".csv")
    data.to_csv(filename, index=False)
