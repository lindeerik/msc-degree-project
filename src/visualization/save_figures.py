"""
Saving figures from Matplotlib
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt


def saveFiguresWithDateTime(path):
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    folder = os.path.join(path, timestamp)
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, figNum in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(figNum)
        fileName = f"figure{i}.png"
        filePath = os.path.join(folder, fileName)
        fig.savefig(filePath)
        print(f"Saved {fileName} to {folder}")
