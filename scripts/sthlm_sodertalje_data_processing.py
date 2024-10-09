"""
Script for cleaning raw data recorded between Stockholm and Södertälje
"""

import pandas as pd

def main():
    dirCsv = "data/raw/sthlm-sodertalje/"
    fileName = "2024.10.04_11.19.11.csv"
    dfRaw = pd.read_csv(dirCsv + fileName, na_values=["", "-"])
    df = dfRaw.dropna(axis=1, how="all")

    # drop rows due to stops in iperf test at 11:40, 11:55, 12:22, 12:35
    # also iperf testing began 10 points late and ends at 12:44
    indicesToDropTestStops = (
        list(range(0, 10))
        + list(range(2160, 2176))
        + list(range(3939, 3949))
        + list(range(6508, 6553))
        + list(range(8063, 8084))
        + list(range(8689, len(df)))
    )

    # drop rows due to tunnels at 11:21, 11:25
    indicesToDropTunnels = list(range(269, 363)) + list(range(530, 711))
    dfClean = df.drop(df.index[indicesToDropTestStops + indicesToDropTunnels])

    dirClean = "data/intermediate/sthlm-sodertalje/"
    dfClean.to_csv(dirClean + fileName)

main()
