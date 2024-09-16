"""
Testing for data loader
"""

import pandas as pd

from data.data_loader import loadDataCsv, loadDataParquet


def test_loadDataCsv(mocker):
    mock_glob = mocker.patch("glob.glob")
    mock_read_csv = mocker.patch("pandas.read_csv")
    mock_glob.return_value = ["file1.csv", "file2.csv"]
    mock_read_csv.side_effect = [
        pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        pd.DataFrame({"A": [5, 6], "B": [7, 8]}),
    ]

    result = loadDataCsv("mock_path", "-")

    expected = pd.DataFrame({"A": [1, 2, 5, 6], "B": [3, 4, 7, 8]})
    pd.testing.assert_frame_equal(result, expected)


def test_loadDataParquet(mocker):
    mock_glob = mocker.patch("glob.glob")
    mock_read_parquet = mocker.patch("pandas.read_parquet")

    mock_glob.return_value = ["file1.parquet", "file2.parquet"]
    mock_read_parquet.side_effect = [
        pd.DataFrame({"X": [10, 20], "Y": [30, 40]}),
        pd.DataFrame({"X": [50, 60], "Y": [70, 80]}),
    ]

    result = loadDataParquet("mock_path")

    expected = pd.DataFrame({"X": [10, 20, 50, 60], "Y": [30, 40, 70, 80]})
    pd.testing.assert_frame_equal(result, expected)
