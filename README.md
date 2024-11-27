# msc-degree-project

This repository contains code for training prediction models on datasets with connectivity parameters, although it is kept general and may be used for other data. Predictions include point-wise estimation and uncertaintly intervals with quantile regression and conformal prediction. Point estimation models include Random Forest from scikit-learn, XGBoost from xgboost, and neural networks with PyTorch. Quantile models are made with Pytorch and sklearn-quantile. Conformal models have a custom implementation. How these are implemented can be referenced to the following articles:

R. Koenker and G. Bassett Jr, “Regression quantiles”, Econometrica: journal of the
Econometric Society, pp. 33–50, 1978.

N. Meinshausen and G. Ridgeway, “Quantile regression forests.”, Journal of machine
learning research, vol. 7, no. 6, 2006.

A. N. Angelopoulos and S. Bates, “Conformal prediction: A gentle introduction”, Found.
Trends Mach. Learn., vol. 16, no. 4, pp. 494–591, 2023, ISSN: 1935-8237. DOI: 10.
1561/2200000101. [Online]. Available: https://doi.org/10.1561/2200000101.

# Quickstart
Assuming the installation is carried out correctly, experiments and results can be generated direclty. As a starting point, the `demo.ipynb` in the `notebooks` directory is recommended. To run a script in the `scripts` folder, you may use 

```shell
python scripts/script_name.py
```


# Table of Contents

- [MSc Degree Project](#msc-degree-project)
- [Quickstart/Demo](#quickstart)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)

# Installation
Begin by cloning the repo to your desired location.

```shell
gh repo clone https://github.com/lindeerik/msc-degree-project.git
```

In the root directory of the project, set up the virtual environment for the project. 

```shell
python -m venv .venv
source .venv/bin/activate
```

You may now install the required dependencies with

```shell
pip install -r requirements.txt
```

This completes the setup and you should be able to run the code. If you at some point want to exit the virtual environment, you can run `deactivate`.

# Usage

Data is found in the `data` directory. Drive tests were conducted between central Stockholm and Hölö, recording uplink throughput using iPerf tests on a Google Pixel 7 device with data logged via the G-NetTrack Pro application.

Reusable code is found in the `src` module. This includes functions for handling data, machine learning modeling, visualizing results, and conducting feature selection. These subdirectories are `data`, `models`, `visualization` and `features` respectively.

As described in [Quickstart](#quickstart), experiments and results can be carried out with the code in `scripts` and `notebooks`. Notebooks are run by selecting `.venv` as the kernel and running cells in the chosen notebook.

For scripts, you can run

```shell
python scripts/script_name.py
```
This generates results to the `experiments` directory. Exact placement within this directory is dependent upon the type of experiment and the timestamp for which it completes. These results can be visualized with visualization scripts, which defaults to saving figures to a `figures` directory.

To ensure working code, several tests are written in the `tests` folder, which are meant to be run with `pytest`. Running all tests can be done simply with 

```shell
pytest .
```


# Development

Development of the project is done with pull requests. Begin by creating a branch for the new work with 

```shell
git branch -c "branch-name-with-conventional-naming"
git switch "branch-name-with-conventional-naming"
```

Commits can be made to this branch with

```shell
git commit
```

When satisfied with the work, ensure code is tested and linted with 

```shell
pytest .
pylint --recursive=y src scripts tests
```

A pull request can then be opened with

```shell
gh pr create
```

If further adjustments must be made, add or change commits and push to the remote branch with

```shell
git push --force-with-lease
```

The branch is ready to be rebased and merged wheen it has has passed tests and after possible code-review.