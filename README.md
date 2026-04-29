## Setup

1. Clone the repo
    ```
    git clone git@github.com:dianapanainte/clustering-project-uniwien.git
    ```
    or
    ```
    git clone https://github.com/dianapanainte/clustering-project-uniwien.git
    ```

2. Create and activate virtual environment
    ```
    cd clustering-project-uniwien
    python3 -m venv venv
    venv\Scripts\activate        # Windows
    source venv/bin/activate     # Mac/Linux
    ```

3. Install dependencies
    ```
    pip install -r requirements.txt
    ```

4. Install the module
    ```
    pip install -e .
    ```

5. Run
    ```
    python3 main.py
    ```
## Structure of the repository

The repository contains the following folders and files:

- **`/clustering_module`**: This directory contains the main source code for the clustering. It is based on the Strategy Pattern and it includes all the necessary classes to preprocess, create features, and cluster the data.

- **`/forecasting`**: This folder contains all the forecasting algorithms we tried on the data obtained from clustering. It contains multiple models that we used as experiments, and the final model that had the best results, XGBoost.

- **`/data`**: This folder is used to store the 2 raw datasets, with data for 2023 and 2024.

- **`/notebooks`**: This directory includes Jupyter notebooks that demonstrate the usage of the clustering algorithms and provide exploratory data analysis.

- **`requirements.txt`**: This file lists all the dependencies required to run the project.

- **`main.py`**: This is the entry point of the application. It contains the main logic to execute KMeans++ clustering followed by the forecasting done with XGBoost.