# project_name

A short description of the project.

# Table of Contents
<!--ts-->
   * [project_name](#project_name)
   * [Table of Contents](#table-of-contents)
   * [Project Organization](#project-organization)

<!-- Added by: humberto, at: Thu Mar 26 20:56:51 AEDT 2020 -->

<!--te-->

# Project Organization


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# References

* [Agg18] Charu C. Aggarwal. "Neural Networks and Deep Learning". Springer, 2018
* [Agg16] Charu C. Aggarwal. "Recommender Systems". Springer, 2016
* [Bra18] Max Bramer. "Principles of Data Mining". Springer. 2018
* [For18] David Forsyth. "Probability and Statistics for Computer Science". Springer. 2018
* [Hun19] John Hunt. "Advanced Guide to Python 3 Programming". Springer. 2019
* [Lee17] Kent D. Lee. "Foundations of Programming Languages". Springer. 2017
* [LS17] Laura Igual and Santi Seguí. "Introduction to Data Science". Springer. 2017
* [Kub17] Miroslav Kubat. "An Introduction to Machine Learning". Springer. 2017
* [ORe17] Gerard O'Regan. "Concise Guide to Software Engineering". Springer. 2017
* [Ski17] Steven S. Skiena. "The Data Science Design Manual". Springer. 2017
