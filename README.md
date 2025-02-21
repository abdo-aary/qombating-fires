<div align="center">
  <h1>Quantum against wildfires</h1>
</div>

<div align="center">

  ![Linting](https://github.com/abdo-aary/qombating-fires/actions/workflows/lint.yml/badge.svg) 
  &nbsp;
  ![Testing](https://github.com/abdo-aary/qombating-fires/actions/workflows/testing.yml/badge.svg) 
  &nbsp;
  ![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

</div>



## Project structure
````
root/
│
├── docs/                       # Documentation of the project 
│
├── notebooks/                  # Jupyter notebooks for exploration and presentation
│
├── data/                       # Scripts to download the data
│   ├── __init__.py             # Makes src a Python module
│   ├── get/                    # Scripts that define how to get the data
│   ├── prep/                   # Data preprocessing scripts
│   ├── utils/                  # Utility tools used for the data handling
│   └── view/                   # Scripts to create data visualizations
|
├── bassir/                     # Source code for the quantum predictor
│   ├── __init__.py             # Makes src a Python module
│   ├── models/                 # Scripts that define the used models
│   ├── train/                  # Scripts used to train the models
│   ├── utils/                  # Utility tools used through src code
│   └── view/                   # Scripts to create exploratory and results oriented visualizations
│
├── guru/                       # Source code for the quantum optimizer (responder)
│   ├── __init__.py             # Makes src a Python module
│   ├── optimizers/             # Scripts that define quantum optimizers
│   ├── utils/                  # Utility tools used through src code
│   └── view/                   # Scripts to create exploratory and results oriented visualizations
│
├── storage/                    # Experiments' results. Directory generated automatically, with content not handled 
│                               # by Git 
│
├── tests/                      # Test cases to ensure the code behaves as expected. These used in CI/CD as well
│
├── requirements.txt            # The dependencies file for reproducing the analysis environment
│
└── README.md                   # The top-level README for developers using this project
````

## Contributing
 
For detailed guidelines on how to contribute—covering our branching strategy, testing requirements, CI pipeline, 
and pull request process—please refer to our [CONTRIBUTING](docs/guides/CONTRIBUTING.md) file.


## Using Docker

For detailed guidelines on how to build a docker image out of the [Dockerfile](Dockerfile), please refer to our 
[DOCKER_SETUP](docs/guides/DOCKER_SETUP.md) file.















<!-- Utility commands -->
<!-- Export python path: ``export PYTHONPATH=${PYTHONPATH}:${pwd}``-->
<!-- Run jupyter-lab server ``jupyter lab --ip 10.44.83.233 --port 8899 --no-browser`` -->

<!-- Run the self-hosted runner via:  -->
