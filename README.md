# qombating-fires
Quantum against wildfires   


## Project structure
````
root/
│
├── data/                       # Scripts to download the data
│
├── notebooks/                  # Jupyter notebooks for exploration and presentation
│
├── src/                        # Source code for the project
│   ├── __init__.py             # Makes src a Python module
│   ├── prep/                   # Scripts to preprocess and handle the data
│   ├── models/                 # Scripts that define the used models
│   ├── qoptim/                 # Scripts used for quantum optimization
│   ├── train/                  # Scripts used to train the models
│   ├── utils/                  # Utility tools used through src code
│   └── view/                   # Scripts to create exploratory and results oriented visualizations
│
├── storage/                    # Experiments' results. Directory generated automatically, with content not handled 
│                               # by Git 
│
├── tests/                      # Test cases to ensure the code behaves as expected
│
├── requirements.txt            # The dependencies file for reproducing the analysis environment
│
└── README.md                   # The top-level README for developers using this project
````

## Contributing
 
For detailed guidelines on how to contribute—covering our branching strategy, testing requirements, CI pipeline, and pull request process—please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) file.

<!-- Utility commands -->
<!-- Export python path: ``export PYTHONPATH=${PYTHONPATH}:${pwd}``-->
<!-- Run jupyter-lab server ``jupyter lab --ip 10.44.83.233 --port 8899 --no-browser`` -->