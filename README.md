# Time series project

## Pre-requisites
1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Download the dataset and place it in the `data` directory. The dataset should be in the form of a PyTorch file named `EEG-ImageNet.pth` 


## Installation

1. Install the project dependencies using uv:
```bash
uv sync
```

2. source the virtual environment:
```bash
source .venv/bin/activate
```

## Example usage
1. there is a notebook named `viz.ipynb` that contains code to visualize the data. You can run it using Jupyter Notebook