# Time series project

## Pre-requisites
1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Download the dataset from [here](https://cloud.tsinghua.edu.cn/d/d812f7d1fc474b14bbd0/) and place it in the `data` directory. 


## Installation

1. Install the project dependencies using uv (One time setup):
```bash
uv venv && uv sync
```

2. source the virtual environment:
```bash
source .venv/bin/activate
```

3. Run a merge of the data files using the following command (One time setup):
```bash
python scripts/merge_dataset.py data/EEG-ImageNet_1.pth data/EEG-ImageNet_2.pth data/EEG-ImageNet.pth
```

## Example usage
1. there is a notebook named `viz.ipynb` that contains code to visualize the data. You can run it using Jupyter Notebook