---
NEURO-PRIOR FOUNDATION MODEL TRAINING DATA SPECIFICATION
---

# Running from `main.py`

First, navigate to your root directory: `.../neuro-prior`

```bash
# Create and activate the environment
conda create -n neuro python=3.12 -y
conda activate neuro
which python

# Install PyTorch dependencies
pip install torch torchvision torchaudio

# Install project requirements and run
pip install -r requirements.txt
python main.py



Command Line Options
You can customize the execution using the following flags:

# Basic execution
python main.py

# With custom options
python main.py --n_cases 50 --vae_epochs 30 --do_steps 200 --output_dir Data

# Without visualizations (faster)
python main.py --skip_plots


Running in Jupyter Notebook
To set up the environment for a notebook, run:

conda create -n debugging python=3.12 -y
conda activate debugging
which python
pip install -r requirements.txt
pip install jupyter
jupyter notebook
