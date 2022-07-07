# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 3/3: Advanced topics and linear optimization
#
# You can run this notebook on your laptop, a GPU is not needed :).
#
# Here we will introduce more advanced formulations of tracking.
#     
# You will learn
# - to set up a **network flow** using `networkx`, which allows to find a global optimum solution for small-scale problems, but without modeling cell divisions.
# - to formulate an **integer linear program (ILP)** to find a global optimum solution for small-scale tracking problems with `cvxopt`.

# %% [markdown]
# ## Install dependencies and import packages

# %%
from pathlib import Path

import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)
from tifffile import imread
from tqdm import tqdm

# %% [markdown]
# ## Load and prepare the dataset

# %%

# %% [markdown]
# ## Extract detections

# %%

# %% [markdown]
# ## Set up network flow

# %% [markdown]
# TODO write brief intro to network flow.

# %% [markdown]
# ## Exercise 3.1
#
# <div class="alert alert-block alert-info"><h3>Exercise 3.1: Set up network flow with simple cost function</h3>
#
# We follow the formulation in [Schulter et al. (2017). Deep network flow for multi-object tracking](https://openaccess.thecvf.com/content_cvpr_2017/papers/Schulter_Deep_Network_Flow_CVPR_2017_paper.pdf), but do not parametrize the costs.
#     
#
# </div>

# %%
