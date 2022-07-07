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
# # Exercise 2/3: Tracking with Linear Assignment Problem (LAP)
#
# You can run this notebook on your laptop, a GPU is not needed :).
#
# Here we will improve the tracking algorithm introduced in exercise 1 by using a linking algorithm that considers more than two frames at a time, the _Linear Assignment Problem_ (LAP).
#     
# You will learn
# - how this formulation addresses **typical challenges of tracking in bioimages**, like cell division and objects temporarily going out of focus.
# - to set up the two **LAP cost matrices** step by step and how to use a solver in python.
# - how to use **_Trackmate_**, a versatile ready-to-go implementation of LAP tracking in ImageJ.

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Install dependencies and import packages

# %%
from pathlib import Path

import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)
from tifffile import imread
from tqdm import tqdm

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Load and prepare the dataset

# %% tags=[]

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Extract detections

# %%
# TODO short version of detection from exercise 1

# %% [markdown]
#

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Introduction to the linear assignment problem (LAP)
#
# In the previous exercise, we have been able to track individual cells over time by linking detections frame-by-frame. However, there are multiple processes that this approach is not able to capture:
# - For tracing cell lineages, we want to capture the connection between mother and daughter cells in cell divisions. To do this, we have to link one object in frame $t$ to two objects in frame $t+1$, but the matching formulation in exercise 1 only allows one-to-one links.
# - If a cell is not detected in just a single frame, its resulting track will be split apart if we only use frame-by-frame linking.
#
# To account for these processes, Jaqaman et al. (2008) have introduced a two-step linking formulation that can be solved using the convenient linear assignment problem (LAP) formulation.
#
# The first step consists of frame-by-frame linking, similar to the matching formulation in exercise 1/3. The output are track segments for individual cells. In the second step, we define costs for linking these track segments beginning to end (gap closing) and beginning to intermediate (cell division).
#
# TODO write more about second step.
#
# <!-- TODO upate!
# The goal of the linear assignment algorithm is to select a set of pairs such that the sum of the selected weights is minimized. In the framework established by Jaqaman et al, the cost matrix is divided into four quadrants. The top left and bottom right corners correspond to direct matches between objects in frame n to objects in frame $n+1$. The bottom left corner is populated by a diagonal matrix where diagonal values correspond to the probability of a cell dividing. The top right corner is filled with another diagonal matrix in which the diagonal values correspond to the probability of cell death. A diagram of the cost matrix is shown below. -->
#
# [Jaqaman et al. (2008). Robust single-particle tracking in live-cell time-lapse sequences. Nature methods, 5(8), 695-702.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2747604/)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Extract features

# %%
# TODO short version of feature extraction from exercise 1

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Frame-by-frame linking

# %%
# TODO cost matrix and setup will be given

# %% [markdown] tags=[]
# ## Segment linking

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Exercise 2.1
#
# <div class="alert alert-block alert-info"><h3>Exercise 2.1: Cost matrix for gap closing</h3>
#
# Set up the cost matrix for gap closing. We will not consider the part for merging/splitting here.
#
# </div>
#
# <img src="figures/LAP_cost_matrix_2.png" width="500"/>
#
# Taken from Jaqaman et al. (2008)

# %% [markdown]
# Run the LAP

# %%

# %% [markdown]
#

# %% [markdown] tags=[]
# ### Visualize results

# %%
# napari

# %% [markdown]
# ## Exercise 2.2
#
# <div class="alert alert-block alert-info"><h3>Exercise 2.2: Cost matrix for modeling cell divisions</h3>
#
# Extend the cost matrix above to include the costs for modeling cell visions.
#
# </div>

# %% [markdown]
# ## Checkpoint: We have implemented an advanced linking algorithm in python :)

# %% [markdown] tags=[]
# ## Run the full linear assignment problem in Trackmate

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Exercise 2.3
#
# <div class="alert alert-block alert-info"><h3>Exercise 2.3: Walk through LAP tracking workflow in Trackmate</h3>
#
# Use Trackmate to conveniently do tracking using the LAP formulation.
#     
# Explore different options (TODO which ones are interesting for the cancer cell migration dataset?)
#     
# </div>

# %% [markdown]
# TODO Fill in screen shots and text

# %% [markdown]
# Install Fiji

# %%

# %% [markdown]
# Install Trackmate + StarDist detection

# %%

# %% [markdown]
# Configure tracking

# %%

# %% [markdown]
# Inspect results

# %%
