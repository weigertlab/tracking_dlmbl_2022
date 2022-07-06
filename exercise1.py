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
# # Exercise 1/3: Tracking by detection and simple frame-by-frame matching

# %% [markdown]
# ## Install dependencies and import packages

# %%
# !pip install -q tensorflow
# !pip install -q stardist

# %%
from urllib.request import urlretrieve
from pathlib import Path

import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)
from tifffile import imread
from tqdm import tqdm

from stardist import fill_label_holes, random_label_cmap
from stardist.plot import render_label
from stardist.models import StarDist2D
from csbdeep.utils import normalize

lbl_cmap = random_label_cmap()


# %% [markdown]
# Some utility functions

# %%
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, gridspec_kw=dict(width_ratios=(1,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    al.imshow(render_label(lbl, img=.3*img, normalize_img=False, cmap=lbl_cmap))
    al.set_title(lbl_title)
    plt.tight_layout()
    
def preprocess(X, Y, axis_norm=(0,1)):
    # normalize channels independently
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X, leave=True, desc="Normalize images")]
    # fill holes in labels
    Y = [fill_label_holes(y) for y in tqdm(Y, leave=True, desc="Fill holes in labels")]
    return X, Y


# %% [markdown] tags=[]
# ## Inspect the dataset

# %% [markdown]
# Download the dataset

# %%
base_path = Path("data/cancer_cell_migration")

if base_path.exists():
    print("Dataset already downloaded.")
else:
    # !curl https://drive.switch.ch/index.php/s/DUwFtY7LAxOFTUW/download --create-dirs -o data/cancer_cell_migration.zip
    # !unzip -q data/cancer_cell_migration.zip -d data

# %% [markdown]
# Load the dataset: Images and tracking annotations.

# %%
x = [imread(xi) for xi in sorted((base_path / "images").glob("*.tif"))]
y = [imread(yi) for yi in sorted((base_path / "gt_tracking").glob("*.tif"))]
assert len(x) == len(y)
print(f"Number of images: {len(x)}")

# %%
# # !pip install ipywidgets
x, y = preprocess(x, y)

# %% [markdown]
# Visualize some images

# %%
idx = 50
plot_img_label(x[0], y[0])
# TODO slider for time series

# %% [markdown]
# Load a pretrained stardist models and detect nuclei

# %%
model = StarDist2D.from_pretrained("2D_versatile_fluo")
labels, details = model.predict_instances(x[0], n_tiles=(2,2))

# %% [markdown]
# Visualize detections and understand them visually

# %%
print("test")

# %% [markdown]
# Extract IoU feature method

# %%

# %% [markdown]
# Greedy linking by nearest neighbor

# %%

# %% [markdown]
# Load ground truth and compute a metric

# %%

# %% [markdown]
# Hungarian matching (scipy.optimize.linear_sum)

# %%

# %% [markdown]
# Compute other features with scikit-image to play with cost function

# %%

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h2>Exercise</h2>
#
# Test
#     
# </div>
