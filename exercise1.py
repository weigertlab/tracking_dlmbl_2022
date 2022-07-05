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

# %%
# !pip install -q tensorflow
# !pip install -q stardist

# %%
from urllib.request import urlretrieve
from pathlib import Path
import stardist
from stardist import fill_label_holes
from csbdeep.utils import normalize
import tifffile


# %% [markdown]
# Some utility functions

# %%
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    al.imshow(render_label(lbl, img=.3*img, normalize_img=False, cmap=lbl_cmap))
    al.set_title(lbl_title)
    plt.tight_layout()
    
def preprocess(X, Y, axis_norm=(0,1)):
    # normalize channels independently

    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X, leave=False, desc="Normalize images")]
    Y = [fill_label_holes(y) for y in tqdm(Y, leave=False, desc="Fill holes in labels")]
    return X, Y


# %% [markdown] tags=[]
# ## Inspect the dataset

# %% [markdown]
# Download the dataset

# %%
# !curl https://zenodo.org/record/5206107/files/P31-crop.tif?download=1 --create-dirs -o data/cancer_cell_migration.tif

# %% [markdown]
# Load the dataset, no splits required, and preprocess it

# %%
num_imgs = 0
data = {}
for split in ["train", "val", "test"]:
    X = sorted((base_path / split / "images").glob("*.tif"))
    X = [imread(x) for x in X]
    Y = sorted((base_path / split / "masks").glob("*.tif"))
    Y = [imread(y) for y in Y]
    data[split] = (X.copy(), Y.copy())
    num_imgs += len(X)
X_trn, Y_trn = data["train"]
X_val, Y_val = data["val"]
X_test, Y_test = data["test"]
print('number of images: %3d' % num_imgs)
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_val))
print('- test:     %3d' % len(X_test))
print(f"Number of channels: {n_channel}")

# %% [markdown]
# Visualize some images

# %%

# %% [markdown]
# Load a pretrained stardist models and detect nuclei

# %%

# %% [markdown]
# Visualize detections and understand them visually

# %%

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
