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
#
# You can run this notebook on your laptop, a GPU is not needed :).

# %% [markdown]
# ## Install dependencies and import packages

# %%
# # !pip install -q tensorflow
# # !pip install -q stardist
# # !pip install nest_asyncio

# %%
from urllib.request import urlretrieve
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams["image.interpolation"] = None
matplotlib.rcParams['figure.figsize'] = (14, 10)
from tifffile import imread
from tqdm import tqdm

from stardist import fill_label_holes, random_label_cmap
from stardist.plot import render_label
from stardist.models import StarDist2D
from stardist import _draw_polygons
from csbdeep.utils import normalize

import nest_asyncio
nest_asyncio.apply()
import napari

lbl_cmap = random_label_cmap()


# %% [markdown]
# Some utility functions

# %%
def plot_img_label(img, lbl, img_title="image", lbl_title="label", **kwargs):
    fig, (ai,al) = plt.subplots(1,2, gridspec_kw=dict(width_ratios=(1,1)))
    im = ai.imshow(img, cmap='gray', clim=(0,1))
    ai.set_title(img_title)
    ai.axis("off")
    al.imshow(render_label(lbl, img=.3*img, normalize_img=False, cmap=lbl_cmap))
    al.set_title(lbl_title)
    al.axis("off")
    plt.tight_layout()
    
def preprocess(X, Y, axis_norm=(0,1)):
    # normalize channels independently
    X = np.array([normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X, leave=True, desc="Normalize images")])
    # fill holes in labels
    Y = np.array([fill_label_holes(y) for y in tqdm(Y, leave=True, desc="Fill holes in labels")])
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
# Load the dataset (images and tracking annotations) from disk.

# %%
x = np.array([imread(xi) for xi in sorted((base_path / "images").glob("*.tif"))])
y = np.array([imread(yi) for yi in sorted((base_path / "gt_tracking").glob("*.tif"))])
assert len(x) == len(y)
print(f"Number of images: {len(x)}")

# %%
x = x[:, 300:, :]
y = y[:, 300:, :]

# %%
# # !pip install ipywidgets
x, y = preprocess(x, y)

# %% [markdown]
# Visualize some images

# %%
idx = 0
plot_img_label(x[idx], y[idx])

# %% [markdown]
# This is ok to take a glimpse, but a dynamic viewer would be much better. Let's use [napari](https://napari.org/tutorials/fundamentals/getting_started.html) for this.
#
# We can easily explore how the nuclei move over time and see that the ground truth annotations are consistent over time. If you zoom in, you will note that the annotations are not perfect segmentations, but rather circular objects placed roughly in the center of each nucleus.

# %%
viewer = napari.Viewer()
viewer.add_image(x)
viewer.add_labels(y)

# %% [markdown]
# Load a pretrained stardist models, detect nuclei in one image and visualize them.

# %% tags=[]
idx = 0
model = StarDist2D.from_pretrained("2D_versatile_fluo")
detections, details = model.predict_instances(x[idx], scale=(1,1))
plot_img_label(x[idx], detections)

# %%
coord, points, prob = details['coord'], details['points'], details['prob']
_draw_polygons(coord, points, prob, show_dist=True)
plt.imshow(x[idx], cmap='gray'); plt.axis('off')
# plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-info"><h2>1) Parameter exploration</h2>
#
# Explore the following aspects of the detection algorithm:
#     
# - The `scale` parameter downscales the images by the given factor before feeding them to the neural network. What happens if you increase it?
# - Inspect false positive and false negative detections. Do you observe pattern?
#     
# </div>

# %% [markdown]
# Detect nuclei in all images of the time lapse.

# %%
pred, details = model.predict_instances(x[idx])
Y_val_pred = [model.predict_instances(x, show_tile_progress=False)[0]
              for x in tqdm(X_val)]

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
