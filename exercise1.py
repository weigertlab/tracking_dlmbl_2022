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
    X = np.stack([normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X, leave=True, desc="Normalize images")])
    # fill holes in labels
    Y = np.stack([fill_label_holes(y) for y in tqdm(Y, leave=True, desc="Fill holes in labels")])
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
x = np.stack([imread(xi) for xi in sorted((base_path / "images").glob("*.tif"))])
y = np.stack([imread(yi) for yi in sorted((base_path / "gt_tracking").glob("*.tif"))])
assert len(x) == len(y)
print(f"Number of images: {len(x)}")
print(f"Image shape: {x[0].shape}")

# %%
# TODO crop the dataset in time and space to reduce runtime
x = x[:20, 300:, :]
y = y[:20, 300:, :]
print(f"Number of images: {len(x)}")
print(f"Image shape: {x[0].shape}")

# %%
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
viewer.add_labels(y);

# %% [markdown]
# Load a pretrained stardist model, detect nuclei in one image and visualize them.

# %% tags=[]
idx = 0
model = StarDist2D.from_pretrained("2D_versatile_fluo")
detections, details = model.predict_instances(x[idx], scale=(1,1))
plot_img_label(x[idx], detections, lbl_title="detections")

# %% [markdown]
# Here we visualize in detail the polygons we have detected with StarDist. TODO more description
#
# Notice that each object comes with a center point, which we can use to compute pairwise euclidian distances between objects.

# %%
coord, points, prob = details['coord'], details['points'], details['prob']
plt.figure(figsize=(20,20))
plt.subplot(211)
plt.title("Predicted Polygons")
_draw_polygons(coord, points, prob, show_dist=True)
plt.imshow(x[idx], cmap='gray'); plt.axis('off')

# plt.subplot(312)
# plt.title("Ground truth tracking anntotations")
# plt.imshow(x[idx], cmap='gray')
# plt.imshow(y[idx], cmap=lbl_cmap, alpha=0.5); plt.axis('off')

plt.subplot(212)
plt.title("Object center probability")
plt.imshow(x[idx], cmap='magma'); plt.axis('off')
plt.tight_layout()
plt.show() 

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Exercise 1.1: Parameter exploration of detection</h3>
#
# Explore the following aspects of the detection algorithm:
#     
# - The `scale` parameter of the function `predict_instances` downscales the images by the given factor before feeding them to the neural network. What happens if you increase it?
# - Inspect false positive and false negative detections. Do you observe patterns?
#     
# </div>

# %% [markdown]
# Detect centers and segment nuclei in all images of the time lapse.

# %%
pred = [model.predict_instances(xi, show_tile_progress=False, scale=(1,1))
              for xi in tqdm(x)]
detections = np.stack([xi[0] for xi in pred])
centers = [xi[1]["points"] for xi in pred]

# %%
print(detections[0].max())
print(len(centers[0]))

# %% [markdown]
# Visualize the unlinked dense detections

# %%
viewer = napari.Viewer()
viewer.add_image(x)
viewer.add_labels(detections);

# %% [markdown]
# We see that the number of detections increases over time, corresponding to the cells that insert the field of view from below during the video.

# %%
plt.figure(figsize=(10,6))
plt.bar(range(len(centers)), [len(xi) for xi in centers])
plt.title("Number of detections in each frame")
# TODO clean up plot
plt.show();


# %% [markdown]
# Function to compute pairwise euclidian distance for detections in two adjacent frames.

# %%
def euclidian_distance(start_points, end_points):
    # TODO vectorize
    dists = []
    for sp in start_points:
        for ep in end_points:
            dists.append(np.sqrt(((sp - ep)**2).sum()))
            
    dists = np.array(dists).reshape(len(start_points), len(end_points))
    return dists


# %%
dist0 = euclidian_distance(centers[0], centers[1])
plt.figure(figsize=(5,5))
plt.title("Pairwise distance matrix")
plt.imshow(dist0);

# %% [markdown]
# Greedy linking by nearest neighbor

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Exercise 1.2: Complete a basic thresholded nearest neighbor linking function</h3>
#
# Given a cost matrix for detections in a pair of frames, implement a neighest neighbor function:    
# - For each detection in frame $t$, find the nearest neighbor in frame $t+1$. If the distance is below a threshold $\tau$, link the two objects.
# - Do the above sequentially for each pair of adjacent frames.
#     
# </div>

# %%
print(detections[0].shape)

# %%
from skimage.segmentation import relabel_sequential

# TODO fix

# TODO introduce gaps in function to fill
def nearest_neighbor_linking(detections, features, cost_function, thres=None):
    """Links the dense detections based on a cost function that takes features of two adjacent frames as input.
    
    TODO docstring
    """
    # TODO clean up
    tracks = [detections[0]]
    track_ids = list(range(1, detections[0].max() + 1))
    n_tracks = detections[0].max()
    # print(track_ids)
    # print(n_tracks)
    for i in tqdm(range(len(detections) - 1)):
        cost_matrix = cost_function(features[i], features[i+1])
        argmin = np.argmin(cost_matrix, axis=1)
        print(argmin.max())
        # print(cost_matrix.shape)
        # print(argmin.shape)
        min_dist = np.take_along_axis(cost_matrix, np.expand_dims(argmin, 1), axis=1)
        # print(min_dist.shape)
        
        linked = []
        new_frame = np.copy(detections[i+1])
        new_ids = []
        for i_in, (md, am) in enumerate(zip(min_dist, argmin)):
            if not thres or md < thres:
                new_frame[new_frame == am + 1] = track_ids[i_in] + 1
                new_ids.append(track_ids[i_in] + 1)
                
        track_ids = new_ids                                
        tracks.append(new_frame)
        
    
        # Start new track for all non-linked tracks
        
    
    return np.stack(tracks)

# %%
tracks = nearest_neighbor_linking(detections, centers, euclidian_distance, thres=20)
print(tracks.shape)

# %% [markdown]
# Visualize results

# %%
viewer = napari.Viewer()
viewer.add_image(x)
viewer.add_labels(tracks);

# TODO visualize each track as line in corresponding color.

# %% [markdown]
# Model the global drift and run nearest neighbor again

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Exercise 1.3: Estimate the global drift of the data</h3>
#
# We observe that all cells move upwards in what appears to be a constant motion. Modify the cost function to take this into account and run the neareast neighbor linking again.
#
# </div>

# %%

# %% [markdown]
# Hungarian matching (scipy.optimize.linear_sum)

# %%
# TODO given

# %% [markdown]
# Load ground truth and compute a metric

# %%
# TODO given

# %% [markdown]
# Compute other features with scikit-image to play with cost function

# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Exercise 1.4: Explore different features for hungarian matching</h3>
#
# Explore running the hungarian matching algorithm from above with different `scikit-image` region properties and inspect the results. 
#     
# </div>

# %%
