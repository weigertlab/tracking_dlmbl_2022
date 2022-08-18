# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 1/3: Tracking by detection and simple frame-by-frame matching
#
# You can run this notebook on your laptop, a GPU is not needed :).
#
# Here we will walk through all basic components of a tracking-by-detection algorithm.
#     
# You will learn
# - to use a robust pretrained deep-learning-based **object detection** algorithm called _StarDist_ (Exercise 1.1).
# - to implement a basic nearest-neighbor linking algorithm (Exercises 1.2 + 1.3).
# - to compute the optimal frame-by-frame linking by setting up a bipartite matching problem and using a python-based solver (Exercise 1.4).
# - to **evaluate the output** of a tracking algorithm against a ground truth annotation.
# - to compute suitable object **features** for the object linking process with `scikit-image` (Exercise 1.5).
#
# <!-- TODO link the learning points to the subsections of the notebook. -->

# %% [raw]
# # TODO input output gif to show the task on this dataset, using napari.

# %% [markdown]
# ![SegmentLocal](figures/trackmate-stardist-tracking.gif "segment")

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Install dependencies and import packages

# %%
# # !pip install -q tensorflow
# # !pip install -q stardist
# # !pip install nest_asyncio

# %%
from urllib.request import urlretrieve
from pathlib import Path
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams["image.interpolation"] = "none"
matplotlib.rcParams['figure.figsize'] = (14, 10)
from tifffile import imread
from tqdm.auto import tqdm
import skimage
import pandas as pd

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
# For pretty tqdm progress bars, run `jupyter nbextension enable --py widgetsnbextension` in the terminal.

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
# TODO write .csv version of man_track to switchdrive
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
# links = np.loadtxt(base_path / "gt_tracking" / "man_track.txt", dtype=int)
links = pd.read_csv(base_path / "gt_tracking" / "man_track.csv")
links[:10]

# %% [markdown]
# Crop the dataset in time and space to reduce runtime

# %%
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
try:
    viewer.close()
except NameError:
    pass
viewer = napari.Viewer()
viewer.add_image(x, name="image");


# %%
def visualize_tracks(viewer, y, links):
    """Utility function to visualize segmentation and tracks"""
    tracks = []
    for t, frame in enumerate(y):
        centers = skimage.measure.regionprops(frame)
        for c in centers:
            tracks.append([c.label, t, int(c.centroid[0]), int(c.centroid[1])])
    tracks = np.array(tracks)
    tracks = tracks[tracks[:, 0].argsort()]
    divisions = links[links[:,3] != 0]
    graph = {}
    for d in divisions:
        if d[0] not in tracks[:, 0] or d[3] not in tracks[:, 0]:
            continue
        graph[d[0]] = [d[3]]
        
    viewer.add_labels(y, name="labels")
    viewer.layers["labels"].contour = 3
    viewer.add_tracks(tracks, name="tracks", graph=graph)
    return tracks
    # TODO coloring by track ID not working.

# This could also be an exercise to get familiar with how these things are stored.
# TODO clean up
def visualize_divisions(viewer, y, links):
    """Utility function to visualize divisions"""
    divisions = links[links[:,3] != 0]
    divisions_dict = defaultdict(list)
    for d in divisions:
        if d[0] not in y or d[3] not in y:
            continue
        divisions_dict[d[3]].append((d[0], d[1]))

    layer = np.zeros_like(y)
    for k, v in divisions_dict.items():
        assert len(v) == 2
        layer[v[0][1]] = (y[v[0][1]] == v[0][0]).astype(int) * v[0][0] + (y[v[0][1]] == v[1][0]).astype(int) * v[1][0]
    
    viewer.add_labels(layer, name="divisions")


# %%
tracks = visualize_tracks(viewer, y, links.to_numpy())

# %%
tracks

# %%
tracks[tracks[:, 0].argsort()]

# %%
visualize_divisions(viewer, y, links.to_numpy())

# %% [markdown] tags=[]
# ## Object detection using a pre-trained neural network

# %% [markdown] tags=[]
# ### Load a pretrained stardist model, detect nuclei in one image and visualize them.
#
# TODO use a model pre-trained on this dataset instead of the general fluorescence nuclei model.

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

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.1
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
scale = (1,1)
pred = [model.predict_instances(xi, show_tile_progress=False, scale=scale)
              for xi in tqdm(x)]
detections = np.stack([xi[0] for xi in pred])
centers = [xi[1]["points"] for xi in pred]

# %% [markdown]
# Visualize the dense detections. They are still not linked.

# %%
try:
    viewer.add_labels(detections, name=f"detections_scale_{scale}");
except NameError:
    viewer = napari.Viewer()
    viewer.add_image(x)
    viewer.add_labels(detections, name=f"detections_scale_{scale}");

# %% [markdown]
# We see that the number of detections increases over time, corresponding to the cells that insert the field of view from below during the video.

# %%
plt.figure(figsize=(10,6))
plt.bar(range(len(centers)), [len(xi) for xi in centers])
plt.title(f"Number of detections in each frame (scale={scale})")
plt.xticks(range(len(centers)))
plt.show();


# %% [markdown]
# ## Checkpoint 1: We have good detections, now on to the linking.

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Greedy linking by nearest neighbor

# %% [markdown] tags=[]
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

# %% [markdown] tags=[]
# ## Exercise 1.2
# <div class="alert alert-block alert-info"><h3>Exercise 1.2: Complete a basic thresholded nearest neighbor linking function</h3>
#
# Given a cost matrix for detections in a pair of frames, implement a neighest neighbor function:    
# - For each detection in frame $t$, find the nearest neighbor in frame $t+1$. If the distance is below a threshold $\tau$, link the two objects.
# - Do the above sequentially for each pair of adjacent frames.
#     
# </div>

# %%
from skimage.segmentation import relabel_sequential

# TODO clean up/rewrite
# TODO how would I want to represent tracks, throughout the exercises? You could leave the detections as is, and output a linking table, which could be applied to the detections to color them appropriately.
# --> Don't worry about this now, standardize later.

# The dict would have {t: {id: parent_id}}
# for iterating, you would get back some

# The tracks layer of napari would also need to be calculated for that.
# The input format isn't too bad. I have tracklets represented dense, and linkings via

# TODO introduce gaps in function to fill
def nearest_neighbor_linking(detections, features, cost_function, thres=None):
    """Links the dense detections based on a cost function that takes features of two adjacent frames as input.
    
    TODO docstring
    """
    # TODO clean up
    tracks = [detections[0]]
    track_ids = list(range(detections[0].max()))  # starting at 0 here
    n_tracks = detections[0].max()  # running index to assign new track numbers
    for i in tqdm(range(len(detections) - 1)):
        cost_matrix = cost_function(features[i], features[i+1])
        argmin = np.argmin(cost_matrix, axis=1)
        min_dist = np.take_along_axis(cost_matrix, np.expand_dims(argmin, 1), axis=1)
        
        linked = []
        new_frame = np.copy(detections[i+1])
        new_ids = np.zeros(len(range(new_frame.max())))
        for i_in, (md, am) in enumerate(zip(min_dist, argmin)):
            if not thres or md < thres:
                new_frame[new_frame == am + 1] = track_ids[i_in] + 1  # +1 offset to account for background in dense
                new_ids[am] = track_ids[i_in]
        
    
        # Start new track for all non-linked tracks
        for ni in range(len(new_ids)):
            if new_ids[ni] == 0:
                new_ids[ni] = n_tracks + 1
                n_tracks += 1
        
        # print(f"Number of total tracks: {n_tracks}")
        track_ids = new_ids          
        tracks.append(new_frame)
    
        # TODO output for napari tracks layers
    return np.stack(tracks)

# %%
tracklets = nearest_neighbor_linking(detections, centers, euclidian_distance, thres=100)


# %%
def visualize_tracklets(viewer, y):
    """Utility function to visualize segmentation and tracks"""
    tracks = []
    for t, frame in enumerate(y):
        centers = skimage.measure.regionprops(frame)
        for c in centers:
            tracks.append([c.label, t, int(c.centroid[0]), int(c.centroid[1])])
    tracks = np.array(tracks)
        
    viewer.add_labels(y, name="detections")
    viewer.layers["labels"].contour = 3
    viewer.add_tracks(tracks, name="tracklets")
    # TODO coloring by track ID not working.



# %% [markdown]
# Visualize results

# %%
try:
    viewer.close()
except NameError:
    pass
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracklets(viewer, tracklets)


# %% [markdown]
# ## Checkpoint 2: We have a working basic tracking algorithm :)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.3
# <div class="alert alert-block alert-info"><h3>Exercise 1.3: Estimate the global drift of the data</h3>
#
# We observe that all cells move upwards in what appears to be a constant motion. Modify the cost function `euclidian distance` from above to take this into account and run the neareast neighbor linking again.
#
# </div>

# %%
def euclidian_distance_drift_correction(start_points, end_points, drift=0):
    """ 
    
    """
    # Insert your solution here
    
    return dists


# %%
tracks_drift_correction = nearest_neighbor_linking(detections, centers, euclidian_distance_drift_correction, thres=100)

# %% [markdown]
# Visualize results

# %%
try:
    viewer.close()
except NameError:
    pass
viewer = napari.Viewer()
viewer.add_image(x)
viewer.add_labels(tracks_drift_correction);

# TODO visualize each track as line in corresponding color.

# %% [markdown] tags=[]
# ## Optimal frame-by-frame matching (*Assignment problem* or *Maximum weighted bipartite matching*)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Exercise 1.4
# <div class="alert alert-block alert-info"><h3>Exercise 1.4: Perform optimal frame-by-frame linking</h3>
#
# Set up the cost matrix such that you can use [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) to solve the matching problem in the bipartite graph.
#     
# </div>

# %%
# TODO insert image for bipartite matching
# TODO use exercise 2021

# %% [markdown]
# Load ground truth and compute a metric

# %%
# TODO given

# %% [markdown] tags=[]
# ## Other suitable features for linking cost function

# %% [markdown]
# Compute other features with scikit-image to play with cost function

# %% [markdown]
# ## Exercise 1.5
#
# <div class="alert alert-block alert-info"><h3>Exercise 1.5: Explore different features for assigment problem</h3>
#
# Explore solving the assignment problem from above with different `scikit-image` region properties and inspect the results. 
#
# Feel free to share tracking runs for which your features improved the results.
#     
# </div>

# %%
