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
import sys
from urllib.request import urlretrieve
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
matplotlib.rcParams["image.interpolation"] = "none"
matplotlib.rcParams['figure.figsize'] = (14, 10)
from tifffile import imread
from tqdm.auto import tqdm
import skimage
import pandas as pd
import scipy

from stardist import fill_label_holes, random_label_cmap
from stardist.plot import render_label
from stardist.models import StarDist2D
from stardist import _draw_polygons
from csbdeep.utils import normalize

# To interact with napari viewer from within a notebook
import nest_asyncio
nest_asyncio.apply()
import napari

lbl_cmap = random_label_cmap()
# Pretty tqdm progress bars 
# ! jupyter nbextension enable --py widgetsnbextension

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
# x = x[:20, 300:, :]
# y = y[:20, 300:, :]
x = x[:20, -256:, -256:]
y = y[:20, -256:, -256:]
print(f"Number of images: {len(x)}")
print(f"Image shape: {x[0].shape}")

# %%
x, y = preprocess(x, y)

# %% [markdown]
# Visualize some images (by changing `idx`).

# %%
idx = 0
plot_img_label(x[idx], y[idx])

# %% [markdown]
# This is ok to take a glimpse, but a dynamic viewer would be much better. Let's use [napari](https://napari.org/tutorials/fundamentals/getting_started.html) for this.

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x, name="image");


# %% [markdown]
# Let's add the ground truth annotations. Now we can easily explore how the cells move over time.
#
# If you zoom in, you will note that the dense annotations are not perfect segmentations, but rather circular objects placed roughly in the center of each nucleus.

# %%
def visualize_tracks(viewer, y, links=None, name=""):
    """Utility function to visualize segmentation and tracks"""
    colorperm = np.random.default_rng(42).permutation((np.arange(1, np.max(y) + 2)))
    tracks = []
    for t, frame in enumerate(y):
        centers = skimage.measure.regionprops(frame)
        for c in centers:
            tracks.append([colorperm[c.label], t, int(c.centroid[0]), int(c.centroid[1])])
    tracks = np.array(tracks)
    tracks = tracks[tracks[:, 0].argsort()]
    
    graph = {}
    if links is not None:
        divisions = links[links[:,3] != 0]
        for d in divisions:
            if colorperm[d[0]] not in tracks[:, 0] or colorperm[d[3]] not in tracks[:, 0]:
                continue
            graph[colorperm[d[0]]] = [colorperm[d[3]]]

    viewer.add_labels(y, name=f"{name}_detections")
    viewer.layers[f"{name}_detections"].contour = 3
    viewer.add_tracks(tracks, name=f"{name}_tracks", graph=graph)
    return tracks


# %%
visualize_tracks(viewer, y, links.to_numpy(), "ground_truth");


# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Exercise 1.1
# <div class="alert alert-block alert-info"><h3>Exercise 1.1: Highlight the cell divisions</h3>
#
# The visualization of the ground truth tracks help our visual system to process this video, it is still hard see sparse events, e.g. the cell divisions. Given the dense annotations `y` and the track links `links`, write a function that highlights the pairs of daughter cells just after mitosis.
#     
# </div>
#
# TODO add example image of output

# %%
def visualize_divisions(viewer, y, links):
    """Utility function to visualize divisions"""
    ### YOUR CODE HERE ###
    divisions = np.zeros_like(y)
    viewer.add_labels(divisions, name="divisions")
    pass


# %%
# Exercise 1.1 solution
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
    return divisions


# %%
visualize_divisions(viewer, y, links.to_numpy());

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
# ## Exercise 1.2
# <div class="alert alert-block alert-info"><h3>Exercise 1.2: Explore the parameters of cell detection</h3>
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
# centers = [xi[1]["points"] for xi in pred]

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


# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Checkpoint 1
# <div class="alert alert-block alert-success"><h3>Checkpoint 1: We have good detections, now on to the linking.</h3></div>

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Greedy linking by nearest neighbor

# %% [markdown] tags=[]
# Function to compute pairwise euclidian distance for detections in two adjacent frames.

# %%
# def euclidian_distance(start_points, end_points):
#     # TODO vectorize
#     dists = []
#     for sp in start_points:
#         for ep in end_points:
#             dists.append(np.sqrt(((sp - ep)**2).sum()))
            
#     dists = np.array(dists).reshape(len(start_points), len(end_points))
#     return dists

# %%
# dist0 = euclidian_distance(centers[0], centers[1])
# plt.figure(figsize=(5,5))
# plt.title("Pairwise distance matrix")
# plt.imshow(dist0);

# %% [markdown] tags=[]
# ## Exercise 1.3
# <div class="alert alert-block alert-info"><h3>Exercise 1.3: Complete a basic thresholded nearest neighbor linking function</h3>
#
# Given a cost matrix for detections in a pair of frames, implement a neighest neighbor function:    
# - For each detection in frame $t$, find the nearest neighbor in frame $t+1$. If the distance is below a threshold $\tau$, link the two objects.
# - Do the above sequentially for each pair of adjacent frames.
#     
# </div>

# %%
# TODO tracks into napari from consistently labelled dense outputs
# OR keep using the man_track.txt format: I have tracklets represented dense, and linkings via man_track.txt

class FrameByFrameLinker(ABC):

    @abstractmethod
    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """TODO
        
        Args:
        
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
            image0 (optional): image corresponding to detections0
            image1 (optional): image corresponding to detections1
        """
        pass
    
    @abstractmethod
    def _link_two_frames(self, cost_matrix):
        """TODO

        Args:

            cost_matrix: m x n matrix

        Returns:

            binary m x n linking matrix


        """
        pass

    @staticmethod
    def relabel_detections(detections, linking_matrices):
        """TODO"""
        # TODO fix
        assert len(detections) - 1 == len(linking_matrices)
        out = [skimage.segmentation.relabel_sequential(detections[0])[0]]
        n_tracks = out[0].max()
        lookup_tables = [{i: i for i in range(1, out[0].max() + 1)}]

        for i in tqdm(range(len(linking_matrices))):
            # old_frame = np.copy(skimage.segmentation.relabel_sequential(detections[i])[0])
            new_frame = np.copy(skimage.segmentation.relabel_sequential(detections[i+1])[0])
            
            lut = {}
            for idx_from, idx_to in zip(linking_matrices[i][0], linking_matrices[i][1]):
                # Copy over ID
                new_frame[new_frame == idx_to] = lookup_tables[i][idx_from]
                lut[idx_to] = lookup_tables[i][idx_from]


            # Start new track for all non-linked tracks
            new_ids = set(range(1, new_frame.max() + 1)) / set(linking_matrices[i][1])
            new_ids = list(new_ids)
                          
            for ni in new_ids:
                n_tracks += 1
                lut[ni] = n_tracks
                new_frame[new_frame == ni] = n_tracks
            lookup_tables.append(lut)
            out.append(new_frame)
                
        return out


    def link(self, detections, images=None):
        """TODO"""
        if images is not None:
            assert len(images) == len(detections)
        else:
            images = [None] * len(detections)

        linking_matrices = []
        for i in tqdm(range(len(images) - 1)):
            detections0, _, _ = skimage.segmentation.relabel_sequential(detections[i])
            detections1, _, _ = skimage.segmentation.relabel_sequential(detections[i+1])
            cost_matrix = self.linking_cost_function(detections0, detections1, images[i], images[i+1])
            links = self._link_two_frames(cost_matrix)
            linking_matrices.append(links)
            
        return linking_matrices


# %%
class NearestNeighborLinkerEuclidian(FrameByFrameLinker):
    def __init__(self, threshold=sys.float_info.max, *args, **kwargs):
        self.threshold = threshold
        super().__init__(*args, **kwargs)
    
    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        regions0 = skimage.measure.regionprops(detections0)
        centroids0 = [np.array(r.centroid) for r in regions0]
        
        regions1 = skimage.measure.regionprops(detections1)
        centroids1 = [np.array(r.centroid) for r in regions1]
        
        # TODO vectorize
        dists = []
        for c0 in centroids0:
            for c1 in centroids1:
                dists.append(np.sqrt(((c0 - c1)**2).sum()))

        dists = np.array(dists).reshape(len(centroids0), len(centroids1))
        return dists
    
    def _link_two_frames(self, cost_matrix):
        """Nearest neighbor linking

        Args:
        
            cost_matrix: m x n matrix with pairwise linking costs > 0 

        Returns:

            binary m x n linking matrix


        """
        cost_matrix = cost_matrix.copy().astype(float)
        assert np.all(cost_matrix > 0)
        cost_matrix[cost_matrix > self.threshold] = self.threshold
        
        idx_from = np.arange(cost_matrix.shape[0])
        idx_to = cost_matrix.argmin(axis=1)
        link = cost_matrix.min(axis=1) != self.threshold
        
        # TODO avoid linking things twice with bidirectional linking --> This can be another exercise?
        
        idx_from = idx_from[link]
        idx_to = idx_to[link]
        
        # Account for +1 offset of the dense labels
        idx_from += 1
        idx_to += 1
        
        return idx_from, idx_to
    
        # links = np.zeros_like(cost_matrix, dtype=bool)
        # links[idx_from, idx_to] = True 
        # return links


# %%
Linker = NearestNeighborLinkerEuclidian(threshold=30)
linking_matrices = Linker.link(detections)
linked_detections = FrameByFrameLinker.relabel_detections(detections, linking_matrices)

# %%

# %%
viewer.close()

# %% [markdown]
# Visualize results

# %%
# try:
#     viewer.close()
# except NameError:
#     pass
viewer = napari.Viewer()
viewer.add_image(x)
viewer.add_labels(linked_detections)
# visualize_tracks(viewer, linked_detections, name="NN")

# %% [markdown]
# ## Checkpoint 2
# <div class="alert alert-block alert-success"><h3>Checkpoint 2: We have a working basic tracking algorithm :).</h3></div>

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.3
# <div class="alert alert-block alert-info"><h3>Exercise 1.3: Estimate the global drift of the data</h3>
#
# We observe that all cells move upwards in what appears to be a constant motion. Modify the cost function `euclidian distance` from above to take this into account and run the neareast neighbor linking again.
#
# </div>

# %%
class NearestNeighborLinkerDriftCorrection(NearestNeighborLinkerEuclidian):
    def __init__(self, drift, *args, **kwargs):
        self.drift = drift
        super().__init__(

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
# ## Optimal frame-by-frame matching (*Assignment problem* or *Weighted bipartite matching*)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Exercise 1.4
# <div class="alert alert-block alert-info"><h3>Exercise 1.4: Perform optimal frame-by-frame linking</h3>
#
# Set up the cost matrix such that you can use [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) to solve the matching problem in the bipartite graph.
#     
# </div>

# %%
# TODO insert image for bipartite matching

# %%
# def minimum_weighted_bipartite_matching(self, cost_matrix):
    #     """TODO"""
    #     row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    #     return np.zeros_like(cost_matrix)

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
