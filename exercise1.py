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
# You could also run this notebook on your laptop, a GPU is not needed.
#
# Here we will walk through all basic components of a tracking-by-detection algorithm.
#
# TODO update
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

# %% [markdown]
# <div class="alert alert-block alert-danger"><h3>Napari from within a jupyter notebook.</h3>
# <ul>
#     <li>To have napari from within a jupyter notebook working, you need to use an up-to-date version of napari, as is the case in the conda environments provided together with this exercise.</li>
#     <li>Whenever you are coding and debugging, close the viewer with `viewer.close()` to avoid problems with the napari and jupyter event loops.</li>
# </ul>
# </div>

# %%
# # !pip install -q tensorflow
# # !pip install -q stardist

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

# Not needed in newer version anymore
# To interact with napari viewer from within a notebook
# import nest_asyncio
# nest_asyncio.apply()


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
# For this exercise we will be working with a fluorenscence microscopy time-lapse of breast cancer cells with stained nuclei (SiR-DNA).
#
# Let's download the dataset to your machine.

# %%
# TODO write .csv version of man_track to switchdrive
base_path = Path("data/cancer_cell_migration")

if base_path.exists():
    print("Dataset already downloaded.")
else:
    # !curl https://drive.switch.ch/index.php/s/DUwFtY7LAxOFTUW/download --create-dirs -o data/cancer_cell_migration.zip
    # !unzip -q data/cancer_cell_migration.zip -d data

# %% [markdown]
# Load the dataset (images and tracking annotations) from disk into this notebook.

# %%
x = np.stack([imread(xi) for xi in sorted((base_path / "images").glob("*.tif"))])
y = np.stack([imread(yi) for yi in sorted((base_path / "gt_tracking").glob("*.tif"))])
assert len(x) == len(y)
print(f"Number of images: {len(x)}")
print(f"Image shape: {x[0].shape}")
# links = np.loadtxt(base_path / "gt_tracking" / "man_track.txt", dtype=int)
links = pd.read_csv(base_path / "gt_tracking" / "man_track.csv")
print("Links")
links

# %% [markdown]
# Crop the dataset in time and space to reduce runtime; preprocess.

# %%
delta_t = 5

x = x[::delta_t, 300:, :]
y = y[::delta_t, 300:, :]
# x = x[:5:delta_t, -64:, -64:]
# y = y[:5:delta_t, -64:, -64:]

links["from"] = (np.ceil(links["from"] / 5)).astype(int)
links["to"] = (np.ceil(links["to"] // 5)).astype(int)
print(f"Number of images: {len(x)}")
print(f"Image shape: {x[0].shape}")
x, y = preprocess(x, y)

# %% [markdown]
# Visualize some images (by changing `idx`).

# %%
idx = 0
plot_img_label(x[idx], y[idx])

# %% [markdown]
# This is ok to take a glimpse, but a dynamic viewer would be much better. Let's use [napari](https://napari.org/tutorials/fundamentals/getting_started.html) for this.

# %%
viewer = napari.Viewer()
viewer.add_image(x, name="image");


# %% [markdown]
# <div class="alert alert-block alert-danger"><h3>Napari in a jupyter notebook:</h3>
#     
# - To have napari working in a jupyter notebook, you need to use up-to-date versions of napari, pyqt and pyqt5, as is the case in the conda environments provided together with this exercise.
# - Whenever you are coding and debugging, close the napari viewer with `viewer.close()` to avoid problems with the asynchronous napari and jupyter event loops.
# </div>

# %% [markdown]
# Let's add the ground truth annotations. Now we can easily explore how the cells move over time.
#
# If you zoom in, you will note that the dense annotations are not perfect segmentations, but rather circles placed roughly in the center of each nucleus.

# %%
def visualize_tracks(viewer, y, links=None, name=""):
    """Utility function to visualize segmentation and tracks"""
    max_label = links.max() if links is not None else y.max()
    colorperm = np.random.default_rng(42).permutation((np.arange(1, max_label + 2)))
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
# The visualization of the ground truth tracks are useful to grasp this video, but it is still hard see the cell divisions. Given the dense annotations `y` and the track links `links`, write a function to create a new layer that highlights the pairs of daughter cells just after mitosis.
#     
# </div>
# Expected outcome:<br>
# <figure style="display:inline-block">
#     <img src="figures/prediv.png" width="400" />
#     <figcaption>t = 6</figcaption>
# </figure>
# <figure style="display:inline-block">
#     <img src="figures/postdiv.png" width="400" />
#     <figcaption>t = 7</figcaption>
# </figure>

# %%
# def visualize_divisions(viewer, y, links):
#     """Utility function to visualize divisions"""
#     ### YOUR CODE HERE ###
#     divisions = np.zeros_like(y)
#     viewer.add_labels(divisions, name="divisions")
#     return divisions

# %%
# Solution Exercise 1.1
def visualize_divisions(viewer, y, links):
    """Utility function to visualize divisions"""
    daughters = links[links[:,3] != 0]
    divisions = np.zeros_like(y)

    for d in daughters:
        if d[0] not in y or d[3] not in y:
            continue
        divisions[d[1]][y[d[1]] == d[0]] = d[0]
                
    viewer.add_labels(divisions, name="divisions")
    return divisions


# %%
visualize_divisions(viewer, y, links.to_numpy());

# %% [markdown] tags=[]
# ## Object detection using a pre-trained neural network

# %% [markdown] tags=[]
# ### Load a pretrained stardist model, detect nuclei in one image and visualize them.

# %% tags=[]
idx = 0
model = StarDist2D.from_pretrained("2D_versatile_fluo")
detections, details = model.predict_instances(x[idx], scale=(1,1))
plot_img_label(x[idx], detections, lbl_title="detections")

# %% [markdown]
# Here we visualize in detail the polygons we have detected with StarDist. TODO some description on how StarDist works.
#
# <!-- Notice that each object comes with a center point, which we can use to compute pairwise euclidian distances between objects. -->

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
# - The `scale` parameter of the function `predict_instances` downscales (< 1) or upscales (> 1) the images by the given factor before feeding them to the neural network. How do the detections change if you adjust it?
# - Inspect false positive and false negative detections. Do you observe patterns?
# - So far we have used a StarDist off the shelf. Luckily, we also have a StarDist model that was trained on a subset of this breast cancer cells dataset. Load it with `model = StarDist2D(None, name="stardist_breast_cancer", basedir="models")` and qualitatively observe differences.
#
# </div>

# %% [markdown]
# Detect centers and segment nuclei in all images of the time lapse.

# %%
scale = (1.0, 1.0)
pred = [model.predict_instances(xi, show_tile_progress=False, scale=scale)
              for xi in tqdm(x)]
detections = [xi[0] for xi in pred]
detections = np.stack([skimage.segmentation.relabel_sequential(d)[0] for d in detections])  # ensure that label ids are contiguous and start at 1 for each frame 
centers = [xi[1]["points"] for xi in pred]

# %% [markdown]
# Visualize the dense detections. They are still not linked.

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
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
#
# TODO write introduction text.

# %% [markdown]
# ## Exercise 1.3
# <div class="alert alert-block alert-info"><h3>Exercise 1.3: Write a function that computes pairwise euclidian distances given two lists of points.</h3></div>

# %%
# def euclidian_distance(points0, points1):
#     dists = np.zeros((len(points0), len(points1)))
#     ### YOUR CODE HERE ###
#     return dists

# %%
# Solution Exercise 1.3

def pairwise_euclidian_distance(points0, points1):
    print("Iterative pairwise euclidian distance")
    dists = []
    for p0 in points0:
        for p1 in points1:
            dists.append(np.sqrt(((p0 - p1)**2).sum()))
            
    dists = np.array(dists).reshape(len(points0), len(points1))
    return dists

# def pairwise_euclidian_distance(points0, points1):
#     # TODO This is slow, make a fast numpy implementation
#     print("Vectorized pairwise euclidian distance")
#     return np.apply_along_axis(
#         np.linalg.norm,
#         2,
#         points0[:, None, :] - points1[None, :, :]
#     )

# def pairwise_euclidian_distance(points0, points1):
#     print("Scipy pairwise euclidian distance")
#     return scipy.spatial.distance.cdist(points0, points1)


# %% [markdown] tags=[]
# Here are two (almost random ;)) lists of points to test your function on.

# %%
green_points = np.load("points.npz")["green"]
cyan_points = np.load("points.npz")["cyan"]

# %%
# %time dists = pairwise_euclidian_distance(green_points, cyan_points)
assert np.allclose(dists, np.load("points.npz")["dists_green_cyan"])


# %% [markdown]
# ## Exercise 1.4
# <div class="alert alert-block alert-info"><h3>Exercise 1.4: Write a function that greedily extracts a nearest neighbors assignment given a cost matrix.</h3></div>

# %%
# def nearest_neighbor(cost_matrix):
#     """Greedy nearest neighbor assignment.
    
#     Each point in both sets can only be assigned once. 
    
#     Args:

#         cost_matrix: m x n matrix with pairwise linking costs of two sets of points.

#     Returns:

#         Determined matches as tuple of lists (ids_of_rows, ids_of_columns).
#     """

#     ids_from = []
#     ids_to = []
#     ### YOUR CODE HERE ###
    
#     return np.array(ids_from), np.array(ids_to)

# %%
# Solution exercise 1.4

def nearest_neighbor(cost_matrix):
    """Greedy nearest neighbor assignment.
    
    Each point in both sets can only be assigned once. 
    
    Args:

        cost_matrix: m x n matrix with pairwise linking costs of two sets of points.

    Returns:

        Tuple of lists (ids frame t, ids frame t+1).
    """
    print("Iterative nearest neighbor")
    A = cost_matrix.copy().astype(float)
    ids_from = []
    ids_to = []
    for i in range(min(A.shape[0], A.shape[1])):
        row, col = np.unravel_index(A.argmin(), A.shape)
        ids_from.append(row)
        ids_to.append(col)
        A[row, :] = cost_matrix.max() + 1
        A[:, col] = cost_matrix.max() + 1

    return np.array(ids_from), np.array(ids_to)


# %% [markdown]
# Test your implementation

# %%
test_matrix = np.array([
    [9, 2, 9],
    [9, 9, 9],
    [1, 9, 9],
    [9, 3, 9],
])
idx_from, idx_to = nearest_neighbor(test_matrix)
assert np.all(idx_from == [2, 0, 1])
assert np.all(idx_to == [0, 1, 2])


# %% [markdown] tags=[]
# ## Exercise 1.5
# <div class="alert alert-block alert-info"><h3>Exercise 1.5: Complete a thresholded nearest neighbor linker using your functions from exercises 1.3 and 1.4.</h3>
#
# You have to write two methods:
#     
# - Method 1 (`linking_cost_function`): Given dense detections in a pair of frames, extract their centroids and calculate pairwise euclidian distances between them. 
# - Method 2 (`_link_two_frames`): For each detection in frame $t$, find the nearest neighbor in frame $t+1$ given the cost matrix. If the distance is below a threshold $\tau$, link the two objects. Explore different values of threshold $\tau$.
# </div>

# %% [markdown]
# Here you are seeing an abstract base class (`ABC`) for linking detections in a video with some local frame-by-frame algorithm.
#
# The class already comes with some useful methods that you won't have to worry about, such as iterating over frames, visualizing linked results as well as sanity checks of inputs.
#
# There are two abstract methods ("gaps") in `FrameByFrameLinker`:
# - `linking_cost_function`
# - `_link_two_frames`
#
# In the exercises 1.5 - 1.8, you will make different subclasses of `FrameByFrameLinker`, in which it will be your job to write these two methods.

# %%
class FrameByFrameLinker(ABC):
    """Abstract base class for linking detections by considering pairs of adjacent frames."""
    
    def link(self, detections, images=None):
        """Links detections in t frames.
        
        Args:
        
            detections:
            
                List of t numpy arrays of shape (x,y) with contiguous label ids. Background = 0.
                
            images (optional):
            
                List of t numpy arrays of shape (x,y).
        
        Returns:
        
            list of linking tuple lists [(ids_frame_0, ids_frame_1), (ids_frame_1, ids_frame_2), ... ]
        """
        if images is not None:
            assert len(images) == len(detections)
        else:
            images = [None] * len(detections)

        links = []
        for i in tqdm(range(len(images) - 1), desc="Linking"):
            detections0 = detections[i]
            detections1 = detections[i+1]
            self._assert_relabeled(detections0)
            self._assert_relabeled(detections1)
            
            cost_matrix = self.linking_cost_function(detections0, detections1, images[i], images[i+1])
            li = self._link_two_frames(cost_matrix)
            links.append(li)
            
        return links

    @abstractmethod
    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """Calculate unary features and extract pairwise costs.
        
        To be overwritten in subclass.
        
        Args:
        
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
            image0 (optional): image corresponding to detections0
            image1 (optional): image corresponding to detections1
            
        Returns:
        
            m x n cost matrix 
        """
        pass
    
    @abstractmethod
    def _link_two_frames(self, cost_matrix):
        """Link two frames.
        
        To be overwritten in subclass.

        Args:

            cost_matrix: m x n matrix

        Returns:

            Tuple of lists (ids frame t, ids frame t+1).
        """
        pass

    def relabel_detections(self, detections, links):
        """Relabel dense detections according to computed links.
        
        # TODO slower in later iterations?
        Args:
        
            detections: 
                 
                 List of t numpy arrays of shape (x,y) with contiguous label ids. Background = 0.
                 
            links:
            
                
 
        """
        
        detections = detections.copy() 
        
        assert len(detections) - 1 == len(links)
        self._assert_relabeled(detections[0])
        out = [detections[0]]
        n_tracks = out[0].max()
        lookup_tables = [{i: i for i in range(1, out[0].max() + 1)}]

        for i in tqdm(range(len(links)), desc="Recoloring detections"):
            new_frame = detections[i+1].copy()
            self._assert_relabeled(new_frame)
            
            lut = {}
            for idx_from, idx_to in zip(links[i][0], links[i][1]):
                # Copy over ID
                new_frame[detections[i+1] == idx_to] = lookup_tables[i][idx_from]
                lut[idx_to] = lookup_tables[i][idx_from]


            # Start new track for all non-linked tracks
            new_ids = set(range(1, new_frame.max() + 1)) - set(links[i][1])
            new_ids = list(new_ids)
                          
            for ni in new_ids:
                n_tracks += 1
                lut[ni] = n_tracks
                new_frame[detections[i+1] == ni] = n_tracks
            # print(lut)
            lookup_tables.append(lut)
            out.append(new_frame)
                
        return np.stack(out)

    def _assert_relabeled(self, x):
        if x.min() < 0:
            raise ValueError("Negative ID in detections.")
        if x.min() == 0:
            n = x.max() + 1
        else:
            n = x.max()
        if n != len(np.unique(x)):
            raise ValueError("Detection IDs are not contiguous.")


# %% [markdown]
# Hints:
# - Check out `skimage.measure.regionprops`.   

# %%
# Solution Exercise 1.5

class NearestNeighborLinkerEuclidian(FrameByFrameLinker):
    """TODO make gaps for exercises"""
    
    def __init__(self, threshold=sys.float_info.max, *args, **kwargs):
        self.threshold = threshold
        super().__init__(*args, **kwargs)
    
    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """ Get centroids from detections and compute pairwise euclidian distances.
                
        Args:
        
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
            
        Returns:
        
            m x n cost matrix 
        """
        # regionprops regions are sorted by label
        regions0 = skimage.measure.regionprops(detections0)
        points0 = [np.array(r.centroid) for r in regions0]
        
        regions1 = skimage.measure.regionprops(detections1)
        points1 = [np.array(r.centroid) for r in regions1]
        
        dists = []
        for p0 in points0:
            for p1 in points1:
                dists.append(np.sqrt(((p0 - p1)**2).sum()))

        dists = np.array(dists).reshape(len(points0), len(points1))
        
        return dists
    
    def _link_two_frames(self, cost_matrix):
        """Greedy nearest neighbor assignment.

        Each point in both sets can only be assigned once. 

        Args:

            cost_matrix: m x n matrix with pairwise linking costs of two sets of points.

        Returns:

            Tuple of lists (ids frame t, ids frame t+1).
            Id indexing needs to start at 1, 0 is reserved for background.
        """
        A = cost_matrix.copy().astype(float)
        ids_from = []
        ids_to = []
        for i in range(min(A.shape[0], A.shape[1])):
            if A.min() >= self.threshold:
                break
            row, col = np.unravel_index(A.argmin(), A.shape)
            ids_from.append(row)
            ids_to.append(col)
            A[row, :] = cost_matrix.max() + 1
            A[:, col] = cost_matrix.max() + 1

        ids_from = np.array(ids_from)
        ids_to = np.array(ids_to)
        
        # Account for +1 offset of the dense labels
        ids_from += 1
        ids_to += 1
        
        births = []
        deaths = []
        
        return ids_from, ids_to
        # return (ids_from, ids_to), births, deaths


# %%
nn_linker = NearestNeighborLinkerEuclidian(threshold=1000) # Explore different values of `threshold`
nn_links = nn_linker.link(detections)
nn_tracks = nn_linker.relabel_detections(detections, nn_links)

# %% [markdown]
# Visualize results

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, nn_tracks, name="nn");


# %%
# TODO metrics
# MOTA
# FLASe div
# false merges

# ANalyse your results visually and quanti.!

# %% [markdown]
# ## Checkpoint 2
# <div class="alert alert-block alert-success"><h3>Checkpoint 2: We have a working basic tracking algorithm :).</h3></div>

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Exercise 1.6
# <div class="alert alert-block alert-info"><h3>Exercise 1.6: Estimate the global drift of the data</h3>
#
# We can observe that all cells move upwards with an approximately constant displacement in each timestep. Write a slightly modified version of `NearestNeighborLinkerEuclidian` with a slightly modified `linking_cost_function` that accounts for this.
#
# </div>

# %%
class NearestNeighborLinkerDriftCorrection(NearestNeighborLinkerEuclidian):
    """.
    
    Args:
        
        drift: tuple of (x,y) drift correction per frame.
    """
    
    def __init__(self, drift, *args, **kwargs):
        self.drift = np.array(drift)
        super().__init__(*args, **kwargs)
        
    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """ Get centroids from detections and compute pairwise euclidian distances with drift correction.
                
        Args:
        
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
            
        Returns:
        
            m x n cost matrix 
        """
        # regionprops regions are sorted by label
        regions0 = skimage.measure.regionprops(detections0)
        points0 = [np.array(r.centroid) for r in regions0]
        
        regions1 = skimage.measure.regionprops(detections1)
        points1 = [np.array(r.centroid) for r in regions1]
        
        dists = []
        for p0 in points0:
            for p1 in points1:
                dists.append(np.sqrt(((p0 + self.drift - p1)**2).sum()))

        dists = np.array(dists).reshape(len(points0), len(points1))
        
        return dists


# %%
# Explore different values of `threshold` and `drift`
# drift_linker = NearestNeighborLinkerDriftCorrection(threshold=30, drift=(-15, 0))
drift_linker = NearestNeighborLinkerDriftCorrection(threshold=1000, drift=(0, 0))
drift_links = drift_linker.link(detections)
drift_tracks = drift_linker.relabel_detections(detections, drift_links)

# %% [markdown]
# Visualize results

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, drift_tracks, name="drift");


# %% [markdown] tags=[]
# ## Optimal frame-by-frame matching (*Linear assignment problem* or *Weighted bipartite matching*)

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Exercise 1.7
# <div class="alert alert-block alert-info"><h3>Exercise 1.7: Perform optimal frame-by-frame linking</h3>
#
# Set up the cost matrix such that you can use [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) to solve the matching problem in the bipartite graph.
#     
# </div>

# %% [markdown]
# ### TODO insert image for bipartite matching
# ### TODO insert image from Jaqaman et al.
#
# Jaqaman, Khuloud, et al. "Robust single-particle tracking in live-cell time-lapse sequences." Nature Methods (2008)

# %%
class BipartiteMatchingLinker(FrameByFrameLinker):
    """TODO make gaps for exercises"""
    
    def __init__(self, threshold=sys.float_info.max, drift=(0,0), *args, **kwargs):
        self.drift = np.array(drift)
        self.threshold = threshold
        super().__init__(*args, **kwargs)
        
    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """ Get centroids from detections and compute pairwise euclidian distances with drift correction.
                
        Args:
        
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
            
        Returns:
        
            m x n cost matrix 
        """
        # regionprops regions are sorted by label
        regions0 = skimage.measure.regionprops(detections0)
        centroids0 = [np.array(r.centroid) for r in regions0]
        
        regions1 = skimage.measure.regionprops(detections1)
        centroids1 = [np.array(r.centroid) for r in regions1]
        
        # TODO vectorize
        dists = []
        for c0 in centroids0:
            for c1 in centroids1:
                dists.append(np.sqrt(((c0 + self.drift - c1)**2).sum()))

        dists = np.array(dists).reshape(len(centroids0), len(centroids1))
        
        return dists
    
#     def _link_two_frames(self, cost_matrix):
#         """Weighted bipartite matching with rectangular matrix.

#         Args:
        
#             cost_matrix: m x n matrix with pairwise linking costs.

#         Returns:

#             Tuple of lists (ids frame t, ids frame t+1).
#         """
#         cost_matrix = cost_matrix.copy().astype(float)
#         cost_matrix[cost_matrix > self.threshold] = sys.float_info.max

#         idx_from, idx_to = scipy.optimize.linear_sum_assignment(cost_matrix)
        
#         # Account for +1 offset of the dense labels
#         idx_from += 1
#         idx_to += 1
        
#         return idx_from, idx_to
    
    def _link_two_frames(self, cost_matrix):
        """Weighted bipartite matching with square matrix from Jaqaman et al (2008).

        Args:
        
            cost_matrix: m x n matrix with pairwise linking costs.

        Returns:

            Tuple of lists (ids frame t, ids frame t+1).
        """
        cost_matrix = cost_matrix.copy().astype(float)
        # print(f"{cost_matrix=}")
        b = d = 1.05 * cost_matrix.max()
        b = d = 1000000
        # prob pf birth / death -> [inf, 0] 
        no_link = b 
        # print(f"{b=}")
        cost_matrix[cost_matrix > self.threshold] = no_link
        lower_right = cost_matrix.transpose()

        deaths = np.full(shape=(cost_matrix.shape[0], cost_matrix.shape[0]), fill_value=no_link)
        np.fill_diagonal(deaths, d)
        births = np.full(shape=(cost_matrix.shape[1], cost_matrix.shape[1]), fill_value=no_link)
        np.fill_diagonal(births, b)
        
        square_cost_matrix = np.block([
            [cost_matrix, deaths],
            [births, lower_right],
        ])
        # print(f"{square_cost_matrix=}")
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(square_cost_matrix)
        # print(f"{row_ind=}")
        # print(f"{col_ind=}")
        
        idx_from = []
        idx_to = []
        
        # Only return links
        for row, col in zip(row_ind, col_ind):
            if row < cost_matrix.shape[0] and col < cost_matrix.shape[1]:
                idx_from.append(row)
                idx_to.append(col)

        idx_from = np.array(idx_from) 
        idx_to = np.array(idx_to) 

        true_matches = cost_matrix[idx_from, idx_to]<no_link
        idx_from, idx_to = idx_from[true_matches], idx_to[true_matches]
        
        print(f'found {np.count_nonzero(~true_matches)} unlinked objects')
                
        # Account for +1 offset of the dense labels
        idx_from, idx_to = idx_from +1, idx_to + 1 
        
        return idx_from, idx_to


# %%
cost_matrix = bm_linker.linking_cost_function(detections[0], detections[1])
self = bm_linker
cost_matrix = cost_matrix.copy().astype(float)
# print(f"{cost_matrix=}")
b = d = 1.05 * cost_matrix.max()
b = d = 1000000
# prob pf birth / death -> [inf, 0] 
no_link = b
# print(f"{b=}")
cost_matrix[cost_matrix > self.threshold] = no_link
lower_right = cost_matrix.transpose()

deaths = np.full(shape=(cost_matrix.shape[0], cost_matrix.shape[0]), fill_value=no_link)
np.fill_diagonal(deaths, d)
births = np.full(shape=(cost_matrix.shape[1], cost_matrix.shape[1]), fill_value=no_link)
np.fill_diagonal(births, b)
square_cost_matrix = np.block([
            [cost_matrix, deaths],
            [births, lower_right],
        ])

# %%
imshow(square_cost_matrix, vmax=100)
cost_matrix.shape


# %%
bm_linker = BipartiteMatchingLinker(threshold=100, drift=(-15, 0))
bm_links = bm_linker.link(detections)
bm_tracks = bm_linker.relabel_detections(detections, bm_links)

# %%
# bm_links

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, bm_tracks, name="bm");


# %% [markdown]
# ### Load ground truth and compute a metric

# %%
# TODO look for library functions.

# %% [markdown] tags=[]
# ## Other suitable features for linking cost function

# %% [markdown]
# ## Exercise 1.8
#
# <div class="alert alert-block alert-info"><h3>Exercise 1.8: Explore different features for assigment problem</h3>
#
# Explore solving the assignment problem based different features and cost functions.
# For example:
# - Different morphological properties of detections (e.g. using `skimage.measure.regionprops`).
# - Extract texture features from the images, e.g. mean intensity for each detection.
# - Pairwise *Intersection over Union (IoU)* of detections.
# - ...
#
# Feel free to share tracking runs for which your features improved the results.    
# </div>

# %%
class YourLinker(BipartiteMatchingLinker):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """ A very smart cost function for frame-by-frame linking.
                
        Args:
        
            detections0: image with background 0 and detections 1, ..., m
            detections1: image with backgruond 0 and detections 1, ..., n
            image0 (optional): image corresponding to detections0
            image1 (optional): image corresponding to detections1
            
        Returns:
        
            m x n cost matrix 
        """
        return np.zeros(detections0.max(), detections1.max())    


# %%
your_linker = YourLinker()
your_links = your_linker.link(detections)
your_tracks = your_linker.relabel_detections(detections, your_links)

# %%
