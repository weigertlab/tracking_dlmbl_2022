# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Import packages

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
import numpy as np
import cvxpy as cp

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
base_path = Path("~/data/celltracking/BF-C2DL-MuSC").expanduser()

#if base_path.exists():
#    print("Dataset already downloaded.")
#else:
# #    !curl https://drive.switch.ch/index.php/s/DUwFtY7LAxOFTUW/download --create-dirs -o data/cancer_cell_migration.zip
# #    !unzip -q data/cancer_cell_migration.zip -d data

# %%
last_n = 50
offset = last_n - len(sorted((base_path/ "01").glob("*.tif")))
x = np.stack([imread(str(p)) for p in sorted((base_path/ "01").glob("*.tif"))[-last_n:]])
y = np.stack([imread(str(p)) for p in sorted((base_path/ "01_GT"/ "TRA").glob("*.tif"))[-last_n:]])
assert len(x) == len(x)
print(f"Number of images: {len(x)}")
print(f"Image shape: {x[0].shape}")

# %%
x = x[:10, 128:384, -384:-128]
y = y[:10, 128:384, -384:-128]
offset -= 0

print(f"Number of images: {len(x)}")
print(f"Image shape: {x[0].shape}")
x, y = preprocess(x, y)

# %% [markdown]
# Load the dataset (images and tracking annotations) from disk into this notebook.

# %%
links = pd.read_csv(base_path / "01_GT" / "TRA"/ "man_track.txt", names=["track_id", "from", "to", "parent_id"], sep=" ")
print("Links")
links["from"] = links["from"] + offset
links["to"] = links["to"] + offset
links[:10]

# %% [markdown]
# Crop the dataset in time and space to reduce runtime; preprocess.

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
# - When you are coding debugging and debugging, close the napari viewer with `viewer.close()` to avoid problems with the asynchronous napari and jupyter event loops. Sometimes you might have to execute a cell twice to get through.
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

# %% [markdown] tags=[]
# ## Object detection using a pre-trained neural network

# %% [markdown] tags=[]
# ### Load a pretrained stardist model, detect nuclei in one image and visualize them.

# %%
idx = 0
plot_img_label(x[idx], y[idx])

# %% tags=[]
idx = 0
# model = StarDist2D.from_pretrained("2D_versatile_fluo")
model = StarDist2D(None, name="BF-C2DL-MuSC", basedir="models")
detections, details = model.predict_instances(x[idx], scale=(1, 1), nms_thresh=0.3, prob_thresh=0.5)
plot_img_label(x[idx], detections, lbl_title="detections")

# %% [markdown]
# Here we visualize in detail the polygons we have detected with StarDist. TODO some description on how StarDist works.
#
# <!-- Notice that each object comes with a center point, which we can use to comnms_thresh=ise eprob_thresh= distances between objects. -->

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
# Detect centers and segment nuclei in all images of the time lapse.

# %%
# TODO adapt thresholds
prob_thres = 0.3
nms_thres = 0.3
scale = (1.0, 1.0)
pred = [model.predict_instances(xi, show_tile_progress=False, scale=scale, nms_thresh=nms_thres, prob_thresh=prob_thres)
              for xi in tqdm(x)]
detections = [xi[0] for xi in pred]
detections = np.stack([skimage.segmentation.relabel_sequential(d)[0] for d in detections])  # ensure that label ids are contiguous and start at 1 for each frame 
centers = [xi[1]["points"] for xi in pred]

# %% [markdown]
# Visualize the dense detections. Note that they are still not linked and therefore randomly colored.

# %%
viewer = napari.viewer.current_viewer()
if viewer:
    viewer.close()
viewer = napari.Viewer()
viewer.add_image(x)
visualize_tracks(viewer, y, links.to_numpy(), "ground_truth");
viewer.add_labels(detections, name=f"detections_scale_{scale}_nmsthres_{nms_thres}");

# %% [markdown]
# We see that the number of detections increases over time, corresponding to the cells that insert the field of view from below during the video.

# %%
plt.figure(figsize=(10,6))
plt.bar(range(len(centers)), [len(xi) for xi in centers])
plt.title(f"Number of detections in each frame (scale={scale})")
plt.xticks(range(len(centers)))
plt.show();


# %%
def build_graph(detections, max_distance=np.finfo(float).max):
    """TODO cleanup"""
    G = nx.DiGraph()
    # e = [v0, v1]

    n_v = 0
    
    luts = []
    draw_positions = {}
    
    for t, d in enumerate(detections):
        frame = skimage.segmentation.relabel_sequential(d)[0]
        regions = skimage.measure.regionprops(frame)
        lut = {}
        for r in regions:
            G.add_node(n_v, time=t, detection_id=r.label, weight=1)
            draw_positions[n_v] = np.array([t, d.shape[0] - r.centroid[0]])
            lut[r.label] = n_v
            n_v += 1
        luts.append(lut)

    n_e = 0
    for t, (d0, d1) in enumerate(zip(detections, detections[1:])):
        f0 = skimage.segmentation.relabel_sequential(d0)[0]
        r0 = skimage.measure.regionprops(f0)
        c0 = [np.array(r.centroid) for r in r0]
        # print(c0)

        f1 = skimage.segmentation.relabel_sequential(d1)[0]
        r1 = skimage.measure.regionprops(f1)
        c1 = [np.array(r.centroid) for r in r1]
        # print(c1)

        for _r0, _c0 in zip(r0, c0):
            for _r1, _c1 in zip(r1, c1):
                dist = np.linalg.norm(_c0 - _c1)
                if dist < max_distance:
                    G.add_edge(
                        luts[t+1][_r1.label],
                        luts[t][_r0.label],
                        # normalized euclidian distance
                        weight = np.linalg.norm(_c0 - _c1) / np.linalg.norm(detections[t].shape),
                        edge_id = n_e,
                    )
                    n_e += 1
    
    return G, draw_positions


# %%
def build_graph_from_tracks(detections, links=None):
    G = nx.DiGraph()
    # e = [v0, v1]

    n_v = 0
    
    luts = []
    draw_positions = {}
    
    for t, d in enumerate(detections):
        frame = d
        regions = skimage.measure.regionprops(frame)
        lut = {}
        for r in regions:
            G.add_node(n_v, time=t, detection_id=r.label, weight=1)
            draw_positions[n_v] = np.array([t, d.shape[0] - r.centroid[0]])
            lut[r.label] = n_v
            n_v += 1
        luts.append(lut)
        
    n_e = 0
    for t, (d0, d1) in enumerate(zip(detections, detections[1:])):
        f0 = d0
        r0 = skimage.measure.regionprops(f0)
        c0 = [np.array(r.centroid) for r in r0]
        # print(c0)

        f1 = d1
        r1 = skimage.measure.regionprops(f1)
        c1 = [np.array(r.centroid) for r in r1]
        # print(c1)

        for _r0, _c0 in zip(r0, c0):
            for _r1, _c1 in zip(r1, c1):
                if _r0.label == _r1.label:
                    G.add_edge(
                        luts[t+1][_r1.label],
                        luts[t][_r0.label],
                        # normalized euclidian distance
                        weight = np.linalg.norm(_c0 - _c1) / np.linalg.norm(detections[t].shape),
                        edge_id = n_e,
                    )
                    n_e += 1
    
    if links is not None:
        divisions = links[links[:,3] != 0]
        for d in divisions:
            if d[1] > 0 and d[1] < detections.shape[0]:
                try:
                    G.add_edge(luts[d[1]][d[0]], luts[d[1] - 1][d[3]])
                    # print("Division edge")
                except KeyError:
                    pass
                    # print(d)
                    # print("Can't find parent in previous frame (cropping, disappearing tracks).")
    
    return G, draw_positions


# %%
def draw_graph(g, pos, title=None):
    fig, ax = plt.subplots()
    plt.title(title)
    nx.draw(g, pos=pos, with_labels=True, ax=ax)

    ax.set_axis_on()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    plt.xlabel("time")
    plt.ylabel("y (spatial)");


# %%
gt_graph, gt_pos = build_graph_from_tracks(y, links.to_numpy())
draw_graph(gt_graph, gt_pos, "Ground truth graph")

# %%
graph, draw_pos = build_graph(detections, max_distance=50)
draw_graph(graph, draw_pos, "Candidate graph")


# %%
def graph2ilp_nodiv(graph, hyperparams):
    """TODO cleanup"""
    edge_to_idx = lambda x: {edge: i for i, edge in enumerate(graph.edges)}[x]
    E = graph.number_of_edges()
    V = graph.number_of_nodes()
    x = cp.Variable(E + 3*V, boolean=True)
    
    c_e = hyperparams["edge_factor"] * np.array([graph.get_edge_data(*e)["weight"] for e in graph.edges])
    # print(c_e)
    c_v = hyperparams["node_offset"] + hyperparams["node_factor"] * np.array([v for k, v in sorted(dict(graph.nodes(data="weight")).items())])
    # print(c_v)
    c_va = np.ones(V) * hyperparams["cost_appear"]
    c_vd = np.ones(V) * hyperparams["cost_disappear"]
    c = np.concatenate([c_e, c_v, c_va, c_vd])
    
    # constraint matrices: {E or V} x (E + 3V)
    # columns: ce, c_v, c_va, c_vd
    
    A0 = np.zeros((E, E + 3 * V))
    A0[:E, :E] = 2 * np.eye(E)
    for edge in graph.edges:
        edge_id = edge_to_idx(edge)
        A0[edge_id, E + edge[0]] = -1
        A0[edge_id, E + edge[1]] = -1
    
    # Appear continuation
    A1 = np.zeros((V, E + 3 * V))
    A1[:, E:E+V] = -np.eye(V)
    A1[:, E+V:E+2*V] = np.eye(V)
    
    for node in graph.nodes:
        out_edges = graph.out_edges(node)
        for edge in out_edges:
            edge_id = edge_to_idx(edge)
            A1[node, edge_id] = 1
     
    # Disappear continuation
    A2 = np.zeros((V, E + 3 * V))
    A2[:, E:E+V] = np.eye(V)
    A2[:, E+2*V:E+3*V] = - np.eye(V)
    
    for node in graph.nodes:
        in_edges = graph.in_edges(node)
        for edge in in_edges:
            edge_id = edge_to_idx(edge)
            A2[node, edge_id] = -1
    
    constraints = [
        A0 @ x <= 0, 
        A1 @ x == 0,
        A2 @ x == 0,
    ]
    
    
    
    # objective = cp.Minimize( c_v.T @ x_v + c_e.T @ x_e)
    objective = cp.Minimize( c.T @ x)

    
    return cp.Problem(objective, constraints)

def solution2graph(solution):
    pass

def graph2links(solution_graph):
    # list of links, births, deaths
    pass

# TODO adapt visualizer


# %%
graph = build_graph(detections)
ilp = graph2ilp_nodiv(graph, hyperparams={"cost_appear": 1, "cost_disappear": 1, "node_offset": 0, "node_factor": -1, "edge_factor": 1})

# %%
ilp.solve()
print("ILP Status: ", ilp.status)
print("The optimal value is", ilp.value)
print("x_e")
E = graph.number_of_edges()
V = graph.number_of_nodes()
print(ilp.variables()[0].value[:E])
print("x_v")
print(ilp.variables()[0].value[E:E+V])
print("x_va")
print(ilp.variables()[0].value[E+V:E+2*V])
print("x_vd")
print(ilp.variables()[0].value[E+2*V:E+3*V])


# %%

# %%
# TODO some tests for the students

# %%
# nx.draw(graph)

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
        
            List of t linking dictionaries, each containing:
                "links": Tuple of lists (ids frame t, ids frame t+1),
                "births": List of ids,
                "deaths": List of ids.
            Ids are one-based, 0 is reserved for background.
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
            self._assert_links(links=li, time=i, detections0=detections0, detections1=detections1) 
            links.append(li)
            
        return links

    @abstractmethod
    def linking_cost_function(self, detections0, detections1, image0=None, image1=None):
        """Calculate features for each detection and extract pairwise costs.
        
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
        
            Linking dictionary:
                "links": Tuple of lists (ids frame t, ids frame t+1),
                "births": List of ids,
                "deaths": List of ids.
            Ids are one-based, 0 is reserved for background.
        """
        pass

    def relabel_detections(self, detections, links):
        """Relabel dense detections according to computed links, births and deaths.
        
        Args:
        
            detections: 
                 
                 List of t numpy arrays of shape (x,y) with contiguous label ids. Background = 0.
                 
            links:
                
                List of t linking dictionaries, each containing:
                    "links": Tuple of lists (ids frame t, ids frame t+1),
                    "births": List of ids,
                    "deaths": List of ids.
                Ids are one-based, 0 is reserved for background.
        """
        detections = detections.copy()
        
        assert len(detections) - 1 == len(links)
        self._assert_relabeled(detections[0])
        out = [detections[0]]
        n_tracks = out[0].max()
        lookup_tables = [{i: i for i in range(1, out[0].max() + 1)}]

        for i in tqdm(range(len(links)), desc="Recoloring detections"):
            (ids_from, ids_to) = links[i]["links"]
            births = links[i]["births"]
            deaths = links[i+1]["deaths"] if i+1 < len(links) else []
            new_frame = np.zeros_like(detections[i+1])
            self._assert_relabeled(detections[i+1])
            
            lut = {}
            for _from, _to in zip(ids_from, ids_to):
                # Copy over ID
                new_frame[detections[i+1] == _to] = lookup_tables[i][_from]
                lut[_to] = lookup_tables[i][_from]

            
            # Start new track for birth tracks
            for b in births:
                if b in deaths:
                    continue
                
                n_tracks += 1
                lut[b] = n_tracks
                new_frame[detections[i+1] == b] = n_tracks
                
            # print(lut)
            lookup_tables.append(lut)
            out.append(new_frame)
                
        return np.stack(out)

    def _assert_links(self, links, time, detections0, detections1):
        if len(links["links"][0]) != len(links["links"][1]):
            raise RuntimeError("Format of links['links'] not correct.")
            
        if sorted([*links["links"][0], *links["deaths"]]) != list(range(1, len(np.unique(detections0)))):
            raise RuntimeError(f"Some detections in frame {time} are not properly assigned as either linked or death.")
            
        if sorted([*links["links"][1], *links["births"]]) != list(range(1, len(np.unique(detections1)))):
            raise RuntimeError(f"Some detections in frame {time + 1} are not properly assigned as either linked or birth.")
            
        for b in links["births"]:
            if b in links["links"][1]:
                raise RuntimeError(f"Links frame {time+1}: Detection {b} marked as birth, but also linked.")
        
        for d in links["deaths"]:
            if d in links["links"][0]:
                raise RuntimeError(f"Links frame {time}: Detection {d} marked as death, but also linked.")
        
        
    def _assert_relabeled(self, x):
        if x.min() < 0:
            raise ValueError("Negative ID in detections.")
        if x.min() == 0:
            n = x.max() + 1
        else:
            n = x.max()
        if n != len(np.unique(x)):
            raise ValueError("Detection IDs are not contiguous.")

# %%
# TODO metrics
# MOTA
# False divs
# False merges

# Analyse your results visually and quantitatively

# %%
