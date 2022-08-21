# Exercise 8 - Tracking

## Setup
TODO Test napari launched from jupyter notebook in final environment on machines.

1. CPU-only environment: `env_cpu.yml`.
1. TODO optional: Environment with GPU support: `env_gpu.yml`.


## Exercises

1. Tracking by detection and simple frame by frame matching

    Here we will walk through all basic components of a tracking-by-detection algorithm.
    
    You will learn
    - to **store and visualize** tracking results with `napari`.
    - to use a robust pretrained deep-learning-based **object detection** algorithm called *StarDist*.
    - to implement a basic **nearest-neighbor** linking algorithm.
    - to compute optimal frame-by-frame linking by setting up a **bipartite matching problem** (also called *linear assignment problem (LAP)*) and using a python-based solver.
    - TODO to **evaluate the output** of a tracking algorithm against ground truth annotations.
    - to compute suitable object **features** for the object linking process with `scikit-image`.


2. Tracking with two-step Linear Assignment Problem (LAP)

    Here we will build upon the tracking algorithm introduced in exercise 1 by using a linking algorithm that considers more than two frames at a time in a second optimization step.
    
    You will learn
    - TODO how this formulation addresses **typical challenges of tracking in bioimages**, like cell division and objects temporarily going out of focus.
    - how to use **Trackmate**, a versatile ready-to-go implementation of two-step LAP tracking in ImageJ/Fiji.

3. TODO Advanced topics 

    Here we will introduce more advanced formulations of tracking.
    
    You will learn
    - (to predict costs with a deep learning approach.)
    - to formulate an **integer linear program (ILP)** to find a global optimum solution for small-scale tracking problems with `cvxopt/linajea`.
    - (to set up a **network flow** using `networkx`, which allows to find a global optimum solution for small-scale problems, but without modeling cell divisions.)
