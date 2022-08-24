# Exercise 8 - Tracking

## Setup
TODO Test napari launched from jupyter notebook in final environment on machines.

1. CPU-only environment: `env_cpu.yml`.


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

    Here we will use an extended version of the tracking algorithm introduced in exercise 1 which uses a linking algorithm that considers more than two frames at a time in a second optimization step.
    
    You will learn
    - how this formulation addresses **typical challenges of tracking in bioimages**, like cell division and objects temporarily going out of focus.
    - how to use **Trackmate**, a versatile ready-to-go implementation of two-step LAP tracking in `ImageJ/Fiji`.



    
3. Tracking with an integer linear program (ILP)

    Here we will introduce a modern formulation of tracking.

    You will learn
    - how linking with global context can be modeled as a **network flow** using `networkx` and solved efficiently as an **integer linear program (ILP)** with `cvxpy` for small-scale problems (Exercise 3.1).
    - to adapt the previous formulation to allow for **arbitrary track starting and ending points** (Exercise 3.2).
    - to extend the ILP to properly model **cell divisions** (Exercise 3.3).
    - to tune the **hyperparameters** of the ILP (Exercise 3.4).
