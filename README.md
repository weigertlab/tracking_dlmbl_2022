# Exercise 8 - Tracking

## Questions

- Notebooks do not need a GPU. Can we ask participants to run locally (and thereby easily use napari for visus)?
- Experience from last year: How much content?

## Setup

TODO

## Exercises:

1. Tracking by detection and simple frame by frame matching

    Here we will walk through all basic components of a tracking-by-detection algorithm.
    
    You will learn
    - to use a robust pretrained deep-learning-based **object detection** algorithm called _StarDist_.
    - to compute suitable object **features** for the object linking process with `scikit-image`.
    - to set up the optimal algorithm for **frame-by-frame matching** called _Hungarian matching_ and to use a solver in python.
    - to **evaluate the output** of a tracking algorithm against a ground truth annotation.

2. Tracking with Linear Assignment Problem (LAP)

    Here we will improve the tracking algorithm introduced in exercise 1 by using a linking algorithm that considers more than two frames at a time, the _Linear Assignment Problem_ (LAP).
    
    You will learn
    - how this formulation addresses **typical challenges of tracking in bioimages**, like cell division and objects temporarily going out of focus.
    - to set up the two **LAP cost matrices** step by step and how to use a solver in python.
    - how to use **_Trackmate_**, a versatile ready-to-go implementation of LAP tracking in ImageJ.

3. OPTIONAL: Advanced topics and linear optimization

    Here we will introduce more advanced formulations of tracking.
    
    You will learn
    - to set up a **network flow** using `networkx`, which allows to find a global optimum solution for small-scale problems, but without modeling cell divisions.
    - to formulate an **integer linear program (ILP)** to find a global optimum solution for small-scale tracking problems with `cvxopt`.

## Internal agenda: (will be removed)

1. Tracking by detection and simple frame-by-frame matching (~2h)
    - Overview of [dataset](https://zenodo.org/record/5206107/files/P31-crop.tif?download=1) (migration of cancer cells) 
    - Detection via stardist, vary scale, show distances etc 
    - Simple IoU based frame-by-frame tracking (hungarian matching)
        * metrics
        * custom features with scikit-image
  
2. Linear Assignment (LAP) tracking (~1.5h)
    - LAP tracking, build two cost matrices M (frame-to-frame linking) and N (gap closing + division) 
    - (LAP++ with missed division)
    - Application workflow with Trackmate (Fiji)
  

3. Advanced topics and linear optimization (~1h)
    - Network flow, how to set up, how to solve with networkx 
    - Full ILP (lineajea, cvxopt)
