# tracking_dlmbl_2022

## Setup (TODO)


## Internal agenda: (will be removed)

1. Tracking by detection and simple frame-by-frame matching 
    - Overview of [dataset](https://zenodo.org/record/5206107/files/P31-crop.tif?download=1) (migration of cancer cells) 
    - Detection via stardist, vary scale, show distances etc 
    - Simple IoU based frame-by-frame tracking (hungarian matching)
        * metrics
        * custom features with scikit-image
  
2. Linear Assignment (LAP) tracking
    - LAP tracking, build two cost matrices M (frame-to-frame linking) and N (gap closing + division) 
    - (LAP++ with missed division)
    - Application workflow with Trackmate (Fiji)
  

3. Advanced topics and linear optimization
    - Network flow, how to set up, how to solve with networkx 
    - Full ILP (lineajea, cvxopt)


## Exercises:

1. Tracking by detection and simple frame by frame matching: 
    Here we will do X and you will learn A, B, C

2. Linear Assignment (LAP) tracking
    Here we will do X and you will learn A, B, C

3. Advanced topics and linear optimization
    Here we will do X and you will learn A, B, C
