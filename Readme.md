# Objective
Implicit interpolation methods represent geological interfaces as isosurfaces of a scalar field, effectively capturing the continuity and structural attitude of conformable strata. However, they struggle to represent geological discontinuities, such as those associated with unconformities, faults, and intrusive rocks. Conversely, lithostratigraphic classification approaches, which frame 3D geological modeling as a classification task, can readily represent these discontinuities but often fail to preserve the inherent continuity of stratigraphic units. Consequently, a significant challenge remains in balancing stratum continuity with geological discontinuities in complex setting. 
# Methods
To address this, we propose a dual-task graph neural network (GNN) framework, introducing a topology-aware boundary sampling strategy to preserve line structural information from geological maps. By incorporating boundaries, attitudes, and lithostratigraphic points into a tetrahedron mesh as an input graph of GNN, the method simultaneously performs lithostratigraphic classification and implicit interpolation. 
# Results 
Experimental results demonstrate that a configuration using a graph attention network (GAT) for interpolation and a graph sample and aggregate (GraphSAGE) network for classification, along with fault encoding, delivers optimal performance. Application to a real-world case study in the Lingnian area of Guangxi, China, and comparison with Hermite radial basis function (HRBF) interpolation, show that our method exhibits superior robustness in complex geological settings. 
# Conclusions
The proposed Dual-GNN framework effectively expresses fault-induced geological discontinuities while preserving the continuity of conformable strata. It enables the one-shot reconstruction of stratigraphic solid models under complex conditions, thereby extending the capabilities of current GNN-based geological modeling methods.

# Code Description
This document comprises the core source code for graph neural network modelling methods and small scripts for data processing procedures.
## tool
Scripts for data pre-processing and post-processing
## output
Functions related to model output and image generation
## model
Model Configuration Class
## input
Function for processing input data
## loss
Loss function
## train
Training Script and Parameter Configuration
## utils
Metric calculations and mathematical functions referenced within other scripts
## environment
This code environment has been configured within the pyg container on the Geo_ML server and is ready for direct invocation.
