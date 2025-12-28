Implicit interpolation methods represent geological interfaces as isosurfaces of a scalar field, effectively capturing the continuity and structural attitude of conformable strata. However, they struggle to represent geological discontinuities, such as those associated with unconformities, faults, and intrusive rocks. Conversely, lithostratigraphic classification approaches, which frame 3D geological modeling as a classification task, can readily represent these discontinuities but often fail to preserve the inherent continuity of stratigraphic units. Consequently, a significant challenge remains in balancing stratum continuity with geological discontinuities in complex setting. (Methods) To address this, we propose a dual-task graph neural network (GNN) framework, introducing a topology-aware boundary sampling strategy to preserve line structural information from geological maps. By incorporating boundaries, attitudes, and lithostratigraphic points into a tetrahedron mesh as an input graph of GNN, the method simultaneously performs lithostratigraphic classification and implicit interpolation. (Results) Experimental results demonstrate that a configuration using a graph attention network (GAT) for interpolation and a graph sample and aggregate (GraphSAGE) network for classification, along with fault encoding, delivers optimal performance. Application to a real-world case study in the Lingnian area of Guangxi, China, and comparison with Hermite radial basis function (HRBF) interpolation, show that our method exhibits superior robustness in complex geological settings. (Conclusions) The proposed Dual-GNN framework effectively expresses fault-induced geological discontinuities while preserving the continuity of conformable strata. It enables the one-shot reconstruction of stratigraphic solid models under complex conditions, thereby extending the capabilities of current GNN-based geological modeling methods.

# 代码说明
本文件为图神经网络建模方法的核心源码及数据处理过程的小脚本
## tool
`数据前处理及后处理的脚本`
## output
`模型输出、成图有关的函数`
## model
`模型设置类`
## input
`对输入数据进行处理的函数`
## loss
`损失函数`
## train
`训练脚本及参数设置`
## utils
`其他脚本里面调用到的指标计算、数学函数`
## environment
该代码环境已配置在Geo_ML服务器的pyg容器中，可以直接调用。
如需重新配置环境，`可参照environment-融合地质拓扑关系知识图谱的图神经网络三维地质建模方法优化文档中的配置教程、environment.yml`进行配置
- 如使用environment.yml进行配置，可通过以下命令构建同一个conda环境
`conda env create -f environment.yml`
