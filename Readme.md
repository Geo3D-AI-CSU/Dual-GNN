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