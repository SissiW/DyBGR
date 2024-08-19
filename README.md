# 《Learning Dynamic Batch-Graph Representation for Deep Representation Learning》IJCV 2024

[paper](https://link.springer.com/article/10.1007/s11263-024-02175-8) &nbsp;&nbsp;

## Abstract
Recently, batch-based image data representation has been demonstrated to be effective for context-enhanced image representation. The core issue for this task is capturing the dependences of image samples within each mini-batch and conducting
message communication among different samples. Existing approaches mainly adopt self-attention or local self-attention
models (on patch dimension) for this task which fail to fully exploit the intrinsic relationships of samples within mini-batch
and also be sensitive to noises and outliers. To address this issue, in this paper, we propose a flexible Dynamic Batch-Graph
Representation (DyBGR) model, to automatically explore the intrinsic relationship of samples for contextual sample representation. Specifically, DyBGR first represents the mini-batch with a graph (termed batch-graph) in which nodes represent
image samples and edges encode the dependences of images. This graph is dynamically learned with the constraint of similarity, sparseness and semantic correlation. Upon this, DyBGR exchanges the sample (node) information on the batch-graph
to update each node representation. Note that, both batch-graph learning and information propagation are jointly optimized to
boost their respective performance. Furthermore, in practical, DyBGR model can be implemented via a simple plug-and-play
block (named DyBGR block) which thus can be potentially integrated into any mini-batch based deep representation learning
schemes. Extensive experiments on deep metric learning tasks demonstrate the effectiveness of DyBGR.

## Overview of existing works and our proposed DyBGR model
![overview](https://github.com/SissiW/DyBGR/blob/main/overview.png)

## Architecture of DyBGR module
![DyBGR](https://github.com/SissiW/DyBGR/blob/main/DyBGR.png)

## Architecture of DyBGR-based metric learning framework
![metric_learning](https://github.com/SissiW/DyBGR/blob/main/metric_learning.png)

## Results on four datasets
![results](https://github.com/SissiW/DyBGR/blob/main/results.png)

## Installation
python==3.9.7, pytorch-metric-learning==1.3.0, numpy==1.21, timm==0.3.2, pytorch==1.9.1, torch-scatter==2.0.8, torchaudio=0.8.0

## training & testing
sh train.sh

## Citation
If you find this project useful, please feel free to leave a star and cite our paper:
```
@article{ijcv2024DyBGR,
  title={Learning Dynamic Batch-Graph Representation for Deep Representation Learning},
  author={Wang, Xixi and Jiang, Bo and Wang, Xiao and Luo, Bin},
  journal={International Journal of Computer Vision},
  year={2024}
}
```

## Acknowledgements
DyBGR-based metric learning framework is mainly built upon [Hyp-metric](https://github.com/htdt/hyp_metric). We gratefully thank the authors for their wonderful works.

