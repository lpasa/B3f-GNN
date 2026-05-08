# Local Learning with Boosting-based Backpropagation-Free Graph Neural Networks

Luca Pasa, Paolo Frazzetto, Nicolò Navarin & Alessandro Sperduti

## Abstract
The framework of Backpropagation-Free Graph Neural Networks (BF-GNNs) enables local learning at the neuron level in GNNs. While BF-GNNs can match the performance of their backpropagation-based counterparts, they may develop redundant internal representations that limit further gains. 

To address this issue, we propose an innovative architecture dubbed Boosting-based Backpropagation-Free GNN (B3F-GNN), where each network module contains multiple backpropagation-free neurons trained locally and combined as a classifier. Within each layer, later modules exploit error signals from earlier trained modules to refine predictions by diversifying internal representations. We implement this approach with two complementary boosting strategies: sample reweighting, in the spirit of AdaBoost, and error-guided prototype selection for gating, which concentrates non-linearities where previous modules struggled. The modular design also enables any-time incremental training by adding more modules on demand within resource constraints. An ablation study and in-depth experimental analysis show that both strategies reduce redundancy and increase specialization, leading to statistically significant accuracy improvements over backpropagation-based counterparts on standard node-classification benchmarks.

Paper: https://link.springer.com/article/10.1007/s10994-026-07041-x

## Citation

If you find this code useful, please cite the following:
﻿@Article{Pasa2026,
author={Pasa, Luca
and Frazzetto, Paolo
and Navarin, Nicol{\`o}
and Sperduti, Alessandro},
title={Local Learning with Boosting-based Backpropagation-Free Graph Neural Networks},
journal={Machine Learning},
year={2026},
month={May},
day={02},
volume={115},
number={5},
pages={111},
abstract={The framework of Backpropagation-Free Graph Neural Networks (BF-GNNs) enables local learning at the neuron level in GNNs. While BF-GNNs can match the performance of their backpropagation-based counterparts, they may develop redundant internal representations that limit further gains. To address this issue, we propose an innovative architecture dubbed Boosting-based Backpropagation-Free GNN (B3F-GNN), where each network module contains multiple backpropagation-free neurons trained locally and combined as a classifier. Within each layer, later modules exploit error signals from earlier trained modules to refine predictions by diversifying internal representations. We implement this approach with two complementary boosting strategies: sample reweighting, in the spirit of AdaBoost, and error-guided prototype selection for gating, which concentrates non-linearities where previous modules struggled. The modular design also enables any-time incremental training by adding more modules on demand within resource constraints. An ablation study and in-depth experimental analysis show that both strategies reduce redundancy and increase specialization, leading to statistically significant accuracy improvements over backpropagation-based counterparts on standard node-classification benchmarks.},
issn={1573-0565},
doi={10.1007/s10994-026-07041-x},
url={https://doi.org/10.1007/s10994-026-07041-x}
}

