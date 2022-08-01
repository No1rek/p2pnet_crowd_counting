Implementation of P2PNet for crowd counting from https://arxiv.org/abs/2107.12746  
Official implementation: https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet  

Model is trained in ShaghaiTech_Part_A dataset, 100 samples for test, 382 for train  

Model path: ./checkpoints/model  
Inference example outputs are stored in ./demo_outputs/  
Training demo: ./demo_train.py  
Inference demo: ./demo_inference.py  

This implementation is more compact than original one and therefore more convenient for learning purposes.  
Also model accuracy is slightly worse because of distinctions in training eval steps and architecture changes along with same hyperparameters from original paper.  

Test metrics: MAE: 69.28, MRSE: 125.1  
