# PowerInsSR: A Large-Scale Dataset for Power Inspection Super-Resolution with RWKV-Based Hierarchical Attention Network
## Abstract
Image super-resolution has demonstrated outstanding performance in high-level vision tasks within power systems, such as equipment detection and fault diagnosis. However, due to the scarcity of high-resolution power images, existing methods often rely on natural scene datasets for training, limiting their adaptability to power-related scenarios. To address this gap, we introduce the Power Inspection Image Super-Resolution (PISR) dataset, which consists of nearly 1000 pairs of high-resolution and low-resolution images specifically tailored for power inspection scenarios. The dataset covers various electrical equipment, including substations and transmission lines, and undergoes rigorous filtering and preprocessing to ensure high quality and diversity, making it more suitable for SR tasks in power image analysis. In addition, we propose a lightweight super-resolution framework for power inspection named SR-RWKV, which leverages a linear attention mechanism to significantly reduce the computational complexity of traditional Transformer-based models. Extensive experiments demonstrate that SR-RWKV achieves state-of-the-art performance across multiple benchmarks while notably lowering computational costs. Notably, compared to models trained on the DIV2K dataset, models trained on PISR achieve a 1.43 dB improvement in PSNR, highlighting its effectiveness in power system applications. 
## Overview
### PISR dataset
We propose a new dataset, the PISR dataset, specifically for the SISR task of power inspection. It is the largest and most comprehensive SR dataset in the power inspection domain, covering a wide range of real-world scenarios. https://pan.baidu.com/s/1QiNUBbOMuHdXEjat6BYh3w?pwd=yteq  
### SR-RWKV
We are first work to adapt RWKV architecture for SISR task. By incorporating a linear attention mechanism, the model effectively addresses the quadratic complexity issue commonly associated with Transformer-based SR models, achieving a balance between computational efficiency and reconstruction performance. SR-RWKV is a lightweight network with a linear attention mechanism which eliminates the quadratic complexity of the Transformer. Extensive experiments have demonstrated that our method outperforms other strong baselines, providing a powerful backbone solution for power inspection SISR task. 


## Dependencies and requirements
```
conda create -n srrwkv python=3.10
conda activate srrwkv
pip install -r requirements.txt
```
## Implementations of key algorithms
### train
`python /SR-RWKV/train.py`  
Then, you can find the weights in the /experiment/Weight  

### test
`python /SR-RWKV/test.py`  
Then, you can find the test result in the /experiment/test_result  
