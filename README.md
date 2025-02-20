# Power Inspection Super-Resolution via Lightweight RWKV Framework and Specialized Dataset
## Abstract
With the continual advancement of modern power systems, there is an increasing demand for high accuracy and reliability in image processing technologies for power inspection tasks. Existing datasets, primarily focusing on natural scenes, fail to adequately address the specific requirements of power inspection. To address this gap, we introduce the Power Inspection Single Image Super-Resolution (PISR) dataset, comprising nearly 1,000 high-resolution (HR) and low-resolution (LR) image pairs tailored for power inspection scenarios. This dataset covers a wide range of electrical equipment in real inspection settings, ensuring data quality and diversity through rigorous screening and preprocessing. Furthermore, we propose a novel lightweight power inspection framework named SR-RWKV, which leverages a linear attention mechanism to mitigate the computational complexity of traditional Transformer-based models. Extensive experiments demonstrate that SR-RWKV achieves state-of-the-art performance across standard benchmarks while significantly reducing computational costs. Notably, the PISR dataset contributes to a PSNR improvement of 1.43 dB compared to the DIV2K dataset. 
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

## citation
@article{PISR,  
   &emsp;&emsp;title={{Power Inspection Super-Resolution via Lightweight RWKV Framework and Specialized Dataset},  
   &emsp;&emsp;author={Liang, Wanyong and Liu, Jie and Deng, Wei and Lei, Xiaoyan and Cao, Weifeng and Qian, Xiaoliang},  
   &emsp;&emsp;journal={The Visual Computer},  
   &emsp;&emsp;pages={1--22},  
   &emsp;&emsp;year={2025},  
   &emsp;&emsp;publisher={Springer}  
}
