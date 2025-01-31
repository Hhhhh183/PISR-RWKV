# Power Inspection Super-Resolution via Lightweight RWKV Framework and Specialized Dataset
## Overview
1.We propose a new dataset, the PISR dataset, specifically for the SISR task of power inspection. It is the largest and most comprehensive SR dataset in the power inspection domain, covering a wide range of real-world scenarios.   
2.We are first work to adapt RWKV architecture for SISR task. By incorporating a linear attention mechanism, the model effectively addresses the quadratic complexity issue commonly associated with Transformer-based SR models, achieving a balance between computational efficiency and reconstruction performance.   
3.Extensive experiments have demonstrated that our method outperforms other strong baselines, providing a powerful backbone solution for power inspection SISR task. 
### PISR dataset
 https://pan.baidu.com/s/1QiNUBbOMuHdXEjat6BYh3w?pwd=yteq
### SR-RWKV

## Dependencies and Installation
```
conda create -n srrwkv python=3.10
conda activate srrwkv
pip install -r requirements.txt
```

## train
`python /SR-RWKV/train.py`  
Then, you can find the weights in the /experiment/Weight  

## test
`python /SR-RWKV/test.py`  
Then, you can find the test result in the /experiment/test_result  
## Citation
