# SVAttack
Spatial-Viewpoint Joint Attack on Skeleton-based Action Recognition

依赖于视点，空间梯度裁剪和分布损失技术，本项目在骨架对抗迁移攻击样本的隐蔽性和攻击成功率上有了非常明显的提升。

本项目运行结果参考：   
|  Algorithm | Proxy Model  | AGCN  | MSG3D | STGCN |
|  ----  | ----  | ----  | ----  | ----  | 
| SVAttack | STGCN | 34.18% | 28.63% | 14.42% |
| I-FGSM  | STGCN | 10.46% | 15.32% | 5.94% |
| MI-FGSM  | STGCN | 2.80% | 3.38% | 1.26% |
| SMART  | STGCN | 8.40% | 7.57% | 5.92% |

|  Algorithm | Proxy Model  | 3D_Perceptual  | 2D_Perceptual | var |
|  ----  | ----  | ----  | ----  | ----  | 
| SVAttack | STGCN | 10.73 | 0.09 | 172 |
| I-FGSM  | STGCN | 20.74 | 0.19 | 319 |
| MI-FGSM  | STGCN | 13.19 | 0.12 | 206 |
| SMART  | STGCN | 24.32 | 0.22 | 355|

|  Algorithm | Proxy Model  | HDGCN  | GAP | SelfGCN |
|  ----  | ----  | ----  | ----  | ----  | 
| SVAttack | CTRGCN | 39.65% | 37.05% | 34.74% |
| I-FGSM  | CTRGCN | 33.33% | 22.74% | 23.60% |
| MI-FGSM  | CTRGCN | 20.80% | 14.89% | 16.75% |
| SMART  | CTRGCN | 8.65% | 5.88% | 10.35% |

|  Algorithm | Proxy Model  | 3D_Perceptual  | 2D_Perceptual | var |
|  ----  | ----  | ----  | ----  | ----  | 
| SVAttack | CTRGCN | 15.51 | 0.11 | 34 |
| I-FGSM  | CTRGCN | 14.64 | 0.13 | 45 |
| MI-FGSM  | CTRGCN | 12.90 | 0.11 | 38 |
| SMART  | CTRGCN | 6.10 | 0.05 | 19 |

本项目目录介绍：  
configs：配置文件目录  
feeder：数据加载器配置目录（对于不同模型可能需要不同的输入数据格式）  
models：模型存放目录   
utils：存放相关视点计算与梯度裁剪计算的相关工具类   


运行命令：  
```python
python attack.py --config ./configs/stgcn.yaml
```
