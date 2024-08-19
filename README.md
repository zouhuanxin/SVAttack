# SVAttack
Spatial-Viewpoint Joint Attack on Skeleton-based Action Recognition

依赖于视点，空间梯度裁剪和分布损失技术，本项目在骨架对抗迁移攻击样本的隐蔽性和攻击成功率上有了非常明显的提升。

本项目运行结果参考：   
|  Algorithm   | AGCN  | MSG3D | STGCN |
|  ----  | ----  | ----  | ----  |
| SVAttack  | 34.18% | 28.63% | 14.42% |
| I-FGSM  | 10.46% | 15.32% | 5.94% |
| MI-FGSM  | 2.80% | 3.38% | 1.26% |
| SSMART  | 8.40% | 7.57% | 5.92% |

本项目目录介绍：  
configs：配置文件目录  
feeder：数据加载器配置目录（对于不同模型可能需要不同的输入数据格式）  
models：模型存放目录   
utils：存放相关视点计算与梯度裁剪计算的相关工具类   


运行命令：  
```python
python attack.py --config ./configs/stgcn.yaml
```
