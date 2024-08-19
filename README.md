# SVAttack
Spatial-Viewpoint Joint Attack on Skeleton-based Action Recognition

依赖于视点，空间梯度裁剪和分布损失技术，本项目在骨架对抗迁移攻击样本的隐蔽性和攻击成功率上有了非常明显的提升。

本项目目录介绍：  
configs：配置文件目录  
feeder：数据加载器配置目录（对于不同模型可能需要不同的输入数据格式）  
models：模型存放目录   
utils：存放相关视点计算与梯度裁剪计算的相关工具类   


运行命令：  
```python
python attack.py --config ./configs/stgcn.yaml
```
