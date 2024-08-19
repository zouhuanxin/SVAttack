# SVAttack
Spatial-Viewpoint Joint Attack on Skeleton-based Action Recognition

By relying on viewpoint, spatial gradient cropping, and distribution loss techniques, this project has significantly improved the concealment and attack success rate of skeleton adversarial transfer attack samples.

## The operating results of this project refer to:  
Among them, 3D-Perceptial represents the perception of the generated sample in 3D, 2D-Perceptial represents the perception of the generated sample in a specific 2D perspective, and var represents the sample variance.
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

|  Algorithm | Proxy Model  | perception  | STGCN | MSG3D | ASGCN |
|  ----  | ----  | ----  | ----  | ----  |  ----  | 
| SVAttack | AGCN | 12.77 | 4.58% | 24.44% | 28.34% |
| S<sub>2</sub>I-FGSM  | AGCN | 27.67 | 5.97% | 7.61% | 9.07% |
| S<sub>2</sub>I-FGSM  | AGCN | 45.67 | 8.84% | 9.99% | 12.29% |
| S<sub>2</sub>I-FGSM  | AGCN | 65.60 | 9.96% | 11.69% | 14.67% |
| S<sub>2</sub>MI-FGSM  | AGCN | 37.46 | 6.66% | 8.39% | 10.90% |
| S<sub>2</sub>MI-FGSM  | AGCN | 62.04 | 9.19% | 12.98% | 14.37% |
| S<sub>2</sub>MI-FGSM  | AGCN | 90.93 | 11.56% | 14.11% | 17.75% |

## Catalog introduction of this project:    
configs：Configuration file directory  
feeder：Data loader configuration directory (different input data formats may be required for different models)  
models：model storage directory  
utils：Tool classes for storing related viewpoint calculations and gradient clipping calculations  


## Run command:    
```python
python attack.py --config ./configs/stgcn.yaml
```
