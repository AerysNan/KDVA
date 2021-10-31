## VIRAT Dataset

|   Model    |   VIRAT 2   |
| :--------: | :---------: |
| MobileNet  | 0.337/0.417 |
|   YOLOv3   | 0.403/0.490 |
|  ResNet50  | 0.448/0.519 |
| ResNet101  | 0.479/0.551 |
| ResNext101 | 0.463/0.532 |

## Distillation SSD

| epoch |    no-op    |   distill   | distill-acc |
| :---: | :---------: | :---------: | :---------: |
|   1   | 0.007/0.096 | 0.014/0.072 | 0.014/0.072 |
|   2   | 0.054/0.113 | 0.214/0.339 | 0.240/0.281 |
|   3   | 0.341/0.410 | 0.059/0.091 | 0.079/0.095 |
|   4   | 0.395/0.456 | 0.417/0.490 | 0.422/0.497 |
|   5   | 0.366/0.435 | 0.377/0.458 | 0.397/0.479 |
|   6   | 0.085/0.225 | 0.163/0.286 | 0.225/0.322 |
|   7   | 0.015/0.111 | 0.132/0.181 | 0.177/0.222 |
|   8   | 0.022/0.108 | 0.121/0.196 | 0.102/0.156 |
|   9   | 0.007/0.111 | 0.220/0.280 | 0.160/0.209 |

| epoch |    no-op    |   distill   | distill-acc |
| :---: | :---------: | :---------: | :---------: |
|   1   | 0.006/0.090 | 0.007/0.105 | 0.007/0.105 |
|   2   | 0.005/0.115 | 0.136/0.246 | 0.143/0.285 |
|   3   | 0.342/0.411 | 0.077/0.104 | 0.083/0.108 |
|   4   | 0.397/0.456 | 0.409/0.486 | 0.408/0.494 |
|   5   | 0.370/0.442 | 0.396/0.464 | 0.435/0.537 |
|   6   | 0.156/0.283 | 0.406/0.475 | 0.353/0.440 |
|   7   | 0.016/0.113 | 0.099/0.218 | 0.143/0.237 |
|   8   | 0.021/0.110 | 0.105/0.197 | 0.156/0.222 |
|   9   | 0.007/0.105 | 0.101/0.195 | 0.230/0.303 |