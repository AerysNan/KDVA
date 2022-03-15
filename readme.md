# 测试方法

## 生成 groundtruth

1. 将 DETRAC XML 转化为 COCO 格式的标注并将多个 DETRAC trace 合并

    `tools/detrac.sh` 提供了一键格式转化与合并的脚本, 输入为需要合并的 trace 以及输出的 trace 名称. 具体操作如下:
    
    首先将 DETRAC 数据集放在 ~/DETRAC 路径下, 其下包含 `annotations` 和 `videos` 两个文件夹. 在 `detrac.sh` 中的 `files` 设定为需要转化与合并的 trace 列表, 在 line 14 将 `-o` 参数设置为输出的 trace 名称. 执行后会在 `./data/annotations/` 下生成 `*.gt.json` 的 COCO 格式标注, 同时 `./data/` 下生成一个和 trace 同名的文件夹存放所有图片

    

2. 将 trace 裁剪为等长

    见 `tools/mv_dataset.py`, 设定起始 index 与终止 index. 具体操作如下:

    假设上一步输出的 trace 名称为 `merge_trace_1`, 那么可执行 `python3 tools/mv_dataset.py -s merge_trace_1 -t detrac_trace_1 -b 0 -e 10000` 将其裁剪为长度为 10000 帧名称为 `detrac_trace_1` 的 trace, 注意输出名称需要不同于输入

## 生成训练用 golden label

首先使用 `tools/model_test.py` 生成 inference 结果, 然后使用 `tools/result_to_annotation.py` 生成 golden label. 具体操作如下:

1. 将从 mmdetection 上下载的 pth 文件放在 `./checkpoints/` 文件夹中, 假设使用 `faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth`, 同时将其对应的配置文件放在 `./configs/` 文件夹中, 假设名为 `faster_rcnn_r101_fpn_1x_coco.py`,则执行 inference `detrac_trace_1` 的命令为:

```
python3 tools/model_test.py configs/faster_rcnn_r101_fpn_1x_coco.py checkpoints/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth -d detrac_trace_1 --out detrac_trace_1.golden.pkl
```

其中 `-d` 指示输入 trace 名称, `--out` 指示输出的结果文件, 结果文件有固定的格式, 如果要使用 efficientdet 作为 golden model 的话需要格式转化

2. 将结果文件转化为 COCO 格式的标注

直接执行 `python3 tools/result_to_annotation.py -p detrac_trace_1.golden.pkl -d detrac_trace_1 -t 0.5` 在 `./data/annotations/` 下生成 `detrac_trace_1.golden.json`. 其中 `-t` 参数表示只将 confidence 超过阈值的 bbox 保留在 golden annotations 中

## 生成各个 interval 对应的数据集

首先在项目根目录下放置 `datasets.json` 文件, 其中包含所有要训练的 trace 及其长度, 示例如下:

```json
{
  "detrac_trace_1": {
    "name": "detrac_trace_1",
    "size": 10000
  },
  "detrac_trace_2": {
    "name": "detrac_trace_2",
    "size": 10000
  },
  "detrac_trace_3": {
    "name": "detrac_trace_3",
    "size": 10000
  },
  "detrac_trace_4": {
    "name": "detrac_trace_4",
    "size": 10000
  }
}
```

执行 `python3 tools/split_dataset.py -s 500 -r 1/25 -o 020` 切分数据集, 其中 `-s` 表示每个 interval 的帧数, `-r` 是采样率, 必须是分数形式, 表示从 `分母` 帧中采样 `分子` 帧, 另外还有 `-o` 用来设定生成数据的后缀, 主要用来防止同时生成不同采样率的数据集时命名冲突的问题, 一般设置成跟采样率一一对应就可以, 比如 `-r 1/25 -o 020` 表示 500 帧中会采样 20 帧, `-r 2/25 -o 040` 等等

## 训练

使用 `tools/model_train.py` 来训练模型, 例如: 

```
python3 tools/model_train.py configs/custom/ssd_all.py --work-dir tmp_detrac_trace_1/ --train-dataset detrac_trace_1_020_train_0 --val-dataset detrac_trace_1_020_test_1
```

表示使用 `./configs/custom/ssd_all.py` 配置文件做训练, 结果会在 `./tmp_detrac_trace_1/` 中, 训练数据集是 `detrac_trace_1_020_train_0`, 即之前以 `1/25` 采样率生成的 interval 0 的训练集, 同时观测在 `detrac_trace_1_020_test_1` 上的 loss 的变化, 如果不需要观测 loss 则设定 `--no-validate`

`./train_test_acc.sh` 可以方便地把所有 interval 一次性都训练好并计算 mAP. 示例如下:

```
./train_test_acc.sh detrac_trace_1 20 0 020 all > result.txt 2> log.txt
```

其中第一个参数为 trace 的名称, 第二个参数为 interval 数量, 第三个参数为 GPU, 第四个参数为之前生成数据集时使用的后缀, 最后一个参数为配置文件的后缀, 命名格式为 `./configs/custom/ssd_*.py`, 例如 `all` 表示使用的配置文件是 `./configs/custom/ssd_all.py`, 对于其说明详见配置文件内的注释
