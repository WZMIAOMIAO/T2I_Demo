# T2I Training Demo
该仓库实现了一个可供学习参考的T2I训练项目，在该项目中默认使用Rectified Flow训练范式配合DiT模型进行训练。

## 环境配置
该项目中使用的torch版本为`torch==2.7.1+cu126`，对应torchvision版本为`torchvision==0.22.1+cu126`（用户可根据自己需求安装其他版本，建议安装大于等于当前的版本），其他依赖可参考`requirements.txt`文件。

## 访问huggingface问题
由于代码中涉及访问huggingface下载VAE权重，如果由于墙问题无法访问huggingface下载权重，可使用国内镜像代理。Linux系统可在终端设置如下临时变量：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 数据下载
数据集下载，这里使用的是kaggle上的数据集，该数据集是从Imagnet中抽取的54W数据并resize到256分辨率。下载地址：
[https://www.kaggle.com/datasets/dimensi0n/imagenet-256](https://www.kaggle.com/datasets/dimensi0n/imagenet-256)
下载完成后解压到`ImageNet256`目录内（根据自己情况设置合适的解压目录），例如：
```bash
unzip archive.zip -d /home/wz/datasets/ImageNet256/
```
（可选）预先抽取好latents数据（VAE encoder将图片从图像域编码到latent域），由于训练过程中数据会迭代上百个epoch可考虑提前抽取好这样能够加速训练并减少显存占用。根据自己存放目录修改`--dataset-path`以及`--save-path`等参数：
```bash
torchrun --master-addr=127.0.0.1 --master-port=12345 --nnodes=1 --nproc-per-node=8 --node-rank=0 extract_latents.py --dataset-path=/home/wz/datasets/ImageNet256 --save-path=/home/wz/datasets/ImageNet256_latents
```

## 修改配置文件
训练相关配置主要在`train_cfg.yaml`中，其中需要根据自己的情况配置好数据集路径`dataset_path`，若没有提前抽取latent文件则`use_extract_latents`应设置为`false`并将`dataset_path`指向原始解压数据集路径，若提前抽取好了latent文件则`use_extract_latents`应设置为`true`并将`dataset_path`指向生成latent的数据集路径。

## 启动训练
启动训练指令如下，可根据自己的训练场景配置训练节点数`--nnodes`，每个节点参与训练的卡数`--nproc-per-node`，对应节点的索引`--node-rank`，以及主节点的ip`--master-addr`和端口`--master-port`：
```bash
torchrun --master-addr=127.0.0.1 --master-port=12345 --nnodes=1 --nproc-per-node=8 --node-rank=0 train.py --cfg-file=train_cfg.yaml
```

## 调试
若需要使用vscode debug进行单步调试，需要配置`launch.json`文件，文件内容可参考：
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug torchrun",
            "type": "python",
            "request": "launch",
            "module": "torch.distributed.run",
            "args": [
                "--master-addr=127.0.0.1",
                "--master-port=12345",
                "--nnodes=1",
                "--nproc-per-node=1",
                "--node-rank=0",
                "train.py",
                "--cfg-file=train_cfg.yaml"
            ],
            "justMyCode": false,
            "console": "integratedTerminal"
        }
    ]
}
```