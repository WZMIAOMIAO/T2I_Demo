如果由于墙问题无法访问huggingface下载权重，可使用国内镜像代理。Linux系统可在终端设置如下临时变量：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

启动训练指令：
```bash
torchrun --master-addr=127.0.0.1 --master-port=12345 --nnodes=1 --nproc-per-node=8 --node-rank=0 train.py --cfg-file=train_cfg.yaml
```

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