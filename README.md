# Fast ImageNet with FP16
This repo is used to reproduce the issue of Fast Imagenet with FP16. That is, the network cannot convergence.

## How to run
```python
python launch.py --gpus 8 train.py --fp16=True --scale_loss=8.0 --data_dir=some/directory
``` 