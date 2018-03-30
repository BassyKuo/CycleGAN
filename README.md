# CycleGAN

## Requirements

* Python3.6
* tensorflow r1.4 (follow the instructions [here](https://github.com/mind/wheels#versions))
* numpy r1.13.3
* scipy
* termcolor
* progress

#### One-Step Installation (GPU version):

```sh
pip install -r requirements.txt
```

## Usage

```sh
python3 main.py -h
```

## Examples

```sh
python3 main.py PHOTO2LABEL --resize '256,256' --gpus 1
```

## Models

- [x] CycleGAN - basic
- [x] CycleGAN - photo2label
    - [x] cityscape: 19 labels
    - [ ] ade


## References

1. https://github.com/junyanz/CycleGAN
2. https://github.com/xhujoy/CycleGAN-tensorflow
3. https://github.com/DrSleep/tensorflow-deeplab-resnet
