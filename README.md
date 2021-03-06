# CycleGAN

## Requirements

* Python3.6
* tensorflow==1.4.0 (check [here](https://github.com/mind/wheels#versions) for more info.)

| Version | URL |
|:-------:|-----|
| CPU | https://github.com/mind/wheels/releases/download/tf1.4-cpu/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl |
| GPU | https://github.com/mind/wheels/releases/download/tf1.4-gpu/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl |

* numpy==1.13.3
* scipy
* termcolor
* progress

##### One-Step Installation (GPU version):

```sh
pip install -r requirements.txt
```

##### MKL issue: https://github.com/mind/wheels#mkl



## Usage

```sh
python3 main.py -h
```



## Examples

```sh
python3 main.py PHOTO2LABEL --crop_size '900,1800' --random_scale False --gpus 1 --bs 1
```



## Models

- [x] CycleGAN - basic
- [x] CycleGAN - photo2label
    - [x] cityscape: 19 labels
    - [ ] ade


## Demos

* basic (RGB2RGB)

    ![demo-rgb2rgb_fullview](./DEMO/demo-rgb2rgb_fullview.png)

* photo2label (RGB2ONEHOT)

    ![demo-photo2label_fullview](./DEMO/demo-photo2label_fullview.png)

## References

1. https://github.com/junyanz/CycleGAN
2. https://github.com/xhujoy/CycleGAN-tensorflow
3. https://github.com/DrSleep/tensorflow-deeplab-resnet
4. https://github.com/mind/wheels
