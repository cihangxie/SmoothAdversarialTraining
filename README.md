
# Smooth Adversarial Training

Code and models for the paper [Smooth Adversarial Training](https://arxiv.org/pdf/2006.14536.pdf).

## Things to do
- [x] ResNet single-GPU inference
- [x] ResNet multi-GPU inference
- [x] ResNet adversarial robustness evaluation
- [ ] ResNet adversarial training
- [x] EfficientNet single-GPU inference
- [ ] EfficientNet multi-GPU inference
- [ ] EfficientNet adversarial robustness evaluation
- [ ] EfficientNet adversarial training

## Introduction

<div align="center">
  <img src="teaser.jpg" width="800px" />
</div>

The widely-used ReLU activation function significantly weakens adversarial training due to its non-smooth nature. In this project, we developed smooth adversarial training (SAT), in which we replace ReLU with its smooth approximations (e.g., SILU, softplus, SmoothReLU) to strengthen adversarial training. 

On ResNet-50, the best result reported by SAT on ImageNet is 69.7% accuracy and 42.3% robustness, beating its ReLU version by 0.9 for accuracy and 9.3 for robustnes.

We also explore the limits of SAT with larger networks. We obtain the best result by using EfficientNet-L1, which achieves 82.2% accuracy and 58.6% robustness on ImageNet.


## Dependencies:

+ TensorFlow ≥ 1.6 with GPU support
+ Tensorpack ≥ 0.9.8
+ OpenCV ≥ 3
+ ImageNet data in [its standard directory structure](https://tensorpack.readthedocs.io/modules/dataflow.dataset.html#tensorpack.dataflow.dataset.ILSVRC12) for ResNet.
+ ImageNet data in [TFRecord format](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) for EfficientNet
+ [gdown] (https://pypi.org/project/gdown/) for downloading pretrained ckpts


## Model Zoo:

Scripts to download [ResNet](ResNet/download_resnet.py) or [EfficientNet](EfficientNet/download_efficientnet.py) from Google Drive.

<table>
<thead>
<tr>
<th align="left" rowspan=2>Model (click for details)</th>
<th align="center">error rate (%)</th>
<th align="center" colspan=3>error rate(%)</th>
</tr>
<tr>
<th align="center">clean images</th>
<th align="center">10-step PGD</th>
<th align="center">20-step PGD</th>
<th align="center">50-step PGD</th>
</tr>
</thead>


<tbody>
<tr>
<td align="left"><details><summary>ReLU </summary> <code>--arch ResNet -d 152</code></details></td>
<td align="center">68.7</td>
<td align="center"></td>
<td align="center">35.3</td>
<td align="center">33.7</td>
<td align="center">33.1</td>
</tr>

<tr>
<td align="left"><details><summary>ReLU </summary> <code>--arch ResNet -d 152</code></details></td>
<td align="center">68.7</td>
<td align="center"></td>
<td align="center">35.3</td>
<td align="center">33.7</td>
<td align="center">33.1</td>
</tr>

<tr>
<td align="left"><details><summary>ReLU </summary> <code>--arch ResNet -d 152</code></details></td>
<td align="center">68.7</td>
<td align="center"></td>
<td align="center">35.3</td>
<td align="center">33.7</td>
<td align="center">33.1</td>
</tr>
</tbody>
</table>


## Acknowledgements

The <b>MAJOR</b> part of this code come from [ImageNet-Adversarial-Training](https://github.com/facebookresearch/ImageNet-Adversarial-Training) and [EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet). We thanks [Yingwei Li](https://yingwei.li/) and [Jieru Mei](https://scholar.google.com/citations?user=nHKExN0AAAAJ&hl) for helping open resource the code and models.

## Citation

If you use our code, models or wish to refer to our results, please use the following BibTex entry:
```
@article{Xie_2020_SAT,
  author = {Xie, Cihang and Tan, Mingxing and Gong, Boqing and Yuille, Alan and Le, Quoc V},
  title  = {Smooth Adversarial Training},
  journal={arXiv preprint arXiv:2006.14536},
  year = {2020}
}
```
