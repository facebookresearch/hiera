# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles

This is the official implementation for our ICML 2023 Oral paper:  
**[Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles][arxiv-link]**  
[Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ)\*,
[Yuan-Ting Hu](https://scholar.google.com/citations?user=aMpbemkAAAAJ)\*,
[Daniel Bolya](https://scholar.google.com/citations?hl=en&user=K3ht_ZUAAAAJ)\*,
[Chen Wei](https://scholar.google.com/citations?hl=en&user=LHQGpBUAAAAJ),
[Haoqi Fan](https://scholar.google.com/citations?hl=en&user=76B8lrgAAAAJ),
[Po-Yao Huang](https://scholar.google.com/citations?hl=en&user=E8K25LIAAAAJ),
[Vaibhav Aggarwal](https://scholar.google.com/citations?hl=en&user=Qwm6ZOYAAAAJ),
[Arkabandhu Chowdhury](https://scholar.google.com/citations?hl=en&user=42v1i_YAAAAJ),
[Omid Poursaeed](https://scholar.google.com/citations?hl=en&user=Ugw9DX0AAAAJ),
[Judy Hoffman](https://scholar.google.com/citations?hl=en&user=mqpjAt4AAAAJ),
[Jitendra Malik](https://scholar.google.com/citations?hl=en&user=oY9R5YQAAAAJ),
[Yanghao Li](https://scholar.google.com/citations?hl=en&user=-VgS8AIAAAAJ)\*,
[Christoph Feichtenhofer](https://scholar.google.com/citations?hl=en&user=UxuqG1EAAAAJ)\*  
_[ICML '23 Oral][icml-link]_ | _[GitHub](https://github.com/facebookresearch/hiera)_ | _[arXiv][arxiv-link]_ | _[BibTeX](https://github.com/facebookresearch/hiera#citation)_

\*: Equal contribution.

## What is Hiera?
**Hiera** is a _hierarchical_ vision transformer that is fast, powerful, and, above all, _simple_. It outperforms the state-of-the-art across a wide array of image and video tasks _while being much faster_. 

<p align="center">
  <img src="https://github.com/facebookresearch/hiera/raw/main/examples/img/inference_speed.png" width="75%">
</p>

## How does it work?
![A diagram of Hiera's architecture.](https://github.com/facebookresearch/hiera/raw/main/examples/img/hiera_arch.png)

Vision transformers like [ViT](https://arxiv.org/abs/2010.11929) use the same spatial resolution and number of features throughout the whole network. But this is inefficient: the early layers don't need that many features, and the later layers don't need that much spatial resolution. Prior hierarchical models like [ResNet](https://arxiv.org/abs/1512.03385) accounted for this by using fewer features at the start and less spatial resolution at the end.

Several domain specific vision transformers have been introduced that employ this hierarchical design, such as [Swin](https://arxiv.org/abs/2103.14030) or [MViT](https://arxiv.org/abs/2104.11227). But in the pursuit of state-of-the-art results using fully supervised training on ImageNet-1K, these models have become more and more complicated as they add specialized modules to make up for spatial biases that ViTs lack. While these changes produce effective models with attractive FLOP counts, under the hood the added complexity makes these models _slower_ overall.

We show that a lot of this bulk is actually _unnecessary_. Instead of manually adding spatial bases through architectural changes, we opt to _teach_ the model these biases instead. By training with [MAE](https://arxiv.org/abs/2111.06377), we can simplify or remove _all_ of these bulky modules in existing transformers and _increase accuracy_ in the process. The result is Hiera, an extremely efficient and simple architecture that outperforms the state-of-the-art in several image and video recognition tasks.

## News
 - **[2023.06.12]** Added more in1k models and some video examples, see inference.ipynb (v0.1.1).
 - **[2023.06.01]** Initial release.

See the [changelog](https://github.com/facebookresearch/hiera/tree/main/CHANGELOG.md) for more details.

## Installation

Hiera requires a reasonably recent version of [torch](https://pytorch.org/get-started/locally/).
After that, you can install hiera through [pip](https://pypi.org/project/hiera-transformer/0.1.0/):
```bash
pip install hiera-transformer
```
This repo _should_ support the latest timm version, but timm is a constantly updating package. Create an issue if you have problems with a newer version of timm.

### Installing from Source

If using [torch hub](#model-zoo), you don't need to install the `hiera` package. But, if you'd like to develop using hiera, it could be a good idea to install it from source:

```bash
git clone https://github.com/facebookresearch/hiera.git
cd hiera
python setup.py build develop
```


## Model Zoo

### Torch Hub

Here we provide model checkpoints for Hiera. Each model listed is accessible on [torch hub](https://pytorch.org/docs/stable/hub.html) even without the `hiera-transformer` package installed, e.g. the following initializes a base model pretrained and finetuned on ImageNet-1k:
```py
model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
```

If you want a model with MAE pretraining only, you can replace the checkpoint with `"mae_in1k"`. Additionally, if you'd like to load the MAE decoder as well (e.g., to continue pretraining), add `mae_` the the start of the model name, e.g.:
```py
model = torch.hub.load("facebookresearch/hiera", model="mae_hiera_base_224", pretrained=True, checkpoint="mae_in1k")
```
**Note:** Our MAE models were trained with a _normalized pixel loss_. That means that the patches were normalized before the network had to predict them. If you want to visualize the predictions, you'll have to unnormalize them using the visible patches (which might work but wouldn't be perfect) or unnormalize them using the ground truth. For model more names and corresponding checkpoint names see below.

### Hugging Face Hub

This repo also has [🤗 hub](https://huggingface.co/docs/hub/index) support. With the `hiera-transformer` and `huggingface-hub` packages installed, you can simply run, e.g.,
```py
from hiera import Hiera
model = Hiera.from_pretrained("facebook/hiera_base_224.mae_in1k_ft_in1k")  # mae pt then in1k ft'd model
model = Hiera.from_pretrained("facebook/hiera_base_224.mae_in1k") # just mae pt, no ft
```
to load a model. Use `<model_name>.<checkpoint_name>` from model zoo below.

If you want to save a model, use `model.config` as the config, e.g.,
```py
model.save_pretrained("hiera-base-224", config=model.config)
```

### Image Models
| Model    | Model Name            | Pretrained Models<br>(IN-1K MAE) | Finetuned Models<br>(IN-1K Supervised) | IN-1K<br>Top-1 (%) | A100 fp16<br>Speed (im/s) |
|----------|-----------------------|----------------------------------|----------------------------------------|:------------------:|:-------------------------:|
| Hiera-T  | `hiera_tiny_224`      | [mae_in1k](https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth)        | [mae_in1k_ft_in1k](https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth)       |       82.8         |            2758           |
| Hiera-S  | `hiera_small_224`     | [mae_in1k](https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth)       | [mae_in1k_ft_in1k](https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth)      |       83.8         |            2211           |
| Hiera-B  | `hiera_base_224`      | [mae_in1k](https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth)        | [mae_in1k_ft_in1k](https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth)       |       84.5         |            1556           |
| Hiera-B+ | `hiera_base_plus_224` | [mae_in1k](https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth)   | [mae_in1k_ft_in1k](https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth)  |       85.2         |            1247           |
| Hiera-L  | `hiera_large_224`     | [mae_in1k](https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth)       | [mae_in1k_ft_in1k](https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth)      |       86.1         |            531            |
| Hiera-H  | `hiera_huge_224`      | [mae_in1k](https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth)        | [mae_in1k_ft_in1k](https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth)       |       86.9         |            274            |

Each model inputs a 224x224 image.
### Video Models
| Model    | Model Name               | Pretrained Models<br>(K400 MAE) | Finetuned Models<br>(K400) | K400 (3x5 views)<br>Top-1 (%) | A100 fp16<br>Speed (clip/s) |
|----------|--------------------------|---------------------------------|----------------------------|:-----------------------------:|:---------------------------:|
| Hiera-B  | `hiera_base_16x224`      | [mae_k400](https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth)       | [mae_k400_ft_k400](https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth)      |              84.0             |            133.6            |
| Hiera-B+ | `hiera_base_plus_16x224` | [mae_k400](https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth)  | [mae_k400_ft_k400](https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth) |              85.0             |             84.1            |
| Hiera-L  | `hiera_large_16x224`     | [mae_k400](https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth)      | [mae_k400_ft_k400](https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth)     |              87.3             |             40.8            |
| Hiera-H  | `hiera_huge_16x224`      | [mae_k400](https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth)       | [mae_k400_ft_k400](https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth)      |              87.8             |             20.9            |

Each model inputs 16 224x224 frames with a temporal stride of 4.

**Note:** the speeds listed here were benchmarked _without_ PyTorch's optimized [scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). If using PyTorch 2.0 or above, your inference speed will probably be faster than what's listed here.

## Usage

This repo implements the code to run Hiera models for inference. This repository is still in progress. Here's what we currently have available and what we have planned:

 - [x] Image Inference
    - [x] MAE implementation
 - [x] Video Inference
    - [x] MAE implementation
 - [x] Full Model Zoo
 - [ ] Training scripts


See [examples](https://github.com/facebookresearch/hiera/tree/main/examples) for examples of how to use Hiera.

### Inference

See [examples/inference](https://github.com/facebookresearch/hiera/blob/main/examples/inference.ipynb) for an example of how to prepare the data for inference.

Instantiate a model with either [torch hub](#model-zoo) or [🤗 hub](#model-zoo) or by [installing hiera](#installing-from-source) and running:
```py
import hiera
model = hiera.hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
```
Then you can run inference like any other model:
```py
output = model(x)
```
Video inference works the same way, just use a `16x224` model instead.

**Note**: for efficiency, Hiera re-orders its tokens at the start of the network (see the `Roll` and `Unroll` modules in `hiera_utils.py`). Thus, tokens _aren't in spatial order_ by default. If you'd like to use intermediate feature maps for a downstream task, pass the `return_intermediates` flag when running the model:
```py
output, intermediates = model(x, return_intermediates=True)
```

#### MAE Inference
By default, the models do not include the MAE decoder. If you would like to use the decoder or compute MAE loss, you can instantiate an mae version by running:
```py
import hiera
model = hiera.mae_hiera_base_224(pretrained=True, checkpoint="mae_in1k")
```
Then when you run inference on the model, it will return a 4-tuple of `(loss, predictions, labels, mask)` where predictions and labels are for the _deleted tokens_ only. The returned mask will be `True` if the token is visible and `False` if it's deleted. You can change the masking ratio by passing it during inference:
```py
loss, preds, labels, mask = model(x, mask_ratio=0.6)
```
The default mask ratio is `0.6` for images, but you should pass in `0.9` for video. See the paper for details.

**Note:** We use _normalized pixel targets_ for MAE pretraining, meaning the patches are each individually normalized before the model model has to predict them. Thus, you have to unnormalize them using the ground truth before visualizing them. See `get_pixel_label_2d` in `hiera_mae.py` for details.

### Benchmarking
We provide a script for easy benchmarking. See [examples/benchmark](https://github.com/facebookresearch/hiera/blob/main/examples/benchmark.ipynb) to see how to use it.

#### Scaled Dot Product Attention
PyTorch 2.0 introduced optimized [scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), which can speed up transformers quite a bit. We didn't use this in our original benchmarking, but since it's a free speed-up this repo will automatically use it if available. To get its benefits, make sure your torch version is 2.0 or above.

### Training

Coming soon.


## Citation
If you use Hiera or this code in your work, please cite:
```
@article{ryali2023hiera,
  title={Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles},
  author={Ryali, Chaitanya and Hu, Yuan-Ting and Bolya, Daniel and Wei, Chen and Fan, Haoqi and Huang, Po-Yao and Aggarwal, Vaibhav and Chowdhury, Arkabandhu and Poursaeed, Omid and Hoffman, Judy and Malik, Jitendra and Li, Yanghao and Feichtenhofer, Christoph},
  journal={ICML},
  year={2023}
}
```

### License
This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

### Contributing
See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

[arxiv-link]: https://arxiv.org/abs/2306.00989/
[icml-link]: https://icml.cc/Conferences/2023
