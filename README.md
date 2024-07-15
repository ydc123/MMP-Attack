## MMP-Attack

This repository is the official repository for our paper "On the Multi-modal Vulnerability of Diffusion Models" 

<p align="center">
  <img src="figures/pipeline.png" alt="bounding box" width="640px">
</p>

### Setup
Install PyTorch, diffusers==0.17.0, and transformers==4.29.1.

Download the [Stable Diffusion model](https://huggingface.co/CompVis/stable-diffusion-v1-4) and [CLIP model](https://huggingface.co/openai/clip-vit-large-patch14).

### Running Commands

You can execute the following command to perform a targeted attack, transforming the category "car" into "bird":
```
python attack.py --ori_sentence "a photo of car" --target_word bird 
```

## Citation

If you benefit from our work in your research, please consider to cite the following paper:
```
@inproceedings{
yang2024on,
title={On the Multi-modal Vulnerability of Diffusion Models},
author={Dingcheng Yang and Yang Bai and Xiaojun Jia and Yang Liu and Xiaochun Cao and Wenjian Yu},
booktitle={Trustworthy Multi-modal Foundation Models and AI Agents (TiFA)},
year={2024},
url={https://openreview.net/forum?id=FuZjlzR7kT}
}
```

Please feel free to contact us if you have any questions.
