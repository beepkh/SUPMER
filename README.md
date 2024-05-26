<h1 align = "center">
Self-supervised Meta-Prompt Learning with Meta-Gradient Regularization for Few-shot Generalization
</h1>

<div align="center">
Kaihang Pan<sup>1</sup>, Juncheng Li<sup>1&dagger;</sup>, Hongye Song<sup>2</sup>, Jun Lin<sup>2</sup>, Xiaozhong Liu<sup>3</sup>, Siliang Tang<sup>1</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>DAMO Academy, Alibaba Group, <sup>3</sup>Worcester Polytechnic Institute

<sup>&dagger;</sup>Corresponding Author

<div align="left">

This repo contains the PyTorch implementation of [Self-supervised Meta-Prompt Learning with Meta-Gradient Regularization for Few-shot Generalization](https://aclanthology.org/2023.findings-emnlp.75/), which is accepted by **EMNLP 2023 (findings)**.

## Installation 

This repos is built based on the repo of [PERFECT](https://arxiv.org/abs/2204.01172). Please refer to [facebookresearch/perfect](https://github.com/facebookresearch/perfect) for the installation of the Python environment.

## Meta-trained checkpoints

* [pretrain_prompt/gradient_edit_t5base.pt](pretrain_prompt/gradient_edit_t5base.pt):  checkpoint of the meta-trained gradient regularization function for t5-base.
* [pretrain_prompt/prompt_t5base.pt](pretrain_prompt/prompt_t5base.pt): checkpoint of the meta-trained soft prompt for t5-base.
* [pretrain_prompt/gradient_edit_flant5xl.pt](pretrain_prompt/gradient_edit_flant5xl.pt):  checkpoint of the meta-trained gradient regularization function for flant5xl.
* [pretrain_prompt/prompt_flant5xl.pt](pretrain_prompt/prompt_flant5xl.pt): checkpoint of the meta-trained soft prompt for flant5xl.

## Run the code

To run the code of meta-training and downstream prompt-tuning, you can refer to the scripts provided at [scripts/meta-train.sh](scripts/meta-train.sh) and [scripts/prompt-tuning.sh](scripts/prompt-tuning.sh).

## Acknowledgment

Our project is developed based on the following repositories:

* [Perfect](https://github.com/facebookresearch/perfect): Prompt-free and Efficient Few-shot Learning with Language Models

* [PPT](https://github.com/thu-coai/PPT): Pre-trained Prompt Tuning for Few-shot Learning

## Citation
If you found this work useful, please consider  citing our paper as follows:
```
@article{pan2023self,
  title={Self-supervised Meta-Prompt Learning with Meta-Gradient Regularization for Few-shot Generalization},
  author={Pan, Kaihang and Li, Juncheng and Song, Hongye and Lin, Jun and Liu, Xiaozhong and Tang, Siliang},
  journal={arXiv preprint arXiv:2303.12314},
  year={2023}
}
```
