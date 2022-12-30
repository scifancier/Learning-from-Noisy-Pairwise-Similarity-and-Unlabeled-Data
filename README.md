# Learning from Noisy Pairwise Similarity and Unlabeled Data
 JMLR'2022/11: Learning from Noisy Pairwise Similarity and Unlabeled Data.



This is the code for the paper:
[Learning from Noisy Pairwise Similarity and Unlabeled Data](https://www.jmlr.org/papers/volume23/21-0946/21-0946.pdf).

Songhua Wu, Tongliang Liu, Bo Han, Jun Yu, Gang Niu, and Masashi Sugiyama.



If you find this code useful for your research, please cite  
```bash
@article{JMLR:v23:21-0946,
  author  = {Songhua Wu and Tongliang Liu and Bo Han and Jun Yu and Gang Niu and Masashi Sugiyama},
  title   = {Learning from Noisy Pairwise Similarity and Unlabeled Data},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {307},
  pages   = {1--34},
  url     = {http://jmlr.org/papers/v23/21-0946.html}
}
```



## Dependencies
We implement our methods by PyTorch on Nvidia GeForce RTX 3090 Ti. The environment is as bellow:
- [Ubuntu 20.04 Desktop](https://ubuntu.com/download)
- [PyTorch](https://PyTorch.org/), version >= 0.4.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 9.0
- [Anaconda3](https://www.anaconda.com/)



## Runing nSU on benchmark datasets 
Here is an example: 

```bash
python3 sudeep.py --mpe 4000 --ns 4000 --nu 2000 --prior 0.7 --noise 0.2 --seed 3 --dataset australian --p 1 --gpu 1



python3 sulearning.py --mpe 4000 --ns 4000 --nu 2000 --prior 0.7 --noise 0.2 --seed 3 --full --dataset australian --p 0
```



