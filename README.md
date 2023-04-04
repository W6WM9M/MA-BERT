# MA-BERT

<p align="justify">This Github respository contains the pre-trained models of MA-BERT models mentioned in the paper <a href="https://openreview.net/forum?id=HtAfbHa7LAL">MA-BERT: Towards Matrix Arithmetic-only BERT Inference by Eliminating Complex Non-linear Functions</a>. In particular, three pretrained checkpoints are released under the <a href="#pretrained-checkpoints">Pretrained Checkpoint</a> section: </p>
<ol>
<li>MA-BERT</li>
<li>MA-BERT (Shared Softmax)</li>
<li>MA-DistilBERT</li>
</ol>
<p align="justify">In MA-BERT, we proposed four correlated techniques that include:</p>

<ol>
<li>Approximating softmax with a two-layer neural network</li>
<li>Replacing GELU with ReLU</li>
<li>Fusing normalization layers with adjacent linear layers</li>
<li>Leveraging knowledge transfer from baseline models </li>
</ol>
<p align="justify">Through these techniques, we were able to eliminate the major non-linear functions in BERT and obtain MA-BERT with only matrix arithmetic and trivial ReLU operations.  Our experimental results show that MA-BERT achieves a more efficient inference with comparable accuracy on many downstream tasks compared to the baseline BERT models.</p>

## Loading Instructions

To load MA-BERT and MA-BERT (Shared Softmax):
1. Download the `ma-bert` folder and its pretrained checkpoint
2. Move the folder to the BERT folder in the transformers library: `transformers/models/bert`
3. Execute the code in the `loading_example.ipynb`

To load MA-DistilBERT:
1. Download the `ma-distilbert` folder and its pretrained checkpoint
2. Move the folder to the DistilBERT folder in the transformers library: `transformers/models/distilbert`
3. Execute the code in the `loading_example.ipynb`

## Pretrained Checkpoints
The following contains the links to our pretrained checkpoints: 

| **Model**         |
| :----------: |
| [MA-BERT](https://drive.google.com/uc?id=16jlFRkuuVsB39yP62k7bnitRW9z9Mb1_&export=download) | 
| [MA-BERT (Shared Softmax)](https://drive.google.com/uc?id=1iuONqg13d2Md8mIDwiBaUhycx5cFrRkm&export=download) |
| [MA-DistilBERT](https://drive.google.com/uc?id=1dvnKAJORjcsH85WPp6g5DyTo_ii1attq&export=download) |


## Citations
```
@inproceedings{
ming2023mabert,
title={{MA}-{BERT}: Towards Matrix Arithmetic-only {BERT} Inference by Eliminating Complex Non-linear Functions},
author={Neo Wei Ming and Zhehui Wang and Cheng Liu and Rick Siow Mong Goh and Tao Luo},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=HtAfbHa7LAL}
}
```
