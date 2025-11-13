
[Deep noise-tolerant hashing for remote sensing image retrieval](https://www.sciencedirect.com/science/article/pii/S0923596525001778)

This paper is accepted for publication with  Signal Processing: Image Communication.

## Training

### Processing dataset
Before training, you need to download the UCMerced dataset http://weegee.vision.ucmerced.edu/datasets/landuse.html, AID dataset from https://captain-whu.github.io/AID ,WHURS dataset from https://captain-whu.github.io/BED4RS.

### Download ViT pretrained model
Pretrained model vit_small_patch16_224.pth is required for loading before training.

### Start
After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset UCMerced --lr 0.0001 --wd 0.0004 --save-dir ./result/ucmd/16 --vit-path ./pretrained/vit_small_patch16_224.pth
> 
### Citation
@article{yan2025deep,

  title={Deep noise-tolerant hashing for remote sensing image retrieval},
  
  author={Yan, Chunyu and Wang, Lei and Qin, Qibing and Dai, Jiangyan and Zhang, Wenfeng},
  
  journal={Signal Processing: Image Communication},
  
  pages={117431},
  
  year={2025},
  
  publisher={Elsevier}
}
