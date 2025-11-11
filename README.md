Deep noise-tolerant hashing for remote sensing image retrieval

This paper is accepted for publication with Signal Processing: Image Communication.

Training
Processing dataset
Before training, you need to download the UCMerced dataset http://weegee.vision.ucmerced.edu/datasets/landuse.html, AID dataset from https://captain-whu.github.io/AID ,WHURS dataset from https://captain-whu.github.io/BED4RS.

Download pretrained model
The pre-trained weight file vit_small_patch16_224.pth is required for model loading before training.

Start
After the dataset has been prepared, we could run the follow command to train.

Python main.py
