# Up-Down-Captioner

Simple yet high-performing image captioning model using Caffe and python. Using image features from [bottom-up attention](https://github.com/peteanderson80/bottom-up-attention), in July 2017 this model achieved state-of-the-art performance on all metrics of the [COCO captions test leaderboard](http://cocodataset.org/#captions-leaderboard)(**SPICE 21.5**, **CIDEr 117.9**, **BLEU_4 36.9**). The architecture (2-layer LSTM with attention) is described in Section 3.2 of:
- [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998). 

### Reference
If you use this code in your research, please cite our paper:
```
@article{Anderson2017up-down,
  author = {Peter Anderson and Xiaodong He and Chris Buehler and Damien Teney and Mark Johnson and Stephen Gould and Lei Zhang},
  title = {Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering},
  journal = {arXiv preprint arXiv:1707.07998},
  year = {2017}
}
```

### License

This code is released under the MIT License (refer to the LICENSE file for details).

### Requirements: software

0. **`Important`** Please use the version of caffe provided as a submodule within this repository. It contains additional layers and features required for captioning.

1.  Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

    **Note:** Caffe *must* be built with support for Python layers and NCCL!

    ```make
    # In your Makefile.config, make sure to have these lines uncommented
    WITH_PYTHON_LAYER := 1
    USE_NCCL := 1
    # Unrelatedly, it's also recommended that you use CUDNN
    USE_CUDNN := 1
    ```
3.  Nvidia's NCCL library which is used for multi-GPU training https://github.com/NVIDIA/nccl

### Requirements: hardware

By default, the provided training scripts assume that two gpus are available, with indices 0,1. Training on two gpus takes around 9 hours. Any NVIDIA GPU with 12GB or larger memory is OK. Training scripts and prototxt files will require minor modifications to train on a single gpu (e.g. set `iter_size` to 2).

### Installation

All instructions are from the top level directory.

1.  Clone the Up-Down-Captioner repository:
    ```Shell
    # Make sure to clone with --recursive
    git clone --recursive https://github.com/peteanderson80/Up-Down-Captioner.git
    ```

    If you forget to clone with the `--recursive` flag, then you'll need to manually clone the submodules:
    ```Shell
    git submodule update --init --recursive
    ```

2.  Build Caffe and pycaffe:
    ```Shell
    cd ./external/caffe

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

3.  Add python layers and caffe build to PYTHONPATH:
    ```Shell
    cd $REPO_ROOT/data
    export PYTHONPATH=${PYTHONPATH}:$(pwd)/layers:$(pwd)/external/caffe/python
    ```
    
4.  Download Stanford CoreNLP (required by the evaluation code):
    ```Shell
    cd ./external/coco-caption
    ./get_stanford_models.sh
    ```

5.  Download the MS COCO train/val image caption annotations. Extract all the json files into one folder `$COCOdata`, then create a symlink to this location:
    ```Shell
    cd $REPO_ROOT/data
    ln -s $COCOdata coco
    ``` 

6.  Pre-process the caption annotations for training (building vocabs etc).
    ```Shell
    cd $REPO_ROOT
    python scripts/preprocess_coco.py
    ``` 
    
7.  Download or generate pretrained image features following the instructions below.

### Pretrained image features

The captioner takes pretrained image features as input (and does not finetune). For best performance, bottom-up attention features should be used. Code for generating these features can be found [here](https://github.com/peteanderson80/bottom-up-attention). For ease-of-use, we provide pretrained features for the [MSCOCO dataset](http://mscoco.org/dataset/#download). Manually download the following tsv file and unzip to `data/tsv/`:
- [2014 Train/Val Image Features (120K / 23GB)](https://storage.googleapis.com/bottom-up-attention/trainval.zip)
To make a test server submission, you would also need these features:
- [2014 Testing Image Features (40K / 7.3GB)](https://storage.googleapis.com/bottom-up-attention/test2014.zip)

Alternatively, to use conventional pretrained features from the ResNet-101 CNN, run:
```Shell
cd $REPO_ROOT
python scripts/generate_baseline.py
``` 

### Training

To train the model on the karpathy training set. 
```Shell
cd $REPO_ROOT
./experiments/caption_lstm/train.sh
```

Trained snapshots are saved under: `snapshots/caption_lstm/`

Logging outputs are saved under: `logs/caption_lstm/`

Generated caption outputs are saved under: `outputs/caption_lstm/`

Scores for the generated captions (on the karpathy test set) are saved under: `scores/caption_lstm/`

Note that if you are running these scripts on a server you may need to add the line `backend : Agg` to `~/.config/matplotlib/matplotlibrc`.

### Results

todo

|                   | objects mAP@0.5     | objects weighted mAP@0.5 | attributes mAP@0.5    | attributes weighted mAP@0.5 |
|-------------------|:-------------------:|:------------------------:|:---------------------:|:---------------------------:|
|Faster R-CNN, ResNet-101 | 10.2%  | 15.1% | 7.8%  | 27.8% |

### Using the model to predict on new images

### Using the model 

### Other useful scripts

1. create_caption_lstm.py
    The version of caffe provided as with this repository includes (amongst other things) a custom `LSTMNode` layer that enables sampling and beam search through LSTM layers. However, the resulting network architecture prototxt files are quite complicated. The file `scripts/create_caption_lstm.py` scaffolds out network structures, such as those in `experiments`.


