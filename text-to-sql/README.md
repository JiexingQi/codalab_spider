# Quick start
This repository uses git submodules. Clone it like this:

```
$ git clone git@github.com:JiexingQi/picard.git
$ cd picard
$ git submodule update --init --recursive
```

# Requirements
Suggested environment to run the code:

python 3.9.7

You can make a new conda envirment using:
```
conda create -n picard python==3.9.7
```
And then, you may need to install these packages using pip：
+ sqlparse==0.4.2
+ nltk==3.6.5
+ wandb==0.12.7
+ transformers==4.13.0
+ datasets==1.16.1
+ tenacity==8.0.1
+ rapidfuzz==1.8.3
+ stanza==1.3.0

or using *requirements.txt*


# Download dataset
Before run the code, you should download dataset files.

First, you should create a dictionary like this:
```
mkdir -p dataset_files/ori_dataset
```

And then you need to download the dataset  file to dataset_files/ and just keep it in zip format. The download link are here:
+ Spider, [link](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0)
+ SParC, [link](https://drive.google.com/uc?export=download&id=13Abvu5SUMSP3SJM-ZIj66mOkeyAquR73)
+ CoSQL, [link](https://drive.google.com/uc?export=download&id=14x6lsWqlu6gR-aYxa6cemslDN3qT3zxP)

Then unzip those dataset file into dataset_files/ori_dataset. Both files in zip format and unzip format are needed:

```
unzip dataset_files/spider.zip -d dataset_files/ori_dataset/
unzip dataset_files/cosql_dataset.zip -d dataset_files/ori_dataset/
unzip dataset_files/sparc.zip -d dataset_files/ori_dataset/
```

# Run code
First, define a config file in /configs, and then use the command to run the code(in this example, the config file is train_0125_example.json):

```
CUDA_VISIBLE_DEVICES="2,3" python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 seq2seq/run_seq2seq.py configs/train_0125_example.json
```

## Note
+ You should set --nproc_per_node=#gpus to make full use of all gpus. 
+ A recommand total_batch_size = #gpus * gradient_accumulation_steps * per_device_train_batch_size is 2048.


# Config file
In config json file, you must set the correct filepath for relation filepath.

```
"lge_relation_path" : "/home/jxqi/text2sql/data"
```

this key-value pair set the relation filepath. 


# Relation file
The relation files are aviliable in Google drive:
https://drive.google.com/drive/folders/1cads4MN02FUj5gUwcwP6mYzrSNkNWD9l?usp=sharing

