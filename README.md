# All for one : Multi-modal Multi-task learning

## Data Preparations and Arrangement

Download the VQAv2.0 dataset from the official website - visualqa.org and the imdb sentiment classification dataset from kaggle. Arrange the data as following

```
    data/
    |
    |____vqa/
    |________images/
    |________raw/
    |____________<questions and annotations files>
    |____imdb/
    |_________<imdb_data csv file>
    |
    |____preprocess_imdb_data.py
    |____convertToVQAFormat.py

```

Run the scripts ```preprocess_imdb_data.py``` and  ```convertToVQAFormat.py``` in order to create two types of new json files. One for the imdb dataset converted to the VQA format and another one for the combined VQA and IMDB dataset. Put both types of json files under ```data/vqa/```.

## Running

Run the following command to run the method on the combined dataset.

```
python run.py --MODEL='all_for_one'\
              --RUN='train'\
              --DATASET='vqa'\
              --VERSION=<string to describe the run>
```

To run the method on the VQA dataset alone or the imdb_dataset alone change the paths in the file ```openvqa/core/path_cfgs.py``` and change the name of the dataset there.

## Acknoledgements

I have used some of the code from the ![openvqa framework][https://github.com/MILVLG/openvqa] and I would like to thank the authors for maintaining such amazing repository.

## Citations

The All for one model was proposed by Bryan McCann and Nat Roth. Link to the paper ![here][https://cs224d.stanford.edu/reports/McCannRoth.pdf].
