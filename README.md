# Machine translation between English and Chinese

## Team
Name:   Zijun Wu    |   Ruiqin Pi  
CCID:   zijun4      |    ruiqin

## Execution instruction
### Step 0:
All codes for data pre-processing, training and testing are in the folder [/code](/code)

Firstly, download the data from : https://conferences.unite.un.org/UNCORPUS/en/DownloadOverview (English-Chinese), then follow the instruction to unzip the dataset.

All raw WMT17 data should be placed in the folder [/data/en-zh/](/data/en-zh/)  
File "UNv1.0.en-zh.zh" records the raw data in Chinese  
File "UNv1.0.en-zh.en" records the raw data in English  
Both are the recorded Conference documents from United Nations  

### Step 1:
```python3 prepare_data.py```  
When finished the execution, you will have "my_en-zh.en" as filtered English sentence "my_en-zh.ch" as filtered Chinese sentence in [/data/cleaned_data/](/data/cleaned_data/)

### Step 2:
For the Spacy embedding dictionary for both Chinese and English ,
```bash
python -m spacy download zh_core_web_md
python -m spacy download en_core_web_md
```

### step 3:
```python3 lang.py```  
After execution, embedding vectors for both English and Chinese data that we filtered. When finished code, you will have
1. "emb_en.npy" as embedded vectors for Engslish sentence
2. "emb_zh.npy" as embedded vectors for Chinese sentence

### Step 4:
Run the RNN encoder decoder with attention mechanism by  
```python3 AttentionMechanism.py```

### Step 5:
Run the RNN encoder decoder with intra-attention by  
```python3 intraAttention.py```