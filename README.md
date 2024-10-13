# SAID

## environment
python>=3.8

```
pip install -r requirements.txt
```

## data
The wildchat data is collected from [wildchat](https://huggingface.co/datasets/allenai/WildChat). We labeled the data by GPT-4 and release it to public for academic use. 

## pretrain
pretrain.py: 
```
python pretrain.py [--base_model roberta-base] [--train_file ./data/wildchat_pretrain.csv] [--mlm_ratio 0.15] [--batch_size 60] [--epochs 3] [--query_only False] [--output output_dir]
```

## finetune 
finetune.py: finetune model which loads checkpoint if able with given data. test results would be saved in output file. 
```
python finetune.py [--base_model google-bert/bert-base-uncased] [--finetune_file wildchat.csv] [--output output_file_name] [--checkpoint pretrained_model_to_load] [--data_dir data_dir] [--epochs 70] [--lr 1e-5] [--batch_size 5] [--hard_prompt]
```
> notice: it will finetune on base_model if you don't give a checkpoint.
> checkpoint can be state_file loaded by torch.load() or a directory loaded by method AutoModel.from_pretrained().
