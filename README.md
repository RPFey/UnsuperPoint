# Unsuper

## requiements

```bash
python -m pip install -r requirements.txt
```

## dataset

```plain
-- Data
    |
    ---- COCO
          |
          ---- patches
          ---- train 2014
          ----  ....
    |
    ---- HPatch
            |
            ----  i_ ..
```

## train

run 

```bash
sh scripts/command_train.sh # multiple gpus
sh scripts/command_single.sh # single gpu train
sh scripts/command_export.sh # single gpu
```