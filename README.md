# OCT Image Understanding

## Dataset

Processed from `boe_chiu` dataset includes train: 86 test: 28. Download link:
[train](https://drive.google.com/file/d/1ge8rux9cSnbjm8OA9HwOomLeypnFjA73/view?usp=sharing) and [test](https://drive.google.com/file/d/1gcTq5quY8OwtTGkLo614qeGFJ4smbFDF/view?usp=sharing).

## Training

Use index to indicate which gpu card you are going to use

```bash
sh script/train_unet.sh [gpu_id]
```
