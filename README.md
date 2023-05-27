# OCT Image Understanding

## Dataset

Processed from `boe_chiu` dataset includes train: 86 test: 28. Download link:
[train](https://drive.google.com/file/d/1PiBaXNbBpKIkKz33EfVTjR3pR2Dx8XH9/view?usp=sharing) and [test](https://drive.google.com/file/d/1TXbxrufaBpq2fWCkl1iMCw8nbIXOep2e/view?usp=sharing).

## Training

### UNet Baseline

Use index to indicate which gpu card you are going to use

```bash
sh script/train_unet.sh [gpu_id]
```

### Probabilistic UNet

```bash
sh script/train_prob_unet.sh [gpu_id]
```

## References
