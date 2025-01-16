
## Process Data
> IMage should be in the 256x256, 512x512, 1024x1024
```bash
python DiffAugment-stylegan2-pytorch/data_process/process_data.py --data_dir data/dummy_data/ --output_dir data/processed_dummy_data
```
## Train

```bash
export CUDA_VISIBLE_DEVICES=1 &&   nohup python DiffAugment-stylegan2-pytorch/train.py --outdir checkpoints --cfg stylegan2 --data data/deposition_data_processed/  > $(date +%m%d)"style_gan_train".log 2>&1 &
```

## Infer-gif(pkl)
```bash
python DiffAugment-stylegan2-pytorch/generate_gif.py  --network checkpoints/00003--low_shot-color-translation-cutout/network-snapshot-000300.pkl  --output output
```