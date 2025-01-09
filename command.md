
## Process Data
> IMage should be in the 256x256, 512x512, 1024x1024
```bash
python DiffAugment-stylegan2-pytorch/data_process/process_data.py --data_dir data/dummy_data/ --output_dir data/processed_dummy_data
```
## Train

```bash
python DiffAugment-stylegan2-pytorch/train.py --outdir checkpoints --data data/processed_dummy_data/
```