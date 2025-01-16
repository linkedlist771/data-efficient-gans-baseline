import legacy
import torch
import dnnlib
import argparse

def load_and_convert_to_pt(source_pkl, dest_pt, force_fp16=False):
    """
    Load a network pickle and convert it to a .pt format.

    Args:
    source_pkl (str): Path to the source pickle file.
    dest_pt (str): Path to save the converted .pt file.
    force_fp16 (bool): Whether to force FP16 precision.
    """
    print(f'Loading "{source_pkl}"...')
    with dnnlib.util.open_url(source_pkl) as f:
        data = legacy.load_network_pkl(f, force_fp16=force_fp16)

    # Extract the G_ema model (usually the one used for inference)
    G_ema = data['G_ema']

    # Save the model in .pt format
    torch.save(G_ema.state_dict(), dest_pt)
    print(f'Saved "{dest_pt}".')


def main():
    parser = argparse.ArgumentParser(description="Convert StyleGAN2 network pickle to PyTorch .pt format.")
    parser.add_argument("--source_pkl", type=str, help="Path to the source pickle file.")
    parser.add_argument("--dest_pt", type=str, help="Path to save the converted .pt file.")
    parser.add_argument("--force-fp16", action="store_true", help="Whether to force FP16 precision.")
    args = parser.parse_args()

    load_and_convert_to_pt(args.source_pkl, args.dest_pt, args.force_fp16)

if __name__ == "__main__":
    main()
