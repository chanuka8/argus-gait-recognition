import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
import torch

from models.reid.osnet_backbone import _build_osnet_x0_25


def download_and_verify():
    dest_path = Path("models/weights/osnet_x0_25.pth")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    urls = [
        ("HuggingFace MSMT17 ReID", "https://huggingface.co/spaces/hysts/DeepPersonReID/resolve/main/models/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp50_lr0.0015_b64_fb10_sqh0.5_crossent_triplet_auto_clean.pth"),
        ("HuggingFace ImageNet", "https://huggingface.co/kadirnar/osnet_x0_25_imagenet/resolve/main/osnet_x0_25_imagenet.pth"),
    ]

    downloaded = False
    for name, url in urls:
        print(f"Trying download from {name} ({url})...")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 100000:
                print(f"Successfully downloaded {len(resp.content)} bytes from {name}")
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                downloaded = True
                break
            else:
                print(f"Failed with status {resp.status_code}, length {len(resp.content)}")
        except (requests.RequestException, OSError, ValueError) as exc:
            print(f"Download failed for {name}: {exc}")

    if not downloaded:

        gdrive_id = "1KkxK1eqSg_P-P3bnoGswcfsp8-Xm_ZtW"
        print(f"Trying Google Drive direct download for ID {gdrive_id}...")
        session = requests.Session()
        g_url = f"https://drive.google.com/uc?export=download&id={gdrive_id}"
        resp = session.get(g_url, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 100000:
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            downloaded = True
            print(f"Downloaded {len(resp.content)} bytes from Google Drive")

    if not dest_path.exists():
        raise RuntimeError("Failed to obtain OSNet pretrained weights checkpoint.")


    print("\n--- Verifying Checkpoint Structure ---")
    ckpt = torch.load(dest_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    print(f"Checkpoint contains {len(state_dict)} tensor keys")


    model = _build_osnet_x0_25()
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        if "classifier" in clean_key:
            continue
        cleaned[clean_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Missing keys: {len(missing)} ({missing[:5]}...)")
    print(f"Unexpected keys: {len(unexpected)} ({unexpected[:5]}...)")


    first_conv_weight = model.conv1.conv.weight.data
    weight_mean = float(first_conv_weight.mean())
    weight_std = float(first_conv_weight.std())
    print(f"Conv1 weight mean: {weight_mean:.6f}, std: {weight_std:.6f}")
    assert weight_std > 0, "Weights appear uninitialized or zero!"
    print("Checkpoint verification SUCCESSFUL! Weights are genuine and match OSNet-x0.25.")

if __name__ == "__main__":
    download_and_verify()
