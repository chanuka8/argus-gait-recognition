"""
Test downloading Google Drive weights with session cookies and confirmation tokens.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
import torch


def download_from_gdrive(file_id: str, dest: Path) -> bool:
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }

    try:
        response = session.get(url, params={"id": file_id, "confirm": "t"}, headers=headers, stream=True, timeout=30)
        
        # Check for Google Drive virus warning token
        token = None
        for key, val in response.cookies.items():
            if key.startswith("download_warning"):
                token = val
                break
        
        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(url, params=params, headers=headers, stream=True, timeout=30)

        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            # Check if there's a confirm link in html
            import re
            match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
            if match:
                token = match.group(1)
                params = {"id": file_id, "confirm": token}
                response = session.get(url, params=params, headers=headers, stream=True, timeout=30)

        # Write chunks
        temp_dest = dest.with_suffix(".tmp")
        bytes_written = 0
        with open(temp_dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

        print(f"Downloaded {bytes_written} bytes (content-type: {response.headers.get('content-type')})")
        
        # Verify it is a valid PyTorch checkpoint
        try:
            ckpt = torch.load(temp_dest, map_location="cpu")
            if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model", "conv1.conv.weight")):
                temp_dest.replace(dest)
                print(f"Successfully verified and saved checkpoint to {dest} ({bytes_written} bytes)")
                return True
            elif hasattr(ckpt, "keys"):
                temp_dest.replace(dest)
                print(f"Successfully verified dict checkpoint and saved to {dest} ({bytes_written} bytes)")
                return True
            else:
                print(f"Object loaded is not a valid state_dict: {type(ckpt)}")
                temp_dest.unlink(missing_ok=True)
                return False
        except (RuntimeError, ValueError, OSError, EOFError) as e:
            print(f"Failed to load downloaded file with torch.load: {e}")
            temp_dest.unlink(missing_ok=True)
            return False

    except (requests.RequestException, OSError, ValueError) as exc:
        print(f"Download exception: {exc}")
        return False


def main():
    dest = Path("models/weights/osnet_x0_25.pth")
    dest.parent.mkdir(parents=True, exist_ok=True)

    gdrive_ids = [
        ("MSMT17 (Person ReID)", "1KkxK1eqSg_P-P3bnoGswcfsp8-Xm_ZtW"),
        ("Market1501 (Person ReID)", "16I0MhyZkM_jbaStspvBtiyauy_U_fXqM"),
        ("ImageNet (Pretrained)", "1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs"),
    ]

    for name, fid in gdrive_ids:
        print(f"\n--- Attempting download of OSNet-x0.25 ({name}) from GDrive ID {fid} ---")
        if download_from_gdrive(fid, dest):
            print(f"SUCCESS! {name} checkpoint is ready at {dest}")
            return True

    print("FAILED to download checkpoint from all Google Drive endpoints.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
