# -*- coding: utf-8 -*-

# ***************************************************
# * File        : download_file.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-10-02
# * Version     : 1.0.100216
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import urllib.request

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


def download_file(url, out_dir=".", backup_url=None):
    """
    Download *url* into *out_dir* with an optional mirror fallback.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.urlparse(url).path).name
    dest = out_dir / filename

    def _download(u):
        try:
            with urllib.request.urlopen(u) as r:
                size_remote = int(r.headers.get("Content-Length", 0))
                if dest.exists() and dest.stat().st_size == size_remote:
                    print(f"✓ {dest} already up-to-date")
                    return True

                block = 1024 * 1024  # 1 MiB
                downloaded = 0
                with open(dest, "wb") as f:
                    while chunk := r.read(block):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if size_remote:
                            pct = downloaded * 100 // size_remote
                            sys.stdout.write(
                                f"\r{filename}: {pct:3d}% "
                                f"({downloaded // (1024*1024)} MiB / {size_remote // (1024*1024)} MiB)"
                            )
                            sys.stdout.flush()
                if size_remote:
                    sys.stdout.write("\n")
            return True
        except (urllib.error.HTTPError, urllib.error.URLError):
            return False

    if _download(url):
        return dest

    if backup_url:
        print(f"Primary URL ({url}) failed. \nTrying backup URL ({backup_url})...,")
        if _download(backup_url):
            return dest

    raise RuntimeError(f"Failed to download {filename} from both mirrors.")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
