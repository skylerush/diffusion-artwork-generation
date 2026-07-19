"""GPU / PyTorch verification for the RTX 5090 (Blackwell, sm_120).

Run inside the project venv:  python environment/verify_gpu.py
Success prints a final line:   VERIFY_OK sm_<cc>
Failure raises (so setup.ps1 marks the install as failed instead of silently passing).
"""
import platform
import sys


def main() -> int:
    print("python      :", platform.python_version())
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print("FATAL: cannot import torch:", e)
        return 1

    print("torch       :", torch.__version__)
    print("torch.cuda  :", torch.version.cuda)
    print("cuda avail  :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available. On a 5090 this usually means the wrong wheel "
              "(need the cu128 build). Reinstall torch from "
              "https://download.pytorch.org/whl/cu128")
        return 2

    i = torch.cuda.current_device()
    name = torch.cuda.get_device_name(i)
    cc = torch.cuda.get_device_capability(i)
    print("device      :", name)
    print("capability  :", f"sm_{cc[0]}{cc[1]}")
    print("bf16 ok     :", torch.cuda.is_bf16_supported())

    # Real compute on the GPU (catches 'installed but cannot run kernels on sm_120').
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul mean :", float(y.mean()))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = (x @ x).sum()
    torch.cuda.synchronize()
    print("autocast ok :", float(z))

    if cc[0] < 12:
        print(f"WARNING: expected Blackwell sm_120, but got sm_{cc[0]}{cc[1]}.")
    print(f"VERIFY_OK sm_{cc[0]}{cc[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
