"""Check the CUDA device and run a small FP16 calculation."""

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; install a CUDA-enabled PyTorch build.")

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(device)}")
    print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 2**30:.2f} GiB")
    print(f"PyTorch CUDA: {torch.version.cuda}")

    a = torch.randn((256, 256), device=device, dtype=torch.float16)
    b = torch.randn((256, 256), device=device, dtype=torch.float16)
    result = a @ b
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise SystemExit("FP16 matrix multiplication produced NaN or infinity. Use FP32.")
    print("FP16 matrix multiplication: finite")


if __name__ == "__main__":
    main()
