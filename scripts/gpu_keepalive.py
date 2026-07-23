"""Low-memory CUDA workload used only when the main pipeline is GPU-idle."""

import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duty-cycle", type=float, default=0.50)
    parser.add_argument("--work-seconds", type=float, default=0.05)
    parser.add_argument("--matrix-size", type=int, default=4096)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU keepalive")
    if not 0.30 <= args.duty_cycle <= 1.0:
        raise ValueError("duty cycle must stay between 0.30 and 1.0")

    device = torch.device("cuda")
    a = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=torch.float16)
    b = torch.randn(args.matrix_size, args.matrix_size, device=device, dtype=torch.float16)
    output = torch.empty_like(a)
    torch.cuda.synchronize()
    print(
        f"GPU keepalive active: matrix={args.matrix_size}, "
        f"target duty={args.duty_cycle:.0%}",
        flush=True,
    )

    while True:
        started = time.monotonic()
        while time.monotonic() - started < args.work_seconds:
            torch.mm(a, b, out=output)
            a, output = output, a
        torch.cuda.synchronize()
        work_time = time.monotonic() - started
        sleep_time = work_time * (1.0 - args.duty_cycle) / args.duty_cycle
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
