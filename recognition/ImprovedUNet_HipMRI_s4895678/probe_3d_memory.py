"""Measure peak GPU memory and step time of the 3D Improved U-Net (random data)."""
import gc
import time

import torch
import torch.nn.functional as F

from modules import ImprovedUNet

dev = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
print("GPU:", props.name, "| total GB:", round(props.total_memory / 1e9, 1))

configs = [((64, 128, 128), 16, True), ((64, 128, 128), 16, False),
           ((128, 256, 256), 8, True), ((128, 256, 256), 8, False),
           ((96, 192, 192), 16, True)]

for shape, base, amp in configs:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    label = f"{shape} base {base} {'bf16' if amp else 'fp32'}"
    try:
        m = ImprovedUNet(dim=3, base=base, depth=5).to(dev).train()
        opt = torch.optim.Adam(m.parameters())
        x = torch.randn(1, 1, *shape, device=dev)
        y = torch.randint(0, 6, (1, *shape), device=dev)
        t0 = time.time()
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = F.cross_entropy(m(x).float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"{label}: peak {gb:.1f} GB | {(time.time() - t0) / 2:.2f} s/step")
    except torch.cuda.OutOfMemoryError:
        print(f"{label}: OUT OF MEMORY")
    m = opt = x = y = loss = None
    gc.collect()
