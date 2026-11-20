import torch
import torch.nn.functional as F
from spikingjelly.clock_driven import functional

from model import SpiLiFormer


def main():
    assert torch.cuda.is_available(), "PyTorch cannot see a CUDA GPU"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("PyTorch:", torch.__version__, "| CUDA build:", torch.version.cuda)

    torch.manual_seed(2021)
    model = SpiLiFormer(T=16).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.06)
    print("Parameters:", sum(p.numel() for p in model.parameters()), flush=True)

    # Synthetic event-count frames: [batch, time, polarity, height, width]
    x = (torch.rand(2, 16, 2, 128, 128, device="cuda") < 0.05).float()
    labels = torch.tensor([0, 1], device="cuda")
    before = model.head.weight.detach().clone()

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)

    first_logits, feedback, tmp = model(x)
    functional.reset_net(model)
    logits, _ = model(tmp, second_forward=feedback)
    assert first_logits.shape == logits.shape == (2, 10)

    loss = 0.5 * F.cross_entropy(first_logits, labels)
    loss += 0.5 * F.cross_entropy(logits, labels)
    assert torch.isfinite(loss).item(), "Non-finite loss"
    loss.backward()

    grad = model.head.weight.grad
    assert grad is not None and torch.isfinite(grad).all().item()
    assert grad.abs().max().item() > 0, "Zero gradient at classification head"

    optimizer.step()
    functional.reset_net(model)
    delta = (model.head.weight.detach() - before).abs().max().item()
    assert delta > 0, "Classification head did not update"

    print("Logits:", tuple(logits.shape), "| Loss:", loss.item())
    print("Max head weight change:", delta)
    print("Peak PyTorch allocated GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print("PASS: two forwards, backward, optimizer step")


if __name__ == "__main__":
    main()