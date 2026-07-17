from pathlib import Path
import torch

checkpoint_dir = Path("checkpoints")

results = []

for path in checkpoint_dir.glob("*.ckpt"):
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        results.append(
            (
                checkpoint.get("epoch", -1),
                checkpoint.get("global_step", -1),
                path.stat().st_mtime,
                path.name,
            )
        )

    except Exception as error:
        print(f"Errore leggendo {path.name}: {error}")

for epoch, step, _, name in sorted(results, reverse=True):
    print(
        f"{name:65s} "
        f"| epoch={epoch:3d} "
        f"| global_step={step}"
    )