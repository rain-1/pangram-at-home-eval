"""Environment check: Python/packages, GPU, disk, HF authentication (never printed), model snapshots, sources."""

from __future__ import annotations

import importlib.metadata as md
import platform
import shutil

from aidet_eval import paths


def doctor() -> int:
    ok = True
    print(f"python {platform.python_version()} on {platform.platform()}")
    for p in ("torch", "transformers", "peft", "bitsandbytes", "datasets", "huggingface-hub", "pyarrow", "pydantic", "emoji"):
        try:
            print(f"  {p:16s} {md.version(p)}")
        except md.PackageNotFoundError:
            print(f"  {p:16s} MISSING")
            ok = False
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"GPU: {torch.cuda.get_device_name(0)}  free {free / 2**30:.1f} / {total / 2**30:.1f} GiB  CUDA {torch.version.cuda}  bf16={torch.cuda.is_bf16_supported()}")
        else:
            print("GPU: none (baseline inference needs CUDA for the reference bf16/NF4 configuration; tests do not)")
    except Exception as e:  # noqa: BLE001
        print(f"GPU check failed: {type(e).__name__}")
    du = shutil.disk_usage(paths.ROOT)
    print(f"disk free at {paths.ROOT}: {du.free / 1e9:.0f} GB")
    try:
        from huggingface_hub import HfApi

        who = HfApi().whoami()
        print(f"HF auth: logged in (account name withheld from logs: {'yes' if who.get('name') else 'unknown'})")
        for repo in ("pangram/editlens_roberta-large", "pangram/editlens_Llama-3.2-3B", "meta-llama/Llama-3.2-3B"):
            try:
                HfApi().model_info(repo)
                print(f"  access {repo}: ok")
            except Exception as e:  # noqa: BLE001
                print(f"  access {repo}: {type(e).__name__} (accept the model's access conditions on its HF page yourself; this tool never does)")
                ok = False
    except Exception as e:  # noqa: BLE001
        print(f"HF auth: not available ({type(e).__name__}); run `! huggingface-cli login` yourself")
    for f in sorted(paths.MODELS.glob("*.snapshot.json")):
        import json

        d = json.loads(f.read_text())
        print(f"model snapshot {d['model_id']}: {d['status']}")
    from aidet_eval.acquire import acquisition_status, check_registry_covers_catalogue

    probs = check_registry_covers_catalogue()
    print("registry covers all catalogue sources" if not probs else f"registry problems: {probs}")
    for r in acquisition_status():
        print(f"  source {r['key']:24s} {r['status']}")
    return 0 if ok else 1
