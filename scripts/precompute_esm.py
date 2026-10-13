from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

def _seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.strip().upper().encode("ascii", errors="replace")).hexdigest()

def _collect_sequences(pair_files: list[str]) -> list[str]:
    from pharmunit.dataset import load_pairs

    seen: set[str] = set()
    seqs: list[str] = []
    for pf in pair_files:
        df = load_pairs(pf)
        if df.empty:
            print(f"WARNING: {pf} parsed to 0 rows — wrong format or empty file.")
        for seq in df["target"].astype(str):
            seq = seq.strip().upper()
            if seq and seq not in seen:
                seen.add(seq)
                seqs.append(seq)
    return seqs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-file", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--max-len", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = _collect_sequences(args.pair_file)
    print(f"Found {len(seqs)} unique sequences across {len(args.pair_file)} files.")
    if not seqs:
        raise SystemExit(
            "ERROR: 0 sequences collected — nothing to cache. Check that the files "
            "exist and are either DrugBAN CSV (SMILES,Protein,Y) or whitespace TXT "
            "(smiles sequence label). Exiting non-zero rather than reporting 'done'."
        )

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    import esm
    model, alphabet = esm.pretrained.__dict__[args.model]()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    repr_layer = model.num_layers

    skipped = 0
    for seq in tqdm(seqs, desc="ESM"):
        s = seq[: args.max_len]
        key = _seq_hash(seq)
        fp = out_dir / f"{key}.npy"
        if fp.exists():
            continue
        try:
            _, _, tokens = batch_converter([("q", s)])
            tokens = tokens.to(device)
            with torch.no_grad():
                out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            rep = out["representations"][repr_layer][0, 1 : 1 + len(s)]
            np.save(fp, rep.cpu().numpy().astype(np.float32))
        except Exception as e:
            print(f"skip {key[:10]} (len {len(s)}): {e!r}")
            skipped += 1

    cached = len(list(out_dir.glob("*.npy")))
    print(f"done. cached files: {cached} / requested: {len(seqs)} / skipped this run: {skipped}")
    if cached < len(seqs):
        print("WARNING: cache is incomplete. Re-run this script to fill the gap; "
              "training will FileNotFoundError on the missing sequences.")

if __name__ == "__main__":
    main()
