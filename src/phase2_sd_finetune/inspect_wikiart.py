"""Inspect the WikiArt dataset schema and locate the Impressionism style.

Anti-hallucination rule: verify the real columns / label names before building the
data-prep pipeline on them. Uses streaming so we don't download the whole dataset
just to read its schema.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

REPO = "huggan/wikiart"


def main():
    print(f"loading {REPO} (streaming) ...")
    ds = load_dataset(REPO, split="train", streaming=True)
    feats = ds.features
    print("columns:", list(feats.keys()) if feats else "unknown")

    for key in ("style", "genre", "artist"):
        f = feats.get(key) if feats else None
        if f is not None and hasattr(f, "names"):
            names = list(f.names)
            print(f"\n{key}: {len(names)} classes")
            if key == "style":
                matches = [(i, n) for i, n in enumerate(names) if "impress" in n.lower()]
                print("  >>> impressionism matches (index, name):", matches)
                print("  all style names:", names)

    print("\n--- first 3 examples ---")
    it = iter(ds)
    for i in range(3):
        ex = next(it)
        desc = {}
        for k, v in ex.items():
            desc[k] = f"PIL size={getattr(v, 'size', None)}" if k == "image" else v
        print(i, desc)


if __name__ == "__main__":
    main()
