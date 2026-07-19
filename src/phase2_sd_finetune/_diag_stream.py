"""Diagnostic: measure WikiArt streaming speed and where Impressionism appears.

Scans labels only (no image decode) to see (a) rows/sec and (b) whether Impressionism
is reachable early or the stream is style-ordered. Informs the data-acquisition strategy.
"""
import sys
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

ds = load_dataset("huggan/wikiart", split="train", streaming=True)
sidx = ds.features["style"].names.index("Impressionism")
print(f"Impressionism index = {sidx}", flush=True)

t0 = time.time()
scanned = found = 0
hist = {}
first_imp_at = None
for ex in ds:
    scanned += 1
    s = ex["style"]
    hist[s] = hist.get(s, 0) + 1
    if s == sidx:
        found += 1
        if first_imp_at is None:
            first_imp_at = scanned
    if scanned % 250 == 0:
        dt = time.time() - t0
        print(f"scanned={scanned} found_imp={found} rate={scanned/dt:.1f} rows/s elapsed={dt:.0f}s", flush=True)
    if scanned >= 5000 or found >= 20:
        break

dt = time.time() - t0
print(f"FINAL scanned={scanned} found_imp={found} first_imp_at={first_imp_at} "
      f"rate={scanned/dt:.1f} rows/s elapsed={dt:.0f}s", flush=True)
print("style histogram (index:count):", dict(sorted(hist.items())), flush=True)
