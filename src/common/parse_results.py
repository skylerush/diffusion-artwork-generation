"""Parse experiments/sweep.log into a markdown FID/CLIP table (final-assembly helper)."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    log = (ROOT / "experiments" / "sweep.log").read_text(encoding="utf-8", errors="ignore")
    rows, cur = {}, None
    for line in log.splitlines():
        m = re.search(r"BEGIN (?:eval|infer):(\S+)", line)
        if m:
            cur = m.group(1)
        m = re.search(r"FID = ([\d.]+)", line)
        if m and cur:
            rows.setdefault(cur, {})["FID"] = float(m.group(1))
        m = re.search(r"CLIPScore = ([\d.]+)", line)
        if m and cur:
            rows.setdefault(cur, {})["CLIP"] = float(m.group(1))
    print("| Model | FID ↓ | CLIP ↑ |")
    print("|---|---|---|")
    for name, v in rows.items():
        print(f"| {name} | {v.get('FID', '—')} | {v.get('CLIP', '—')} |")


if __name__ == "__main__":
    main()
