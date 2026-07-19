"""Print every training/evaluation metric in the project as terminal tables.

    python src/common/show_metrics.py

A note on "accuracy" / "val AUC": those are CLASSIFICATION metrics — they need a right answer to
compare against. Generative models invent images, so neither exists here *by construction*. What we
report instead:
  * training loss  — the diffusion objective is MSE on predicting the added noise ε ~ N(0,1), whose
                     variance is 1.0. So (1 - loss) reads as "% of the noise variance explained":
                     0.011 → ≈98.9% (learned), 0.98 → ≈2% (learned nothing).
  * FID            — distance between distributions of generated vs real paintings (lower better).
                     ONLY meaningful vs the real-vs-real floor at the same sample size (we measured
                     floor = 156.7 at N≈150/300 — which invalidated our v1 numbers; hence eval v2).
  * CLIP score     — cosine(image embedding, prompt embedding)×100 (higher = follows the prompt).
"""
import json
import pathlib
import re
import sys

# Windows consoles often default to cp1252, which cannot print arrows/dashes — force UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]

HEADNOTE = """A note on "accuracy" / "val AUC": those are CLASSIFICATION metrics — they need a right
answer to compare against. Generative models invent images, so neither exists here by construction.
What we report instead: training loss (1-loss = "% of noise variance explained"), FID vs the
real-vs-real floor, and CLIP score (prompt adherence). Details in each table below."""

P1_NOTES = {
    "p1a_butterflies":   "main sanity run — EMA bug found & fixed here",
    "p1a_speedcheck":    "throughput probe (200→500 img/s experiment)",
    "p1_r01_lr2e3":      "DELIBERATE FAILURE: lr 10x too high",
    "p1b_impressionism": "on-theme: learns palette/texture, not scenes",
    "run":               "smoke test (16px, fake data)",
}

# v1 trainable-parameter counts (measured at train time)
V1_PARAMS = {"base": "0", "lora_r4": "0.8M", "lora_r16": "3.2M", "lora_r64": "12.8M",
             "full_ft": "860M", "dreambooth": "3.2M"}

# lora_r16's in-sweep infer crashed on a Windows file lock (JOURNEY D4) and its logged eval ran on
# 16 stale images (FID 317). These are its corrected values from the post-sweep re-infer (256 imgs).
V1_CORRECTIONS = {"lora_r16": {"fid": 164.07, "clip": 33.15, "note": "corrected re-infer (D4)"}}


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def phase1_table():
    hr("PHASE 1 — from-scratch DDPM training runs  (source: TensorBoard logs)")
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        print("  tensorboard not importable in this env"); return
    rows = []
    for run in sorted((ROOT / "outputs" / "phase1").glob("*/tb")):
        name = run.parent.name
        try:
            ea = EventAccumulator(str(run), size_guidance={"scalars": 0}); ea.Reload()
            if "loss" not in ea.Tags().get("scalars", []):
                continue
            ev = ea.Scalars("loss")
            first, final = ev[0].value, ev[-1].value
            mn = min(e.value for e in ev)
            explained = max(0.0, (1.0 - final)) * 100
            rows.append((name, ev[-1].step, f"{first:.3f}", f"{final:.4f}", f"{mn:.4f}",
                         f"{explained:5.1f}%", P1_NOTES.get(name, "")))
        except Exception as e:  # noqa: BLE001
            rows.append((name, "-", "-", "-", "-", "-", f"unreadable: {e}"))
    print(f"  {'run':<20}{'last step':>10}{'first':>8}{'final':>9}{'min':>9}{'noise expl.':>13}   note")
    print("  " + "-" * 96)
    for r in rows:
        print(f"  {r[0]:<20}{r[1]:>10}{r[2]:>8}{r[3]:>9}{r[4]:>9}{r[5]:>13}   {r[6]}")
    print("\n  'noise expl.' = (1 - final MSE)x100 — the honest analogue of accuracy for a diffusion")
    print("  model (predicting ε with variance 1.0). There is NO val-AUC for generative models.")


def v1_table():
    hr("PHASE 2 — evaluation v1  (256 imgs/model vs 300 refs — LATER SHOWN UNRELIABLE)")
    log = ROOT / "experiments" / "sweep.log"
    if not log.exists():
        print("  sweep.log missing"); return
    rows, cur = {}, None
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"BEGIN eval:(\S+)", line)
        if m:
            cur = m.group(1)
        m = re.search(r"FID = ([\d.]+)", line)
        if m and cur:
            rows.setdefault(cur, {})["fid"] = float(m.group(1))
        m = re.search(r"CLIPScore = ([\d.]+)", line)
        if m and cur:
            rows.setdefault(cur, {})["clip"] = float(m.group(1))
    for k, v in V1_CORRECTIONS.items():
        rows[k] = {"fid": v["fid"], "clip": v["clip"], "note": v["note"]}
    print(f"  {'model':<14}{'FID ↓':>9}{'CLIP ↑':>9}{'trainable':>11}   note")
    print("  " + "-" * 66)
    for name in ("base", "lora_r4", "lora_r16", "lora_r64", "full_ft", "dreambooth"):
        v = rows.get(name, {})
        print(f"  {name:<14}{v.get('fid', float('nan')):>9.2f}{v.get('clip', float('nan')):>9.2f}"
              f"{V1_PARAMS.get(name, '?'):>11}   {v.get('note', '')}")
    print("\n  ⚠ v1 verdict: REAL paintings score FID 156.7 vs OTHER real paintings at this sample")
    print("  size — every model above sits within ~10 points of that floor, so v1 FID could not")
    print("  distinguish the methods. That discovery motivated evaluation v2 (below).")


def v2_table():
    hr("PHASE 2 — evaluation v2  (2048 imgs/model, ~2800 refs, neutral prompts, tuned scale)")
    res = ROOT / "experiments" / "eval_v2_results.json"
    if res.exists():
        data = json.loads(res.read_text())
        print(f"  reference: {data['reference_n']} real paintings   |   "
              f"REAL-vs-REAL floor at this N: FID {data['real_vs_real_floor']:.2f}\n")
        print(f"  {'model':<18}{'N':>6}{'FID ↓':>9}{'vs floor':>10}{'CLIP ↑':>9}")
        print("  " + "-" * 54)
        for name, v in data["models"].items():
            clip = f"{v['clip']:.2f}" if v.get("clip") is not None else "n/a"
            print(f"  {name:<18}{v['n']:>6}{v['fid']:>9.2f}{v['fid_above_floor']:>+10.2f}{clip:>9}")
        print("\n  'vs floor' is the honest number: distance ABOVE what real art scores against")
        print("  real art. ~0 would mean statistically indistinguishable from real Impressionism.")
        return
    # not finished -> show live progress
    log = ROOT / "experiments" / "eval_v2.log"
    print("  status: RUNNING (results file not written yet)")
    if log.exists():
        status = [ln for ln in log.read_text(encoding="utf-8", errors="ignore").splitlines()
                  if re.search(r"(BEGIN|END|START|DONE)", ln)]
        for ln in status[-3:]:
            print("    " + ln.strip())
    v2 = ROOT / "outputs" / "phase2_eval_v2"
    if v2.exists():
        for d in sorted(v2.glob("*/eval_samples")):
            n = len(list(d.glob("*.jpg")))
            print(f"    generated  {d.parent.name:<18} {n:>5} / 2048")
    nref = len(list((ROOT / "data" / "impressionism_512_ref" / "heldout").glob("*.jpg")))
    print(f"    reference extraction: {nref} / 2500")


def zoo():
    hr("MODEL ZOO — every trained checkpoint on disk")
    print(f"  {'model':<26}{'size':>10}   path")
    print("  " + "-" * 84)
    for p in sorted((ROOT / "outputs").rglob("lora_last.pt")):
        mb = p.stat().st_size / 1e6
        print(f"  {p.parent.parent.name + ' (LoRA)':<26}{mb:>8.1f}MB   {p.relative_to(ROOT)}")
    for p in sorted((ROOT / "outputs").rglob("unet_last")):
        if "_memtest" in str(p):
            continue
        sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
        print(f"  {p.parent.parent.name + ' (full UNet)':<26}{sz:>8.2f}GB   {p.relative_to(ROOT)}")
    for p in sorted((ROOT / "outputs" / "phase1").glob("*/ckpt/last.pt")):
        mb = p.stat().st_size / 1e6
        print(f"  {p.parent.parent.name + ' (scratch)':<26}{mb:>8.1f}MB   {p.relative_to(ROOT)}")


if __name__ == "__main__":
    print(HEADNOTE)
    phase1_table()
    v1_table()
    v2_table()
    zoo()
    print()
