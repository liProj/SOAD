"""Gate statistics of the tiled geometry model on held-out tiles, with clean and fully masked curvature maps.
The fusion is f * (1 + alpha * gate); this measures how much the gate actually varies. -> results/gate_probe.json"""
import sys, json, torch, numpy as np
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/third_party/Geosphere-DETR-run")
from ultralytics import RTDETR
NB = 25


class Stop(Exception):
    pass


out = {}
for name, miss in (("clean", 0.0), ("masked", 1.0)):
    m = RTDETR(f"{B}/runs/T_E4_aligned_s0/weights/last.pt"); net = m.model; net.geom_missing_ratio = miss
    st = {i: [] for i in range(len(net.gate_heads))}; nb = [0]

    def mk(i):
        def hook(mod, inp, o):
            g = torch.sigmoid(o.float()); st[i].append((g.mean().item(), g.std().item(), g.flatten(1).std(1).mean().item(), g.min().item(), g.max().item()))
            if i == len(net.gate_heads) - 1:
                nb[0] += 1
                if nb[0] >= NB: raise Stop
        return hook
    hs = [h.register_forward_hook(mk(i)) for i, h in enumerate(net.gate_heads)]
    try:
        m.val(data=f"{B}/configs/tiles_geomBN_aligned.yaml", split="test", imgsz=640, batch=16, conf=0.001, device=0, plots=False, verbose=False, project=f"{B}/runs/_probe", name="gate", exist_ok=True)
    except Stop:
        pass
    for h in hs: h.remove()
    out[name] = {f"level{i}": dict(zip(("mean", "std_all", "std_within_image", "min", "max"), np.mean(v, 0).round(4).tolist())) for i, v in st.items()}
    out[name]["batches"] = nb[0]
    print(name, json.dumps(out[name]), flush=True)
out["alpha"] = float(net.geo_alpha); out["fusion"] = "f * (1 + alpha * gate)"
json.dump(out, open(f"{B}/results/gate_probe.json", "w"), indent=1)
