"""Abstract-level screening of the 200 AIIA papers (full text was not machine-readable: ScienceDirect
answers scripts with 403 and a browser visit with a CAPTCHA). Every row says what evidence it rests on."""
import re, pandas as pd
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
d = pd.read_csv(f"{B}/latest200.csv").fillna("")
# verified by hand on 2026-10-12 (GitHub API search + repository listing + Hugging Face API)
VERIFIED = {
 "10.1016/j.aiia.2026.08.010": ("yes", "HF ariel0818/citrus-fruit-rgbd, CC BY 4.0, 13,213 files, 14.5 GB; code github.com/Ariel0818/Geosphere-DETR", "checked: dataset + code both downloadable"),
 "10.1016/j.aiia.2026.08.016": ("partial", "github.com/geonextgis/CropFusionNet has code only; README: data/output excluded from version control", "checked: code yes, data no"),
 "10.1016/j.aiia.2025.04.009": ("unknown", "github.com/grahman20/ADeepWeed holds a README only ('Source code: coming soon')", "checked: repo empty"),
}
REVIEW = r"\breview\b|survey|bibliometric|perspective|status, bottleneck|challenges and future directions|recent advances"
CATS = [("review", REVIEW), ("livestock/aquaculture", r"\b(pigs?|piglets?|sows?|swine|cattle|cows?|sheep|goats?|chickens?|hens|poultry|shrimp|aquaculture|livestock)\b"),
        ("genomics", r"genomic|genome|\bSNP"), ("LLM/agents", r"language model|\bLLMs?\b|generative AI|multi-modal models"),
        ("robotics/harvest/navigation", r"robot|harvest|picking|grasp|navigation|SLAM|obstacle|spray|roguing|pruning|packag|seeding quantity|fertilizer"),
        ("3D phenotyping/point cloud", r"point cloud|\b3D\b|NeRF|Gaussian|LiDAR"),
        ("spectral/hyperspectral", r"hyperspectral|spectroscop|spectra\b|\bNIR\b|\bSIF\b"),
        ("disease/pest/weed vision", r"disease|pest|weed|lesion|mildew|scab|blight"),
        ("remote sensing/yield", r"\bUAV\b|satellite|remote sensing|yield|\bLAI\b|biomass|NDVI|Sentinel")]
PUB = r"publicly available|public dataset|public benchmark|open-source|github|zenodo|kaggle|figshare|hugging ?face|PlantVillage|MODIS|Sentinel|released|IP102|benchmark"
OWN = r"we (collected|constructed|built|established)|was collected|were collected|self-built|field experiment|field trial|our dataset|a dataset (of|comprising|containing)|plots"
rows = []
for i, r in d.iterrows():
    txt = r.title + " " + r.abstract_oa
    cat = next((c for c, p in CATS if re.search(p, r.title, re.I)), None) or next((c for c, p in CATS[1:] if re.search(p, txt, re.I)), "other")
    pub = sorted(set(m.group(0).lower() for m in re.finditer(PUB, r.abstract_oa, re.I)))
    own = bool(re.search(OWN, r.abstract_oa, re.I))
    if r.doi in VERIFIED: public, where, ev = VERIFIED[r.doi]
    elif cat == "review": public, where, ev = "n/a", "", "abstract only"
    elif not r.abstract_oa: public, where, ev = "unknown", "", "title only (no abstract in OpenAlex)"
    elif pub: public, where, ev = "claimed/likely", "abstract mentions: " + ", ".join(pub), "abstract only"
    else: public, where, ev = "not stated" + (" (own field/lab data)" if own else ""), "", "abstract only"
    heavy = bool(re.search(r"robot|field trial|SLAM|3D reconstruction|LLM|language model|UAV|hyperspectral|point cloud", txt, re.I))
    repro = "n/a" if cat == "review" else ("hard: hardware/field campaign or private sensing data" if heavy and public != "yes" else "medium: standard deep-learning training")
    s_data = {"yes": 5, "partial": 2, "claimed/likely": 2}.get(public, 0)
    s_time = 0 if cat == "review" else (1 if heavy else 2)
    score = s_data * 2 + s_time + (1 if r.doi in VERIFIED and public == "yes" else 0)
    newm = "n/a" if cat == "review" else ("yes - data and code open, pretrained backbones/foundation features applicable" if public == "yes"
            else "possible in principle, but not testable without the data")
    beat = "testable (see results/)" if public == "yes" else ("not testable" if cat != "review" else "n/a")
    rows.append(dict(idx=i, doi=r.doi, pii=r.pii, published=r.pub_date, title=r.title, category=cat, data_public=public,
                     data_where_size=where, downloadable_checked=ev, reproducibility=repro, new_method_feasible=newm,
                     can_beat_all_datasets=beat, score=score, evidence=ev, url=f"https://doi.org/{r.doi}"))
s = pd.DataFrame(rows).sort_values(["score", "published"], ascending=[False, False])
s.to_csv(f"{B}/screening.csv", index=False)
s[s.score >= 5].to_csv(f"{B}/results/shortlist.csv", index=False)
print(s.category.value_counts().to_string()); print(s.data_public.value_counts().to_string())
print(s.head(12)[["idx", "doi", "data_public", "score", "title"]].to_string())
