"""Keep the 200 newest research articles and attach OpenAlex abstracts / PII / OA licence.

ScienceDirect returns 403 to scripts and the KeAi site carries tdm-reservation=1, so the
full text is not bulk-downloaded here; abstracts come from OpenAlex, PII from Crossref.
"""
import re, time, requests, pandas as pd
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
UA = {"User-Agent": "mdpi-repro-screening/0.2 (mailto:ireneliuxin@outlook.com)"}
d = pd.read_csv(f"{B}/latest50_crossref.csv")
nonart = d.title.fillna("").str.contains(r"^(?:Erratum|Corrigendum|Editorial Board|Retraction|Correction|Publisher)", case=False) | d.title.isna()
d["is_article"] = ~nonart
d.to_csv(f"{B}/all_fetched.csv", index=False)
d = d[d.is_article].head(200).reset_index(drop=True)
rows = []
for i in range(0, len(d), 50):
    chunk = d.doi[i:i + 50]
    r = requests.get("https://api.openalex.org/works", headers=UA, timeout=120, params={
        "filter": "doi:" + "|".join(chunk), "per-page": 50,
        "select": "doi,abstract_inverted_index,type,publication_date,biblio,primary_location,keywords,topics"})
    r.raise_for_status()
    for w in r.json()["results"]:
        inv = w.get("abstract_inverted_index") or {}
        pos = sorted((p, t) for t, ps in inv.items() for p in ps)
        rows.append(dict(doi=w["doi"].replace("https://doi.org/", "").lower(),
                         abstract_oa=" ".join(t for _, t in pos), oa_type=w.get("type"),
                         pub_date=w.get("publication_date"),
                         volume_oa=(w.get("biblio") or {}).get("volume"),
                         licence=(w.get("primary_location") or {}).get("license"),
                         keywords="; ".join(k["display_name"] for k in w.get("keywords") or []),
                         topic=((w.get("topics") or [{}])[0]).get("display_name", "")))
    time.sleep(0.5)
d = d.merge(pd.DataFrame(rows).drop_duplicates("doi"), on="doi", how="left")
pii = []
for doi in d.doi:
    try:
        m = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=60).json()["message"]
        pii.append((m.get("alternative-id") or [""])[0])
    except Exception:
        pii.append("")
    time.sleep(0.05)
d["pii"] = pii
d.to_csv(f"{B}/latest200.csv", index=False)
print(len(d), "articles;", (d.abstract_oa.fillna("").str.len() > 200).sum(), "with abstract;", (d.pii != "").sum(), "with PII")
print(d.oa_type.value_counts().to_dict()); print(d.pub_date.min(), d.pub_date.max())
