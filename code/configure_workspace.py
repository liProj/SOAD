"""Resolve dataset YAML paths after cloning; no training or data download."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
for p in (root / 'configs').glob('*.yaml'):
    s = p.read_text(encoding='utf-8')
    def replace(m):
        raw = m.group(1).strip().strip('\"\'')
        match = re.search(r'(?:^|/)data/(.+)$', raw.replace('\\', '/'))
        return 'path: ' + (root / 'data' / match.group(1)).as_posix() if match else m.group(0)
    updated = re.sub(r'^path:\s*(.+)$', replace, s, flags=re.M)
    if updated != s:
        p.write_text(updated, encoding='utf-8')
for d in ['results/det', 'results/dets', 'logs', 'runs', 'weights', 'data']:
    (root / d).mkdir(parents=True, exist_ok=True)
print('Workspace paths configured at', root)
