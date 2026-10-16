from pathlib import Path
import sys,re,json
r=Path(__file__).resolve().parents[1]
src=(r/'paper/manuscript_zh.tex').read_text(encoding='utf-8')
lines=src.splitlines()
if len(sys.argv)>1:
    lo,hi=map(int,sys.argv[1:3])
    for n,s in enumerate(lines,1):
        if lo<=n<=hi: print(f'{n}: {s}')
else:
    raw=(r/'paper/translation_en.txt').read_text(encoding='utf-8')
    parts=re.split(r'^@@(\d+)\s*$',raw,flags=re.M)
    changes={}
    for i in range(1,len(parts),2):
        n=int(parts[i]);assert n not in changes,n
        changes[n]=parts[i+1].strip()
    for n,v in changes.items():lines[n-1]=v
    out='\n'.join(lines)+'\n'
    out=out.replace(r'\{0,427,853,1280\}',r'\{0,\allowbreak 427,\allowbreak 853,\allowbreak 1280\}')
    out=out.replace(r'\{0,280,560\}',r'\{0,\allowbreak 280,\allowbreak 560\}')
    out=out.replace(r'\tabcolsep}{1.1pt}',r'\tabcolsep}{3pt}')
    out=out.replace(r'\tabcolsep}{2.1pt}',r'\tabcolsep}{4pt}')
    cjk=re.findall(r'[\u3400-\u9fff]',out)
    assert not cjk, 'Untranslated text: '+''.join(cjk[:100])
    # Constrain only tables wider than the page; preserve native size otherwise.
    out=out.replace(r'\begin{tabular}',r'\begin{adjustbox}{max width=\linewidth}'+'\n'+r'\begin{tabular}')
    out=out.replace(r'\end{tabular}',r'\end{tabular}'+'\n'+r'\end{adjustbox}')
    assert re.findall(r'\\includegraphics[^\n]+',src)==re.findall(r'\\includegraphics[^\n]+',out)
    for tag in ['label','cite','bibitem']:
        a=re.findall(r'\\'+tag+r'\{([^}]+)\}',src)
        b=re.findall(r'\\'+tag+r'\{([^}]+)\}',out)
        assert a==b,tag
    assert re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',src,re.S)==re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}',out,re.S)
    # All numeric table cells must remain identical, including bold formatting.
    rows_a=[s.split('&')[1:] for s in src.splitlines() if '&' in s and re.search(r'&\s*(?:\\textbf\{)?[+0-9$]',s)]
    rows_b=[s.split('&')[1:] for s in out.splitlines() if '&' in s and re.search(r'&\s*(?:\\textbf\{)?[+0-9$]',s)]
    nums=lambda rows:[[re.findall(r'[-+]?\d+(?:\.\d+)?',c) for c in row] for row in rows]
    assert nums(rows_a)==nums(rows_b),'Table numeric mismatch'
    (r/'paper/manuscript_en.tex').write_text(out,encoding='utf-8')
    print(f'Translated {len(changes)} source lines; {len(rows_a)} numeric table rows verified.')
    print('Figures, equation blocks, citation keys, reference entries and labels preserved; no CJK text remains.')
