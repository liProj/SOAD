from pathlib import Path
import re,fitz
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parents[1]
doc=fitz.open(r/'paper/manuscript_en.pdf')
texts=[p.get_text() for p in doc]
cjk=[(i+1,re.findall(r'[\u3400-\u9fff]',s)) for i,s in enumerate(texts) if re.search(r'[\u3400-\u9fff]',s)]
print('Pages:',len(doc),'CJK in PDF:',cjk)
assert not cjk
for needle in ['Overall framework of','Comparison of SOAD with','Paired performance increments','Effect of calibration evidence','Paired differences between']:
    hits=[i+1 for i,s in enumerate(texts) if needle in s]
    print(needle,hits)
    for n in hits:doc[n-1].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(r/f'paper/_en_check_{n:02}.png')
doc[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7)).save(r/'paper/_en_check_01.png')
sheet=Image.new('RGB',(1200,470*5),'#dddddd')
d=ImageDraw.Draw(sheet)
for i,p in enumerate(doc):
    pix=p.get_pixmap(matrix=fitz.Matrix(.48,.48))
    im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
    im.thumbnail((290,435))
    x=(i%4)*300+5;y=(i//4)*470+22
    sheet.paste(im,(x,y));d.text((x,y-18),f'Page {i+1}',fill='black')
sheet.save(r/'paper/_en_contact.png')
print('Rendered layout previews.')
