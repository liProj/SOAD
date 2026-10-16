"""Vector reconstruction of the user-selected exec-5c6bb5a2 illustration."""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
SOURCE=OUT/'fig_pipeline_selected_reference.png'
# Source artwork is stored alongside the vector figure.
im=Image.open(SOURCE).convert('RGB')
W,H=1717,916
matplotlib.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','image.composite_image':False})
fig=plt.figure(figsize=(17.17,9.16),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
INK='#101d50'; EDGE='#304a83'; ORANGE='#f2692a'
def text(x,y,s,fs=14,bold=False,ha='center',maxw=None,color=INK,italic=False):
    a=ax.text(x,y,s,fontsize=fs,ha=ha,va='center',weight='bold' if bold else 'normal',color=color,
              style='italic' if italic else 'normal',linespacing=1.1,zorder=9)
    if maxw:
        width=a.get_window_extent(fig.canvas.get_renderer()).width
        if width>maxw: a.set_fontsize(fs*maxw/width)
    return a
def box(x,y,w,h,fill='white',edge=EDGE,r=5,dash=False,lw=1,z=2):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fill,
       edgecolor=edge,lw=lw,linestyle=(0,(5,3)) if dash else '-',zorder=z)
    ax.add_patch(p)
def panel(x,y,w,h,title,fill,edge):
    box(x,y,w,h,fill,edge,11,True,1,0)
    text(x+17,y+29,title,20,True,'left',w-34)
def line(points,color=INK,lw=1.1,dash=False):
    x,y=zip(*points);ax.plot(x,y,color=color,lw=lw,ls=(0,(5,3)) if dash else '-',zorder=5)
def arrow(points,color=INK,lw=1.1,dash=False):
    if len(points)>2:line(points[:-1],color,lw,dash)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=12,
       color=color,lw=lw,linestyle=(0,(5,3)) if dash else '-',shrinkA=0,shrinkB=0,zorder=6))
def photo(crop,x,y,w,h):
    a=im.crop(crop)
    ax.imshow(a,extent=(x,x+w,y+h,y),aspect='auto',interpolation='lanczos',zorder=3)
    ax.add_patch(Rectangle((x,y),w,h,fill=False,ec='#92a9b6',lw=.7,zorder=4))
def slab(x,y,w,h,d=22,front='#6dafd5',side='#4585b2',top='#c0ddf0'):
    for pts,col in [([(x,y),(x+d,y-d),(x+w+d,y-d),(x+w,y)],top),
                    ([(x+w,y),(x+w+d,y-d),(x+w+d,y+h-d),(x+w,y+h)],side),
                    ([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],front)]:
        ax.add_patch(Polygon(pts,closed=True,fc=col,ec='#46749d',lw=.9,zorder=5))
def stack(x,y,w=38,n=7):
    for j in reversed(range(n)):slab(x,y+j*4,w,7,14)
def node(x,y,w,h,label,fs=12,fill='#f7fcff'):
    box(x,y,w,h,fill);text(x+w/2,y+h/2,label,fs,maxw=w-10)
def tree(x,y):
    for x1,y1,x2,y2 in [(25,0,10,19),(25,0,40,19),(10,19,0,39),(10,19,19,39),(40,19,33,39),(40,19,52,39)]:
        line([(x+x1,y+y1),(x+x1,y+y1+8),(x+x2,y+y1+8),(x+x2,y+y2)],'#20a15c',1.2)
    for xx,yy in [(25,0),(10,19),(40,19),(0,39),(19,39),(33,39),(52,39)]:
        box(x+xx-5,y+yy-5,10,11,'#52d78a','#229b58',2,lw=.6,z=6)
def detections(x,y,w,h,color='#a5ee35'):
    for xx,yy,ww,hh in [(.22,.22,.13,.12),(.41,.40,.14,.13),(.64,.31,.12,.13),(.28,.64,.13,.12),(.65,.70,.14,.12),(.52,.56,.1,.1)]:
        ax.add_patch(Rectangle((x+xx*w,y+yy*h),ww*w,hh*h,fill=False,ec=color,lw=1.1,zorder=7))
def brace(x,y,w):
    verts=[(x,y),(x,y+8),(x+5,y+8),(x+w/2-6,y+8),(x+w/2,y+15),(x+w/2+6,y+8),(x+w-5,y+8),(x+w,y+8),(x+w,y)]
    codes=[MPath.MOVETO,MPath.CURVE3,MPath.CURVE3,MPath.LINETO,MPath.LINETO,MPath.LINETO,MPath.LINETO,MPath.CURVE3,MPath.CURVE3]
    ax.add_patch(PathPatch(MPath(verts,codes),fill=False,ec='#1477b0',lw=1,zorder=6))

panel(12,25,435,428,'(A) Native-pixel observations','#f6edff','#b068f0')
panel(457,25,643,428,'(B) Shared RGB detector','#eaf8ff','#38a7ee')
panel(1110,25,595,428,'(C) Instance aggregation','#effcf4','#19bb78')
panel(12,492,1118,347,'(D) Evidence calibration (EC)','#fff4e6','#f58a47')
panel(1140,492,565,347,'(E) Aligned geometry control','#f5f4fa','#9990c1')

# A: preserve the selected photo placement and overlapping thumbnail layout.
text(220,87,'RGB frame / 1920 × 1200',15)
photo((37,103,345,262),36,103,310,160)
for x in np.linspace(37,345,5):line([(x,104),(x,262)],'white',1,True)
for y in np.linspace(104,262,4):line([(37,y),(345,y)],'white',1,True)
text(394,169,'4 × 3\noverlapping\nwindows',13,maxw=86)
arrow([(80,273),(80,306)])
text(261,295,'12 overlapping tiles / 640 × 640',14,maxw=329)
tile=(489,111,590,206)
for x,y in [(38,313),(84,321),(132,327),(179,316),(230,329),(342,320),(356,332)]:
    photo(tile,x,y,74,71)
text(324,363,'…',24)
node(28,410,402,36,'Area-based labels; fruit-free tile sampling',14,'#fbf6ff')
arrow([(431,190),(477,190)])

# B: backbone, multi-scale feature maps, encoder and decoder.
text(540,94,'Original',15,True)
photo(tile,487,110,104,99)
text(575,240,'Shared\nweights',12.5,True,maxw=104)
text(563,278,'(same encoder–decoder)',10.5,maxw=167)
arrow([(503,218),(503,263)])
arrow([(503,263),(503,218)])
text(547,295,'Horizontal flip',14,True)
photo((492,310,598,405),490,309,108,96)
line([(597,153),(621,153),(621,207),(638,207)])
line([(601,355),(638,355),(638,299)])
line([(638,207),(638,299)])
arrow([(638,250),(655,250)])
text(750,150,'RT-DETR-L backbone',16,True,maxw=210)
for x,y,w,h,d in [(659,203,24,125,34),(695,218,23,96,28),(729,237,22,62,25),(783,211,24,99,24)]:slab(x,y,w,h,d)
text(914,144,'Multi-scale\nfeatures',14)
for yy in [185,240,302]:stack(868,yy,34)
for yy in [199,322]:arrow([(831,249),(842,249),(842,yy),(865,yy)])
arrow([(842,249),(865,249)])
text(933,193,'P3',12);text(941,323,'P5',12)
arrow([(916,249),(936,249)])
node(937,211,59,78,'Hybrid\nencoder',12,'#f2edfc')
arrow([(996,249),(1013,249)])
box(999,187,96,116,'#f4effc','#8185b0')
text(1047,212,'Decoder +\ndetection heads',11.5,maxw=89)
stack(1018,247,40,9)
arrow([(1056,187),(1056,139),(1125,139)])
arrow([(1056,303),(1056,347),(1125,347)])

# C: paired per-view processing and mirror fusion.
for yy,title in [(79,'Original view'),(253,'Mirrored view')]:
    box(1120,yy,421,137,'#f6fff9','#4daa8e',6,True,.9,1)
    text(1150,yy+20,title,14,True,'left')
    photo(tile,1127,yy+43,68,73)
    for xx,lab in [(1202,'Internal-\nedge\nfiltering'),(1276,'Full-frame\ncoordinates'),(1363,'Per-view\nNMS')]:
        node(xx,yy+47,68 if xx!=1276 else 80,70,lab,12)
    for x1,x2 in [(1195,1202),(1270,1276),(1356,1363),(1431,1448)]:arrow([(x1,yy+82),(x2,yy+82)])
    text(1482,yy+23,'Candidates\n(original)' if yy==79 else 'Candidates\n(mirrored)',11,maxw=80)
    photo(tile,1449,yy+47,73,74);detections(1449,yy+47,73,74)
node(1524,208,74,70,'Mirror\nfusion\n(MF)',14,'#f2edfc')
arrow([(1522,159),(1561,159),(1561,208)])
arrow([(1522,335),(1561,335),(1561,278)])
arrow([(1598,245),(1618,245)])
text(1655,190,'Aggregated\ncandidates',13,maxw=97)
photo(tile,1619,212,72,73);detections(1619,212,72,73)
arrow([(1652,285),(1652,474),(449,474),(449,491)])

# D: original photo crops, feature-vector brackets, two tree estimators.
text(140,568,'Candidate from detector\n(in full-frame coordinates)',14,maxw=245)
photo((31,598,160,726),29,597,132,130)
ax.add_patch(Rectangle((75,644),29,30,fill=False,ec=ORANGE,lw=1.4,zorder=7))
photo((181,615,269,710),180,612,91,100)
line([(75,644),(179,612)],ORANGE,1.4);line([(104,674),(179,712)],ORANGE,1.4)
arrow([(276,652),(303,652)])
text(429,605,'39 candidate features',15,True)
box(306,628,248,43,'#fbfeff','#60a0b8',3)
for i in range(25):
    if 14<=i<=17:continue
    col='#9dd0fa' if i<7 else '#ffbb94' if i<14 else '#a5dbac'
    ax.add_patch(Rectangle((311+i*9.5,633),7,33,fc=col,ec='#4e97bc' if i<7 else '#da7e53' if i<14 else '#54a679',lw=.65,zorder=6))
text(463,650,'…',17)
for x,w,lab in [(310,67,'Score +\ncontext'),(383,91,'Box +\nappearance'),(483,68,'Depth +\ncurvature')]:
    brace(x,678,w);text(x+w/2,714,lab,10.5,maxw=w)
box(589,510,381,242,'#fffaf2','#777d85',7,True,1,1)
text(779,527,'Detector weights frozen during calibration',13,italic=True,maxw=358)
for yy,title,loss in [(543,'Correctness\nestimator','BCE'),(661,'Localization\nquality','Squared loss')]:
    box(625,yy,190,68,'#f0fcf6')
    tree(644,yy+14)
    text(759,yy+25,title,12.5,True,maxw=103)
    text(759,yy+55,loss,12,maxw=104)
arrow([(558,650),(574,650),(574,575),(624,575)])
arrow([(574,650),(574,702),(624,702)])
box(857,612,105,93,'#fbfcff')
text(910,630,'Score fusion',13,True,maxw=94)
text(910,662,r'$s=p^{(1-\alpha)}u^\alpha$',16,maxw=96)
text(910,690,r'$\alpha=0.5$',15)
arrow([(815,575),(837,575),(837,623),(856,623)])
arrow([(815,702),(838,702),(838,667),(856,667)])
arrow([(963,651),(991,651)])
text(1055,574,'Full-frame detections',13,maxw=139)
photo(tile,994,588,122,132);detections(994,588,122,132)
box(587,768,134,64,'#f7fbff','#6994aa',4)
text(654,782,'Validation labels',12.5,maxw=125)
photo((594,796,715,828),594,796,120,30)
text(882,783,'Fit after detector training',13,maxw=247)
# Four terminating arrows make both supervised losses explicit, as in the selected figure.
arrow([(667,768),(667,733)],ORANGE,1.4,True)
arrow([(722,788),(758,788),(758,733)],ORANGE,1.4,True)
arrow([(667,659),(667,613)],ORANGE,1.4,True)
arrow([(758,659),(758,613)],ORANGE,1.4,True)

# E: preserve RGB, depth, B and N image grid from the selected source.
text(1691,549,'Separate comparison model',11,ha='right',italic=True,maxw=185)
text(1223,570,'RGB (colour)',13)
text(1348,564,'Depth\n(0.5–6 m)',13)
photo((1163,586,1273,680),1163,585,111,97)
photo((1293,586,1399,677),1292,585,109,94)
text(1219,706,'Convexity map B',12.5,maxw=123)
text(1353,708,'Bumpiness map N',12.5,maxw=133)
photo((1162,722,1268,809),1162,722,108,88)
photo((1295,726,1404,809),1294,725,110,85)
node(1445,650,113,71,'Synchronized\ntransforms',13)
arrow([(1405,616),(1423,616),(1423,666),(1445,666)])
arrow([(1274,637),(1283,637),(1283,689),(1445,689)])
arrow([(1349,680),(1349,695)])
arrow([(1270,754),(1283,754),(1283,691),(1349,691)])
arrow([(1404,761),(1428,761),(1428,712),(1445,712)])
arrow([(1558,686),(1577,686)])
text(1632,626,'GeoSphere-DETR\ncontrol',14,maxw=147)
for x,y,w,h in [(1578,669,14,65),(1600,675,14,57),(1624,702,13,30),(1651,680,15,51)]:
    slab(x,y,w,h,18,'#b198d5','#876cb0','#dfcdf3')
text(1593,814,'Same tiles and output processing',11.5,italic=True,maxw=213)

arrow([(392,874),(489,874)],lw=1.4);text(569,874,'Inference flow',15)
arrow([(698,874),(793,874)],ORANGE,1.4,True);text(935,874,'Calibration supervision',15)
line([(1080,855),(1080,889)])
text(1481,874,'Photo insets illustrate the workflow; feature stacks are schematic.',10.5,maxw=435)

for ext in ['pdf','png','svg']:
    old=OUT/f'fig_pipeline.{ext}'
    backup=OUT/f'fig_pipeline_third_layout.{ext}'
    if old.exists() and not backup.exists():shutil.copy2(old,backup)
fig.savefig(OUT/'fig_pipeline.pdf',metadata={'Title':'SOAD — vector reconstruction of selected layout'})
fig.savefig(OUT/'fig_pipeline.svg')
fig.savefig(OUT/'fig_pipeline.png',dpi=160)
plt.close(fig)
p=fitz.open(OUT/'fig_pipeline.pdf')[0]
print('Vector paths:',len(p.get_drawings()),'Text characters:',len(p.get_text()),'Photo objects:',len(p.get_images()))
assert len(p.get_drawings())>250 and len(p.get_text())>1200
