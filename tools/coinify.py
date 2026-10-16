import numpy as np
from PIL import Image, ImageDraw, ImageFilter
LOGO='/cursor/stores/store-30184ac7-dc26-4aeb-931e-9409e9d44de3/grok-bot/agents/bf1440c9-ec0a-474a-b259-2a3dd3c88b4e/attachments/01M52YDV78V0N0VG3SK21JX3WH.png'
S=4  # orig->display factor: coords given in 1024-wide display units
lg=Image.open(LOGO).convert('L')
# coin face in logo: center ~ (250,248) r ~ 243 of 500
lg=lg.crop((250-238,248-238,250+238,248+238)).resize((512,512),Image.LANCZOS)
G=np.asarray(lg,np.float32)/255.
G=(G-G.mean())/ (G.std()+1e-6)  # relief signal

def place(im, cx, cy, rx, ry, ang=0, strength=.55, inner=.9):
    cx,cy,rx,ry=[v*1.25 for v in (cx,cy,rx,ry)]
    rx*=inner; ry*=inner
    W=int(2*max(rx,ry)*1.5)+4
    # relief texture scaled to ellipse, rotated
    t=Image.fromarray(((G*0.25+0.5).clip(0,1)*255).astype(np.uint8)).resize((max(2,int(2*rx)),max(2,int(2*ry))),Image.LANCZOS)
    m=Image.new('L',t.size,0); ImageDraw.Draw(m).ellipse((0,0,t.size[0]-1,t.size[1]-1),fill=255)
    m=m.filter(ImageFilter.GaussianBlur(max(1,min(rx,ry)*.06)))
    t=t.rotate(-ang,Image.BICUBIC,expand=True,fillcolor=128); m=m.rotate(-ang,Image.BICUBIC,expand=True)
    x0=int(cx-t.size[0]/2); y0=int(cy-t.size[1]/2)
    box=(x0,y0,x0+t.size[0],y0+t.size[1])
    reg=np.asarray(im.crop(box),np.float32)
    rel=(np.asarray(t,np.float32)/255.-.5)/.25           # z-score relief
    a=np.asarray(m,np.float32)[...,None]/255.
    # local lighting: heavily blurred original region
    light=np.asarray(im.crop(box).filter(ImageFilter.GaussianBlur(max(2,min(rx,ry)*.35))),np.float32)
    lum=light.mean(-1,keepdims=True)
    out=light*(1+strength*0.35*rel[...,None])
    # keep a bit of original fine detail (specular grain)
    out=out*0.85+reg*0.15
    res=reg*(1-a)+out.clip(0,255)*a
    im.paste(Image.fromarray(res.astype(np.uint8)),box[:2])

P={
 4:[(622,355,178,190,-4,.7)],
 9:[(240,77,33,33,0,.6)],
 7:[(355,380,95,58,-3,.55),(805,292,100,70,12,.6),(480,288,103,82,-14,.6),(610,363,116,94,-4,.65)],
 1:[(178,410,28,33,0,.6),(170,482,29,33,0,.6),(447,290,34,38,0,.6),(415,370,31,36,0,.6),(400,455,29,34,0,.6)],
 6:[(432,180,50,62,28,.75),(378,430,17,17,0,.5),(328,478,15,15,0,.5),(512,300,13,12,0,.5),(470,375,13,12,0,.5)],
 5:[(150,55,48,44,-20,.55),(813,93,24,21,0,.5),(975,200,42,40,-15,.55),(52,237,24,28,-30,.5),(610,140,14,14,0,.5)],
 10:[(472,227,52,42,0,.55),(335,402,26,17,0,.5),(373,417,26,17,0,.5),(427,428,27,18,0,.5),(437,374,24,15,0,.5),(502,95,22,14,0,.5),(557,96,22,14,0,.5),(95,527,28,12,0,.5),(182,515,26,12,0,.5),(250,533,26,12,0,.5),(355,540,27,12,0,.5),(430,552,27,12,0,.5),(560,555,27,12,0,.5)],
 3:[(440,410,22,12,0,.5),(486,432,20,11,0,.5),(360,437,22,12,0,.5),(560,455,20,11,0,.5)],
 2:[(450,337,16,24,0,.5),(453,440,16,25,0,.5)],
}
import sys
for i,pl in P.items():
    im=Image.open(f'/workspace/t1/orig/g{i}.jpg').convert('RGB')
    for (cx,cy,rx,ry,ang,s) in pl: place(im,cx,cy,rx,ry,ang,s)
    im.save(f'/workspace/t1/img/g{i}.jpg',quality=86,optimize=True,progressive=True)
print('done')
