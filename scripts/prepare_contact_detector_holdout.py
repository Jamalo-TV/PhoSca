"""Create a source-disjoint, evaluator-ready held-out stock ownership test.

The independently authored renderer imports no PhoSca training or fixture code.
Do not use these image roots, artwork tiles, or seed IDs for training or selection.
"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
WIDTH, HEIGHT = 1352, 1008
SEEDS = tuple(range(1_449_600, 1_449_612))

def digest(data):
    return sha256(data).hexdigest()

def render_tex(width,height,rng,colors):
    picture=np.full((height,width,3),colors,np.float32)
    coarse=rng.normal(0,1,(11,13)).astype(np.float32)
    coarse=cv2.resize(coarse,(width,height),interpolation=cv2.INTER_CUBIC)
    picture += coarse[:,:,None]*rng.uniform(7,29)
    for _ in range(int(rng.integers(7,21))):
        a=tuple(map(int,rng.integers([0,0],[width,height])))
        b=tuple(map(int,rng.integers([0,0],[width,height])))
        c=tuple(np.clip(np.rint(colors+rng.uniform(-31,58)),0,255).tolist())
        cv2.line(picture,a,b,c,int(rng.integers(1,5)),cv2.LINE_AA)
    noise=rng.normal(0,2,picture.shape[:2]).astype(np.float32)
    return np.clip(picture+noise[:,:,None],0,255).astype(np.uint8)

def film_page(width,height,rows,columns,paper,seed):
    rng=np.random.default_rng(seed)
    y_top=int(rng.integers(35,62))
    y_inner=height-y_top-31
    x_inner=width-int(rng.integers(51,78))
    x0=(width-x_inner)//2
    y_height=y_inner//rows
    x_width=x_inner//columns
    art=np.full((height,width,3),paper,np.uint8)
    for r in range(rows):
        y1=y_top+r*y_height
        y2=y1+y_height-13
        for c in range(columns):
            x1=x0+x_width*c
            x2=x1+x_width-7
            strength=int(rng.integers(37,145))
            tones=rng.uniform(.76,1.16,3)*strength
            art[y1+12:y2-10,x1:x2]=render_tex(x2-x1,y2-y1-22,rng,tones)
        cv2.rectangle(art,(x0,y1),(x0+x_inner-3,y1+11),(31,28,29),-1)
        cv2.rectangle(art,(x0,y2-9),(x0+x_inner-3,y2),(28,30,28),-1)
        pitch=int(rng.integers(12,18))
        for x in range(x0+4,x0+x_inner-8,pitch):
            cv2.ellipse(art,(x,y1+6),(3,2),0,0,360,(218,226,224),-1,cv2.LINE_AA)
            cv2.ellipse(art,(x,y2-4),(3,2),0,0,360,(198,209,206),-1,cv2.LINE_AA)
    cv2.putText(art,f"TEST PAPER {seed%10000}",(x0,y_top-11),cv2.FONT_HERSHEY_SIMPLEX,.43,(53,69,60),1,cv2.LINE_AA)
    split=y_top+(rows//2)*y_height-7
    return art,split

def scene(seed,separate):
    rng=np.random.default_rng(seed)
    height=int(rng.integers(785,837))
    width=int(rng.integers(552,595))
    x=int(rng.integers(678,722))
    y=int(rng.integers(82,114))
    rows=int(rng.choice([6,7,8]))
    columns=int(rng.choice([2,3,4]))
    gray=int(rng.integers(172,228))
    base=np.array([gray-12,gray+2,gray+8])
    paper=np.clip(base+rng.uniform(5,34,3),0,254)
    page=render_tex(WIDTH,HEIGHT,rng,base)
    # A geometric strip with no photographic paper face is a conspicuous distractor.
    strip_width=int(rng.integers(295,412))
    strip_height=int(rng.integers(710,827))
    sx=x-strip_width-int(rng.integers(10,50))
    sy=min(y+int(rng.integers(15,75)),HEIGHT-strip_height-6)
    negative,_=film_page(strip_width,strip_height,rows+1,columns,
                         [71,80,74],seed+400_000)
    page[sy:sy+strip_height,sx:sx+strip_width]=negative
    cv2.rectangle(page,(sx,sy),(sx+strip_width-1,sy+strip_height-1),(85,100,89),1,cv2.LINE_AA)

    tile,cut=film_page(width,height,rows,columns,paper,seed+700_000)
    gap=int(rng.choice([3,4,6,8]))
    plate=np.zeros((HEIGHT,WIDTH),np.uint8)
    dx=int(rng.choice([-3,0,3]))
    if separate:
        cut_top=cut-gap//2
        cut_bottom=cut+(gap-gap//2)
        lower=tile[cut_bottom:].copy()
        tint=np.int16([2,-2,3])
        stock=np.max(np.abs(lower.astype(np.int16)-paper[None,None,:]),axis=2)<8
        lower[stock]=np.clip(lower[stock].astype(np.int16)+tint,0,255)
        overlays=[(x,y,tile[:cut_top]),(x+dx,y+cut_bottom,lower)]
    else:
        overlays=[(x,y,tile)]
    for n,(x1,y1,image) in enumerate(overlays,1):
        h,w=image.shape[:2]
        # Contact shadows remain outside their source ownership masks.
        shade=np.zeros((HEIGHT,WIDTH),np.uint8)
        cv2.rectangle(shade,(x1+2,y1+3),(x1+w+4,y1+h+5),45,-1)
        shade=cv2.GaussianBlur(shade,(11,11),1.6)
        page=np.rint(page.astype(np.float32)*(1-shade[:,:,None]/255)).astype(np.uint8)
        page[y1:y1+h,x1:x1+w]=image
        plate[y1:y1+h,x1:x1+w]=n

    q=int((seed-SEEDS[0])%4)
    source=np.float32([[0,0],[WIDTH-1,0],[WIDTH-1,HEIGHT-1],[0,HEIGHT-1]])
    target=source+np.float32(rng.uniform(-26,26,(4,2)))
    plane=cv2.getPerspectiveTransform(source,target)
    page=cv2.warpPerspective(page,plane,(WIDTH,HEIGHT),borderMode=cv2.BORDER_REFLECT)
    plate=cv2.warpPerspective(plate,plane,(WIDTH,HEIGHT),flags=cv2.INTER_NEAREST)
    factor=float(rng.choice([.43,.53,.65,.76]))
    dims=(round(WIDTH*factor),round(HEIGHT*factor))
    page=cv2.resize(page,dims,interpolation=cv2.INTER_AREA)
    plate=cv2.resize(plate,dims,interpolation=cv2.INTER_NEAREST)
    page=np.ascontiguousarray(np.rot90(page,q))
    plate=np.ascontiguousarray(np.rot90(plate,q))
    quality=int(rng.choice([55,65,75,88]))
    return page,plate,dict(physical_state="two_separate_stock_masks" if separate else "one_contiguous_stock_mask",
                          source_seed=seed,visual_donor_seeds=[seed+400_000,seed+700_000],
                          rectified_sheet_pixels=[width,height],nominal_gap_pre_scale_px=gap if separate else 0,
                          projective_camera=plane.tolist(),rotation_deg=q*90,jpeg_quality=quality)

def labels_for(plate,expected):
    h,w=plate.shape
    labels=[]
    for i in range(1,expected+1):
        mask=(plate==i).astype(np.uint8)
        assert mask.sum()>100
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours)==1
        polygon=cv2.approxPolyDP(contours[0],.55,True).reshape(-1,2) / [w-1,h-1]
        labels.append("0 "+" ".join(f"{x:.9f} {y:.9f}" for x,y in polygon))
    return ("\n".join(labels)+"\n").encode()

def run(root):
    (root/"images").mkdir(parents=True,exist_ok=False)
    (root/"labels").mkdir()
    (root/"masks").mkdir()
    pages=[]
    for seed in SEEDS:
        for separate in (False,True):
            name=f"{seed}-{'separate' if separate else 'joined'}"
            art,plate,provenance=scene(seed,separate)
            expected=2 if separate else 1
            ok,image=cv2.imencode(".jpg",art,[cv2.IMWRITE_JPEG_QUALITY,provenance["jpeg_quality"]])
            assert ok
            ok,mask=cv2.imencode(".png",plate)
            assert ok
            items={"image":(f"images/{name}.jpg",image.tobytes()),
                   "annotations":(f"labels/{name}.txt",labels_for(plate,expected)),
                   "exact_ownership_mask":(f"masks/{name}.png",mask.tobytes())}
            row=dict(id=name,source_group=f"independent-arcs-stock-{seed}",expected_count=expected,
                     annotation_provenance="generated_source_ownership",data_scope="heldout_synthetic_only",
                     difficulty_tags=["multipanel_print" if not separate else "adjacent_independent_prints",
                                      "non_print_negative_distractor","low_resolution_printed_stock"],
                     source_raster_pixels=list(art.shape[1::-1]),**provenance)
            for role,(rel,data) in items.items():
                (root/rel).write_bytes(data)
                row[role]=rel
                row[role+"_sha256"]=digest(data)
            pages.append(row)
    meta=dict(role="development",training_excluded=True,selection_excluded=True,album_independent=False,
              generator_sha256=digest(Path(__file__).read_bytes()),opencv=cv2.__version__,numpy=np.__version__)
    (root/"heldout.json").write_text(json.dumps(dict(split_policy=meta,pages=pages),indent=2)+"\n")

if __name__=="__main__":
    p=ArgumentParser()
    p.add_argument("root",type=Path)
    run(p.parse_args().root)
