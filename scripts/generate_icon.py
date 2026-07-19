#!/usr/bin/env python3
"""Generate a 512x512 app icon: bottle-shaped slime face on transparent background."""
from pathlib import Path
import math, sys
try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("Pillow required: pip install Pillow")
    sys.exit(1)

SIZE = 512
RENDER_SIZE = SIZE * 4  # supersample
OUT = Path(__file__).parent.parent / "assets" / "bubby_icon.png"

BODY_TOP = (80, 180, 220)
BODY_BOTTOM = (30, 100, 160)
BODY_OUTLINE = (15, 50, 90)
SHADOW = (0, 0, 0, 60)
EYE_WHITE = (255, 255, 255, 255)
PUPIL = (20, 30, 50, 255)
HIGHLIGHT = (255, 255, 255, 220)
BLUSH = (255, 140, 170, 100)
MOUTH = (25, 45, 60, 200)

def radial_grad(w, h, inner, outer, cy_ratio=0.35):
    img = Image.new("RGBA", (w, h), (0,0,0,0))
    cx, cy = w/2, h*cy_ratio
    max_d = math.sqrt(cx*cx + max(cy*cy, (h-cy)*(h-cy)))
    for y in range(h):
        for x in range(w):
            dx, dy = x-cx, y-cy
            t = min(1.0, math.sqrt(dx*dx+dy*dy)/max_d)
            t = t*t*(3-2*t)
            r = int(inner[0]+(outer[0]-inner[0])*t)
            g = int(inner[1]+(outer[1]-inner[1])*t)
            b = int(inner[2]+(outer[2]-inner[2])*t)
            img.putpixel((x,y), (r,g,b,255))
    return img

def main():
    s = RENDER_SIZE
    img = Image.new("RGBA", (s, s), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    # shadow
    sl = Image.new("RGBA", (s,s), (0,0,0,0))
    sd = ImageDraw.Draw(sl)
    sd.ellipse([int(s*.15), int(s*.78), int(s*.85), int(s*.93)], fill=SHADOW)
    img = Image.alpha_composite(img, sl.filter(ImageFilter.GaussianBlur(s*.03)))

    # body mask (bottle shape)
    mask = Image.new("L", (s,s), 0)
    md = ImageDraw.Draw(mask)
    bw, bh = int(s*.48), int(s*.60)
    bx0 = (s-bw)//2
    by0 = int(s*.22)
    bx1, by1 = bx0+bw, by0+bh
    cr = int(bw*.22)
    md.rounded_rectangle([bx0,by0,bx1,by1], radius=cr, fill=255)
    # bump
    bump_w, bump_h = int(bw*.52), int(s*.12)
    bump_x0 = (s-bump_w)//2
    bump_y0 = by0-bump_h+int(s*.02)
    bcr = int(bump_w*.35)
    md.rounded_rectangle([bump_x0,bump_y0,bump_x0+bump_w,bump_y0+bump_h+int(s*.06)], radius=bcr, fill=255)
    # bottom plump
    btw, bth = int(bw*1.06), int(s*.10)
    btx0 = (s-btw)//2
    bty0 = by1-bth+int(s*.01)
    md.ellipse([btx0,bty0,btx0+btw,bty0+bth], fill=255)

    # gradient
    grad = radial_grad(s, s, BODY_TOP+(255,), BODY_BOTTOM+(255,), cy_ratio=0.35)
    body = Image.new("RGBA", (s,s), (0,0,0,0))
    body.paste(grad, (0,0), mask)
    img = Image.alpha_composite(img, body)

    # outline
    from PIL import ImageOps
    blur = mask.filter(ImageFilter.GaussianBlur(s*.022))
    inv = ImageOps.invert(blur)
    out = Image.new("RGBA", (s,s), BODY_OUTLINE+(80,))
    out.putalpha(inv)
    img = Image.alpha_composite(img, out)

    # eyes
    eye_y, eye_sp, eye_r = int(s*.35), int(s*.12), int(s*.10)
    for cx in (s//2-eye_sp, s//2+eye_sp):
        draw.ellipse([cx-eye_r, eye_y-eye_r, cx+eye_r, eye_y+eye_r], fill=EYE_WHITE)
        pr = int(eye_r*.48)
        ox = int(s*.010)
        draw.ellipse([cx-pr+ox, eye_y-pr, cx+pr+ox, eye_y+pr], fill=PUPIL)
        hl = max(2, int(eye_r*.30))
        hox, hoy = int(s*.018), -int(s*.015)
        draw.ellipse([cx+hox-hl, eye_y+hoy-hl, cx+hox+hl, eye_y+hoy+hl], fill=HIGHLIGHT)

    # blush
    br, by, bsp = int(s*.05), int(s*.42), int(s*.20)
    for bx in (s//2-bsp, s//2+bsp):
        tmp = Image.new("RGBA", (s,s), (0,0,0,0))
        t = ImageDraw.Draw(tmp)
        t.ellipse([bx-br, by-br, bx+br, by+br], fill=BLUSH)
        img = Image.alpha_composite(img, tmp.filter(ImageFilter.GaussianBlur(s*.025)))

    # mouth (happy arc)
    mc, my = s//2, int(s*.43)
    mw, mh = int(s*.08), int(s*.03)
    draw.arc([mc-mw, my, mc+mw, my+mh], 0, 180, fill=MOUTH, width=max(2,int(s*.010)))

    # downsample
    out_img = img.resize((SIZE, SIZE), Image.LANCZOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(str(OUT))
    print(f"Icon saved: {OUT} ({SIZE}x{SIZE} RGBA)")

if __name__ == "__main__":
    main()