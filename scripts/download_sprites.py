#!/usr/bin/env python3
"""
Generate high-quality procedural sprite frames for the Bubby companion.

Draws a cute bottle-shaped slime/robot character using supersampled
anti-aliasing, radial gradients, drop shadows, and smooth curves.
All frames have transparent backgrounds (RGBA alpha channel).

Character: Blue, bottle-shaped/rounded-rectangle body with a small bump
on top. Large, perfectly round white eyes with dark pupils and
catch-light highlights.

Animations generated:
  - idle   : gentle breathing + occasional eye blink
  - walk   : horizontal stretch/squash with vertical bounce
  - wave   : body tilt side-to-side (happy wiggle)
  - talk   : mouth open/close with subtle body bounce

Usage:
    python3 scripts/download_sprites.py
"""

from pathlib import Path
import math
import shutil
import sys

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)


SPRITE_DIR = Path(__file__).parent.parent / "sprites"
SIZE = 128           # output frame size in pixels
SUPER_SCALE = 4      # render at 4x then downsample for anti-aliasing
RENDER_SIZE = SIZE * SUPER_SCALE

# ── Slime colour palette ──────────────────────────────────────────
BODY_COLOR_TOP = (80, 180, 220)       # light blue
BODY_COLOR_BOTTOM = (30, 100, 160)    # darker blue
BODY_OUTLINE = (15, 50, 90)           # dark outline tint
SHADOW_COLOR = (0, 0, 0, 60)          # drop shadow
EYE_WHITE = (255, 255, 255, 255)
PUPIL_COLOR = (20, 30, 50, 255)
HIGHLIGHT = (255, 255, 255, 220)      # eye catch-light
BLUSH_COLOR = (255, 140, 170, 90)
MOUTH_COLOR = (25, 45, 60, 200)


def _create_radial_gradient(width: int, height: int,
                            inner_color: tuple, outer_color: tuple,
                            center_y_ratio: float = 0.5) -> Image.Image:
    """Create a smooth radial gradient image (RGBA).
    
    center_y_ratio: vertical position of the gradient center (0=top, 1=bottom).
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    cx, cy = width / 2, height * center_y_ratio
    max_dist = math.sqrt(cx * cx + max(cy * cy, (height - cy) * (height - cy)))

    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy) / max_dist
            t = min(1.0, dist)
            t = t * t * (3 - 2 * t)  # smoothstep
            r = int(inner_color[0] + (outer_color[0] - inner_color[0]) * t)
            g = int(inner_color[1] + (outer_color[1] - inner_color[1]) * t)
            b = int(inner_color[2] + (outer_color[2] - inner_color[2]) * t)
            img.putpixel((x, y), (r, g, b, 255))
    return img


def _draw_body_mask(draw: ImageDraw.Draw, size: int, squash_x: float = 1.0,
                     squash_y: float = 1.0, offset_y: int = 0) -> None:
    """
    Draw a bottle-shaped body mask: tall rounded rectangle with a small
    bump on top. Wider at the bottom, slightly narrower in the middle,
    and a curved top.
    """
    # Body dimensions
    bw = int(size * 0.44 * squash_x)    # body width
    bh = int(size * 0.56 * squash_y)    # body height
    bx0 = (size - bw) // 2
    by0 = int(size * 0.28) + offset_y
    bx1 = bx0 + bw
    by1 = by0 + bh

    # Main body: rounded rectangle
    corner_r = int(bw * 0.22)  # rounded corners
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=corner_r, fill=255)

    # Bump on top: a smaller rounded rect protruding upward from center
    bump_w = int(bw * 0.52)
    bump_h = int(size * 0.12 * squash_y)
    bump_x0 = (size - bump_w) // 2
    bump_y0 = by0 - bump_h + int(size * 0.03)
    bump_x1 = bump_x0 + bump_w
    bump_y1 = bump_y0 + bump_h + int(size * 0.06)
    bump_corner = int(bump_w * 0.35)
    draw.rounded_rectangle([bump_x0, bump_y0, bump_x1, bump_y1],
                           radius=bump_corner, fill=255)

    # Slightly wider bottom (plumpness)
    bottom_w = int(bw * 1.06)
    bottom_h = int(size * 0.10 * squash_y)
    bottom_x0 = (size - bottom_w) // 2
    bottom_y0 = by1 - bottom_h + int(size * 0.01)
    draw.ellipse([bottom_x0, bottom_y0, bottom_x0 + bottom_w, bottom_y0 + bottom_h],
                 fill=255)


def _draw_slime_frame(size: int, phase: float = 0.0,
                      mouth_open: float = 0.0,
                      squash_x: float = 1.0, squash_y: float = 1.0,
                      offset_y: int = 0, eye_blink: float = 0.0) -> Image.Image:
    """
    Draw a single slime frame at RENDER_SIZE resolution.

    Args:
        size: Render canvas size (RENDER_SIZE).
        phase: Animation phase 0.0–1.0.
        mouth_open: 0.0 = closed, 1.0 = fully open.
        squash_x: Horizontal squash/stretch factor.
        squash_y: Vertical squash/stretch factor.
        offset_y: Vertical offset for bounce.
        eye_blink: 0.0 = fully open, 1.0 = fully closed.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # ── 1. Drop shadow ─────────────────────────────────────────────
    shadow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    sw = int(size * 0.40)
    sh = int(size * 0.15)
    sx0 = (size - sw) // 2
    sy0 = size - sh - int(size * 0.04) + offset_y
    shadow_draw.ellipse([sx0, sy0, sx0 + sw, sy0 + sh], fill=SHADOW_COLOR)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=size * 0.035))
    img = Image.alpha_composite(img, shadow_layer)

    # ── 2. Body mask ───────────────────────────────────────────────
    body_mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(body_mask)
    _draw_body_mask(mask_draw, size, squash_x, squash_y, offset_y)

    # ── 3. Body radial gradient (light from above) ─────────────────
    gradient = _create_radial_gradient(size, size,
                                       BODY_COLOR_TOP + (255,),
                                       BODY_COLOR_BOTTOM + (255,),
                                       center_y_ratio=0.35)
    body_grad = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    body_grad.paste(gradient, (0, 0), body_mask)
    img = Image.alpha_composite(img, body_grad)

    # ── 4. Body outline/edge darkening ─────────────────────────────
    outline_blur = body_mask.filter(ImageFilter.GaussianBlur(radius=size * 0.022))
    from PIL import ImageOps
    outline_inv = ImageOps.invert(outline_blur)
    outline_dark = Image.new("RGBA", (size, size), BODY_OUTLINE + (70,))
    outline_dark.putalpha(outline_inv)
    img = Image.alpha_composite(img, outline_dark)

    # ── 5. Eyes — large, perfectly round ──────────────────────────
    eye_draw = ImageDraw.Draw(img)
    eye_y = int(size * 0.37) + offset_y
    eye_spacing = int(size * 0.11)
    eye_radius = int(size * 0.09)  # large round eyes

    blink_scale = max(0.08, 1.0 - eye_blink * 0.92)
    eye_actual_ry = max(2, int(eye_radius * blink_scale))

    left_cx = size // 2 - eye_spacing
    right_cx = size // 2 + eye_spacing

    for cx in (left_cx, right_cx):
        # White
        eye_draw.ellipse(
            [cx - eye_radius, eye_y - eye_actual_ry,
             cx + eye_radius, eye_y + eye_actual_ry],
            fill=EYE_WHITE,
        )
        if eye_blink < 0.7:
            # Pupil
            pupil_r = int(eye_radius * 0.48)
            pupil_ox = int(size * 0.010)
            eye_draw.ellipse(
                [cx - pupil_r + pupil_ox, eye_y - pupil_r,
                 cx + pupil_r + pupil_ox, eye_y + pupil_r],
                fill=PUPIL_COLOR,
            )
            # Catch-light highlight
            hl_r = max(2, int(eye_radius * 0.30))
            hl_ox = int(size * 0.018)
            hl_oy = -int(size * 0.015)
            eye_draw.ellipse(
                [cx + hl_ox - hl_r, eye_y + hl_oy - hl_r,
                 cx + hl_ox + hl_r, eye_y + hl_oy + hl_r],
                fill=HIGHLIGHT,
            )

    # ── 6. Blush ───────────────────────────────────────────────────
    blush_r = int(size * 0.05)
    blush_y = int(size * 0.43) + offset_y
    blush_spacing = int(size * 0.18)
    for bx in (size // 2 - blush_spacing, size // 2 + blush_spacing):
        blush_tmp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        blush_d = ImageDraw.Draw(blush_tmp)
        blush_d.ellipse(
            [bx - blush_r, blush_y - blush_r,
             bx + blush_r, blush_y + blush_r],
            fill=BLUSH_COLOR,
        )
        blush_tmp = blush_tmp.filter(ImageFilter.GaussianBlur(radius=size * 0.025))
        img = Image.alpha_composite(img, blush_tmp)

    # ── 7. Mouth ───────────────────────────────────────────────────
    mouth_cx = size // 2
    mouth_y = int(size * 0.46) + offset_y

    if mouth_open < 0.05:
        # Happy arc (closed)
        mw = int(size * 0.08)
        mh = int(size * 0.03)
        eye_draw.arc(
            [mouth_cx - mw, mouth_y, mouth_cx + mw, mouth_y + mh],
            start=0, end=180,
            fill=MOUTH_COLOR,
            width=max(2, int(size * 0.010)),
        )
    else:
        mw = int(size * 0.06 + size * 0.03 * mouth_open)
        mh = int(size * 0.025 + size * 0.06 * mouth_open)
        eye_draw.ellipse(
            [mouth_cx - mw, mouth_y, mouth_cx + mw, mouth_y + mh],
            fill=MOUTH_COLOR,
        )

    # ── 8. Downsample with Lanczos ─────────────────────────────────
    return img.resize((SIZE, SIZE), Image.LANCZOS)


# ── Animation generators ───────────────────────────────────────────

def generate_idle(count: int = 8) -> list:
    frames = []
    for i in range(count):
        t = i / count
        breath = 1.0 + 0.022 * math.sin(t * 2 * math.pi)
        squash_x = 1.0 + 0.008 * math.sin(t * 2 * math.pi)
        squash_y = breath
        blink = 0.9 if i in (3, 7) else 0.0
        offset_y = int(RENDER_SIZE * 0.015 * math.sin(t * 2 * math.pi))
        frames.append(_draw_slime_frame(
            RENDER_SIZE, phase=t, mouth_open=0.0,
            squash_x=squash_x, squash_y=squash_y,
            offset_y=offset_y, eye_blink=blink,
        ))
    return frames


def generate_walk(count: int = 8) -> list:
    frames = []
    for i in range(count):
        t = i / count
        bounce = abs(math.sin(t * 2 * math.pi))
        offset_y = int(RENDER_SIZE * -0.05 * bounce)
        squash_x = 1.0 + 0.06 * bounce - 0.03
        squash_y = 1.0 - 0.04 * bounce + 0.02
        frames.append(_draw_slime_frame(
            RENDER_SIZE, phase=t, mouth_open=0.0,
            squash_x=squash_x, squash_y=squash_y,
            offset_y=offset_y,
        ))
    return frames


def generate_wave(count: int = 6) -> list:
    frames = []
    for i in range(count):
        t = i / count
        wiggle = math.sin(t * 2 * math.pi)
        squash_x = 1.0 + 0.05 * abs(wiggle)
        squash_y = 1.0 - 0.025 * abs(wiggle)
        bounce = abs(math.sin(t * 2 * math.pi))
        offset_y = int(RENDER_SIZE * 0.035 * bounce)
        mouth = 0.12 + 0.08 * bounce
        frames.append(_draw_slime_frame(
            RENDER_SIZE, phase=t, mouth_open=mouth,
            squash_x=squash_x, squash_y=squash_y,
            offset_y=offset_y,
        ))
    return frames


def generate_talk(count: int = 4) -> list:
    frames = []
    for i in range(count):
        t = i / count
        mouth = 0.4 + 0.5 * math.sin(t * 2 * math.pi)
        bounce = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 2)
        offset_y = int(RENDER_SIZE * 0.012 * bounce)
        squash_y = 1.0 + 0.015 * bounce
        squash_x = 1.0 - 0.008 * bounce
        frames.append(_draw_slime_frame(
            RENDER_SIZE, phase=t, mouth_open=mouth,
            squash_x=squash_x, squash_y=squash_y,
            offset_y=offset_y,
        ))
    return frames


ANIMATIONS = {
    "idle": generate_idle,
    "walk": generate_walk,
    "wave": generate_wave,
    "talk": generate_talk,
}


def main():
    if SPRITE_DIR.exists():
        shutil.rmtree(SPRITE_DIR)
        print(f"Deleted old sprites/ directory")
    SPRITE_DIR.mkdir(exist_ok=True)
    print(f"Generating high-quality sprite frames in {SPRITE_DIR}/ ...")
    print(f"  Render resolution: {RENDER_SIZE}x{RENDER_SIZE} (downsampled to {SIZE}x{SIZE})")
    print(f"  Character: bottle-shaped blue slime with large round eyes + drop shadow")

    for name, generator in ANIMATIONS.items():
        out_dir = SPRITE_DIR / name
        out_dir.mkdir(exist_ok=True)
        frames = generator()
        for idx, img in enumerate(frames):
            path = out_dir / f"{idx:02d}.png"
            img.save(str(path))
        print(f"  {name}: {len(frames)} frames saved to {out_dir}/")

    print(f"\nDone! {len(ANIMATIONS)} animations generated.")
    print(f"Sprites ready for use. Run: PYTHONPATH=. python3 src/app.py")


if __name__ == "__main__":
    main()