#!/usr/bin/env python3
"""
Generate high-quality procedural sprite frames for the Bubby companion.

Draws a cute slime/blob character using supersampled anti-aliasing,
radial gradients, drop shadows, and smooth curves. All frames have
transparent backgrounds (RGBA alpha channel).

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
FPS = 12

# ── Slime colour palette ──────────────────────────────────────────
BODY_COLOR_TOP = (80, 200, 220)       # light teal
BODY_COLOR_BOTTOM = (40, 140, 160)    # darker teal
BODY_OUTLINE = (20, 80, 100)          # dark outline tint
SHADOW_COLOR = (0, 0, 0, 80)          # drop shadow
EYE_WHITE = (255, 255, 255, 255)
PUPIL_COLOR = (30, 50, 60, 255)
HIGHLIGHT = (255, 255, 255, 200)      # eye catch-light
BLUSH_COLOR = (255, 140, 160, 100)
MOUTH_COLOR = (30, 50, 60, 200)


def _create_radial_gradient(width: int, height: int,
                            inner_color: tuple, outer_color: tuple) -> Image.Image:
    """Create a smooth radial gradient image (RGBA)."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    cx, cy = width / 2, height / 2
    max_dist = math.sqrt(cx * cx + cy * cy)

    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy) / max_dist
            t = min(1.0, dist)
            # Smoothstep for nicer falloff
            t = t * t * (3 - 2 * t)
            r = int(inner_color[0] + (outer_color[0] - inner_color[0]) * t)
            g = int(inner_color[1] + (outer_color[1] - inner_color[1]) * t)
            b = int(inner_color[2] + (outer_color[2] - inner_color[2]) * t)
            a = 255
            img.putpixel((x, y), (r, g, b, a))
    return img


def _draw_body_mask(draw: ImageDraw.Draw, size: int, squash_x: float = 1.0,
                     squash_y: float = 1.0, offset_y: int = 0) -> None:
    """
    Draw the slime body shape as a filled white ellipse on a mask layer.
    The body is a rounded teardrop/blob — wider at the bottom, narrower at top.
    """
    w = int(size * 0.68 * squash_x)
    h = int(size * 0.72 * squash_y)
    x0 = (size - w) // 2
    y0 = (size - h) // 2 - int(size * 0.04) + offset_y
    x1 = x0 + w
    y1 = y0 + h

    # Main body ellipse
    draw.ellipse([x0, y0, x1, y1], fill=255)

    # Extra plumpness at the bottom — a wider ellipse overlapping lower half
    bw = int(size * 0.62 * squash_x)
    bh = int(size * 0.35 * squash_y)
    bx0 = (size - bw) // 2
    by0 = y1 - bh + int(size * 0.02)
    draw.ellipse([bx0, by0, bx0 + bw, by0 + bh], fill=255)

    # Slight point at the top (small ellipse centred near the top)
    tw = int(size * 0.30 * squash_x)
    th = int(size * 0.24 * squash_y)
    tx0 = (size - tw) // 2
    ty0 = y0 - th + int(size * 0.06)
    draw.ellipse([tx0, ty0, tx0 + tw, ty0 + th], fill=255)


def _draw_slime_frame(size: int, phase: float = 0.0,
                      mouth_open: float = 0.0,
                      squash_x: float = 1.0, squash_y: float = 1.0,
                      offset_y: int = 0, eye_blink: float = 0.0) -> Image.Image:
    """
    Draw a single slime frame at RENDER_SIZE resolution.

    Args:
        size: Render canvas size (RENDER_SIZE).
        phase: Animation phase 0.0–1.0 (used for subtle details).
        mouth_open: 0.0 = closed, 1.0 = fully open.
        squash_x: Horizontal squash/stretch factor.
        squash_y: Vertical squash/stretch factor.
        offset_y: Vertical offset for bounce.
        eye_blink: 0.0 = fully open, 1.0 = fully closed (blink).
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # ── 1. Drop shadow ─────────────────────────────────────────────
    shadow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    sw = int(size * 0.55)
    sh = int(size * 0.18)
    sx0 = (size - sw) // 2
    sy0 = size - sh - int(size * 0.04) + offset_y
    shadow_draw.ellipse([sx0, sy0, sx0 + sw, sy0 + sh],
                        fill=SHADOW_COLOR)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=size * 0.04))
    img = Image.alpha_composite(img, shadow_layer)

    # ── 2. Body mask ───────────────────────────────────────────────
    body_mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(body_mask)
    _draw_body_mask(mask_draw, size, squash_x, squash_y, offset_y)

    # ── 3. Body radial gradient ────────────────────────────────────
    gradient = _create_radial_gradient(size, size,
                                       BODY_COLOR_TOP + (255,),
                                       BODY_COLOR_BOTTOM + (255,))
    # Shift gradient centre up slightly for lighting from above
    grad_shift = int(size * 0.10)
    gradient_shifted = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gradient_shifted.paste(gradient, (0, -grad_shift))
    gradient_shifted.paste(gradient.crop((0, 0, size, grad_shift)),
                           (0, size - grad_shift))

    # Composite gradient through body mask
    body_grad = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    body_grad.paste(gradient_shifted, (0, 0), body_mask)
    img = Image.alpha_composite(img, body_grad)

    # ── 4. Body outline (subtle inner shadow) ──────────────────────
    # Blur the mask and use it to darken edges
    outline_blur = body_mask.filter(ImageFilter.GaussianBlur(radius=size * 0.025))
    outline_layer = Image.new("RGBA", (size, size), BODY_OUTLINE + (60,))
    outline_layer.putalpha(outline_blur)
    # Invert: we want the body area to stay bright, edges to darken
    from PIL import ImageOps
    outline_inv = ImageOps.invert(outline_blur)
    outline_layer_final = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    outline_dark = Image.new("RGBA", (size, size), BODY_OUTLINE + (80,))
    outline_dark.putalpha(outline_inv)
    img = Image.alpha_composite(img, outline_dark)

    # ── 5. Eyes ────────────────────────────────────────────────────
    eye_draw = ImageDraw.Draw(img)
    eye_y = int(size * 0.38) + offset_y
    eye_spacing = int(size * 0.13)
    eye_radius_x = int(size * 0.10)
    eye_radius_y = int(size * 0.12)

    blink_scale = 1.0 - eye_blink * 0.92  # eyes nearly close
    eye_ry = max(2, int(eye_radius_y * blink_scale))

    left_eye_cx = size // 2 - eye_spacing
    right_eye_cx = size // 2 + eye_spacing

    for cx in (left_eye_cx, right_eye_cx):
        # White of eye
        eye_draw.ellipse(
            [cx - eye_radius_x, eye_y - eye_ry,
             cx + eye_radius_x, eye_y + eye_ry],
            fill=EYE_WHITE,
        )
        if eye_blink < 0.7:  # only draw pupils when not mostly closed
            # Pupil
            pupil_rx = int(eye_radius_x * 0.45)
            pupil_ry = int(eye_ry * 0.50)
            pupil_off_x = int(size * 0.012)
            eye_draw.ellipse(
                [cx - pupil_rx + pupil_off_x, eye_y - pupil_ry,
                 cx + pupil_rx + pupil_off_x, eye_y + pupil_ry],
                fill=PUPIL_COLOR,
            )
            # Catch-light highlight
            hl_r = max(2, int(eye_radius_x * 0.32))
            hl_off_x = int(size * 0.022)
            hl_off_y = -int(size * 0.018)
            eye_draw.ellipse(
                [cx + hl_off_x - hl_r, eye_y + hl_off_y - hl_r,
                 cx + hl_off_x + hl_r, eye_y + hl_off_y + hl_r],
                fill=HIGHLIGHT,
            )

    # ── 6. Blush spots ─────────────────────────────────────────────
    blush_rx = int(size * 0.06)
    blush_ry = int(size * 0.04)
    blush_y = int(size * 0.44) + offset_y
    blush_spacing = int(size * 0.22)
    # Use a blurred ellipse for soft blush
    for bx in (size // 2 - blush_spacing, size // 2 + blush_spacing):
        blush_tmp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        blush_tmp_draw = ImageDraw.Draw(blush_tmp)
        blush_tmp_draw.ellipse(
            [bx - blush_rx, blush_y - blush_ry,
             bx + blush_rx, blush_y + blush_ry],
            fill=BLUSH_COLOR,
        )
        blush_tmp = blush_tmp.filter(ImageFilter.GaussianBlur(radius=size * 0.025))
        img = Image.alpha_composite(img, blush_tmp)

    # ── 7. Mouth ───────────────────────────────────────────────────
    mouth_cx = size // 2
    mouth_y = int(size * 0.48) + offset_y

    if mouth_open < 0.05:
        # Closed: a thin, happy arc
        mw = int(size * 0.12)
        mh = int(size * 0.04)
        eye_draw.arc(
            [mouth_cx - mw, mouth_y, mouth_cx + mw, mouth_y + mh],
            start=0, end=180,
            fill=MOUTH_COLOR,
            width=max(2, int(size * 0.012)),
        )
    else:
        # Open: an ellipse that widens with mouth_open
        mw = int(size * 0.08 + size * 0.04 * mouth_open)
        mh = int(size * 0.03 + size * 0.07 * mouth_open)
        eye_draw.ellipse(
            [mouth_cx - mw, mouth_y, mouth_cx + mw, mouth_y + mh],
            fill=MOUTH_COLOR,
        )
        # Tongue / inner
        if mouth_open > 0.3:
            tw = int(mw * 0.55)
            th = int(mh * 0.45)
            ty = mouth_y + mh - th
            eye_draw.ellipse(
                [mouth_cx - tw, ty, mouth_cx + tw, ty + th],
                fill=(255, 150, 150, 180),
            )

    # ── 8. Downsample with Lanczos ─────────────────────────────────
    output = img.resize((SIZE, SIZE), Image.LANCZOS)
    return output


# ── Animation generators ───────────────────────────────────────────


def generate_idle(count: int = 8) -> list:
    """
    Idle breathing animation.
    Gentle scale oscillation (breathing) with occasional eye blink.
    """
    frames = []
    for i in range(count):
        t = i / count
        # Breathing: subtle vertical squash/stretch
        breath = 1.0 + 0.025 * math.sin(t * 2 * math.pi)
        squash_x = 1.0 + 0.01 * math.sin(t * 2 * math.pi)
        squash_y = breath

        # Blink on some frames (briefly)
        blink = 0.0
        if i in (3, 7):  # blink on these frames
            blink = 0.9

        # Slight float offset
        offset_y = int(RENDER_SIZE * 0.02 * math.sin(t * 2 * math.pi))

        frame = _draw_slime_frame(
            RENDER_SIZE, phase=t,
            mouth_open=0.0,
            squash_x=squash_x, squash_y=squash_y,
            offset_y=offset_y,
            eye_blink=blink,
        )
        frames.append(frame)
    return frames


def generate_walk(count: int = 8) -> list:
    """
    Walking animation: horizontal stretch/squash and vertical bounce.
    """
    frames = []
    for i in range(count):
        t = i / count
        # Bounce: up and down
        bounce = abs(math.sin(t * 2 * math.pi))
        offset_y = int(RENDER_SIZE * -0.05 * bounce)

        # Horizontal squash/stretch (anticipation and follow-through)
        # Stretch wider when at bottom, squash narrower when at top
        squash_x = 1.0 + 0.08 * bounce - 0.04
        squash_y = 1.0 - 0.06 * bounce + 0.03

        # Slight horizontal wobble
        phase = t

        frame = _draw_slime_frame(
            RENDER_SIZE, phase=phase,
            mouth_open=0.0,
            squash_x=squash_x, squash_y=squash_y,
            offset_y=offset_y,
        )
        frames.append(frame)
    return frames


def generate_wave(count: int = 6) -> list:
    """
    Waving animation: happy body wiggle side-to-side.
    Simulated by asymmetric squash and offset.
    """
    frames = []
    for i in range(count):
        t = i / count
        # Side-to-side wiggle via asymmetric squash
        wiggle = math.sin(t * 2 * math.pi)
        squash_x = 1.0 + 0.06 * abs(wiggle)
        squash_y = 1.0 - 0.03 * abs(wiggle)

        # Subtle vertical bounce
        bounce = abs(math.sin(t * 2 * math.pi))
        offset_y = int(RENDER_SIZE * 0.04 * bounce)

        # Mouth slightly open on peak (happy)
        mouth = 0.15 + 0.10 * bounce

        frame = _draw_slime_frame(
            RENDER_SIZE, phase=t,
            mouth_open=mouth,
            squash_x=squash_x, squash_y=squash_y,
            offset_y=offset_y,
        )
        frames.append(frame)
    return frames


def generate_talk(count: int = 4) -> list:
    """
    Talking animation: mouth open/close cycle with subtle body bounce.
    """
    frames = []
    for i in range(count):
        t = i / count
        # Mouth opens and closes
        mouth = 0.5 + 0.5 * math.sin(t * 2 * math.pi)

        # Subtle body bounce on syllables
        bounce = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 2)  # double frequency
        offset_y = int(RENDER_SIZE * 0.015 * bounce)

        squash_y = 1.0 + 0.02 * bounce
        squash_x = 1.0 - 0.01 * bounce

        frame = _draw_slime_frame(
            RENDER_SIZE, phase=t,
            mouth_open=mouth,
            squash_x=squash_x, squash_y=squash_y,
            offset_y=offset_y,
        )
        frames.append(frame)
    return frames


ANIMATIONS = {
    "idle": generate_idle,
    "walk": generate_walk,
    "wave": generate_wave,
    "talk": generate_talk,
}


def main():
    # Force-clean old sprites to ensure new ones take effect
    if SPRITE_DIR.exists():
        shutil.rmtree(SPRITE_DIR)
        print(f"Deleted old sprites/ directory")
    SPRITE_DIR.mkdir(exist_ok=True)
    print(f"Generating high-quality sprite frames in {SPRITE_DIR}/ ...")
    print(f"  Render resolution: {RENDER_SIZE}x{RENDER_SIZE} (downsampled to {SIZE}x{SIZE})")
    print(f"  Character: cute slime/blob with gradients + drop shadow")

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