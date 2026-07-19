#!/usr/bin/env python3
"""
Generate placeholder sprite frames for the Bubby companion.

Creates simple colored circle frames in sprites/[state]/ directories
so the sprite animation system can be tested without external assets.

Usage:
    python3 scripts/download_sprites.py
"""

from pathlib import Path
import math
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow not installed. Install with: pip install Pillow")
    sys.exit(1)


SPRITE_DIR = Path(__file__).parent.parent / "sprites"
SIZE = 128  # pixel size of each frame
FPS = 12


def make_circle_frame(draw, size, color, offset_x=0, offset_y=0, scale=1.0):
    """Draw a simple colored circle with eyes."""
    cx = size // 2 + offset_x
    cy = size // 2 + offset_y
    r = int((size // 3) * scale)

    # Body
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    # Eyes
    eye_r = max(3, r // 5)
    eye_offset = r // 3
    # White of eyes
    draw.ellipse(
        [cx - eye_offset - eye_r, cy - eye_r - eye_r // 2,
         cx - eye_offset + eye_r, cy - eye_r + eye_r // 2],
        fill=(255, 255, 255),
    )
    draw.ellipse(
        [cx + eye_offset - eye_r, cy - eye_r - eye_r // 2,
         cx + eye_offset + eye_r, cy - eye_r + eye_r // 2],
        fill=(255, 255, 255),
    )
    # Pupils
    pupil_r = max(2, eye_r // 2)
    draw.ellipse(
        [cx - eye_offset - pupil_r, cy - eye_r // 2 - pupil_r,
         cx - eye_offset + pupil_r, cy - eye_r // 2 + pupil_r],
        fill=(0, 0, 0),
    )
    draw.ellipse(
        [cx + eye_offset - pupil_r, cy - eye_r // 2 - pupil_r,
         cx + eye_offset + pupil_r, cy - eye_r // 2 + pupil_r],
        fill=(0, 0, 0),
    )


def generate_idle(count: int = 8) -> list:
    """Generate idle breathing frames."""
    frames = []
    for i in range(count):
        img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Subtle scale oscillation for breathing effect
        scale = 1.0 + 0.03 * math.sin(i * 2 * math.pi / count)
        make_circle_frame(draw, SIZE, (100, 200, 255, 255), scale=scale)
        frames.append(img)
    return frames


def generate_walk(count: int = 8) -> list:
    """Generate walking frames with horizontal bounce."""
    frames = []
    for i in range(count):
        img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Horizontal oscillation + vertical bounce
        ox = int(8 * math.sin(i * 2 * math.pi / count))
        oy = int(-3 * abs(math.sin(i * 2 * math.pi / count)))
        make_circle_frame(draw, SIZE, (100, 200, 255, 255), offset_x=ox, offset_y=oy)
        frames.append(img)
    return frames


def generate_wave(count: int = 6) -> list:
    """Generate waving frames."""
    frames = []
    for i in range(count):
        img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Small horizontal shake for wave
        ox = int(10 * math.sin(i * 2 * math.pi / count))
        make_circle_frame(draw, SIZE, (255, 200, 100, 255), offset_x=ox)
        frames.append(img)
    return frames


def generate_talk(count: int = 4) -> list:
    """Generate talking frames (mouth open/close)."""
    frames = []
    for i in range(count):
        img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        scale = 1.0 + 0.05 * (i % 2)  # two-frame open/close
        make_circle_frame(draw, SIZE, (255, 255, 150, 255), scale=scale)
        frames.append(img)
    return frames


ANIMATIONS = {
    "idle": generate_idle,
    "walk": generate_walk,
    "wave": generate_wave,
    "talk": generate_talk,
}


def main():
    SPRITE_DIR.mkdir(exist_ok=True)
    print(f"Generating placeholder sprites in {SPRITE_DIR}/ ...")

    for name, generator in ANIMATIONS.items():
        out_dir = SPRITE_DIR / name
        out_dir.mkdir(exist_ok=True)

        frames = generator()
        for idx, img in enumerate(frames):
            path = out_dir / f"{idx:02d}.png"
            img.save(str(path))
        print(f"  {name}: {len(frames)} frames saved to {out_dir}/")

    print(f"\nDone! {len(ANIMATIONS)} animations generated.")
    print(f"Sprites ready for testing. Run: PYTHONPATH=. python3 src/app.py")


if __name__ == "__main__":
    main()