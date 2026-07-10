#!/usr/bin/env python3
"""Generate title/closing cards + native 1080 section stills for thesis demo video."""
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "fullpage_probe.png")   # 1x render, 1920 wide, matches CSS 1:1
W, H = 1920, 1080

AV = "/System/Library/Fonts/Avenir Next.ttc"
IDX = {"heavy": 8, "bold": 0, "demi": 2, "medium": 5, "regular": 7,
       "demi_it": 3, "medium_it": 6}

def font(kind, size):
    return ImageFont.truetype(AV, size, index=IDX[kind])

NAVY = (31, 78, 121)          # #1F4E79
NAVY_DARK = (23, 58, 92)
OFFWHITE = (236, 242, 249)
WHITE = (255, 255, 255)
ACCENT = (137, 183, 226)      # light blue
DIM = (176, 198, 222)

def draw_centered(draw, cy, text, fnt, fill, spacing=0):
    if spacing:
        # letter-spaced rendering
        widths = [draw.textlength(ch, font=fnt) for ch in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = (W - total) / 2
        asc, desc = fnt.getmetrics()
        y = cy - (asc + desc) / 2
        for ch, wch in zip(text, widths):
            draw.text((x, y), ch, font=fnt, fill=fill)
            x += wch + spacing
        return
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    asc, desc = fnt.getmetrics()
    draw.text(((W - tw) / 2, cy - (asc + desc) / 2), text, font=fnt, fill=fill)

def gradient_bg():
    img = Image.new("RGB", (W, H), NAVY)
    px = img.load()
    for y in range(H):
        t = y / H
        r = int(NAVY[0] * (1 - t) + NAVY_DARK[0] * t)
        g = int(NAVY[1] * (1 - t) + NAVY_DARK[1] * t)
        b = int(NAVY[2] * (1 - t) + NAVY_DARK[2] * t)
        for x in range(W):
            px[x, y] = (r, g, b)
    return img

def title_card():
    img = gradient_bg()
    d = ImageDraw.Draw(img)
    # top + bottom accent hairlines
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    d.rectangle([0, H - 6, W, H], fill=ACCENT)
    draw_centered(d, 250, "M . T E C H   T H E S I S   D E M O", font("medium", 30), DIM, spacing=2)
    draw_centered(d, 360, "Tinker RL Lab", font("heavy", 132), WHITE)
    draw_centered(d, 470, "Reinforcement Learning for LLMs — Benchmark & Ablations", font("medium", 36), OFFWHITE)
    # accent rule
    d.rectangle([(W // 2 - 160), 545, (W // 2 + 160), 549], fill=ACCENT)
    draw_centered(d, 620, "Arvind C R  ·  PES University", font("demi", 46), OFFWHITE)
    draw_centered(d, 685, "Guide: Ramesh Prakash Guledgudd", font("regular", 38), DIM)
    draw_centered(d, 830, "Notebook executed live — all outputs real", font("demi_it", 40), ACCENT)
    img.save(os.path.join(HERE, "card_title.png"))

def closing_card():
    img = gradient_bg()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    d.rectangle([0, H - 6, W, H], fill=ACCENT)
    draw_centered(d, 300, "Papers P1–P8", font("heavy", 104), WHITE)
    d.rectangle([(W // 2 - 160), 400, (W // 2 + 160), 404], fill=ACCENT)
    draw_centered(d, 490, "NeurIPS 2026 workshop submission", font("medium", 46), OFFWHITE)
    draw_centered(d, 660, "github.com/arvindcr4/tinker-rl-lab", font("demi", 44), ACCENT)
    draw_centered(d, 770, "Full notebook, results tables, and reproduction scripts included",
                  font("regular", 32), DIM)
    img.save(os.path.join(HERE, "card_closing.png"))

# CSS-y top of each 1080 window, with duration (s). Matches measured element offsets.
WINDOWS = [
    (0,    6.0),   # W1  notebook title + env setup + overview
    (930,  7.0),   # W2  overview + load consolidated + first tables
    (1560, 7.5),   # W3  consolidated results tables
    (2540, 7.0),   # W4  end of consolidated + headline heading
    (3160, 9.0),   # W5  HEADLINE result + GRPO table
    (4230, 8.0),   # W6  held-out accuracy + PPO vs GRPO + table
    (5140, 9.0),   # W7  PPO vs GRPO reward-curve plot
    (6000, 9.0),   # W8  ZVF diagnostic + plot
    (7060, 7.5),   # W9  ZVF summary table + group-size heading
    (8000, 9.0),   # W10 group-size sweep plot
    (8740, 7.5),   # W11 fraud study P8 tables
    (9560, 9.0),   # W12 fraud ROC plot + closing + repro table
]
F = 1.00705  # 1x-render row / CSS-px

def slice_windows():
    src = Image.open(SRC).convert("RGB")
    paths = []
    for i, (top, dur) in enumerate(WINDOWS, 1):
        row = int(round(top * F))
        crop = src.crop((0, row, W, row + H))
        p = os.path.join(HERE, f"w{i:02d}.png")
        crop.save(p)
        paths.append((f"w{i:02d}.png", dur))
    return paths

def write_concat(win_paths):
    lines = []
    lines.append("file 'card_title.png'")
    lines.append("duration 6.0")
    for name, dur in win_paths:
        lines.append(f"file '{name}'")
        lines.append(f"duration {dur}")
    lines.append("file 'card_closing.png'")
    lines.append("duration 6.0")
    # repeat last so its duration is honored by concat demuxer
    lines.append("file 'card_closing.png'")
    with open(os.path.join(HERE, "concat_v4.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    total = 6.0 + sum(d for _, d in win_paths) + 6.0
    print("total duration (s):", total, "frames:", len(win_paths) + 2)

if __name__ == "__main__":
    title_card()
    closing_card()
    wp = slice_windows()
    write_concat(wp)
    print("done")
