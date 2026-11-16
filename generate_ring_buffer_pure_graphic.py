# -*- coding: utf-8 -*-
"""
generate_ring_buffer_pure_graphic.py
Generates a PURE GRAPHICAL ASSET (GIF + PNG) for Zero-Copy Ring Buffer:
- NO PPT titles, NO subtitles, NO divider lines.
- NO bottom Traditional vs Proposed text blocks (those belong in PPT text).
- Pure visual animation: Camera -> 3-Slot Circular Ring -> Virtual AI Tensor.
- Clean rotating pointers, highlighting the active slot.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image
import numpy as np

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']

def draw_pure_graphic_frame(step_idx, theme='light', filename=None):
    dark = (theme == 'dark')
    bg_color = '#0f172a' if dark else '#ffffff'
    card_bg = '#1e293b' if dark else '#f8fafc'
    border_color = '#475569' if dark else '#cbd5e1'
    text_main = '#f8fafc' if dark else '#0f172a'
    text_muted = '#94a3b8' if dark else '#64748b'
    
    # Modern high-contrast accents
    c_blue = '#38bdf8' if dark else '#0284c7'
    c_green = '#4ade80' if dark else '#16a34a'
    c_amber = '#fbbf24' if dark else '#d97706'
    c_purple = '#c084fc' if dark else '#7e22ce'
    
    fig, ax = plt.subplots(figsize=(14, 6.8), dpi=200)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.8)
    ax.axis('off')
    
    # 4 Steps Data
    steps_data = [
        {
            "tag": "Step 1: Initial State",
            "incoming": "Frame 3",
            "active_slot": 2,
            "slots": ["Frame 1", "Frame 2", "Frame 3"],
            "ptrs": [0, 1, 2] # [Past, Middle, Current]
        },
        {
            "tag": "Step 2: Frame 4 Overwrites Slot 0",
            "incoming": "Frame 4",
            "active_slot": 0,
            "slots": ["Frame 4", "Frame 2", "Frame 3"],
            "ptrs": [1, 2, 0] # Past=Slot 1, Mid=Slot 2, Cur=Slot 0
        },
        {
            "tag": "Step 3: Frame 5 Overwrites Slot 1",
            "incoming": "Frame 5",
            "active_slot": 1,
            "slots": ["Frame 4", "Frame 5", "Frame 3"],
            "ptrs": [2, 0, 1] # Past=Slot 2, Mid=Slot 0, Cur=Slot 1
        },
        {
            "tag": "Step 4: Frame 6 Overwrites Slot 2",
            "incoming": "Frame 6",
            "active_slot": 2,
            "slots": ["Frame 4", "Frame 5", "Frame 6"],
            "ptrs": [0, 1, 2] # Past=Slot 0, Mid=Slot 1, Cur=Slot 2
        }
    ]
    curr = steps_data[step_idx]

    # Clean Top Status Pill (Minimal)
    pill = FancyBboxPatch((5.0, 6.05), 4.0, 0.55, boxstyle="round,pad=0.08",
                          facecolor=card_bg, edgecolor=c_amber, lw=1.5)
    ax.add_patch(pill)
    ax.text(7.0, 6.32, curr['tag'], color=c_amber, fontsize=11, fontweight='bold', ha='center', va='center')

    # =========================================================================
    # 1. LEFT: Camera Input
    # =========================================================================
    cam_card = FancyBboxPatch((0.5, 0.8), 2.8, 5.0, boxstyle="round,pad=0.1",
                             facecolor=card_bg, edgecolor=c_blue, linewidth=2)
    ax.add_patch(cam_card)
    ax.text(1.9, 5.35, "Camera Stream", color=c_blue, fontsize=13, fontweight='bold', ha='center')
    ax.text(1.9, 4.98, "(30 FPS Video)", color=text_muted, fontsize=9.5, ha='center')

    # Current Frame Thumbnail
    f_box = FancyBboxPatch((0.9, 2.6), 2.0, 2.0, boxstyle="round,pad=0.06",
                           facecolor='#0284c7' if not dark else '#0369a1', edgecolor=border_color, lw=1.5)
    ax.add_patch(f_box)
    ax.text(1.9, 3.8, f"{curr['incoming']}", color='white', fontsize=16, fontweight='bold', ha='center', va='center')
    ax.text(1.9, 3.1, "Incoming Frame", color='#e0f2fe', fontsize=9, ha='center', va='center')

    # 1x Resize Pill
    resize_pill = FancyBboxPatch((0.9, 1.3), 2.0, 0.8, boxstyle="round,pad=0.06",
                                 facecolor=bg_color, edgecolor=c_green, linewidth=1.5)
    ax.add_patch(resize_pill)
    ax.text(1.9, 1.7, "Resize 1x", color=c_green, fontsize=11, fontweight='bold', ha='center', va='center')

    # Arrow from Camera to Ring Buffer
    ax.annotate('', xy=(3.9, 3.6), xytext=(3.3, 3.6),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", color=c_blue, lw=3))

    # =========================================================================
    # 2. CENTER: 3-Slot Circular Ring Buffer in RAM
    # =========================================================================
    ring_outer = FancyBboxPatch((4.0, 0.8), 6.0, 5.0, boxstyle="round,pad=0.12",
                                facecolor=card_bg, edgecolor=border_color, linewidth=1.5, linestyle='--')
    ax.add_patch(ring_outer)
    ax.text(7.0, 5.4, "Physical RAM Slots (Ring Buffer)", color=text_main, fontsize=13, fontweight='bold', ha='center')

    # Geometry for 3 Slots in a Ring
    # Slot 0 (Top Center), Slot 1 (Bottom Right), Slot 2 (Bottom Left)
    slot_centers = [
        (7.0, 4.05),  # Slot 0
        (8.4, 2.05),  # Slot 1
        (5.6, 2.05)   # Slot 2
    ]
    slot_labels = ["Slot 0", "Slot 1", "Slot 2"]

    # Circular connecting dashed arrows
    circ_1 = FancyArrowPatch((7.7, 3.85), (8.4, 2.75), connectionstyle="arc3,rad=-0.25",
                             arrowstyle="->,head_width=0.25,head_length=0.3", color=border_color, lw=2, linestyle=':')
    circ_2 = FancyArrowPatch((8.1, 1.65), (5.9, 1.65), connectionstyle="arc3,rad=-0.25",
                             arrowstyle="->,head_width=0.25,head_length=0.3", color=border_color, lw=2, linestyle=':')
    circ_3 = FancyArrowPatch((5.6, 2.75), (6.3, 3.85), connectionstyle="arc3,rad=-0.25",
                             arrowstyle="->,head_width=0.25,head_length=0.3", color=border_color, lw=2, linestyle=':')
    ax.add_patch(circ_1)
    ax.add_patch(circ_2)
    ax.add_patch(circ_3)

    # Draw each of the 3 slots
    for i in range(3):
        cx, cy = slot_centers[i]
        is_active = (i == curr['active_slot'])
        
        s_edge = c_green if is_active else border_color
        s_face = ('#064e3b' if dark else '#dcfce7') if is_active else bg_color
        s_lw = 2.8 if is_active else 1.2
        
        box = FancyBboxPatch((cx - 1.1, cy - 0.55), 2.2, 1.1, boxstyle="round,pad=0.08",
                             facecolor=s_face, edgecolor=s_edge, linewidth=s_lw, zorder=3)
        ax.add_patch(box)
        
        ax.text(cx, cy + 0.22, f"{slot_labels[i]}", color=text_muted, fontsize=8.5, fontweight='bold', ha='center', zorder=4)
        ax.text(cx, cy - 0.18, f"{curr['slots'][i]}", 
                color=c_green if is_active else text_main, fontsize=12, fontweight='bold', ha='center', zorder=4)
        
        if is_active:
            glow = FancyBboxPatch((cx - 0.9, cy + 0.65), 1.8, 0.42, boxstyle="round,pad=0.04",
                                  facecolor=c_green, edgecolor=c_green, zorder=5)
            ax.add_patch(glow)
            ax.text(cx, cy + 0.86, "OVERWRITE", color='white', fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=6)

    # =========================================================================
    # 3. RIGHT: AI Input Tensor (Pointers Only)
    # =========================================================================
    out_card = FancyBboxPatch((10.6, 0.8), 2.9, 5.0, boxstyle="round,pad=0.1",
                             facecolor=card_bg, edgecolor=c_purple, linewidth=2)
    ax.add_patch(out_card)
    ax.text(12.05, 5.35, "AI Input Tensor", color=c_purple, fontsize=13, fontweight='bold', ha='center')
    ax.text(12.05, 4.98, "(Virtual 3-Frame Stack)", color=text_muted, fontsize=9.5, ha='center')

    # The 3 Virtual Slices: Past, Middle, Current
    ptr_past, ptr_mid, ptr_cur = curr['ptrs']
    slice_data = [
        {"role": "Past", "slot": ptr_past, "frame": curr['slots'][ptr_past], "color": '#38bdf8', "y": 3.7},
        {"role": "Middle", "slot": ptr_mid, "frame": curr['slots'][ptr_mid], "color": '#fbbf24', "y": 2.5},
        {"role": "Current", "slot": ptr_cur, "frame": curr['slots'][ptr_cur], "color": '#4ade80', "y": 1.3}
    ]

    for s_info in slice_data:
        sy = s_info['y']
        s_box = FancyBboxPatch((10.9, sy), 2.3, 0.95, boxstyle="round,pad=0.05",
                               facecolor=bg_color, edgecolor=s_info['color'], linewidth=1.8, zorder=3)
        ax.add_patch(s_box)
        ax.text(12.05, sy + 0.65, f"[{s_info['role']}]", color=s_info['color'], fontsize=8.5, fontweight='bold', ha='center')
        ax.text(12.05, sy + 0.25, f"Slot {s_info['slot']} ({s_info['frame']})", 
                color=text_main, fontsize=9.5, fontweight='bold', ha='center')

        # Connect pointer arrow from Ring Buffer slot to Tensor slice
        src_cx, src_cy = slot_centers[s_info['slot']]
        p_arrow = FancyArrowPatch((src_cx + 1.1, src_cy), (10.85, sy + 0.48),
                                  connectionstyle="arc3,rad=-0.12",
                                  arrowstyle="->,head_width=0.22,head_length=0.22",
                                  color=s_info['color'], lw=2.0, linestyle='--')
        ax.add_patch(p_arrow)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200, facecolor=bg_color, bbox_inches='tight')
        plt.close()
    else:
        return fig

def main():
    light_imgs = []
    dark_imgs = []
    
    for i in range(4):
        fn_light = f"E:/Thesis/pure_ring_buffer_{i}_light.png"
        fn_dark = f"E:/Thesis/pure_ring_buffer_{i}_dark.png"
        draw_pure_graphic_frame(i, theme='light', filename=fn_light)
        draw_pure_graphic_frame(i, theme='dark', filename=fn_dark)
        light_imgs.append(Image.open(fn_light))
        dark_imgs.append(Image.open(fn_dark))
        
    gif_light = "E:/Thesis/zero_copy_ring_buffer_pure.gif"
    gif_dark = "E:/Thesis/zero_copy_ring_buffer_pure_dark.gif"
    
    # 2.2s per frame, infinite loop
    light_imgs[0].save(gif_light, save_all=True, append_images=light_imgs[1:], duration=2200, loop=0)
    dark_imgs[0].save(gif_dark, save_all=True, append_images=dark_imgs[1:], duration=2200, loop=0)
    print(f"Generated {gif_light} and {gif_dark}")

    # Save Step 2 as pure static master diagram
    draw_pure_graphic_frame(1, theme='light', filename="E:/Thesis/zero_copy_ring_buffer_pure_diagram_light.png")
    draw_pure_graphic_frame(1, theme='dark', filename="E:/Thesis/zero_copy_ring_buffer_pure_diagram_dark.png")
    print("Generated pure master diagrams.")

if __name__ == '__main__':
    main()
