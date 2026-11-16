# -*- coding: utf-8 -*-
"""
generate_fsm_clean_gif.py
Generates a pure visual animated GIF for the Dual-Engine Tracking State Machine (FSM).
Follows user rules:
- Light background only (no dark mode, no PNGs).
- Pure visual graphic: No wall of text, no formulas, clear glowing states and transition arrows.
- Saved directly into E:/Thesis/report_materials/tracking_fsm_mechanism.gif.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import numpy as np

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']

def draw_fsm_frame(phase_idx, filename=None):
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    
    # Palette
    c_neutral_bg = '#f8fafc'
    c_neutral_border = '#cbd5e1'
    c_text_main = '#0f172a'
    c_text_muted = '#64748b'
    
    c_purple = '#7e22ce'
    c_amber = '#d97706'
    c_green = '#16a34a'
    c_cyan = '#0284c7'
    c_red = '#dc2626'
    
    states = [
        {
            "id": 0, "name": "00 | SEARCH_LOST", "sub": "Global Radar (256x144)",
            "x": 1.2, "y": 4.4, "w": 3.8, "h": 2.0, "color": c_purple, "bg": '#faf5ff'
        },
        {
            "id": 1, "name": "01 | TENTATIVE_HOLD", "sub": "2-Hit Motion Filter",
            "x": 1.2, "y": 1.2, "w": 3.8, "h": 2.0, "color": c_amber, "bg": '#fffbeb'
        },
        {
            "id": 2, "name": "02 | LOCAL_TRACK", "sub": "Fast Cruise (128x128 | 89 FPS)",
            "x": 6.2, "y": 1.8, "w": 4.0, "h": 3.6, "color": c_green, "bg": '#f0fdf4'
        },
        {
            "id": 3, "name": "03 | SLEEP / FREEZE", "sub": "Dead-Ball Freeze (v = 0)",
            "x": 11.2, "y": 2.4, "w": 2.2, "h": 2.4, "color": c_cyan, "bg": '#f0f9ff'
        }
    ]
    
    phase_titles = [
        ("Phase 1: Startup -> Global Radar Scans Full Court (256x144)", c_purple),
        ("Phase 2: 2-Hit Motion Filter -> Rejects Static Glare & White Shoes", c_amber),
        ("Phase 3: High-Speed In-Rally Cruise (128x128 Local Window | 89 FPS)", c_green),
        ("Phase 4: Local Miss -> Instant Same-Frame Rescue (0-Frame Delay!)", c_red),
        ("Phase 5: Dead Ball / Pre-Serve -> Low-Power Sleep Mode (Freeze v = 0)", c_cyan)
    ]
    title_text, title_color = phase_titles[phase_idx]
    
    # Status Pill at top
    pill = FancyBboxPatch((3.0, 6.7), 8.0, 0.6, boxstyle="round,pad=0.08",
                          facecolor='#f8fafc', edgecolor=title_color, lw=1.8)
    ax.add_patch(pill)
    ax.text(7.0, 7.0, title_text, color=title_color, fontsize=11.5, fontweight='bold', ha='center', va='center')

    # Draw State Boxes
    for s in states:
        sid = s['id']
        is_active = False
        if phase_idx == 0 and sid == 0: is_active = True
        elif phase_idx == 1 and sid == 1: is_active = True
        elif phase_idx == 2 and sid == 2: is_active = True
        elif phase_idx == 3 and (sid == 2 or sid == 0): is_active = True
        elif phase_idx == 4 and sid == 3: is_active = True
        
        edge_col = s['color'] if is_active else c_neutral_border
        bg_col = s['bg'] if is_active else c_neutral_bg
        lw = 2.8 if is_active else 1.2
        
        box = FancyBboxPatch((s['x'], s['y']), s['w'], s['h'], boxstyle="round,pad=0.1",
                             facecolor=bg_col, edgecolor=edge_col, linewidth=lw, zorder=3)
        ax.add_patch(box)
        
        cx = s['x'] + s['w']/2
        cy = s['y'] + s['h']/2
        
        if sid == 2:
            ax.text(cx, s['y'] + s['h'] - 0.45, s['name'], color=s['color'] if is_active else c_text_main,
                    fontsize=12, fontweight='bold', ha='center', zorder=4)
            ax.text(cx, s['y'] + s['h'] - 0.85, s['sub'], color=c_text_muted,
                    fontsize=9, fontstyle='italic', ha='center', zorder=4)
            
            b1 = FancyBboxPatch((s['x'] + 0.3, s['y'] + 1.4), s['w'] - 0.6, 0.65, boxstyle="round,pad=0.04",
                                facecolor='#ffffff', edgecolor=c_green if is_active else c_neutral_border, lw=1.2, zorder=4)
            ax.add_patch(b1)
            ax.text(cx, s['y'] + 1.72, "Smash Dynamic Expansion (160 px)", color=c_green if is_active else c_text_muted,
                    fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=5)
            
            b2 = FancyBboxPatch((s['x'] + 0.3, s['y'] + 0.5), s['w'] - 0.6, 0.65, boxstyle="round,pad=0.04",
                                facecolor='#ffffff', edgecolor=c_green if is_active else c_neutral_border, lw=1.2, zorder=4)
            ax.add_patch(b2)
            ax.text(cx, s['y'] + 0.82, "Kinematic Reversal Damping", color=c_green if is_active else c_text_muted,
                    fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=5)
        else:
            ax.text(cx, cy + 0.3, s['name'], color=s['color'] if is_active else c_text_main,
                    fontsize=11.5, fontweight='bold', ha='center', zorder=4)
            ax.text(cx, cy - 0.2, s['sub'], color=c_text_muted,
                    fontsize=9, fontstyle='italic', ha='center', zorder=4)
            
        if is_active and sid != 2:
            tag = FancyBboxPatch((cx - 1.2, s['y'] + s['h'] + 0.05), 2.4, 0.4, boxstyle="round,pad=0.03",
                                 facecolor=s['color'], edgecolor=s['color'], zorder=6)
            ax.add_patch(tag)
            ax.text(cx, s['y'] + s['h'] + 0.25, "CURRENT STATE", color='#ffffff', fontsize=7.5, fontweight='bold', ha='center', va='center', zorder=7)

    # Transitions
    a1_active = (phase_idx == 0 or phase_idx == 1)
    a1 = FancyArrowPatch((3.1, 4.3), (3.1, 3.3),
                         arrowstyle="->,head_width=0.3,head_length=0.35",
                         color=c_purple if a1_active else c_neutral_border, lw=2.5 if a1_active else 1.5, zorder=2)
    ax.add_patch(a1)
    ax.text(3.3, 3.8, "Candidate Found", color=c_purple if a1_active else c_text_muted,
            fontsize=8.5, fontweight='bold' if a1_active else 'normal', va='center')

    a2_active = (phase_idx == 1 or phase_idx == 2)
    a2 = FancyArrowPatch((5.1, 2.2), (6.1, 2.8),
                         arrowstyle="->,head_width=0.3,head_length=0.35",
                         color=c_amber if a2_active else c_neutral_border, lw=2.5 if a2_active else 1.5, zorder=2)
    ax.add_patch(a2)
    ax.text(5.5, 2.3, "Motion Confirmed", color=c_amber if a2_active else c_text_muted,
            fontsize=8.5, fontweight='bold' if a2_active else 'normal', ha='center', va='top')

    a_loop_active = (phase_idx == 2)
    a_loop = FancyArrowPatch((8.8, 5.5), (7.4, 5.5),
                             connectionstyle="arc3,rad=0.6",
                             arrowstyle="->,head_width=0.3,head_length=0.35",
                             color=c_green if a_loop_active else c_neutral_border, lw=2.5 if a_loop_active else 1.5, zorder=2)
    ax.add_patch(a_loop)
    ax.text(8.1, 6.35, "Normal Rally Loop (89 FPS)", color=c_green if a_loop_active else c_text_muted,
            fontsize=8.5, fontweight='bold' if a_loop_active else 'normal', ha='center')

    a3_active = (phase_idx == 4)
    a_freeze = FancyArrowPatch((10.3, 4.0), (11.1, 4.0),
                               arrowstyle="->,head_width=0.25,head_length=0.3",
                               color=c_cyan if a3_active else c_neutral_border, lw=2.2 if a3_active else 1.2, zorder=2)
    ax.add_patch(a_freeze)
    ax.text(10.7, 4.3, "Static (4f)", color=c_cyan if a3_active else c_text_muted,
            fontsize=8.0, fontweight='bold' if a3_active else 'normal', ha='center')

    a_wake = FancyArrowPatch((11.1, 2.9), (10.3, 2.9),
                             arrowstyle="->,head_width=0.25,head_length=0.3",
                             color=c_green if a3_active else c_neutral_border, lw=2.2 if a3_active else 1.2, zorder=2)
    ax.add_patch(a_wake)
    ax.text(10.7, 2.6, "Serve Hit", color=c_green if a3_active else c_text_muted,
            fontsize=8.0, fontweight='bold' if a3_active else 'normal', ha='center')

    a_rescue_active = (phase_idx == 3)
    a_rescue = FancyArrowPatch((6.3, 4.8), (5.1, 5.4),
                               connectionstyle="arc3,rad=-0.25",
                               arrowstyle="->,head_width=0.35,head_length=0.4",
                               color=c_red if a_rescue_active else c_neutral_border,
                               lw=3.0 if a_rescue_active else 1.5, linestyle='--' if not a_rescue_active else '-', zorder=5)
    ax.add_patch(a_rescue)
    
    a_return = FancyArrowPatch((5.1, 5.0), (6.3, 4.2),
                               connectionstyle="arc3,rad=-0.25",
                               arrowstyle="->,head_width=0.35,head_length=0.4",
                               color=c_green if a_rescue_active else c_neutral_border,
                               lw=3.0 if a_rescue_active else 1.5, linestyle='--' if not a_rescue_active else '-', zorder=5)
    ax.add_patch(a_return)

    if a_rescue_active:
        rescue_badge = FancyBboxPatch((4.5, 4.4), 2.4, 0.8, boxstyle="round,pad=0.06",
                                      facecolor='#fef2f2', edgecolor=c_red, lw=1.8, zorder=7)
        ax.add_patch(rescue_badge)
        ax.text(5.7, 4.95, "SAME-FRAME RESCUE", color=c_red, fontsize=8.5, fontweight='bold', ha='center', va='center', zorder=8)
        ax.text(5.7, 4.62, "0-Frame Delay (30.9 ms)", color='#991b1b', fontsize=7.5, ha='center', va='center', zorder=8)
    else:
        ax.text(5.7, 4.8, "Instant Rescue", color=c_text_muted, fontsize=8.0, ha='center', va='center')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200, facecolor='#ffffff', bbox_inches='tight')
        plt.close()
    else:
        return fig

def main():
    out_dir = "E:/Thesis/report_materials"
    os.makedirs(out_dir, exist_ok=True)
    
    frames = []
    for p in range(5):
        fn = f"{out_dir}/fsm_step_{p}.png"
        draw_fsm_frame(p, filename=fn)
        frames.append(Image.open(fn))
        
    gif_path = f"{out_dir}/tracking_fsm_mechanism.gif"
    durations = [2000, 2000, 2400, 2400, 2200]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"Generated {gif_path}")
    
    for p in range(5):
        try:
            os.remove(f"{out_dir}/fsm_step_{p}.png")
        except:
            pass
    print("Cleaned temp frames. Done!")

if __name__ == '__main__':
    main()
