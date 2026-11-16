# -*- coding: utf-8 -*-
"""
generate_fsm_portrait_gif.py
Generates a pristine, compact PORTRAIT (直向) pure visual animated GIF for the FSM.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import numpy as np

plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']

def draw_portrait_fsm_frame(phase_idx, filename=None):
    fig, ax = plt.subplots(figsize=(6.0, 9.6), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 9.6)
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
    
    # 4 States Geometry (Vertical Stack)
    states = [
        {
            "id": 0, "name": "00 | SEARCH_LOST", "sub": "Global Radar (256x144)",
            "x": 1.2, "y": 7.35, "w": 4.2, "h": 1.3, "color": c_purple, "bg": '#faf5ff'
        },
        {
            "id": 1, "name": "01 | TENTATIVE_HOLD", "sub": "2-Hit Motion Filter",
            "x": 1.2, "y": 5.35, "w": 4.2, "h": 1.3, "color": c_amber, "bg": '#fffbeb'
        },
        {
            "id": 2, "name": "02 | LOCAL_TRACK", "sub": "Fast Cruise (128x128 | 89 FPS)",
            "x": 1.2, "y": 2.30, "w": 4.2, "h": 2.35, "color": c_green, "bg": '#f0fdf4'
        },
        {
            "id": 3, "name": "03 | SLEEP / FREEZE", "sub": "Dead-Ball Freeze (v = 0)",
            "x": 1.2, "y": 0.50, "w": 4.2, "h": 1.05, "color": c_cyan, "bg": '#f0f9ff'
        }
    ]
    
    phase_titles = [
        ("Phase 1: Global Radar Scan", c_purple),
        ("Phase 2: 2-Hit Motion Check", c_amber),
        ("Phase 3: Fast In-Rally Cruise (89 FPS)", c_green),
        ("Phase 4: Same-Frame Instant Rescue", c_red),
        ("Phase 5: Sleep Mode (Freeze v = 0)", c_cyan)
    ]
    title_text, title_color = phase_titles[phase_idx]
    
    # Top Status Pill
    pill = FancyBboxPatch((0.8, 9.0), 4.4, 0.45, boxstyle="round,pad=0.06",
                          facecolor='#f8fafc', edgecolor=title_color, lw=1.8)
    ax.add_patch(pill)
    ax.text(3.0, 9.22, title_text, color=title_color, fontsize=10.5, fontweight='bold', ha='center', va='center')

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
        
        box = FancyBboxPatch((s['x'], s['y']), s['w'], s['h'], boxstyle="round,pad=0.06",
                             facecolor=bg_col, edgecolor=edge_col, linewidth=lw, zorder=3)
        ax.add_patch(box)
        
        cx = s['x'] + s['w']/2
        
        if sid == 2:
            ax.text(cx, s['y'] + s['h'] - 0.35, s['name'], color=s['color'] if is_active else c_text_main,
                    fontsize=11.5, fontweight='bold', ha='center', zorder=4)
            ax.text(cx, s['y'] + s['h'] - 0.68, s['sub'], color=c_text_muted,
                    fontsize=8.5, fontstyle='italic', ha='center', zorder=4)
            
            # Sub-badges inside Local Track
            b1 = FancyBboxPatch((s['x'] + 0.3, s['y'] + 0.95), s['w'] - 0.6, 0.52, boxstyle="round,pad=0.03",
                                facecolor='#ffffff', edgecolor=c_green if is_active else c_neutral_border, lw=1.2, zorder=4)
            ax.add_patch(b1)
            ax.text(cx, s['y'] + 1.21, "Smash Dynamic Expansion (160 px)", color=c_green if is_active else c_text_muted,
                    fontsize=8.0, fontweight='bold', ha='center', va='center', zorder=5)
            
            b2 = FancyBboxPatch((s['x'] + 0.3, s['y'] + 0.28), s['w'] - 0.6, 0.52, boxstyle="round,pad=0.03",
                                facecolor='#ffffff', edgecolor=c_green if is_active else c_neutral_border, lw=1.2, zorder=4)
            ax.add_patch(b2)
            ax.text(cx, s['y'] + 0.54, "Kinematic Reversal Damping", color=c_green if is_active else c_text_muted,
                    fontsize=8.0, fontweight='bold', ha='center', va='center', zorder=5)
        else:
            ax.text(cx, s['y'] + s['h'] - 0.45, s['name'], color=s['color'] if is_active else c_text_main,
                    fontsize=11.5, fontweight='bold', ha='center', zorder=4)
            ax.text(cx, s['y'] + 0.38, s['sub'], color=c_text_muted,
                    fontsize=8.5, fontstyle='italic', ha='center', zorder=4)

    # -------------------------------------------------------------
    # Transitions
    # -------------------------------------------------------------
    # Arrow 1: S0 -> S1 (Candidate Found)
    a1_active = (phase_idx == 0 or phase_idx == 1)
    a1 = FancyArrowPatch((3.3, 7.30), (3.3, 6.70),
                         arrowstyle="->,head_width=0.25,head_length=0.3",
                         color=c_purple if a1_active else c_neutral_border, lw=2.2 if a1_active else 1.3, zorder=2)
    ax.add_patch(a1)
    ax.text(3.55, 7.0, "Candidate Found", color=c_purple if a1_active else c_text_muted,
            fontsize=7.8, fontweight='bold' if a1_active else 'normal', va='center')

    # Arrow 2: S1 -> S2 (Motion Confirmed)
    a2_active = (phase_idx == 1 or phase_idx == 2)
    a2 = FancyArrowPatch((3.3, 5.30), (3.3, 4.70),
                         arrowstyle="->,head_width=0.25,head_length=0.3",
                         color=c_amber if a2_active else c_neutral_border, lw=2.2 if a2_active else 1.3, zorder=2)
    ax.add_patch(a2)
    ax.text(3.55, 5.0, "Motion Confirmed (disp >= 2px)", color=c_amber if a2_active else c_text_muted,
            fontsize=7.8, fontweight='bold' if a2_active else 'normal', va='center')

    # Arrow 3: S2 <-> S3 (Freeze & Wakeup)
    a3_active = (phase_idx == 4)
    # Down arrow (S2 -> S3)
    a_freeze = FancyArrowPatch((2.6, 2.25), (2.6, 1.60),
                               arrowstyle="->,head_width=0.22,head_length=0.28",
                               color=c_cyan if a3_active else c_neutral_border, lw=2.0 if a3_active else 1.2, zorder=2)
    ax.add_patch(a_freeze)
    ax.text(2.0, 1.92, "Static (4f)", color=c_cyan if a3_active else c_text_muted,
            fontsize=7.5, fontweight='bold' if a3_active else 'normal', ha='center')

    # Up arrow (S3 -> S2)
    a_wake = FancyArrowPatch((4.0, 1.60), (4.0, 2.25),
                             arrowstyle="->,head_width=0.22,head_length=0.28",
                             color=c_green if a3_active else c_neutral_border, lw=2.0 if a3_active else 1.2, zorder=2)
    ax.add_patch(a_wake)
    ax.text(4.6, 1.92, "Serve Hit", color=c_green if a3_active else c_text_muted,
            fontsize=7.5, fontweight='bold' if a3_active else 'normal', ha='center')

    # Self Loop on S2 (In-Rally Loop on right edge)
    a_loop_active = (phase_idx == 2)
    a_loop = FancyArrowPatch((5.45, 4.1), (5.45, 2.9),
                             connectionstyle="arc3,rad=-0.65",
                             arrowstyle="->,head_width=0.25,head_length=0.3",
                             color=c_green if a_loop_active else c_neutral_border, lw=2.2 if a_loop_active else 1.3, zorder=2)
    ax.add_patch(a_loop)
    ax.text(5.9, 3.5, "Rally\n(89 FPS)", color=c_green if a_loop_active else c_text_muted,
            fontsize=7.5, fontweight='bold' if a_loop_active else 'normal', ha='left', va='center')

    # Arrow 4: SAME-FRAME RESCUE (Left side curve from S2 up to S0 and back)
    a_rescue_active = (phase_idx == 3)
    a_rescue = FancyArrowPatch((1.15, 3.6), (1.15, 7.9),
                               connectionstyle="arc3,rad=0.35",
                               arrowstyle="->,head_width=0.3,head_length=0.35",
                               color=c_red if a_rescue_active else c_neutral_border,
                               lw=2.5 if a_rescue_active else 1.2, linestyle='-' if a_rescue_active else ':', zorder=5)
    ax.add_patch(a_rescue)
    
    a_return = FancyArrowPatch((1.15, 7.9), (1.15, 4.1),
                               connectionstyle="arc3,rad=0.25",
                               arrowstyle="->,head_width=0.3,head_length=0.35",
                               color=c_green if a_rescue_active else c_neutral_border,
                               lw=2.5 if a_rescue_active else 1.2, linestyle='-' if a_rescue_active else ':', zorder=5)
    ax.add_patch(a_return)

    if a_rescue_active:
        rescue_badge = FancyBboxPatch((0.08, 5.4), 0.95, 1.15, boxstyle="round,pad=0.04",
                                      facecolor='#fef2f2', edgecolor=c_red, lw=1.5, zorder=7)
        ax.add_patch(rescue_badge)
        ax.text(0.55, 6.1, "SAME\nFRAME", color=c_red, fontsize=7.2, fontweight='bold', ha='center', va='center', zorder=8)
        ax.text(0.55, 5.65, "RESCUE", color='#991b1b', fontsize=6.8, fontweight='bold', ha='center', va='center', zorder=8)
    else:
        ax.text(0.5, 5.75, "Rescue\nFallback", color=c_text_muted, fontsize=7.0, ha='center', va='center')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=180, facecolor='#ffffff', bbox_inches='tight')
        plt.close()
    else:
        return fig

def main():
    out_dir = "E:/Thesis/report_materials"
    os.makedirs(out_dir, exist_ok=True)
    
    frames = []
    for p in range(5):
        fn = f"{out_dir}/fsm_p_step_{p}.png"
        draw_portrait_fsm_frame(p, filename=fn)
        frames.append(Image.open(fn))
        
    gif_path = f"{out_dir}/tracking_fsm_mechanism.gif"
    durations = [2000, 2000, 2400, 2400, 2200]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"Generated portrait GIF: {gif_path}")
    
    for p in range(5):
        try:
            os.remove(f"{out_dir}/fsm_p_step_{p}.png")
        except:
            pass
    print("Done!")

if __name__ == '__main__':
    main()
