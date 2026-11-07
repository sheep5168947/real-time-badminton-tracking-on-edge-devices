import os
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

def create_presentation():
    prs = pptx.Presentation()
    # 16:9 Widescreen (10.0 x 5.625 inches) matching user's original presentation
    prs.slide_width = Inches(10.0)
    prs.slide_height = Inches(5.625)
    blank_layout = prs.slide_layouts[6]

    # Colors matching the user's original clean modern styling
    C_BG = RGBColor(255, 255, 255)         # Pure White
    C_CARD_BG = RGBColor(248, 250, 252)    # Slate 50 #F8FAFC
    C_CARD_BORDER = RGBColor(226, 232, 240)# Slate 200 #E2E8F0
    C_NAVY = RGBColor(30, 41, 59)          # Slate 800 #1E293B (Main Title)
    C_BLUE = RGBColor(37, 99, 235)         # Blue 600 #2563EB (Accent)
    C_CYAN_BG = RGBColor(239, 246, 255)    # Blue 50 #EFF6FF
    C_MUTED = RGBColor(100, 116, 139)      # Slate 500 #64748B
    C_RED = RGBColor(225, 29, 72)          # Rose 600 #E11D48
    C_RED_BG = RGBColor(255, 241, 242)     # Rose 50 #FFF1F2
    C_GREEN = RGBColor(16, 185, 129)       # Emerald 500 #10B981
    C_GREEN_BG = RGBColor(236, 253, 245)   # Emerald 50 #ECFDF5
    C_DARK = RGBColor(15, 23, 42)          # Slate 900 #0F172A

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(10.0), Inches(5.625))
        bg.fill.solid()
        bg.fill.fore_color.rgb = C_BG
        bg.line.fill.background()
        return bg

    def add_slide_header(slide, tag_text, title_text, highlight_text=""):
        # Header text box
        tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(8.8), Inches(0.85))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        # Category Tag
        p_tag = tf.paragraphs[0]
        p_tag.text = tag_text.upper()
        p_tag.font.name = "Inter"
        p_tag.font.size = Pt(10)
        p_tag.font.bold = True
        p_tag.font.color.rgb = C_BLUE

        # Title
        p_title = tf.add_paragraph()
        run1 = p_title.add_run()
        run1.text = title_text + " "
        run1.font.name = "Inter"
        run1.font.size = Pt(20)
        run1.font.bold = True
        run1.font.color.rgb = C_NAVY

        if highlight_text:
            run2 = p_title.add_run()
            run2.text = highlight_text
            run2.font.name = "Inter"
            run2.font.size = Pt(20)
            run2.font.bold = True
            run2.font.color.rgb = C_BLUE

    assets_dir = r"E:\Thesis\tracknet-series-pytorch\thesis_manuscript\assets"

    # ==========================================================
    # SLIDE 1: Title Slide (Cover)
    # ==========================================================
    s0 = prs.slides.add_slide(blank_layout)
    set_slide_background(s0)

    # Decorative top bar
    bar = s0.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.8), Inches(1.2), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_BLUE
    bar.line.fill.background()

    # Title box
    t_box = s0.shapes.add_textbox(Inches(0.6), Inches(1.1), Inches(8.8), Inches(2.2))
    tf0 = t_box.text_frame
    tf0.word_wrap = True
    tf0.margin_left = tf0.margin_top = tf0.margin_right = tf0.margin_bottom = 0

    p_t1 = tf0.paragraphs[0]
    p_t1.text = "Real-Time Badminton Tracking on Edge Devices"
    p_t1.font.name = "Inter"
    p_t1.font.size = Pt(28)
    p_t1.font.bold = True
    p_t1.font.color.rgb = C_DARK

    p_t2 = tf0.add_paragraph()
    p_t2.text = "A Lightweight ROI-based Architecture"
    p_t2.font.name = "Inter"
    p_t2.font.size = Pt(20)
    p_t2.font.bold = True
    p_t2.font.color.rgb = C_BLUE

    p_t3 = tf0.add_paragraph()
    p_t3.text = "Chapter 1: Motivation, Trends & The Edge Dilemma"
    p_t3.font.name = "Inter"
    p_t3.font.size = Pt(13)
    p_t3.font.color.rgb = C_MUTED

    # Bottom Info Card
    c_info = s0.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(3.6), Inches(8.8), Inches(1.4))
    c_info.fill.solid()
    c_info.fill.fore_color.rgb = C_CARD_BG
    c_info.line.color.rgb = C_CARD_BORDER
    c_info.line.width = Pt(1)

    itb = s0.shapes.add_textbox(Inches(0.9), Inches(3.75), Inches(8.2), Inches(1.1))
    itf = itb.text_frame
    itf.word_wrap = True
    itf.margin_left = itf.margin_top = itf.margin_right = itf.margin_bottom = 0

    ip1 = itf.paragraphs[0]
    ip1.text = "Master's Thesis Oral Defense Presentation"
    ip1.font.name = "Inter"
    ip1.font.size = Pt(13)
    ip1.font.bold = True
    ip1.font.color.rgb = C_NAVY

    ip2 = itf.add_paragraph()
    ip2.text = "Target Hardware: NVIDIA Jetson Nano 4GB (10W MAXN Mode)  |  Standard: 30.0 FPS Zero-Drop Ingestion"
    ip2.font.name = "Inter"
    ip2.font.size = Pt(11)
    ip2.font.color.rgb = C_MUTED

    ip3 = itf.add_paragraph()
    ip3.text = "Key Milestone: 96.12% Precision, 30 FPS Real-Time, Verified Across 146 Tournament Clips & Amateur Courts"
    ip3.font.name = "Inter"
    ip3.font.size = Pt(11)
    ip3.font.bold = True
    ip3.font.color.rgb = C_GREEN

    # ==========================================================
    # SLIDE 2: Motivation & The Accessibility Gap
    # ==========================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)
    add_slide_header(s1, "Chapter 1 : Introduction", "Motivation & The Accessibility Gap", "(Why Edge AI for Badminton?)")

    # Three-Column Storyboard Cards
    col_w = Inches(2.73)
    col_gap = Inches(0.28)
    left0 = Inches(0.6)
    c_top = Inches(1.3)
    c_h = Inches(3.9)

    # ---------------- Card 1: Badminton Popularity & Analytics ----------------
    c1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left0, c_top, col_w, c_h)
    c1.fill.solid()
    c1.fill.fore_color.rgb = C_CARD_BG
    c1.line.color.rgb = C_CARD_BORDER
    c1.line.width = Pt(1)

    # Badge 1
    b1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left0 + Inches(0.15), c_top + Inches(0.15), Inches(2.43), Inches(0.32))
    b1.fill.solid()
    b1.fill.fore_color.rgb = C_CYAN_BG
    b1.line.color.rgb = C_BLUE
    b1.line.width = Pt(0.75)
    b1.text_frame.paragraphs[0].text = "01 / MASSIVE POPULARITY"
    b1.text_frame.paragraphs[0].font.name = "Inter"
    b1.text_frame.paragraphs[0].font.size = Pt(9.5)
    b1.text_frame.paragraphs[0].font.bold = True
    b1.text_frame.paragraphs[0].font.color.rgb = C_BLUE
    b1.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # Image 1 (Court Trajectory)
    img1_path = os.path.join(assets_dir, "slide5_img_3.png")
    if os.path.exists(img1_path):
        s1.shapes.add_picture(img1_path, left0 + Inches(0.2), c_top + Inches(0.55), width=Inches(2.33))

    tb1 = s1.shapes.add_textbox(left0 + Inches(0.15), c_top + Inches(2.2), Inches(2.43), Inches(1.9))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = 0

    p = tf1.paragraphs[0]
    p.text = "#1 Ball Sport in Taiwan"
    p.font.name = "Inter"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    p2 = tf1.add_paragraph()
    p2.text = "> 3 Million active players (9.4% population share), surpassing Basketball (2.1M) & Table Tennis (1.7M)."
    p2.font.name = "Inter"
    p2.font.size = Pt(10)
    p2.font.color.rgb = C_MUTED

    p3 = tf1.add_paragraph()
    p3.text = "\nWhy Track Trajectories?"
    p3.font.name = "Inter"
    p3.font.size = Pt(11)
    p3.font.bold = True
    p3.font.color.rgb = C_BLUE

    p4 = tf1.add_paragraph()
    p4.text = "• Weakness discovery & shot patterns\n• Automated line-call judging\n• Objective tactical coaching"
    p4.font.name = "Inter"
    p4.font.size = Pt(9.5)
    p4.font.color.rgb = C_DARK

    # ---------------- Card 2: The Elite Standard vs. Prohibitive Cost ----------------
    left1 = left0 + col_w + col_gap
    c2 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left1, c_top, col_w, c_h)
    c2.fill.solid()
    c2.fill.fore_color.rgb = C_CARD_BG
    c2.line.color.rgb = C_CARD_BORDER
    c2.line.width = Pt(1)

    # Badge 2
    b2 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left1 + Inches(0.15), c_top + Inches(0.15), Inches(2.43), Inches(0.32))
    b2.fill.solid()
    b2.fill.fore_color.rgb = C_RED_BG
    b2.line.color.rgb = C_RED
    b2.line.width = Pt(0.75)
    b2.text_frame.paragraphs[0].text = "02 / THE PROHIBITIVE COST"
    b2.text_frame.paragraphs[0].font.name = "Inter"
    b2.text_frame.paragraphs[0].font.size = Pt(9.5)
    b2.text_frame.paragraphs[0].font.bold = True
    b2.text_frame.paragraphs[0].font.color.rgb = C_RED
    b2.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # Image 2 (Hawk-Eye multi-camera setup)
    img2_path = os.path.join(assets_dir, "slide5_img_5.png")
    if os.path.exists(img2_path):
        s1.shapes.add_picture(img2_path, left1 + Inches(0.35), c_top + Inches(0.55), width=Inches(2.03))

    tb2 = s1.shapes.add_textbox(left1 + Inches(0.15), c_top + Inches(2.2), Inches(2.43), Inches(1.9))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = 0

    p = tf2.paragraphs[0]
    p.text = "Commercial Hawk-Eye Dilemma"
    p.font.name = "Inter"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    p2 = tf2.add_paragraph()
    p2.text = "Cost: > $30,000+ USD (~NT$ 1M+)\nRequires 8-12 synchronized high-speed cameras, optical fiber, & high-end servers."
    p2.font.name = "Inter"
    p2.font.size = Pt(10)
    p2.font.color.rgb = C_RED

    p3 = tf2.add_paragraph()
    p3.text = "\nThe Accessibility Gap:"
    p3.font.name = "Inter"
    p3.font.size = Pt(11)
    p3.font.bold = True
    p3.font.color.rgb = C_DARK

    p4 = tf2.add_paragraph()
    p4.text = "Extremely high barriers prevent adoption in general badminton venues, school gyms, and amateur clubs."
    p4.font.name = "Inter"
    p4.font.size = Pt(9.5)
    p4.font.color.rgb = C_MUTED

    # ---------------- Card 3: Our Mission - Affordable Edge AI ----------------
    left2 = left1 + col_w + col_gap
    c3 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left2, c_top, col_w, c_h)
    c3.fill.solid()
    c3.fill.fore_color.rgb = C_GREEN_BG
    c3.line.color.rgb = C_GREEN
    c3.line.width = Pt(1.5)

    # Badge 3
    b3 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left2 + Inches(0.15), c_top + Inches(0.15), Inches(2.43), Inches(0.32))
    b3.fill.solid()
    b3.fill.fore_color.rgb = C_GREEN
    b3.line.fill.background()
    b3.text_frame.paragraphs[0].text = "03 / OUR RESEARCH MISSION"
    b3.text_frame.paragraphs[0].font.name = "Inter"
    b3.text_frame.paragraphs[0].font.size = Pt(9.5)
    b3.text_frame.paragraphs[0].font.bold = True
    b3.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    b3.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    tb3 = s1.shapes.add_textbox(left2 + Inches(0.15), c_top + Inches(0.65), Inches(2.43), Inches(3.1))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    tf3.margin_left = tf3.margin_top = tf3.margin_right = tf3.margin_bottom = 0

    p = tf3.paragraphs[0]
    p.text = "Democratizing Badminton AI"
    p.font.name = "Inter"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    p2 = tf3.add_paragraph()
    p2.text = "\nWhat If We Could Track With:"
    p2.font.name = "Inter"
    p2.font.size = Pt(11)
    p2.font.bold = True
    p2.font.color.rgb = C_GREEN

    p3 = tf3.add_paragraph()
    p3.text = "1. ONE Standard Camera\n   (Single monocular broadcast view)\n2. ONE Accessible Edge Device\n   (NVIDIA Jetson Nano ~$100 in lab)\n3. Standalone Real-Time AI\n   (Zero-drop 30 FPS processing)"
    p3.font.name = "Inter"
    p3.font.size = Pt(10)
    p3.font.color.rgb = C_DARK

    p4 = tf3.add_paragraph()
    p4.text = "\n★ The Vision:"
    p4.font.name = "Inter"
    p4.font.size = Pt(11)
    p4.font.bold = True
    p4.font.color.rgb = C_NAVY

    p5 = tf3.add_paragraph()
    p5.text = "Enable portable, low-cost Hawk-Eye tracking in ANY sports venue, empowering amateur players with elite-level sports science!"
    p5.font.name = "Inter"
    p5.font.size = Pt(9.5)
    p5.font.color.rgb = C_MUTED

    # ==========================================================
    # SLIDE 3: Background & Trends: The Edge Dilemma
    # ==========================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_slide_header(s2, "Chapter 1 : Introduction", "Background & Trends : The Edge Dilemma", "(Why Naive Baselines Fail)")

    # Top Row: 3 Strategy Summary Cards
    top_w = Inches(2.73)
    t_h = Inches(1.15)
    t_top = Inches(1.25)

    strat_data = [
        ("Full-Frame Deep Model", "TrackNetV4 on Jetson Nano", "1,100 ms / frame (< 1 FPS)", C_RED, C_RED_BG),
        ("Knowledge Distillation (KD)", "Shrinks network parameters", "0.9 FPS → 4.8 FPS (Still Laggy)", C_BLUE, C_CYAN_BG),
        ("Region of Interest (Mini-ROI)", "Cuts image workload (128x128)", "Successfully reaches 30 FPS!", C_GREEN, C_GREEN_BG)
    ]

    for idx, (t1, t2, t3, col, bg_col) in enumerate(strat_data):
        lx = left0 + idx * (top_w + col_gap)
        box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, lx, t_top, top_w, t_h)
        box.fill.solid()
        box.fill.fore_color.rgb = bg_col
        box.line.color.rgb = col
        box.line.width = Pt(1)

        stb = s2.shapes.add_textbox(lx + Inches(0.12), t_top + Inches(0.1), top_w - Inches(0.24), t_h - Inches(0.2))
        stf = stb.text_frame
        stf.word_wrap = True
        stf.margin_left = stf.margin_top = stf.margin_right = stf.margin_bottom = 0

        p = stf.paragraphs[0]
        p.text = t1
        p.font.name = "Inter"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = C_DARK

        p2 = stf.add_paragraph()
        p2.text = t2
        p2.font.name = "Inter"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED

        p3 = stf.add_paragraph()
        p3.text = t3
        p3.font.name = "Inter"
        p3.font.size = Pt(11)
        p3.font.bold = True
        p3.font.color.rgb = col

    # Bottom Row: Real Embedded GIF / Video Comparison
    bot_top = Inches(2.55)
    gif_w = Inches(4.25)
    gif_h = Inches(2.25)

    # Left Frame Box: Native TrackNetV4
    lx1 = Inches(0.6)
    box_l = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, lx1, bot_top, gif_w, Inches(2.65))
    box_l.fill.solid()
    box_l.fill.fore_color.rgb = C_CARD_BG
    box_l.line.color.rgb = C_CARD_BORDER
    box_l.line.width = Pt(1)

    # GIF 1
    gif1_path = os.path.join(assets_dir, "slide9_img_4.gif")
    if os.path.exists(gif1_path):
        s2.shapes.add_picture(gif1_path, lx1 + Inches(0.1), bot_top + Inches(0.1), width=Inches(4.05))

    tb_l = s2.shapes.add_textbox(lx1 + Inches(0.1), bot_top + Inches(2.1), Inches(4.05), Inches(0.5))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = 0
    p = tf_l.paragraphs[0]
    p.text = "Native TrackNetV4 (Full Frame) | 0.9 FPS"
    p.font.name = "Inter"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_RED
    p2 = tf_l.add_paragraph()
    p2.text = "❌ 1.1s latency causes >75% hardware frame drop; buffer overflows immediately."
    p2.font.name = "Inter"
    p2.font.size = Pt(9)
    p2.font.color.rgb = C_MUTED

    # Right Frame Box: KD + ROI Baseline
    lx2 = Inches(5.15)
    box_r = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, lx2, bot_top, gif_w, Inches(2.65))
    box_r.fill.solid()
    box_r.fill.fore_color.rgb = C_CARD_BG
    box_r.line.color.rgb = C_CARD_BORDER
    box_r.line.width = Pt(1)

    # GIF 2
    gif2_path = os.path.join(assets_dir, "slide9_img_5.gif")
    if os.path.exists(gif2_path):
        s2.shapes.add_picture(gif2_path, lx2 + Inches(0.1), bot_top + Inches(0.1), width=Inches(4.05))

    tb_r = s2.shapes.add_textbox(lx2 + Inches(0.1), bot_top + Inches(2.1), Inches(4.05), Inches(0.5))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0
    p = tf_r.paragraphs[0]
    p.text = "TrackNetV4 + KD + Naive ROI | 30.0 FPS"
    p.font.name = "Inter"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = C_BLUE
    p2 = tf_r.add_paragraph()
    p2.text = "⚠️ 30 FPS reached, BUT: Tail-Lag on smashes, Glare Lock on floor, & Hit Inversion Overshoot!"
    p2.font.name = "Inter"
    p2.font.size = Pt(9)
    p2.font.color.rgb = C_RED

    # Bottom Takeaway Callout Bar
    bar_callout = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(5.25), Inches(8.8), Inches(0.3))
    bar_callout.fill.solid()
    bar_callout.fill.fore_color.rgb = C_DARK
    bar_callout.line.fill.background()
    b_tf = bar_callout.text_frame
    bp = b_tf.paragraphs[0]
    bp.text = "💡 Core Dilemma: Reaching 30 FPS is only half the battle. Real-world edge tracking demands robust physical & optical defenses!"
    bp.font.name = "Inter"
    bp.font.size = Pt(9.5)
    bp.font.bold = True
    bp.font.color.rgb = RGBColor(255, 255, 255)
    bp.alignment = PP_ALIGN.CENTER

    out_dir = r"E:\Thesis\tracknet-series-pytorch\thesis_manuscript"
    out_path = os.path.join(out_dir, "Chapter1_Story_Presentation_EN.pptx")
    prs.save(out_path)
    print(f"Presentation successfully saved to: {out_path}")

if __name__ == "__main__":
    create_presentation()
