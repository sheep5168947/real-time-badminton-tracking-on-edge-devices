import os
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

def create_presentation():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # Modern Academic Dark Slate & Tech Accent Palette
    C_BG = RGBColor(15, 23, 42)          # Deep Slate Navy #0F172A
    C_CARD = RGBColor(30, 41, 59)        # Card Navy #1E293B
    C_CARD_BORDER = RGBColor(51, 65, 85) # Border #334155
    C_CYAN = RGBColor(56, 189, 248)      # Tech Cyan #38BDF8
    C_EMERALD = RGBColor(52, 211, 153)   # Mint Emerald #34D399
    C_AMBER = RGBColor(251, 191, 36)     # Amber Orange #FBBF24
    C_WHITE = RGBColor(248, 250, 252)    # Pure White #F8FAFC
    C_MUTED = RGBColor(148, 163, 184)    # Slate Muted #94A3B8
    C_TAG_BG = RGBColor(30, 58, 138)     # Deep Blue Tag #1E3A8A
    
    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = C_BG
        bg.line.fill.background() # no line
        return bg

    def add_header(slide, tag_text, title_text, subtitle_text=None):
        # Category Tag
        tag = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.5), Inches(3.2), Inches(0.38))
        tag.fill.solid()
        tag.fill.fore_color.rgb = C_TAG_BG
        tag.line.color.rgb = C_CYAN
        tag.line.width = Pt(1)
        tf_t = tag.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = tag_text
        p_t.font.name = "Microsoft JhengHei"
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = C_CYAN
        p_t.alignment = PP_ALIGN.CENTER
        
        # Main Slide Title
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.95), Inches(11.7), Inches(0.7))
        tf = tx_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Microsoft JhengHei"
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = C_WHITE
        
        if subtitle_text:
            p2 = tf.add_paragraph()
            p2.text = subtitle_text
            p2.font.name = "Microsoft JhengHei"
            p2.font.size = Pt(13)
            p2.font.color.rgb = C_MUTED
            
    # ==========================================
    # SLIDE 1: TITLE SLIDE (口試首頁)
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)
    
    # Outer Tech Glow Box
    card1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.0), Inches(10.933), Inches(5.5))
    card1.fill.solid()
    card1.fill.fore_color.rgb = C_CARD
    card1.line.color.rgb = C_CARD_BORDER
    card1.line.width = Pt(1.5)
    
    # Badge
    b1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.8), Inches(1.5), Inches(3.8), Inches(0.45))
    b1.fill.solid()
    b1.fill.fore_color.rgb = C_TAG_BG
    b1.line.color.rgb = C_CYAN
    b1.text_frame.paragraphs[0].text = "國立大學資訊工程研究所 碩士論文"
    b1.text_frame.paragraphs[0].font.name = "Microsoft JhengHei"
    b1.text_frame.paragraphs[0].font.size = Pt(13)
    b1.text_frame.paragraphs[0].font.bold = True
    b1.text_frame.paragraphs[0].font.color.rgb = C_CYAN
    b1.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    # Title Text Box
    t_box = s1.shapes.add_textbox(Inches(1.8), Inches(2.2), Inches(9.8), Inches(2.2))
    tf = t_box.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "基於邊緣運算裝置之即時羽球視覺追蹤系統"
    p.font.name = "Microsoft JhengHei"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = C_WHITE
    
    p_en = tf.add_paragraph()
    p_en.text = "Real-Time Badminton Visual Tracking on Edge Devices"
    p_en.font.name = "Calibri"
    p_en.font.size = Pt(20)
    p_en.font.color.rgb = C_CYAN
    
    p_sec = tf.add_paragraph()
    p_sec.text = "▶ 第一章 緒論 (Chapter 1: Introduction) 口試與專案匯報"
    p_sec.font.name = "Microsoft JhengHei"
    p_sec.font.size = Pt(16)
    p_sec.font.bold = True
    p_sec.font.color.rgb = C_EMERALD
    
    # Divider line
    div = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.8), Inches(4.5), Inches(9.7), Inches(0.02))
    div.fill.solid()
    div.fill.fore_color.rgb = C_CARD_BORDER
    div.line.fill.background()
    
    # Info Box
    info_box = s1.shapes.add_textbox(Inches(1.8), Inches(4.7), Inches(9.7), Inches(1.4))
    itf = info_box.text_frame
    itf.word_wrap = True
    
    ip1 = itf.paragraphs[0]
    ip1.text = "指導教授：王教授 博士   |   研究生：陳同學"
    ip1.font.name = "Microsoft JhengHei"
    ip1.font.size = Pt(15)
    ip1.font.bold = True
    ip1.font.color.rgb = C_WHITE
    
    ip2 = itf.add_paragraph()
    ip2.text = "實機環境：NVIDIA Jetson Nano 4GB (10W MAXN)   |   即時串流標準：30.0 FPS 零掉幀"
    ip2.font.name = "Microsoft JhengHei"
    ip2.font.size = Pt(13)
    ip2.font.color.rgb = C_MUTED

    # ==========================================
    # SLIDE 2: RESEARCH BACKGROUND & DEMANDS
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "01.1 研究背景與實務需求", "羽球運動之極限動態挑戰與大眾化即時追蹤需求", "從昂貴的多相機鷹眼系統走向低成本普及化的單視角邊緣 AI 分析")
    
    cards_data_s2 = [
        ("【羽球極限運動特徵】", [
            ("極限擊球速度", "殺球瞬間初速動輒超過 300 km/h，金氏世界紀錄達 493 km/h。"),
            ("嚴重動態模糊", "標準 30/60 FPS 相機曝光下被拉伸為半透明長條殘影。"),
            ("超微小像素目標", "在 720p 轉播視角中球體僅佔 2x2 至 6x6 像素（萬分之幾）。")
        ], C_CYAN),
        ("【商業鷹眼系統之痛點】", [
            ("設備成本高昂", "需 8~12 台高速同步工業相機與伺服器（耗資數百萬）。"),
            ("場地標定繁雜", "依賴繁複空間幾何校正與多視角交會，僅限封閉場館。"),
            ("無法普惠基層", "基層學校、基層體育館與社區球友完全無法負擔。")
        ], C_AMBER),
        ("【本研究核心切入點】", [
            ("單一常規攝影機", "僅依賴單一普通視角（Monocular），無需昂貴陣列。"),
            ("邊緣即時運算", "直接在低成本邊緣裝置獨立推論，無需連線工作站。"),
            ("大眾智慧運動應用", "賦能基層業餘賽事落點自動判決、打擊質量即時回饋與戰術分析。")
        ], C_EMERALD)
    ]
    
    lefts = [0.8, 4.8, 8.8]
    for idx, (title, items, col) in enumerate(cards_data_s2):
        c = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(lefts[idx]), Inches(1.8), Inches(3.7), Inches(5.1))
        c.fill.solid()
        c.fill.fore_color.rgb = C_CARD
        c.line.color.rgb = col
        c.line.width = Pt(1.5)
        
        tb = s2.shapes.add_textbox(Inches(lefts[idx] + 0.15), Inches(1.95), Inches(3.4), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Microsoft JhengHei"
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = col
        
        for item_t, item_d in items:
            p_t = tf.add_paragraph()
            p_t.text = f"• {item_t}"
            p_t.font.name = "Microsoft JhengHei"
            p_t.font.size = Pt(14)
            p_t.font.bold = True
            p_t.font.color.rgb = C_WHITE
            
            p_d = tf.add_paragraph()
            p_d.text = f"   {item_d}"
            p_d.font.name = "Microsoft JhengHei"
            p_d.font.size = Pt(12)
            p_d.font.color.rgb = C_MUTED

    # ==========================================
    # SLIDE 3: THE EDGE DILEMMA
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "01.2 邊緣計算之困境與研究動機", "現存深度學習追蹤邁向邊緣端的三大本質瓶頸", "全圖卷積算力牆、離線評測掩蓋的掉幀陷阱、以及雙重懲罰下的假陽性反光崩潰")
    
    cards_data_s3 = [
        ("困境一：全圖卷積算力牆", "Compute Bottleneck", [
            "學界主流 TrackNet 系列多採全圖輸入（512x288 或 640x360）。",
            "在 10W 邊緣 GPU (Jetson Nano) 即使經 TensorRT FP16 加速，單幀仍需 130ms (7~8 FPS)。",
            "遠遠落後於相機 33.3ms (30 FPS) 攝入要求，計算瓶頸極端嚴峻。"
        ], C_AMBER),
        ("困境二：離線評測掩蓋掉幀陷阱", "Hardware Frame-Drop Trap", [
            "既有文獻普遍採「離線逐幀餵入（Offline）」評測，掩蓋了時間剛性維度。",
            "相機每 33.3ms 強制推入新影格，推論不若引發佇列溢位，硬體強制掉幀率 >75%。",
            "高速下每掉 1 幀等同球體失蹤 1~3 公尺，時序卷積特徵完全斷裂，引發連鎖脫軌。"
        ], C_AMBER),
        ("困境三：雙重懲罰下假陽性崩潰", "Strict Dual-Penalty Metric", [
            "國際標準要求偏差 >15px 同時計為一次 FP 與一次 FN（雙重重罰）。",
            "球場頂棚強光在地坪形成鏡面過曝反光，低階追蹤器盲目被反光吸附。",
            "缺乏物理動態驗證時，準確率狂崩至 30.6%（如 match10 爆出 1,773 幀 FP）。"
        ], C_AMBER)
    ]
    
    for idx, (title, sub, bullets, col) in enumerate(cards_data_s3):
        c = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(lefts[idx]), Inches(1.8), Inches(3.7), Inches(5.1))
        c.fill.solid()
        c.fill.fore_color.rgb = C_CARD
        c.line.color.rgb = C_CARD_BORDER
        c.line.width = Pt(1.5)
        
        tb = s3.shapes.add_textbox(Inches(lefts[idx] + 0.15), Inches(1.95), Inches(3.4), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Microsoft JhengHei"
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = C_WHITE
        
        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = C_CYAN
        
        for b in bullets:
            p_b = tf.add_paragraph()
            p_b.text = f"▶ {b}"
            p_b.font.name = "Microsoft JhengHei"
            p_b.font.size = Pt(12)
            p_b.font.color.rgb = C_MUTED

    # ==========================================
    # SLIDE 4: RESEARCH OBJECTIVES & COMMITMENTS
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "01.3 研究目標與硬體承諾", "兼顧極限低延遲、硬體零掉幀與高純度準確率之邊緣系統", "在僅 10W 功耗之 NVIDIA Jetson Nano 上建立真即時、高純度羽球追蹤新基準")
    
    # Top Banner
    ban = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.733), Inches(0.8))
    ban.fill.solid()
    ban.fill.fore_color.rgb = C_TAG_BG
    ban.line.color.rgb = C_CYAN
    btf = ban.text_frame
    bp = btf.paragraphs[0]
    bp.text = "🎯 核心目標：以超輕量階層式架構，攻克「算力限制」與「光學噪聲」，實現端到端脫機自足運作"
    bp.font.name = "Microsoft JhengHei"
    bp.font.size = Pt(15)
    bp.font.bold = True
    bp.font.color.rgb = C_WHITE
    bp.alignment = PP_ALIGN.CENTER
    
    # 4 Metric Cards
    metrics_data = [
        ("30.0 FPS", "硬體串流即時性", "單影格總處理延遲嚴格限制於 33.3ms 以內，相機持續攝入毫不卡頓。", C_CYAN),
        ("0.00%", "硬體掉幀率", "雙執行緒環形佇列與鎖頁記憶體協同，全程 0 掉幀，時序特徵無縫銜接。", C_EMERALD),
        ("96.12%", "全量實測準確率", "嚴格驗證閘門與雙影格飛行確認，跨越 95% 門檻，消滅 3,776 個偽陽性。", C_EMERALD),
        ("Strict 15px", "雙重懲罰標準", "偏差 >15px 嚴格同罰 FP 與 FN，拒絕放水，真實反映鷹眼落點可信度。", C_AMBER)
    ]
    
    m_lefts = [0.8, 3.8, 6.8, 9.8]
    for idx, (val, title, desc, col) in enumerate(metrics_data):
        mc = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(m_lefts[idx]), Inches(2.8), Inches(2.733), Inches(4.1))
        mc.fill.solid()
        mc.fill.fore_color.rgb = C_CARD
        mc.line.color.rgb = col
        mc.line.width = Pt(1.5)
        
        mtb = s4.shapes.add_textbox(Inches(m_lefts[idx] + 0.1), Inches(2.95), Inches(2.533), Inches(3.8))
        mtf = mtb.text_frame
        mtf.word_wrap = True
        
        p_val = mtf.paragraphs[0]
        p_val.text = val
        p_val.font.name = "Calibri"
        p_val.font.size = Pt(30)
        p_val.font.bold = True
        p_val.font.color.rgb = col
        p_val.alignment = PP_ALIGN.CENTER
        
        p_tit = mtf.add_paragraph()
        p_tit.text = title
        p_tit.font.name = "Microsoft JhengHei"
        p_tit.font.size = Pt(15)
        p_tit.font.bold = True
        p_tit.font.color.rgb = C_WHITE
        p_tit.alignment = PP_ALIGN.CENTER
        
        p_desc = mtf.add_paragraph()
        p_desc.text = f"\n{desc}"
        p_desc.font.name = "Microsoft JhengHei"
        p_desc.font.size = Pt(12)
        p_desc.font.color.rgb = C_MUTED

    # ==========================================
    # SLIDE 5: FOUR CORE CONTRIBUTIONS
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "01.3 本論文核心創新貢獻", "突破邊緣即時性與高精準度矛盾的四大核心技術支柱", "兼具理論物理動力學創新與硬體原生實作驗證")
    
    contribs_data = [
        ("貢獻一：階層式粗細兩階段邊緣追蹤架構", "Two-Stage Coarse-to-Fine Pipeline", [
            "局部 128 Mini-ROI (11ms) 高速循航 + 全域 256 (22ms) 粗尺度應急找回。",
            "增量零拷貝環形緩衝區（<0.9ms），全圖卷積開銷消除 85%，攻克算力牆。"
        ], C_CYAN),
        ("貢獻二：物理運動學動態門檻與反向阻尼", "Smash Gate & Kinematic Damping", [
            "動態殺球搜尋窗擴展（v x 1.6 Gate），徹底杜絕 300 km/h 極速扣殺脫靶。",
            "向量內積方向突變檢驗（v_dot < 0），瞬間斬斷球拍擊球與反彈之慣性過衝。"
        ], C_CYAN),
        ("貢獻三：高純度微型驗證與雙影格物理確認", "Precision >= 95% Defense Matrix", [
            "0.35 嚴格微型驗證門檻，徹底廢除粗候選點盲目接入，地坪反光被視為無效。",
            "動態飛行雙影格物理確認（2.0px <= jump <= Gate）+ 溯及補傳，消滅 97.2% 反光。"
        ], C_EMERALD),
        ("貢獻四：全量 146 支影片真實邊緣大數據實證", "Comprehensive 146-Video Hardware Benchmark", [
            "拒絕離線數值模擬，全量 23 場錦標賽、146 支影片（69,396 影格）實機 Native 運算。",
            "實體裝置全量實測：96.12% Precision、29.94 FPS、0 掉幀，泛化至跨域業餘球場。"
        ], C_EMERALD)
    ]
    
    c_coords = [
        (0.8, 1.8), (6.8, 1.8),
        (0.8, 4.4), (6.8, 4.4)
    ]
    
    for idx, (title, sub, bullets, col) in enumerate(contribs_data):
        x, y = c_coords[idx]
        card = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(5.7), Inches(2.45))
        card.fill.solid()
        card.fill.fore_color.rgb = C_CARD
        card.line.color.rgb = col
        card.line.width = Pt(1.5)
        
        tb = s5.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.1), Inches(5.4), Inches(2.25))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Microsoft JhengHei"
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = col
        
        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = C_CYAN
        
        for b in bullets:
            p_b = tf.add_paragraph()
            p_b.text = f"• {b}"
            p_b.font.name = "Microsoft JhengHei"
            p_b.font.size = Pt(12)
            p_b.font.color.rgb = C_WHITE

    # ==========================================
    # SLIDE 6: THESIS ORGANIZATION
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "01.4 論文架構概述", "由淺入深、理論與實機部署並重之六大章節組織", "邏輯縝密的研究進程與實證支撐體系")
    
    chapters_data = [
        ("第一章 緒論", "Introduction", "研究背景、邊緣運算三大瓶頸、四大核心貢獻與組織結構。"),
        ("第二章 相關研究", "Related Work", "高速球體追蹤演進、模型輕量化/蒸餾、邊緣串流掉幀評測缺陷。"),
        ("第三章 方法論", "Methodology", "階層式架構、動態殺球門檻、反向阻尼、微驗證與雙影格確認。"),
        ("第四章 邊緣部署", "Edge Implementation", "Jetson Nano 特性、TensorRT FP16 鎖頁記憶體、雙執行緒防掉幀佇列。"),
        ("第五章 實驗評估", "Experiments", "全量 146 支錦標賽影片大數據、SOTA 橫向對比、極端反光消融實證。"),
        ("第六章 結論展望", "Conclusion & Future", "研究成果總結、基層落地價值、未來 3D 雙目空間立體追蹤展望。")
    ]
    
    ch_coords = [
        (0.8, 1.8), (4.8, 1.8), (8.8, 1.8),
        (0.8, 4.4), (4.8, 4.4), (8.8, 4.4)
    ]
    
    for idx, (title, sub, desc) in enumerate(chapters_data):
        x, y = ch_coords[idx]
        card = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(3.7), Inches(2.45))
        card.fill.solid()
        card.fill.fore_color.rgb = C_CARD
        card.line.color.rgb = C_CARD_BORDER
        card.line.width = Pt(1.5)
        
        tb = s6.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.15), Inches(3.4), Inches(2.15))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Microsoft JhengHei"
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = C_WHITE
        
        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = C_CYAN
        
        p_desc = tf.add_paragraph()
        p_desc.text = f"\n{desc}"
        p_desc.font.name = "Microsoft JhengHei"
        p_desc.font.size = Pt(12)
        p_desc.font.color.rgb = C_MUTED
        
    out_dir = r"E:\Thesis\tracknet-series-pytorch\thesis_manuscript"
    out_path = os.path.join(out_dir, "Chapter1_Introduction_Presentation.pptx")
    prs.save(out_path)
    print(f"Presentation successfully saved to: {out_path}")

if __name__ == "__main__":
    create_presentation()
