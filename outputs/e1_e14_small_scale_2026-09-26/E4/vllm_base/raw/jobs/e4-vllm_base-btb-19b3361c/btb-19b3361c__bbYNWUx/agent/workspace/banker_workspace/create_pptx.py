from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from datetime import datetime

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color scheme
    NAVY = RGBColor(0x00, 0x2B, 0x5C)
    DARK_BLUE = RGBColor(0x00, 0x3D, 0x7A)
    ACCENT_BLUE = RGBColor(0x00, 0x72, 0xCE)
    LIGHT_BLUE = RGBColor(0xD6, 0xE9, 0xF8)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    BLACK = RGBColor(0x00, 0x00, 0x00)
    DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
    LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
    ORANGE = RGBColor(0xE8, 0x6C, 0x00)
    GREEN = RGBColor(0x00, 0x8A, 0x00)
    RED = RGBColor(0xC0, 0x00, 0x00)

    def add_background(slide, color):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_shape(slide, left, top, width, height, fill_color, border_color=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1)
        else:
            shape.line.fill.background()
        return shape

    def add_textbox(slide, left, top, width, height, text, font_size=12, bold=False, color=BLACK, alignment=PP_ALIGN.LEFT, font_name='Calibri'):
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.name = font_name
        p.alignment = alignment
        return txBox

    def add_bullet_list(slide, left, top, width, height, items, font_size=11, color=DARK_GRAY, spacing=Pt(6)):
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            if i == 0:
                p = tf.paragraphs[0]
            else:
                p = tf.add_paragraph()
            p.text = item
            p.font.size = Pt(font_size)
            p.font.color.rgb = color
            p.font.name = 'Calibri'
            p.space_after = spacing
            p.level = 0
        return txBox

    def add_header_bar(slide, title_text):
        add_shape(slide, Inches(0), Inches(0), Inches(13.333), Inches(1.2), NAVY)
        add_textbox(slide, Inches(0.5), Inches(0.2), Inches(12), Inches(0.8), title_text, 28, True, WHITE)
        add_shape(slide, Inches(0), Inches(1.2), Inches(13.333), Inches(0.05), ACCENT_BLUE)

    def add_footer(slide, slide_num, total_slides=8):
        add_shape(slide, Inches(0), Inches(7.1), Inches(13.333), Inches(0.4), LIGHT_GRAY)
        add_textbox(slide, Inches(0.5), Inches(7.15), Inches(6), Inches(0.3), "Confidential - M&A Due Diligence", 8, False, DARK_GRAY)
        add_textbox(slide, Inches(11), Inches(7.15), Inches(2), Inches(0.3), f"{slide_num}/{total_slides}", 8, False, DARK_GRAY, PP_ALIGN.RIGHT)

    # ============================================
    # SLIDE 1: Title Slide
    # ============================================
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide1, NAVY)
    add_shape(slide1, Inches(0), Inches(3), Inches(13.333), Inches(0.08), ACCENT_BLUE)
    add_textbox(slide1, Inches(1), Inches(1.5), Inches(11), Inches(1.5), "Sell-Side M&A Due Diligence", 44, True, WHITE, PP_ALIGN.CENTER)
    add_textbox(slide1, Inches(1), Inches(3.3), Inches(11), Inches(1), "Timeline & Deliverable Summary", 28, False, LIGHT_BLUE, PP_ALIGN.CENTER)
    add_textbox(slide1, Inches(1), Inches(4.5), Inches(11), Inches(0.8), "Key Milestones: Apr 6 - Jun 5, 2025", 18, False, RGBColor(0xA0, 0xC4, 0xE8), PP_ALIGN.CENTER)
    add_textbox(slide1, Inches(1), Inches(5.5), Inches(11), Inches(0.6), "Prepared by: [Advisory Team]", 16, False, RGBColor(0x80, 0xA0, 0xC0), PP_ALIGN.CENTER)

    # ============================================
    # SLIDE 2: Executive Summary
    # ============================================
    slide2 = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide2, WHITE)
    add_header_bar(slide2, "Executive Summary")
    add_footer(slide2, 2)

    # Key metrics box
    add_shape(slide2, Inches(0.5), Inches(1.6), Inches(12.333), Inches(0.6), LIGHT_BLUE, ACCENT_BLUE)
    add_textbox(slide2, Inches(0.8), Inches(1.65), Inches(11), Inches(0.5), "Transaction Timeline: 60 days from CIM release to signing | 4 key phases | 8 workstreams", 14, True, NAVY)

    # Summary points
    summary_items = [
        "Transaction Overview: Sell-side M&A process with accelerated 60-day timeline from initial CIM release to definitive agreement signing",
        "Key Milestones: CIM Release (Apr 6) → IOI Submission (Apr 24) → LOI (May 8) → Signing (Jun 5)",
        "Data Room Access: Partial access granted post-IOI (Apr 27), full access post-LOI (May 11)",
        "Exclusivity Period: Begins upon LOI acceptance (May 8), enabling comprehensive due diligence",
        "Critical Path: Financial, commercial, legal, operational, and governance workstreams running in parallel during exclusivity",
        "Risk Focus: Early identification of potential deal-breakers in tax, litigation, and environmental matters"
    ]
    add_bullet_list(slide2, Inches(0.8), Inches(2.5), Inches(11.5), Inches(4.2), summary_items, 12, DARK_GRAY, Pt(10))

    # ============================================
    # SLIDE 3: Key Milestones
    # ============================================
    slide3 = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide3, WHITE)
    add_header_bar(slide3, "Key Milestones & Timeline")
    add_footer(slide3, 3)

    milestones = [
        ("Apr 6", "CIM Release", "Confidential Information Memorandum distributed to potential buyers", ACCENT_BLUE),
        ("Apr 24", "IOI Submission", "Buyers submit Indicative Offers with preliminary valuation ranges", NAVY),
        ("Apr 27", "Partial Data Room", "Limited access granted to qualified buyers for initial diligence", DARK_BLUE),
        ("May 8", "LOI Submission", "Non-binding Letter of Intent from shortlisted buyers", ORANGE),
        ("May 11", "Full Data Room", "Complete access to financial, legal, and operational documents", GREEN),
        ("Jun 5", "Signing", "Definitive agreement execution, closing expected 30-60 days later", RED)
    ]

    y_start = 1.7
    for i, (date, title, desc, color) in enumerate(milestones):
        y = y_start + (i * 0.85)
        # Date box
        date_shape = add_shape(slide3, Inches(0.8), Inches(y), Inches(1.2), Inches(0.7), color)
        add_textbox(slide3, Inches(0.8), Inches(y + 0.1), Inches(1.2), Inches(0.5), date, 14, True, WHITE, PP_ALIGN.CENTER)
        # Title
        add_textbox(slide3, Inches(2.3), Inches(y + 0.05), Inches(2.5), Inches(0.4), title, 14, True, NAVY)
        # Description
        add_textbox(slide3, Inches(5), Inches(y + 0.05), Inches(7.5), Inches(0.6), desc, 11, False, DARK_GRAY)
        # Connector line
        if i < len(milestones) - 1:
            add_shape(slide3, Inches(1.35), Inches(y + 0.7), Inches(0.05), Inches(0.15), LIGHT_GRAY)

    # ============================================
    # SLIDE 4: Financial & Commercial Diligence
    # ============================================
    slide4 = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide4, WHITE)
    add_header_bar(slide4, "Financial & Commercial Diligence")
    add_footer(slide4, 4)

    # Financial workstream
    add_shape(slide4, Inches(0.5), Inches(1.6), Inches(6), Inches(0.5), NAVY)
    add_textbox(slide4, Inches(0.7), Inches(1.62), Inches(5.5), Inches(0.45), "Financial Diligence", 16, True, WHITE)

    financial_items = [
        "• Historical financial statement analysis (3-5 years)",
        "• Quality of earnings (QoE) review and adjustments",
        "• Working capital normalization and trends",
        "• Debt and capital structure analysis",
        "• Forecast accuracy and assumption validation",
        "• Tax compliance and potential exposures",
        "• Pension and post-retirement benefit obligations"
    ]
    add_bullet_list(slide4, Inches(0.7), Inches(2.3), Inches(5.5), Inches(3.5), financial_items, 11, DARK_GRAY, Pt(6))

    # Commercial workstream
    add_shape(slide4, Inches(6.8), Inches(1.6), Inches(6), Inches(0.5), ACCENT_BLUE)
    add_textbox(slide4, Inches(7), Inches(1.62), Inches(5.5), Inches(0.45), "Commercial Diligence", 16, True, WHITE)

    commercial_items = [
        "• Revenue concentration and customer retention analysis",
        "• Market size, growth rates, and competitive positioning",
        "• Key customer and supplier contract reviews",
        "• Pricing power and margin trends by segment",
        "• Pipeline and backlog analysis",
        "• Go-to-market strategy assessment",
        "• Synergy identification and validation"
    ]
    add_bullet_list(slide4, Inches(7), Inches(2.3), Inches(5.5), Inches(3.5), commercial_items, 11, DARK_GRAY, Pt(6))

    # Key items box
    add_shape(slide4, Inches(0.5), Inches(5.2), Inches(12.333), Inches(1.6), LIGHT_BLUE, ACCENT_BLUE)
    add_textbox(slide4, Inches(0.8), Inches(5.25), Inches(11), Inches(0.4), "Key Financial & Commercial Items Requiring Attention", 14, True, NAVY)
    key_items = [
        "→ Revenue recognition policies and any changes in accounting methods",
        "→ Related-party transactions and non-recurring items requiring add-backs",
        "→ Customer concentration risk (any single customer >10% of revenue)",
        "→ Channel stuffing or unusual quarter-end revenue spikes"
    ]
    add_bullet_list(slide4, Inches(0.8), Inches(5.7), Inches(11), Inches(1.1), key_items, 11, DARK_GRAY, Pt(4))

    # ============================================
    # SLIDE 5: Legal, Operational & Governance Diligence
    # ============================================
    slide5 = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide5, WHITE)
    add_header_bar(slide5, "Legal, Operational & Governance Diligence")
    add_footer(slide5, 5)

    # Legal column
    add_shape(slide5, Inches(0.5), Inches(1.6), Inches(3.8), Inches(0.5), NAVY)
    add_textbox(slide5, Inches(0.7), Inches(1.62), Inches(3.3), Inches(0.45), "Legal Diligence", 14, True, WHITE)

    legal_items = [
        "• Corporate structure and cap table",
        "• Material contracts and obligations",
        "• Litigation and regulatory matters",
        "• Intellectual property portfolio",
        "• Employment agreements and benefits",
        "• Environmental compliance (if applicable)",
        "• Data privacy and cybersecurity"
    ]
    add_bullet_list(slide5, Inches(0.7), Inches(2.3), Inches(3.5), Inches(3.5), legal_items, 10, DARK_GRAY, Pt(5))

    # Operational column
    add_shape(slide5, Inches(4.6), Inches(1.6), Inches(3.8), Inches(0.5), ACCENT_BLUE)
    add_textbox(slide5, Inches(4.8), Inches(1.62), Inches(3.3), Inches(0.45), "Operational Diligence", 14, True, WHITE)

    operational_items = [
        "• IT systems and technology stack",
        "• Supply chain resilience",
        "• Manufacturing/facility condition",
        "• Key personnel and org structure",
        "• ESG initiatives and metrics",
        "• Insurance coverage review",
        "• Real estate leases and ownership"
    ]
    add_bullet_list(slide5, Inches(4.8), Inches(2.3), Inches(3.5), Inches(3.5), operational_items, 10, DARK_GRAY, Pt(5))

    # Governance column
    add_shape(slide5, Inches(8.7), Inches(1.6), Inches(4.1), Inches(0.5), DARK_BLUE)
    add_textbox(slide5, Inches(8.9), Inches(1.62), Inches(3.6), Inches(0.45), "Governance Diligence", 14, True, WHITE)

    governance_items = [
        "• Board composition and charters",
        "• Related-party transaction history",
        "• Internal controls (SOX if applicable)",
        "• Shareholder agreements",
        "• Change of control provisions",
        "• Regulatory approvals needed",
        "• Antitrust filing requirements"
    ]
    add_bullet_list(slide5, Inches(8.9), Inches(2.3), Inches(3.6), Inches(3.5), governance_items, 10, DARK_GRAY, Pt(5))

    # Critical legal items
    add_shape(slide5, Inches(0.5), Inches(5.2), Inches(12.333), Inches(1.6), LIGHT_BLUE, ACCENT_BLUE)
    add_textbox(slide5, Inches(0.8), Inches(5.25), Inches(11), Inches(0.4), "Critical Legal & Governance Items Requiring Attention", 14, True, NAVY)
    critical_items = [
        "→ Pending or threatened litigation with potential material financial impact",
        "→ Change of control clauses in key customer/supplier contracts",
        "→ Unresolved environmental liabilities or regulatory investigations",
        "→ IP ownership disputes or reliance on third-party licenses"
    ]
    add_bullet_list(slide5, Inches(0.8), Inches(5.7), Inches(11), Inches(1.1), critical_items, 11, DARK_GRAY, Pt(4))

    # ============================================
    # SLIDE 6: Timeline Phases
    # ============================================
    slide6 = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide6, WHITE)
    add_header_bar(slide6, "Timeline Phases & Workstream Dependencies")
    add_footer(slide6, 6)

    phases = [
        ("Phase 1: Pre-CIM", "Apr 1-5", "Preparation", "Finalize CIM, prepare data room, identify buyer universe, coordinate with legal counsel", ACCENT_BLUE),
        ("Phase 2: Pre-IOI", "Apr 6-23", "Marketing", "Distribute CIM, conduct buyer presentations, manage Q&A process, track buyer interest", NAVY),
        ("Phase 3: Pre-LOI", "Apr 24-May 7", "Evaluation", "Evaluate IOIs, select shortlist, negotiate exclusivity, prepare partial data room access", DARK_BLUE),
        ("Phase 4: Exclusivity", "May 8-Jun 5", "Deep Diligence", "Full diligence across all workstreams, negotiate LOI terms, draft definitive agreements", ORANGE)
    ]

    y_start = 1.6
    for i, (phase, dates, focus, activities, color) in enumerate(phases):
        y = y_start + (i * 1.3)
        # Phase box
        phase_shape = add_shape(slide6, Inches(0.5), Inches(y), Inches(2.2), Inches(1.1), color)
        add_textbox(slide6, Inches(0.6), Inches(y + 0.05), Inches(2), Inches(0.4), phase, 13, True, WHITE)
        add_textbox(slide6, Inches(0.6), Inches(y + 0.45), Inches(2), Inches(0.3), dates, 11, False, LIGHT_BLUE)
        # Focus area
        add_textbox(slide6, Inches(3), Inches(y + 0.05), Inches(1.5), Inches(0.3), "Focus:", 11, True, NAVY)
        add_textbox(slide6, Inches(3), Inches(y + 0.3), Inches(1.5), Inches(0.3), focus, 11, False, DARK_GRAY)
        # Activities
        add_textbox(slide6, Inches(5), Inches(y + 0.05), Inches(1.5), Inches(0.3), "Key Activities:", 11, True, NAVY)
        add_textbox(slide6, Inches(5), Inches(y + 0.3), Inches(7.5), Inches(0.8), activities, 10, False, DARK_GRAY)
        # Arrow connector
        if i < len(phases) - 1:
            add_textbox(slide6, Inches(1.5), Inches(y + 1.15), Inches(0.5), Inches(0.2), "→", 16, True, ACCENT_BLUE, PP_ALIGN.CENTER)

    # Dependencies note
    add_shape(slide6, Inches(0.5), Inches(6.5), Inches(12.333), Inches(0.4), LIGHT_GRAY)
    add_textbox(slide6, Inches(0.8), Inches(6.52), Inches(11), Inches(0.35), "Dependency: Each phase gates on completion of prior phase | Exclusivity phase enables full data room and definitive agreement drafting", 10, False, DARK_GRAY)

    # ============================================
    # SLIDE 7: Risks & Mitigation
    # ============================================
    slide7 = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide7, WHITE)
    add_header_bar(slide7, "Key Risks & Mitigation Strategies")
    add_footer(slide7, 7)

    # Risk table header
    add_shape(slide7, Inches(0.5), Inches(1.6), Inches(12.333), Inches(0.5), NAVY)
    add_textbox(slide7, Inches(0.7), Inches(1.62), Inches(3), Inches(0.45), "Risk Category", 13, True, WHITE)
    add_textbox(slide7, Inches(4), Inches(1.62), Inches(4), Inches(0.45), "Potential Impact", 13, True, WHITE)
    add_textbox(slide7, Inches(8.5), Inches(1.62), Inches(4), Inches(0.45), "Mitigation Strategy", 13, True, WHITE)

    risks = [
        ("Due Diligence Findings", "Unexpected liabilities or quality of earnings issues could reduce valuation or kill deal", "Pre-emptive internal review; prepare QoE adjustments in advance; identify and disclose material issues early"),
        ("Buyer Financing Risk", "LOI buyer unable to secure acquisition financing, causing deal failure", "Verify financing commitments pre-signing; require proof of funds with LOI; identify backup buyers"),
        ("Competing Bids", "Multiple buyers could accelerate timeline or inflate expectations unrealistically", "Manage buyer expectations on timeline; maintain disciplined process; avoid overmarketing"),
        ("Data Room Leaks", "Confidential information leakage could compromise competitive position", "Strict NDA enforcement; watermark documents; track access logs; limit sensitive info initially"),
        ("Regulatory Approval Delays", "Antitrust or sector-specific approvals could delay or block closing", "Pre-filing regulatory consultation; prepare merger control filings early; identify remedies in advance"),
        ("Key Employee Retention", "Talent flight during process could impact business value", "Prepare retention bonus plans; identify critical personnel; communicate post-deal strategy to team")
    ]

    y_start = 2.2
    for i, (risk, impact, mitigation) in enumerate(risks):
        y = y_start + (i * 0.75)
        row_color = LIGHT_GRAY if i % 2 == 0 else WHITE
        add_shape(slide7, Inches(0.5), Inches(y), Inches(12.333), Inches(0.7), row_color)
        add_textbox(slide7, Inches(0.7), Inches(y + 0.05), Inches(3), Inches(0.6), risk, 11, True, NAVY)
        add_textbox(slide7, Inches(4), Inches(y + 0.05), Inches(4), Inches(0.6), impact, 10, False, DARK_GRAY)
        add_textbox(slide7, Inches(8.5), Inches(y + 0.05), Inches(4), Inches(0.6), mitigation, 10, False, DARK_GRAY)

    # ============================================
    # SLIDE 8: Next Steps
    # ============================================
    slide8 = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide8, WHITE)
    add_header_bar(slide8, "Next Steps & Action Items")
    add_footer(slide8, 8)

    # Immediate actions
    add_shape(slide8, Inches(0.5), Inches(1.6), Inches(6), Inches(0.5), NAVY)
    add_textbox(slide8, Inches(0.7), Inches(1.62), Inches(5.5), Inches(0.45), "Immediate Actions (This Week)", 16, True, WHITE)

    immediate_items = [
        "✓ Finalize and approve CIM content with management team",
        "✓ Complete data room organization and document indexing",
        "✓ Execute NDAs with target buyer list and distribute CIM",
        "✓ Schedule buyer presentation meetings for Week of Apr 7",
        "✓ Prepare Q&A response templates for common buyer questions",
        "✓ Coordinate with legal counsel on IOI evaluation criteria"
    ]
    add_bullet_list(slide8, Inches(0.7), Inches(2.3), Inches(5.5), Inches(3), immediate_items, 12, DARK_GRAY, Pt(8))

    # Near-term priorities
    add_shape(slide8, Inches(6.8), Inches(1.6), Inches(6), Inches(0.5), ACCENT_BLUE)
    add_textbox(slide8, Inches(7), Inches(1.62), Inches(5.5), Inches(0.45), "Near-Term Priorities (Next 2 Weeks)", 16, True, WHITE)

    near_term_items = [
        "→ Monitor buyer engagement levels and schedule follow-up meetings",
        "→ Prepare preliminary IOI evaluation framework and scoring matrix",
        "→ Begin internal coordination for partial data room access setup",
        "→ Draft exclusivity agreement terms and LOI template",
        "→ Identify and engage key advisors (legal, tax, environmental)",
        "→ Develop closing checklist and anticipated timeline for signing"
    ]
    add_bullet_list(slide8, Inches(7), Inches(2.3), Inches(5.5), Inches(3), near_term_items, 12, DARK_GRAY, Pt(8))

    # Key contacts box
    add_shape(slide8, Inches(0.5), Inches(5.2), Inches(12.333), Inches(1.6), LIGHT_BLUE, ACCENT_BLUE)
    add_textbox(slide8, Inches(0.8), Inches(5.25), Inches(11), Inches(0.4), "Key Stakeholders & Communication Protocol", 14, True, NAVY)
    stakeholder_items = [
        "• Seller Management Team: Daily updates during exclusivity; weekly updates during marketing phase",
        "• Investment Banker: Lead process coordination; IOI/LOI negotiation; buyer communications",
        "• Legal Counsel: Document preparation; contract review; regulatory filing management",
        "• External Advisors: QoE provider, environmental consultant, IP specialist as needed"
    ]
    add_bullet_list(slide8, Inches(0.8), Inches(5.7), Inches(11), Inches(1.1), stakeholder_items, 11, DARK_GRAY, Pt(4))

    # Save presentation
    output_path = "/home/agent/workspace/banker_workspace/deliverables/MA_SellSide_DueDiligence_Summary.pptx"
    prs.save(output_path)
    print(f"PowerPoint saved to: {output_path}")

if __name__ == "__main__":
    create_presentation()