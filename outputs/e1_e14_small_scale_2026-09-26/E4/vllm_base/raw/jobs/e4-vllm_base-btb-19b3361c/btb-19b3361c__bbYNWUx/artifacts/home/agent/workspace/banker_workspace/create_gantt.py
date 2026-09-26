import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime, timedelta

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "M&A Sell-Side Timeline"

# ── Color palette ──
PHASE_FILLS = {
    "Pre-CIM / Preparation": PatternFill("solid", fgColor="4472C4"),
    "Pre-IOI":               PatternFill("solid", fgColor="548235"),
    "Pre-LOI":               PatternFill("solid", fgColor="BF8F00"),
    "Exclusivity":           PatternFill("solid", fgColor="C00000"),
    "Key Milestone":         PatternFill("solid", fgColor="404040"),
}
CATEGORY_FILLS = {
    "Deal Preparation":            PatternFill("solid", fgColor="D6E4F0"),
    "Marketing & Buyer Engagement": PatternFill("solid", fgColor="E2EFDA"),
    "Financial Diligence":         PatternFill("solid", fgColor="FFF2CC"),
    "Commercial Diligence":        PatternFill("solid", fgColor="FCE4D6"),
    "Legal & Regulatory":          PatternFill("solid", fgColor="E4DFEC"),
    "Operational Diligence":       PatternFill("solid", fgColor="D9E1F2"),
    "Transaction & Closing":       PatternFill("solid", fgColor="F2DCDB"),
    "Governance & Cap Table":      PatternFill("solid", fgColor="E0E0E0"),
}

# ── Styles ──
HEADER_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
TITLE_FONT = Font(name="Calibri", bold=True, size=16, color="1F4E79")
PHASE_FONT = Font(name="Calibri", bold=True, size=10, color="FFFFFF")
TASK_FONT = Font(name="Calibri", size=10)
MILESTONE_FONT = Font(name="Calibri", bold=True, size=10, color="C00000")
THIN_BORDER = Border(
    left=Side(style="thin", color="B4C6E7"),
    right=Side(style="thin", color="B4C6E7"),
    top=Side(style="thin", color="B4C6E7"),
    bottom=Side(style="thin", color="B4C6E7"),
)
MEDIUM_BORDER = Border(
    left=Side(style="medium", color="1F4E79"),
    right=Side(style="medium", color="1F4E79"),
    top=Side(style="medium", color="1F4E79"),
    bottom=Side(style="medium", color="1F4E79"),
)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)

# ── Timeline data ──
# (phase, category, task_name, start_date, duration_days, dependency, deliverables_notes)
tasks = [
    # PHASE 1: PRE-CIM / PREPARATION (Mar 16 - Apr 5)
    ("Pre-CIM / Preparation", "Deal Preparation", "Engagement letter execution & team onboarding", datetime(2026, 3, 16), 3, None,
     "Banker team, legal counsel, and advisors engaged"),
    ("Pre-CIM / Preparation", "Deal Preparation", "Management interview & business overview sessions", datetime(2026, 3, 19), 5, "Engagement letter execution & team onboarding",
     "Deep-dive sessions with CEO, CFO, and key executives"),
    ("Pre-CIM / Preparation", "Deal Preparation", "CIM (Confidential Information Memorandum) drafting", datetime(2026, 3, 23), 9, "Management interview & business overview sessions",
     "Comprehensive CIM covering business overview, financials, growth drivers, market position"),
    ("Pre-CIM / Preparation", "Deal Preparation", "Financial model build & sanity check", datetime(2026, 3, 23), 9, "Management interview & business overview sessions",
     "3-statement model, valuation scenarios, sensitivity analysis"),
    ("Pre-CIM / Preparation", "Deal Preparation", "Datapack compilation & quality review", datetime(2026, 3, 26), 7, "Financial model build & sanity check",
     "Historical financials, KPIs, customer/contributor data"),
    ("Pre-CIM / Preparation", "Deal Preparation", "Investor target list development & approval", datetime(2026, 3, 30), 4, "Engagement letter execution & team onboarding",
     "Identify 5-8 strategic and financial buyers"),
    ("Pre-CIM / Preparation", "Legal & Regulatory", "NDA template preparation", datetime(2026, 3, 23), 3, None,
     "Standard NDA for buyer outreach"),
    ("Pre-CIM / Preparation", "Legal & Regulatory", "Initial data room organization (virtual)", datetime(2026, 3, 26), 7, "NDA template preparation",
     "Organize corporate records, cap table, material contracts"),
    ("Pre-CIM / Preparation", "Governance & Cap Table", "Cap table analysis & clean-up", datetime(2026, 3, 30), 4, None,
     "Current cap table, option pool, investor rights analysis"),
    ("Pre-CIM / Preparation", "Governance & Cap Table", "Identify potential deal-breaker items in cap table", datetime(2026, 4, 1), 3, "Cap table analysis & clean-up",
     "ROFR, drag-along, tag-along, liquidation preferences"),

    # KEY MILESTONE: CIM & Datapack Release (Apr 6)
    ("Key Milestone", "Deal Preparation", "*** CIM & Datapack Release to Prospective Buyers ***", datetime(2026, 4, 6), 1,
     "CIM drafting; Datapack compilation; Investor target list approval",
     "NDA execution required; CIM and datapack distributed"),

    # PHASE 2: PRE-IOI (Apr 6 - Apr 23)
    ("Pre-IOI", "Marketing & Buyer Engagement", "Buyer outreach & NDA execution", datetime(2026, 4, 6), 14,
     "*** CIM & Datapack Release to Prospective Buyers ***",
     "Initial outreach calls, NDA signing, CIM distribution"),
    ("Pre-IOI", "Marketing & Buyer Engagement", "Management presentation preparation", datetime(2026, 4, 8), 7, "Buyer outreach & NDA execution",
     "Refine CIM into management presentation deck"),
    ("Pre-IOI", "Marketing & Buyer Engagement", "Management meetings & roadshow (initial round)", datetime(2026, 4, 13), 8, "Management presentation preparation",
     "Introductory meetings with shortlisted buyers; high-level model guidance"),
    ("Pre-IOI", "Deal Preparation", "High-level model guidance to buyers", datetime(2026, 4, 13), 8, "Financial model build & sanity check",
     "Provide model assumptions, key drivers, and guidance document"),
    ("Pre-IOI", "Legal & Regulatory", "Q&A log setup & tracking", datetime(2026, 4, 13), 11, "Buyer outreach & NDA execution",
     "Centralized tracker for all buyer questions"),
    ("Pre-IOI", "Deal Preparation", "Competitive process framework setup", datetime(2026, 4, 6), 10,
     "*** CIM & Datapack Release to Prospective Buyers ***",
     "Process letter, bid matrix, timeline communication to buyers"),

    # KEY MILESTONE: IOI (Apr 24)
    ("Key Milestone", "Marketing & Buyer Engagement", "*** IOI Submission Deadline ***", datetime(2026, 4, 24), 1,
     "Buyer outreach; Management meetings & roadshow",
     "Non-binding IOIs received from all 5-8 parties; price range and structure"),

    # PHASE 3: PRE-LOI (Apr 24 - May 7)
    ("Pre-LOI", "Marketing & Buyer Engagement", "IOI evaluation & buyer shortlisting", datetime(2026, 4, 24), 5,
     "*** IOI Submission Deadline ***",
     "Scorecard evaluation: price, certainty, terms, strategic fit"),
    ("Pre-LOI", "Marketing & Buyer Engagement", "Buyer feedback & pricing analysis", datetime(2026, 4, 29), 4, "IOI evaluation & buyer shortlisting",
     "Summarize IOI terms, pricing, and conditions for management"),
    ("Pre-LOI", "Marketing & Buyer Engagement", "Shortlist notification & LOI invitation", datetime(2026, 4, 29), 2, "Buyer feedback & pricing analysis",
     "Notify 5-8 parties of invitation to submit LOI"),
    ("Pre-LOI", "Marketing & Buyer Engagement", "Management meetings (LOI round) & site visits", datetime(2026, 4, 30), 5, "Shortlist notification & LOI invitation",
     "Deeper management sessions, facility tours for shortlisted buyers"),

    # KEY MILESTONE: Partial Data Room (Apr 27)
    ("Key Milestone", "Financial Diligence", "*** Partial Data Room Access for Shortlisted Buyers ***", datetime(2026, 4, 27), 1,
     "Initial data room org; IOI evaluation & buyer shortlisting",
     "Access granted to shortlisted buyers only"),

    ("Pre-LOI", "Financial Diligence", "QoE (Quality of Earnings) preparation", datetime(2026, 4, 27), 10,
     "*** Partial Data Room Access for Shortlisted Buyers ***",
     "Engage QoE advisor; normalize EBITDA; working capital analysis"),
    ("Pre-LOI", "Financial Diligence", "Financial model update with Q1 actuals", datetime(2026, 4, 27), 7, "QoE preparation",
     "Incorporate Q1 2026 actuals into financial model"),
    ("Pre-LOI", "Financial Diligence", "Budget driver analysis & 2025 budget prep", datetime(2026, 4, 29), 6, "Financial model update with Q1 actuals",
     "Detail revenue drivers, cost structure, margin assumptions for 2025 budget"),
    ("Pre-LOI", "Operational Diligence", "Census data & org structure compilation", datetime(2026, 4, 27), 7,
     "*** Partial Data Room Access for Shortlisted Buyers ***",
     "Employee headcount, compensation, benefits, org chart, key personnel"),
    ("Pre-LOI", "Operational Diligence", "Raw customer data & contract renewal status", datetime(2026, 4, 27), 7,
     "*** Partial Data Room Access for Shortlisted Buyers ***",
     "Customer concentration, churn rates, contract terms, renewal pipeline"),
    ("Pre-LOI", "Legal & Regulatory", "Off-the-shelf legal documents preparation", datetime(2026, 4, 27), 7,
     "Initial data room organization (virtual)",
     "SPA template, disclosure schedule, corp docs, material contracts"),
    ("Pre-LOI", "Legal & Regulatory", "Q&A response & data room updates", datetime(2026, 4, 29), 6, "Q&A log setup & tracking",
     "Address buyer Q&A, update data room with new materials"),

    # KEY MILESTONE: LOI (May 8)
    ("Key Milestone", "Marketing & Buyer Engagement", "*** LOI Submission Deadline ***", datetime(2026, 5, 8), 1,
     "Management meetings (LOI round); IOI evaluation",
     "Binding LOIs from 5-8 parties; valuation, structure, conditions"),

    # KEY MILESTONE: Full Data Room (May 11)
    ("Key Milestone", "Financial Diligence", "*** Full Data Room Access (All LOI Parties) ***", datetime(2026, 5, 11), 1,
     "*** LOI Submission Deadline ***",
     "Full data room opened to all LOI-submitting parties"),

    # PHASE 4: EXCLUSIVITY (May 8 - Jun 26, post-signing Jun 5)
    ("Exclusivity", "Financial Diligence", "QoE diligence & buyer review", datetime(2026, 5, 8), 25,
     "*** LOI Submission Deadline ***",
     "QoE advisor finalizes report; buyer financial due diligence"),
    ("Exclusivity", "Financial Diligence", "Tax diligence (historical & projected)", datetime(2026, 5, 8), 25,
     "*** LOI Submission Deadline ***",
     "Engage tax advisor; review historical filings; identify tax exposures"),
    ("Exclusivity", "Financial Diligence", "Working capital normalization analysis", datetime(2026, 5, 11), 20,
     "*** Full Data Room Access (All LOI Parties) ***",
     "Define target working capital; analyze historical trends"),
    ("Exclusivity", "Financial Diligence", "Final financial model & deal projections", datetime(2026, 5, 11), 20,
     "Full data room access",
     "Updated projections incorporating all diligence findings"),

    ("Exclusivity", "Legal & Regulatory", "Legal diligence & due diligence review", datetime(2026, 5, 8), 25,
     "*** LOI Submission Deadline ***",
     "Legal counsel reviews all contracts, litigation, IP, compliance"),
    ("Exclusivity", "Legal & Regulatory", "Regulatory review & antitrust assessment", datetime(2026, 5, 11), 18,
     "*** Full Data Room Access (All LOI Parties) ***",
     "Hart-Scott-Rodino analysis; industry-specific regulatory review"),
    ("Exclusivity", "Legal & Regulatory", "SPA negotiation & drafting", datetime(2026, 5, 15), 18,
     "Legal diligence & due diligence review",
     "Share purchase agreement drafting, negotiation, and finalization"),
    ("Exclusivity", "Legal & Regulatory", "Disclosure schedule preparation", datetime(2026, 5, 18), 15,
     "SPA negotiation & drafting",
     "Comprehensive disclosure schedule attached to SPA"),
    ("Exclusivity", "Legal & Regulatory", "Buyer diligence Q&A & responses", datetime(2026, 5, 8), 25,
     "*** LOI Submission Deadline ***",
     "Ongoing buyer Q&A, data room updates, supplemental requests"),

    ("Exclusivity", "Commercial Diligence", "Customer reference calls & feedback", datetime(2026, 5, 8), 20,
     "Raw customer data & contract renewal status",
     "Coordinate customer references; gather buyer feedback on customer base"),
    ("Exclusivity", "Commercial Diligence", "Competitive landscape update", datetime(2026, 5, 8), 15,
     "Competitive process framework setup",
     "Updated competitive analysis for SPA representations"),
    ("Exclusivity", "Commercial Diligence", "Commercial due diligence support", datetime(2026, 5, 11), 20,
     "*** Full Data Room Access (All LOI Parties) ***",
     "Support buyer CDD advisor; market sizing, growth trends"),

    ("Exclusivity", "Operational Diligence", "IT systems & cybersecurity review", datetime(2026, 5, 11), 20,
     "*** Full Data Room Access (All LOI Parties) ***",
     "IT infrastructure assessment, cybersecurity audit, data privacy review"),
    ("Exclusivity", "Operational Diligence", "ESG diligence (environmental, social, governance)", datetime(2026, 5, 11), 20,
     "*** Full Data Room Access (All LOI Parties) ***",
     "ESG risk assessment, compliance review, sustainability reporting"),
    ("Exclusivity", "Operational Diligence", "Capex analysis & capital expenditure review", datetime(2026, 5, 11), 20,
     "*** Full Data Room Access (All LOI Parties) ***",
     "Historical capex trends, future capex requirements, maintenance vs growth"),
    ("Exclusivity", "Operational Diligence", "Employee retention & transition planning", datetime(2026, 5, 18), 15,
     "Census data & org structure compilation",
     "Key employee retention strategies, transition plan for management"),

    ("Exclusivity", "Governance & Cap Table", "Final cap table review & investor consent", datetime(2026, 5, 11), 15,
     "Identify potential deal-breaker items in cap table",
     "Obtain investor consents, waive ROFR if applicable"),
    ("Exclusivity", "Governance & Cap Table", "Pension & benefits review", datetime(2026, 5, 11), 15,
     "Census data & org structure compilation",
     "Review pension obligations, employee benefit plans"),

    ("Exclusivity", "Transaction & Closing", "Selection of final buyer", datetime(2026, 5, 29), 1,
     "IOI evaluation; QoE diligence & buyer review",
     "Evaluate final bids, select preferred buyer based on price, certainty, terms"),
    ("Exclusivity", "Transaction & Closing", "Exclusivity period management", datetime(2026, 5, 29), 28,
     "Selection of final buyer",
     "Manage exclusive negotiation period with selected buyer"),
    ("Exclusivity", "Transaction & Closing", "Final SPA negotiation & execution", datetime(2026, 6, 1), 4,
     "SPA negotiation & drafting",
     "Final terms negotiation, SPA signing"),

    # KEY MILESTONE: Signing (Jun 5)
    ("Key Milestone", "Transaction & Closing", "*** Signing (SPA Execution) ***", datetime(2026, 6, 5), 1,
     "Final SPA negotiation; Exclusivity period management",
     "SPA signed; deal announced (if agreed); closing conditions defined"),

    # Post-Signing (Jun 5 - Jun 26)
    ("Exclusivity", "Transaction & Closing", "Closing conditions satisfaction", datetime(2026, 6, 5), 21,
     "*** Signing (SPA Execution) ***",
     "Satisfy all conditions precedent: regulatory approvals, financing, third-party consents"),
    ("Exclusivity", "Transaction & Closing", "Closing preparation & fund transfer", datetime(2026, 6, 22), 4,
     "Closing conditions satisfaction",
     "Final closing documents, fund transfer, share transfer"),
    ("Exclusivity", "Transaction & Closing", "Post-closing transition support", datetime(2026, 6, 26), 10,
     "Closing preparation & fund transfer",
     "Transition services agreement, 90-day support period"),
]

# ═══════════════════════════════════════════════════════════
# ROW LAYOUT
#   Row 1: Title
#   Row 2: Headers
#   Row 3+: Tasks
# ═══════════════════════════════════════════════════════════

# ── Title ──
ws.merge_cells("A1:H1")
title_cell = ws.cell(row=1, column=1, value="SELL-SIDE M&A DUE DILIGENCE TIMELINE")
title_cell.font = Font(name="Calibri", bold=True, size=16, color="1F4E79")
title_cell.alignment = CENTER

# No subtitle needed - title spans A1:H1

# ── Headers (row 2) ──
headers = ["Phase", "Category", "Workstream / Task", "Start Date", "Duration (Days)", "End Date", "Dependency", "Key Deliverables / Diligence Items"]
col_widths = [22, 22, 48, 14, 14, 14, 45, 52]

for col_idx, (header, width) in enumerate(zip(headers, col_widths), 1):
    cell = ws.cell(row=2, column=col_idx, value=header)
    cell.font = HEADER_FONT
    cell.fill = HEADER_FILL
    cell.alignment = CENTER
    cell.border = MEDIUM_BORDER
    ws.column_dimensions[get_column_letter(col_idx)].width = width

# ── Write task rows (start at row 3) ──
row = 3
for phase, category, task_name, start, duration, dependency, notes in tasks:
    end = start + timedelta(days=duration - 1)

    ws.cell(row=row, column=1, value=phase)
    ws.cell(row=row, column=2, value=category)
    ws.cell(row=row, column=3, value=task_name)
    ws.cell(row=row, column=4, value=start)
    ws.cell(row=row, column=4).number_format = "MM/DD/YYYY"
    ws.cell(row=row, column=5, value=duration)
    ws.cell(row=row, column=6, value=end)
    ws.cell(row=row, column=6).number_format = "MM/DD/YYYY"
    ws.cell(row=row, column=7, value=dependency if dependency else "None")
    ws.cell(row=row, column=8, value=notes)

    # Apply styles based on phase
    if phase == "Key Milestone":
        for c in range(1, 9):
            ws.cell(row=row, column=c).font = MILESTONE_FONT
            ws.cell(row=row, column=c).fill = PHASE_FILLS.get(phase, PatternFill())
    elif phase in PHASE_FILLS:
        ws.cell(row=row, column=1).font = PHASE_FONT
        ws.cell(row=row, column=1).fill = PHASE_FILLS[phase]
        ws.cell(row=row, column=1).alignment = CENTER
        ws.cell(row=row, column=2).fill = CATEGORY_FILLS.get(category, PatternFill())
        ws.cell(row=row, column=3).font = TASK_FONT
        ws.cell(row=row, column=3).alignment = LEFT
        ws.cell(row=row, column=4).alignment = CENTER
        ws.cell(row=row, column=5).alignment = CENTER
        ws.cell(row=row, column=6).alignment = CENTER
        ws.cell(row=row, column=7).font = TASK_FONT
        ws.cell(row=row, column=7).alignment = LEFT
        ws.cell(row=row, column=8).font = TASK_FONT
        ws.cell(row=row, column=8).alignment = LEFT

    for c in range(1, 9):
        ws.cell(row=row, column=c).border = THIN_BORDER

    row += 1

total_rows = row - 1  # last task row

# ═══════════════════════════════════════════════════════════
# GANTT CHART BARS (columns 9-35, one column per day)
# Chart period: Apr 6 (day 0) to Jun 26 (day 81) = 82 days
# ═══════════════════════════════════════════════════════════

# Set Gantt area column widths
for c in range(9, 36):
    ws.column_dimensions[get_column_letter(c)].width = 4

# Write Gantt date marker headers in row 2 (alongside main headers)
date_markers = [
    (9, "Apr 6"), (13, "Apr 20"), (17, "May 1"), (21, "May 15"),
    (25, "May 29"), (29, "Jun 5"), (33, "Jun 18"),
]
for col_idx, label in date_markers:
    cell = ws.cell(row=2, column=col_idx, value=label)
    cell.font = Font(name="Calibri", bold=True, size=8, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="1F4E79")
    cell.alignment = CENTER
    cell.border = THIN_BORDER

# Draw Gantt bars
chart_start = datetime(2026, 4, 6)
chart_span = 82  # Apr 6 to Jun 26

for r in range(3, total_rows + 1):
    phase = ws.cell(row=r, column=1).value
    start = ws.cell(row=r, column=4).value
    duration = ws.cell(row=r, column=5).value

    if not start or not duration:
        continue

    days_from_start = (start - chart_start).days

    # Only show tasks that overlap the chart period
    if days_from_start + duration <= 0 or days_from_start >= chart_span:
        continue

    # Calculate bar position
    bar_start_col = 9 + max(0, days_from_start)
    bar_end_col = 9 + min(days_from_start + duration, chart_span) - 1

    bar_start_col = max(bar_start_col, 9)
    bar_end_col = min(bar_end_col, 35)

    if bar_start_col > bar_end_col:
        continue

    fill = PHASE_FILLS.get(phase, PatternFill("solid", fgColor="B4C6E7"))

    # Fill individual cells
    for c in range(bar_start_col, bar_end_col + 1):
        cell = ws.cell(row=r, column=c)
        cell.fill = fill
        cell.border = THIN_BORDER

    # Merge for cleaner look (only if bar spans multiple columns)
    if bar_end_col - bar_start_col > 0:
        ws.merge_cells(start_row=r, start_column=bar_start_col, end_row=r, end_column=bar_end_col)
        merged_cell = ws.cell(row=r, column=bar_start_col)
        merged_cell.fill = fill
        merged_cell.border = THIN_BORDER

# ═══════════════════════════════════════════════════════════
# CONDITIONAL FORMATTING (dynamic updates)
# ═══════════════════════════════════════════════════════════

# Overdue tasks (end date < today) → red highlight
overdue_formula = f'AND($F3<TODAY(),$A3<>"Key Milestone",$A3<>"")'
red_fill = PatternFill("solid", fgColor="FF0000")
red_font = Font(name="Calibri", size=10, color="FFFFFF", bold=True)

ws.conditional_formatting.add(
    f"A3:H{total_rows}",
    FormulaRule(formula=[overdue_formula], fill=red_fill, font=red_font)
)

# Upcoming tasks (within 7 days) → yellow highlight
upcoming_formula = f'AND($F3>=TODAY(),$F3<=TODAY()+7,$A3<>"Key Milestone",$A3<>"")'
yellow_fill = PatternFill("solid", fgColor="FFFF00")
ws.conditional_formatting.add(
    f"A3:H{total_rows}",
    FormulaRule(formula=[upcoming_formula], fill=yellow_fill)
)

# ═══════════════════════════════════════════════════════════
# FORMATTING & LAYOUT
# ═══════════════════════════════════════════════════════════

# Freeze panes (freeze title + header)
ws.freeze_panes = "A3"

# Auto-filter
ws.auto_filter.ref = f"A2:H{total_rows}"

# Print setup
ws.sheet_properties.pageSetUpPr = openpyxl.worksheet.properties.PageSetupProperties(fitToPage=True)
ws.page_setup.fitToWidth = 1
ws.page_setup.fitToHeight = 0
ws.page_setup.orientation = "landscape"
ws.page_setup.paperSize = ws.PAPERSIZE_TABLOID
ws.page_setup.scale = 100

# ── Save ──
output_path = "/home/agent/workspace/banker_workspace/deliverables/MA_SellSide_DueDiligence_Timeline.xlsx"
wb.save(output_path)
print(f"Excel Gantt chart saved to {output_path}")
print(f"Total tasks: {total_rows - 2}")
print(f"Date range: Mar 16, 2026 - Jun 26, 2026")
print(f"Gantt chart columns: 9-35 (Apr 6 - Jun 26, 2026)")