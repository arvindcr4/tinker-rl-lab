from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import CellIsRule
from datetime import datetime, timedelta
import json

wb = Workbook()

# ============================================================
# SHEET 1: GANTT CHART
# ============================================================
ws = wb.active
ws.title = "Gantt Chart"

# --- Color palette ---
DARK_NAVY = PatternFill(start_color="1B2A4A", end_color="1B2A4A", fill_type="solid")
MEDIUM_BLUE = PatternFill(start_color="2E5090", end_color="2E5090", fill_type="solid")
LIGHT_BLUE = PatternFill(start_color="5B8DD5", end_color="5B8DD5", fill_type="solid")
PALE_BLUE = PatternFill(start_color="B8D4F0", end_color="B8D4F0", fill_type="solid")
WHITE_FILL = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
LIGHT_GRAY = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
YELLOW_MILESTONE = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")
GREEN_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
ORANGE_FILL = PatternFill(start_color="ED7D31", end_color="ED7D31", fill_type="solid")
RED_FILL = PatternFill(start_color="C00000", end_color="C00000", fill_type="solid")
TEAL_FILL = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
PURPLE_FILL = PatternFill(start_color="7030A0", end_color="7030A0", fill_type="solid")
GOLD_FILL = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
ROSE_FILL = PatternFill(start_color="E25822", end_color="E25822", fill_type="solid")
CYAN_FILL = PatternFill(start_color="00B0F0", end_color="00B0F0", fill_type="solid")

# --- Fonts ---
TITLE_FONT = Font(name="Calibri", size=18, bold=True, color="1B2A4A")
SECTION_FONT = Font(name="Calibri", size=12, bold=True, color="FFFFFF")
SUBSECTION_FONT = Font(name="Calibri", size=11, bold=True, color="1B2A4A")
NORMAL_FONT = Font(name="Calibri", size=10)
BOLD_FONT = Font(name="Calibri", size=10, bold=True)
HEADER_FONT = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
KEY_DATE_FONT = Font(name="Calibri", size=10, bold=True, color="1B2A4A")

# --- Alignment ---
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
LEFT_BOLD = Alignment(horizontal="left", vertical="center")

# --- Borders ---
THIN_BORDER = Border(
    left=Side(style="thin", color="B0B0B0"),
    right=Side(style="thin", color="B0B0B0"),
    top=Side(style="thin", color="B0B0B0"),
    bottom=Side(style="thin", color="B0B0B0"),
)

# ============================================================
# CONFIGURATION: Key dates and workstreams
# ============================================================
# Key dates
key_dates = {
    "Process Kickoff / CIM & Datapack Release": datetime(2025, 4, 6),
    "IOI Submission": datetime(2025, 4, 24),
    "Partial Data Room Access": datetime(2025, 4, 27),
    "LOI Submission": datetime(2025, 5, 8),
    "Full Data Room Access": datetime(2025, 5, 11),
    "Signing": datetime(2025, 6, 5),
}

# Workstreams: (name, start_date, duration_days, section, fill, dependencies, deliverables/description)
workstreams = [
    # --- MARKETING & PREPARATION (Pre-IOI) ---
    ("Prepare CIM & Marketing Materials", datetime(2025, 3, 17), 16, "Marketing & Preparation", MEDIUM_BLUE, [],
     "Confidential Information Memorandum, teaser, management summary"),
    ("Build / Refresh Financial Model", datetime(2025, 3, 17), 20, "Marketing & Preparation", MEDIUM_BLUE, [],
     "High-level guidance on model assumptions, 3-statement model"),
    ("Prepare Datapack", datetime(2025, 3, 24), 10, "Marketing & Preparation", MEDIUM_BLUE, [],
     "Historical financials, KPIs, market overview, competitive landscape"),
    ("Management Presentation Prep", datetime(2025, 3, 31), 6, "Marketing & Preparation", MEDIUM_BLUE, [],
     "Management slide deck for buyer meetings"),
    ("Management Training", datetime(2025, 4, 3), 3, "Marketing & Preparation", MEDIUM_BLUE, [],
     "Q&A prep, data room navigation training"),

    # --- BUYER IDENTIFICATION & OUTREACH (Pre-IOI) ---
    ("Identify & Prioritize Buyer List (8+ parties)", datetime(2025, 3, 17), 14, "Buyer Identification & Outreach", LIGHT_BLUE, [],
     "Long list of 8-12 potential buyers, prioritization matrix"),
    ("Prepare NDA Templates", datetime(2025, 3, 24), 5, "Buyer Identification & Outreach", LIGHT_BLUE, [],
     "Standard NDA, confidentiality agreements"),
    ("Send Teasers & NDAs", datetime(2025, 3, 31), 10, "Buyer Identification & Outreach", LIGHT_BLUE, ["Prepare NDA Templates"],
     "Outreach to all 8 parties, track responses"),
    ("Schedule Initial Management Meetings", datetime(2025, 4, 10), 10, "Buyer Identification & Outreach", LIGHT_BLUE, ["Send Teasers & NDAs"],
     "Coordinate management calendars, set up meetings with interested parties"),

    # --- IOI STAGE (Pre-LOI) ---
    ("Release CIM & Datapack to Buyers", datetime(2025, 4, 6), 1, "IOI Stage", GREEN_FILL, ["Prepare CIM & Marketing Materials", "Prepare Datapack", "Send Teasers & NDAs"],
     "CIM and datapack distributed to executed NDAs"),
    ("Pre-IOI Q&A Process", datetime(2025, 4, 6), 18, "IOI Stage", GREEN_FILL, ["Release CIM & Datapack"],
     "High-level guidance on model, answer initial buyer questions"),
    ("Open Partial Data Room", datetime(2025, 4, 27), 14, "IOI Stage", GREEN_FILL, ["Pre-IOI Q&A Process"],
     "Limited access: high-level financials, key contracts summary"),
    ("Manage Buyer Inquiries (Pre-IOI)", datetime(2025, 4, 6), 22, "IOI Stage", GREEN_FILL, ["Release CIM & Datapack"],
     "Track and respond to buyer questions, coordinate management responses"),
    ("IOI Evaluation & Scoring", datetime(2025, 4, 24), 14, "IOI Stage", GREEN_FILL, ["IOI Submission"],
     "Score and rank IOIs from 5-8 parties, prepare summary for client"),

    # --- LOI STAGE ---
    ("LOI Submission Deadline", datetime(2025, 5, 8), 1, "LOI Stage", ORANGE_FILL, ["IOI Evaluation & Scoring"],
     "Final LOIs due from shortlisted parties (5-8 parties)"),
    ("Open Full Data Room", datetime(2025, 5, 11), 25, "LOI Stage", ORANGE_FILL, ["LOI Submission Deadline"],
     "Full access: QoE financials, census, org structure, Q1 actuals, 2025 budget drivers, raw customer data with contract renewal statuses, off-the-shelf legal"),
    ("Provide In-Depth Financials to Buyers", datetime(2025, 5, 11), 20, "LOI Stage", ORANGE_FILL, ["Open Full Data Room"],
     "QoE-ready financials, census data, org structure, Q1 actuals, 2025 budget drivers"),
    ("Provide Customer Data & Contract Analysis", datetime(2025, 5, 11), 20, "LOI Stage", ORANGE_FILL, ["Open Full Data Room"],
     "Raw customer data with contract renewal statuses, churn analysis"),
    ("Provide Legal Documents (Off-the-Shelf)", datetime(2025, 5, 11), 15, "LOI Stage", ORANGE_FILL, ["Open Full Data Room"],
     "Off-the-shelf legal: SPA templates, employment agreements, IP assignments"),
    ("LOI Evaluation & Shortlisting", datetime(2025, 5, 8), 21, "LOI Stage", ORANGE_FILL, ["LOI Submission Deadline"],
     "Evaluate all LOIs, select 1 party for exclusivity"),
    ("Select Exclusive Buyer", datetime(2025, 5, 29), 1, "LOI Stage", ORANGE_FILL, ["LOI Evaluation & Shortlisting"],
     "Negotiate terms, select one party for 4-week exclusivity"),

    # --- EXCLUSIVITY PERIOD (Full Diligence) ---
    ("Execute Exclusivity Agreement", datetime(2025, 5, 29), 1, "Exclusivity Period", PURPLE_FILL, ["Select Exclusive Buyer"],
     "Exclusivity agreement signed with selected buyer"),
    ("Financial Diligence (QoE Support)", datetime(2025, 5, 12), 28, "Exclusivity Period", PURPLE_FILL, ["Open Full Data Room"],
     "Revenue recognition, working capital analysis, quality of earnings support"),
    ("Tax Diligence", datetime(2025, 5, 12), 28, "Exclusivity Period", PURPLE_FILL, ["Open Full Data Room"],
     "Historical tax returns, tax provision analysis, transfer pricing"),
    ("Legal Diligence", datetime(2025, 5, 12), 28, "Exclusivity Period", PURPLE_FILL, ["Open Full Data Room"],
     "Corporate structure, litigation review, material contracts, IP due diligence"),
    ("Capex Diligence", datetime(2025, 5, 12), 21, "Exclusivity Period", PURPLE_FILL, ["Open Full Data Room"],
     "Capital expenditure analysis, maintenance vs. expansion capex, future capex needs"),
    ("Regulatory Diligence", datetime(2025, 5, 12), 21, "Exclusivity Period", PURPLE_FILL, ["Open Full Data Room"],
     "Regulatory approvals needed, compliance review, industry-specific regulations"),
    ("Cap Table Diligence", datetime(2025, 5, 12), 14, "Exclusivity Period", PURPLE_FILL, ["Open Full Data Room"],
     "Capitalization table review, option pool, convertible securities"),
    ("IT Diligence", datetime(2025, 5, 12), 28, "Exclusivity Period", PURPLE_FILL, ["Open Full Data Room"],
     "Technology stack review, cybersecurity, data privacy, software licenses"),
    ("ESG Diligence", datetime(2025, 5, 12), 21, "Exclusivity Period", PURPLE_FILL, ["Open Full Data Room"],
     "Environmental compliance, social governance, sustainability reporting"),
    ("Manage Exclusivity Q&A", datetime(2025, 5, 12), 28, "Exclusivity Period", PURPLE_FILL, ["Execute Exclusivity Agreement"],
     "Coordinate and respond to all buyer diligence questions"),
    ("Negotiate SPA & Final Terms", datetime(2025, 5, 26), 10, "Exclusivity Period", PURPLE_FILL, ["Manage Exclusivity Q&A"],
     "Share Purchase Agreement negotiation, final price adjustment"),

    # --- CLOSING ---
    ("Signing", datetime(2025, 6, 5), 1, "Closing", GOLD_FILL, ["Negotiate SPA & Final Terms"],
     "Transaction signing - SPA executed"),
    ("Post-Signing / Closing Conditions", datetime(2025, 6, 5), 21, "Closing", GOLD_FILL, ["Signing"],
     "Satisfy closing conditions, regulatory filings, final adjustments"),
]

# ============================================================
# WRITE HEADER ROWS
# ============================================================
# Row 1: Title
ws.merge_cells("A1:K1")
ws["A1"] = "M&A Sell-Side Transaction — Deal Timeline & Process Timeline"
ws["A1"].font = TITLE_FONT
ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
ws.row_dimensions[1].height = 36

# Row 2: Subtitle
ws.merge_cells("A2:K2")
ws["A2"] = "Process Kickoff: April 6, 2025  |  Target Signing: June 5, 2025"
ws["A2"].font = Font(name="Calibri", size=11, italic=True, color="555555")
ws["A2"].alignment = Alignment(horizontal="left", vertical="center")
ws.row_dimensions[2].height = 24

# Row 3: Empty spacer
ws.row_dimensions[3].height = 8

# Row 4: Column headers
headers = [
    ("Task / Workstream", 38),
    ("Phase", 16),
    ("Dependencies", 22),
    ("Start Date", 14),
    ("Duration (Days)", 14),
    ("End Date", 14),
    ("Key Deliverables / Diligence Items", 40),
]

for col_idx, (header_text, col_width) in enumerate(headers, 1):
    cell = ws.cell(row=4, column=col_idx, value=header_text)
    cell.font = HEADER_FONT
    cell.fill = DARK_NAVY
    cell.alignment = CENTER
    cell.border = THIN_BORDER
    ws.column_dimensions[get_column_letter(col_idx)].width = col_width

# ============================================================
# WRITE WORKSTREAM ROWS
# ============================================================
current_section = ""
row = 5
section_row = 0

for i, (name, start, duration, section, fill, deps, deliverables) in enumerate(workstreams):
    # Section header
    if section != current_section:
        current_section = section
        section_row = row
        # Merge section header cells
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
        cell = ws.cell(row=row, column=1, value=section.upper())
        cell.font = SECTION_FONT
        cell.fill = DARK_NAVY
        cell.alignment = LEFT
        cell.border = THIN_BORDER
        for c in range(2, 8):
            ws.cell(row=row, column=c).fill = DARK_NAVY
            ws.cell(row=row, column=c).border = THIN_BORDER
        ws.row_dimensions[row].height = 24
        row += 1

    # Data row
    deps_str = ", ".join(deps) if deps else "—"
    
    # Alternate row shading
    row_fill = LIGHT_GRAY if (row % 2 == 0) else WHITE_FILL
    
    # Column A: Task name
    cell_a = ws.cell(row=row, column=1, value=name)
    cell_a.font = BOLD_FONT if any(d in name for d in ["Release", "Submission", "Signing", "Select", "Execute"]) else NORMAL_FONT
    cell_a.fill = row_fill
    cell_a.alignment = LEFT_BOLD
    cell_a.border = THIN_BORDER
    
    # Column B: Phase
    cell_b = ws.cell(row=row, column=2, value=section)
    cell_b.font = NORMAL_FONT
    cell_b.fill = row_fill
    cell_b.alignment = CENTER
    cell_b.border = THIN_BORDER
    
    # Column C: Dependencies
    cell_c = ws.cell(row=row, column=3, value=deps_str)
    cell_c.font = Font(name="Calibri", size=9, color="666666")
    cell_c.fill = row_fill
    cell_c.alignment = LEFT
    cell_c.border = THIN_BORDER
    
    # Column D: Start Date
    cell_d = ws.cell(row=row, column=4, value=start)
    cell_d.font = NORMAL_FONT
    cell_d.fill = row_fill
    cell_d.alignment = CENTER
    cell_d.border = THIN_BORDER
    cell_d.number_format = "MM/DD/YYYY"
    
    # Column E: Duration
    cell_e = ws.cell(row=row, column=5, value=duration)
    cell_e.font = NORMAL_FONT
    cell_e.fill = row_fill
    cell_e.alignment = CENTER
    cell_e.border = THIN_BORDER
    cell_e.number_format = "0"
    
    # Column F: End Date (formula: Start + Duration - 1)
    cell_f = ws.cell(row=row, column=6)
    cell_f.value = f'=D{row}+E{row}-1'
    cell_f.font = NORMAL_FONT
    cell_f.fill = row_fill
    cell_f.alignment = CENTER
    cell_f.border = THIN_BORDER
    cell_f.number_format = "MM/DD/YYYY"
    
    # Column G: Deliverables
    cell_g = ws.cell(row=row, column=7, value=deliverables)
    cell_g.font = Font(name="Calibri", size=9)
    cell_g.fill = row_fill
    cell_g.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    cell_g.border = THIN_BORDER
    
    # Apply fill color to task name cell for visual distinction
    cell_a.fill = fill
    
    # Milestone rows (duration = 1) get special formatting
    if duration == 1:
        cell_d.fill = YELLOW_MILESTONE
        cell_e.fill = YELLOW_MILESTONE
        cell_f.fill = YELLOW_MILESTONE
        cell_b.fill = YELLOW_MILESTONE
        cell_c.fill = YELLOW_MILESTONE
        cell_g.fill = YELLOW_MILESTONE
        row_fill = YELLOW_MILESTONE
    
    ws.row_dimensions[row].height = 28
    row += 1

# ============================================================
# KEY DATES MILESTONE TABLE (below workstreams)
# ============================================================
row += 2
ws.merge_cells(f"A{row}:G{row}")
cell = ws.cell(row=row, column=1, value="KEY TRANSACTION MILESTONES")
cell.font = SECTION_FONT
cell.fill = DARK_NAVY
cell.alignment = LEFT
for c in range(1, 8):
    ws.cell(row=row, column=c).fill = DARK_NAVY
    ws.cell(row=row, column=c).border = THIN_BORDER
ws.row_dimensions[row].height = 24
row += 1

key_date_headers = ["Milestone", "", "", "Date", "", "", ""]
for col_idx, header_text in enumerate(key_date_headers, 1):
    if header_text:
        cell = ws.cell(row=row, column=col_idx, value=header_text)
        cell.font = HEADER_FONT
        cell.fill = MEDIUM_BLUE
        cell.alignment = CENTER
        cell.border = THIN_BORDER

row += 1
for milestone_name, milestone_date in key_dates.items():
    cell_a = ws.cell(row=row, column=1, value=milestone_name)
    cell_a.font = KEY_DATE_FONT
    cell_a.fill = WHITE_FILL
    cell_a.alignment = LEFT_BOLD
    cell_a.border = THIN_BORDER
    
    cell_d = ws.cell(row=row, column=4, value=milestone_date)
    cell_d.font = BOLD_FONT
    cell_d.fill = WHITE_FILL
    cell_d.alignment = CENTER
    cell_d.border = THIN_BORDER
    cell_d.number_format = "MM/DD/YYYY"
    
    for c in [2, 3, 5, 6, 7]:
        ws.cell(row=row, column=c).border = THIN_BORDER
        ws.cell(row=row, column=c).fill = WHITE_FILL
    
    ws.row_dimensions[row].height = 24
    row += 1

# ============================================================
# PHASE LEGEND
# ============================================================
row += 2
ws.merge_cells(f"A{row}:G{row}")
cell = ws.cell(row=row, column=1, value="PHASE LEGEND")
cell.font = SECTION_FONT
cell.fill = DARK_NAVY
cell.alignment = LEFT
for c in range(1, 8):
    ws.cell(row=row, column=c).fill = DARK_NAVY
    ws.cell(row=row, column=c).border = THIN_BORDER
ws.row_dimensions[row].height = 24
row += 1

legend_items = [
    ("Marketing & Preparation", MEDIUM_BLUE),
    ("Buyer Identification & Outreach", LIGHT_BLUE),
    ("IOI Stage", GREEN_FILL),
    ("LOI Stage", ORANGE_FILL),
    ("Exclusivity Period", PURPLE_FILL),
    ("Closing", GOLD_FILL),
    ("Milestone (1-day event)", YELLOW_MILESTONE),
]

for legend_name, legend_fill in legend_items:
    cell = ws.cell(row=row, column=1, value=legend_name)
    cell.font = NORMAL_FONT
    cell.fill = legend_fill
    cell.alignment = LEFT
    cell.border = THIN_BORDER
    for c in range(2, 8):
        ws.cell(row=row, column=c).border = THIN_BORDER
        ws.cell(row=row, column=c).fill = WHITE_FILL
    ws.row_dimensions[row].height = 20
    row += 1

# ============================================================
# DYNAMIC UPDATE INSTRUCTIONS
# ============================================================
row += 2
ws.merge_cells(f"A{row}:G{row}")
cell = ws.cell(row=row, column=1, value="HOW TO UPDATE THIS TIMELINE DYNAMICALLY")
cell.font = Font(name="Calibri", size=11, bold=True, color="1B2A4A")
ws.row_dimensions[row].height = 24
row += 1

instructions = [
    "1. Update the 'Start Date' column (Column D) as dates change.",
    "2. Update the 'Duration (Days)' column (Column E) if task timelines shift.",
    "3. The 'End Date' column (Column F) auto-calculates using the formula: Start Date + Duration - 1.",
    "4. All Gantt chart bars and dependent tasks will automatically adjust.",
    "5. Key Milestones (highlighted in gold) should be updated manually if dates change.",
    "6. This timeline is designed for the sell-side M&A process with 5-8 potential buyers.",
]

for instr in instructions:
    cell = ws.cell(row=row, column=1, value=instr)
    cell.font = Font(name="Calibri", size=9, color="555555")
    cell.alignment = LEFT
    ws.row_dimensions[row].height = 18
    row += 1

# ============================================================
# FREEZE PANES & PRINT SETUP
# ============================================================
ws.freeze_panes = "A5"
ws.sheet_properties.pageSetUpPr = None

ws.page_setup.orientation = "landscape"
ws.page_setup.fitToWidth = 1
ws.page_setup.fitToHeight = 0
ws.page_margins.left = 0.5
ws.page_margins.right = 0.5
ws.page_margins.top = 0.5
ws.page_margins.bottom = 0.5

# ============================================================
# SHEET 2: WORKSTREAM DETAILS
# ============================================================
ws2 = wb.create_sheet("Workstream Details")

# Title
ws2.merge_cells("A1:F1")
ws2["A1"] = "M&A Sell-Side — Workstream Details & Diligence Items"
ws2["A1"].font = TITLE_FONT
ws2["A1"].alignment = Alignment(horizontal="left", vertical="center")
ws2.row_dimensions[1].height = 36

ws2.merge_cells("A2:F2")
ws2["A2"] = "Detailed breakdown of each workstream, key deliverables, and expected timelines"
ws2["A2"].font = Font(name="Calibri", size=11, italic=True, color="555555")
ws2["A2"].alignment = Alignment(horizontal="left", vertical="center")

# Headers
detail_headers = [
    ("Phase / Workstream", 30),
    ("Timeline", 18),
    ("Key Activities", 45),
    ("Key Diligence Items", 45),
    ("Responsible Party", 20),
    ("Status", 12),
]

row = 4
for col_idx, (header_text, col_width) in enumerate(detail_headers, 1):
    cell = ws2.cell(row=row, column=col_idx, value=header_text)
    cell.font = HEADER_FONT
    cell.fill = DARK_NAVY
    cell.alignment = CENTER
    cell.border = THIN_BORDER
    ws2.column_dimensions[get_column_letter(col_idx)].width = col_width

ws2.row_dimensions[row].height = 28

# Workstream detail data
details = [
    # Marketing & Preparation
    ("Marketing & Preparation", "Mar 17 – Apr 6",
     "• Draft and finalize CIM\n• Build/refresh financial model\n• Prepare datapack with historicals\n• Prepare management presentation\n• Conduct management training",
     "• Historical 3-year financials\n• Revenue breakdown by segment\n• Customer concentration analysis\n• Market sizing and trends\n• Competitive positioning",
     "Investment Banker\nManagement", "Complete"),
    
    # Buyer Identification & Outreach
    ("Buyer Identification & Outreach", "Mar 17 – Apr 24",
     "• Identify 8-12 potential buyers\n• Prepare NDA templates\n• Send teasers and execute NDAs\n• Schedule initial management meetings\n• Track buyer interest levels",
     "• Buyer profile and fit analysis\n• Strategic vs. financial buyer assessment\n• Estimated valuation ranges\n• Process timing confirmation",
     "Investment Banker", "Complete"),
    
    # IOI Stage
    ("IOI Stage", "Apr 6 – May 8",
     "• Release CIM & datapack to buyers\n• Open partial data room (Apr 27)\n• Manage pre-IOI Q&A process\n• Respond to buyer inquiries\n• Evaluate and score IOIs",
     "• High-level financial guidance only\n• Key contract summaries\n• Revenue and customer metrics\n• Initial management meetings\n• Model guidance (high-level)",
     "Investment Banker\nManagement", "In Progress"),
    
    # LOI Stage
    ("LOI Stage", "Apr 24 – May 29",
     "• Receive and evaluate IOIs\n• Set LOI deadline (May 8)\n• Open full data room (May 11)\n• Provide in-depth financials\n• Provide customer data & legal docs\n• Evaluate LOIs and select exclusive buyer",
     "• QoE-ready financial statements\n• Census data and org structure\n• Q1 2025 actuals\n• 2025 budget drivers\n• Raw customer data with renewal statuses\n• Off-the-shelf legal documents\n• Employment agreements\n• IP assignment documents",
     "Investment Banker\nManagement\nLegal Counsel", "Pending"),
    
    # Exclusivity Period
    ("Exclusivity Period", "May 12 – Jun 5+",
     "• Execute exclusivity agreement\n• Full financial diligence (QoE)\n• Tax, legal, capex, regulatory diligence\n• Cap Table review\n• IT diligence\n• ESG diligence\n• Manage exclusivity Q&A\n• Negotiate SPA and final terms",
     "• Financial: Revenue recognition, working capital, QoE support\n• Tax: Historical returns, tax provisions, transfer pricing\n• Legal: Corporate structure, litigation, material contracts, IP\n• Capex: Maintenance vs. expansion, future needs\n• Regulatory: Approvals, compliance, industry-specific\n• Cap Table: Options, convertibles, ownership\n• IT: Technology stack, cybersecurity, data privacy\n• ESG: Environmental, social governance, sustainability",
     "Investment Banker\nManagement\nAdvisors (QoE, Legal, Tax)", "Pending"),
    
    # Closing
    ("Closing", "Jun 5 – Jun 26+",
     "• Transaction signing\n• Satisfy closing conditions\n• Regulatory filings\n• Final purchase price adjustment\n• Closing and fund transfer",
     "• Executed Share Purchase Agreement\n• Closing deliverables checklist\n• Regulatory approval confirmations\n• Final working capital calculation\n• Transition services agreement",
     "Investment Banker\nManagement\nLegal Counsel", "Pending"),
]

row = 5
for phase, timeline, activities, diligence_items, responsible, status in details:
    row_fill = LIGHT_GRAY if (row % 2 == 0) else WHITE_FILL
    
    cell_a = ws2.cell(row=row, column=1, value=phase)
    cell_a.font = BOLD_FONT
    cell_a.fill = MEDIUM_BLUE
    cell_a.alignment = LEFT_BOLD
    cell_a.border = THIN_BORDER
    cell_a.font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    
    cell_b = ws2.cell(row=row, column=2, value=timeline)
    cell_b.font = NORMAL_FONT
    cell_b.fill = row_fill
    cell_b.alignment = CENTER
    cell_b.border = THIN_BORDER
    
    cell_c = ws2.cell(row=row, column=3, value=activities)
    cell_c.font = Font(name="Calibri", size=9)
    cell_c.fill = row_fill
    cell_c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    cell_c.border = THIN_BORDER
    
    cell_d = ws2.cell(row=row, column=4, value=diligence_items)
    cell_d.font = Font(name="Calibri", size=9)
    cell_d.fill = row_fill
    cell_d.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    cell_d.border = THIN_BORDER
    
    cell_e = ws2.cell(row=row, column=5, value=responsible)
    cell_e.font = Font(name="Calibri", size=9)
    cell_e.fill = row_fill
    cell_e.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    cell_e.border = THIN_BORDER
    
    # Status with color coding
    status_colors = {
        "Complete": PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid"),
        "In Progress": PatternFill(start_color="ED7D31", end_color="ED7D31", fill_type="solid"),
        "Pending": PatternFill(start_color="A5A5A5", end_color="A5A5A5", fill_type="solid"),
    }
    cell_f = ws2.cell(row=row, column=6, value=status)
    cell_f.font = Font(name="Calibri", size=9, bold=True, color="FFFFFF")
    cell_f.fill = status_colors.get(status, WHITE_FILL)
    cell_f.alignment = CENTER
    cell_f.border = THIN_BORDER
    
    ws2.row_dimensions[row].height = 100
    row += 1

# Freeze panes
ws2.freeze_panes = "A5"

# ============================================================
# SAVE
# ============================================================
output_path = "/home/agent/workspace/banker_workspace/deliverables/MA_SellSide_Deadline_Timeline.xlsx"
wb.save(output_path)
print(f"Saved to {output_path}")
print("Done!")