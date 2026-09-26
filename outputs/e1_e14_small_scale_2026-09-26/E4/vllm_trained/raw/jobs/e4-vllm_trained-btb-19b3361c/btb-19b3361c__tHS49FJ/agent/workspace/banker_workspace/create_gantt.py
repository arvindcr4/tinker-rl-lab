import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.series import Series
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers
from openpyxl.utils import get_column_letter
from datetime import datetime, timedelta
from copy import copy

wb = openpyxl.Workbook()

# ============================================================
# SHEET 1: GANTT CHART
# ============================================================
ws = wb.active
ws.title = "M&A Deal Timeline"

# --- Color palette ---
DARK_BLUE = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
MED_BLUE = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
LIGHT_BLUE = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
TEAL = PatternFill(start_color="548235", end_color="548235", fill_type="solid")
GREEN_FILL = PatternFill(start_color="A9D18E", end_color="A9D18E", fill_type="solid")
ORANGE = PatternFill(start_color="ED7D31", end_color="ED7D31", fill_type="solid")
YELLOW_FILL = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
RED_FILL = PatternFill(start_color="C00000", end_color="C00000", fill_type="solid")
WHITE_FILL = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
LIGHT_GRAY = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
VERY_LIGHT_BLUE = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
MILESTONE_FILL = PatternFill(start_color="BF8F00", end_color="BF8F00", fill_type="solid")

HEADER_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
TITLE_FONT = Font(name="Calibri", bold=True, color="1F3864", size=18)
SECTION_FONT = Font(name="Calibri", bold=True, color="1F3864", size=11)
SUBSECTION_FONT = Font(name="Calibri", bold=True, color="2E75B6", size=10)
NORMAL_FONT = Font(name="Calibri", size=10)
SMALL_FONT = Font(name="Calibri", size=9, color="666666")
MILESTONE_FONT = Font(name="Calibri", bold=True, color="FFFFFF", size=10)

THIN_BORDER = Border(
    left=Side(style="thin", color="B4C6E7"),
    right=Side(style="thin", color="B4C6E7"),
    top=Side(style="thin", color="B4C6E7"),
    bottom=Side(style="thin", color="B4C6E7"),
)

# --- Configuration: Change these dates to update the chart dynamically ---
config = {
    "start_date": datetime(2025, 4, 6),
    "end_date": datetime(2025, 8, 1),  # ~16 weeks after kickoff
    "key_dates": {
        "CIM & Datapack Release": datetime(2025, 4, 6),
        "IOI Submission Deadline": datetime(2025, 4, 24),
        "Partial Data Room Access": datetime(2025, 4, 27),
        "LOI Submission Deadline": datetime(2025, 5, 8),
        "Full Data Room Access": datetime(2025, 5, 11),
        "Definitive Agreement Signing": datetime(2025, 6, 5),
    },
}

# --- Workstreams with dependencies ---
# Each entry: (task_name, start_date, duration_days, category, dependency, is_milestone)
workstreams = [
    # === PHASE 1: PRE-IOI (Preparation Phase) ===
    ("Pre-IOI Preparation Phase", datetime(2025, 4, 6), 18, "Phase 1: Pre-IOI", None, False, DARK_BLUE),
    # Sub-tasks under Pre-IOI
    ("  CIM Preparation & Finalization", datetime(2025, 4, 6), 10, "Phase 1: Pre-IOI", None, False, LIGHT_BLUE),
    ("  Data Pack Assembly & Quality Review", datetime(2025, 4, 6), 10, "Phase 1: Pre-IOI", None, False, LIGHT_BLUE),
    ("  Financial Model - High-Level Guidance", datetime(2025, 4, 6), 12, "Phase 1: Pre-IOI", None, False, LIGHT_BLUE),
    ("  Management Team - Availability & Briefing Prep", datetime(2025, 4, 6), 8, "Phase 1: Pre-IOI", None, False, LIGHT_BLUE),
    ("  Buyer List Identification & Targeting", datetime(2025, 4, 6), 14, "Phase 1: Pre-IOI", None, False, LIGHT_BLUE),
    ("  NDAs Execution", datetime(2025, 4, 6), 18, "Phase 1: Pre-IOI", None, False, LIGHT_BLUE),
    ("  Management Presentation Deck Prep", datetime(2025, 4, 8), 12, "Phase 1: Pre-IOI", None, False, LIGHT_BLUE),
    ("  Teaser Distribution to Potential Buyers", datetime(2025, 4, 10), 14, "Phase 1: Pre-IOI", None, False, LIGHT_BLUE),
    ("MILESTONE: CIM & Datapack Release", datetime(2025, 4, 6), 0, "Phase 1: Pre-IOI", None, True, MILESTONE_FILL),

    # === PHASE 2: IOI TO LOI ===
    ("IOI to LOI Phase", datetime(2025, 4, 24), 17, "Phase 2: IOI - LOI", None, False, MED_BLUE),
    ("  IOI Review & Evaluation (8 parties)", datetime(2025, 4, 24), 10, "Phase 2: IOI - LOI", "IOI Submission Deadline", False, TEAL),
    ("  Buyer Q&A Coordination", datetime(2025, 4, 24), 14, "Phase 2: IOI - LOI", None, False, TEAL),
    ("  Management Meetings & Presentations", datetime(2025, 4, 27), 10, "Phase 2: IOI - LOI", "Partial Data Room Access", False, TEAL),
    ("MILESTONE: Partial Data Room Access", datetime(2025, 4, 27), 0, "Phase 2: IOI - LOI", None, True, MILESTONE_FILL),
    ("  QoE Financial Analysis Preparation", datetime(2025, 4, 27), 10, "Phase 2: IOI - LOI", "Partial Data Room Access", False, TEAL),
    ("  Census Data & Org Structure Compilation", datetime(2025, 4, 27), 8, "Phase 2: IOI - LOI", "Partial Data Room Access", False, TEAL),
    ("  Q1 Actuals Preparation", datetime(2025, 4, 27), 8, "Phase 2: IOI - LOI", "Partial Data Room Access", False, TEAL),
    ("  2025 Budget Drivers Analysis", datetime(2025, 4, 27), 8, "Phase 2: IOI - LOI", "Partial Data Room Access", False, TEAL),
    ("  Off-the-Shelf Legal Documents Prep", datetime(2025, 4, 27), 6, "Phase 2: IOI - LOI", "Partial Data Room Access", False, TEAL),
    ("  Raw Customer Data & Contract Renewal Status", datetime(2025, 4, 27), 8, "Phase 2: IOI - LOI", "Partial Data Room Access", False, TEAL),
    ("  LOI Preparation & Submission Support", datetime(2025, 5, 1), 7, "Phase 2: IOI - LOI", None, False, TEAL),
    ("MILESTONE: LOI Submission Deadline", datetime(2025, 5, 8), 0, "Phase 2: IOI - LOI", None, True, MILESTONE_FILL),

    # === PHASE 3: LOI TO SIGNING ===
    ("LOI to Signing Phase", datetime(2025, 5, 8), 28, "Phase 3: LOI - Signing", None, False, ORANGE),
    ("  LOI Evaluation & Selection of One Party", datetime(2025, 5, 8), 5, "Phase 3: LOI - Signing", "LOI Submission Deadline", False, GREEN_FILL),
    ("MILESTONE: Full Data Room Access", datetime(2025, 5, 11), 0, "Phase 3: LOI - Signing", None, True, MILESTONE_FILL),
    ("  Financial Diligence - Complete QoE", datetime(2025, 5, 11), 20, "Phase 3: LOI - Signing", "Full Data Room Access", False, GREEN_FILL),
    ("  Tax Diligence", datetime(2025, 5, 11), 20, "Phase 3: LOI - Signing", "Full Data Room Access", False, GREEN_FILL),
    ("  Legal Diligence", datetime(2025, 5, 11), 20, "Phase 3: LOI - Signing", "Full Data Room Access", False, GREEN_FILL),
    ("  Capex Analysis & Review", datetime(2025, 5, 11), 15, "Phase 3: LOI - Signing", "Full Data Room Access", False, GREEN_FILL),
    ("  Regulatory Compliance Review", datetime(2025, 5, 11), 15, "Phase 3: LOI - Signing", "Full Data Room Access", False, GREEN_FILL),
    ("  Cap Table Analysis", datetime(2025, 5, 11), 10, "Phase 3: LOI - Signing", "Full Data Room Access", False, GREEN_FILL),
    ("  IT Systems & Cybersecurity Diligence", datetime(2025, 5, 11), 18, "Phase 3: LOI - Signing", "Full Data Room Access", False, GREEN_FILL),
    ("  ESG Assessment", datetime(2025, 5, 11), 15, "Phase 3: LOI - Signing", "Full Data Room Access", False, GREEN_FILL),
    ("  Definitive Agreement Negotiation", datetime(2025, 5, 19), 14, "Phase 3: LOI - Signing", None, False, GREEN_FILL),
    ("MILESTONE: Definitive Agreement Signing", datetime(2025, 6, 5), 0, "Phase 3: LOI - Signing", None, True, MILESTONE_FILL),

    # === PHASE 4: EXCLUSIVITY ===
    ("Exclusivity Period (4 weeks)", datetime(2025, 6, 5), 28, "Phase 4: Exclusivity", None, False, RED_FILL),
    ("  Complete Financial Diligence", datetime(2025, 6, 5), 20, "Phase 4: Exclusivity", "Definitive Agreement Signing", False, YELLOW_FILL),
    ("  Complete Tax Diligence", datetime(2025, 6, 5), 20, "Phase 4: Exclusivity", "Definitive Agreement Signing", False, YELLOW_FILL),
    ("  Complete Legal Diligence", datetime(2025, 6, 5), 20, "Phase 4: Exclusivity", "Definitive Agreement Signing", False, YELLOW_FILL),
    ("  Complete Capex Review", datetime(2025, 6, 5), 18, "Phase 4: Exclusivity", "Definitive Agreement Signing", False, YELLOW_FILL),
    ("  Complete Regulatory Review", datetime(2025, 6, 5), 18, "Phase 4: Exclusivity", "Definitive Agreement Signing", False, YELLOW_FILL),
    ("  Cap Table Finalization", datetime(2025, 6, 5), 14, "Phase 4: Exclusivity", "Definitive Agreement Signing", False, YELLOW_FILL),
    ("  IT & Cybersecurity Deep Dive", datetime(2025, 6, 5), 20, "Phase 4: Exclusivity", "Definitive Agreement Signing", False, YELLOW_FILL),
    ("  ESG Full Assessment", datetime(2025, 6, 5), 18, "Phase 4: Exclusivity", "Definitive Agreement Signing", False, YELLOW_FILL),
    ("  Closing Preparation & Conditions Precedent", datetime(2025, 6, 23), 10, "Phase 4: Exclusivity", None, False, YELLOW_FILL),
]

# --- Write headers ---
headers = ["Workstream", "Category", "Start Date", "Duration (Days)", "End Date", "Dependency", "Status"]
col_widths = [45, 22, 14, 15, 14, 28, 15]

for col_idx, (header, width) in enumerate(zip(headers, col_widths), 1):
    cell = ws.cell(row=2, column=col_idx, value=header)
    cell.font = HEADER_FONT
    cell.fill = DARK_BLUE
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = THIN_BORDER
    ws.column_dimensions[get_column_letter(col_idx)].width = width

# --- Write data rows ---
row = 3
for task in workstreams:
    task_name, start, duration, category, dependency, is_milestone, bar_color = task

    # Calculate end date
    if is_milestone:
        end = start
    else:
        end = start + timedelta(days=duration) - timedelta(days=1)

    # Status column - default to "Not Started"
    status = "Not Started"

    ws.cell(row=row, column=1, value=task_name).font = (
        MILESTONE_FONT if is_milestone else (
            Font(name="Calibri", bold=True, size=10, color="FFFFFF") if duration <= 1 and task_name.startswith("Phase")
            else Font(name="Calibri", bold=True, size=10) if task_name.startswith("  ") == False and not is_milestone
            else NORMAL_FONT
        )
    )
    ws.cell(row=row, column=2, value=category).font = NORMAL_FONT
    ws.cell(row=row, column=3, value=start).font = NORMAL_FONT
    ws.cell(row=row, column=3).number_format = 'MM/DD/YYYY'
    ws.cell(row=row, column=4, value=duration if not is_milestone else "Milestone").font = NORMAL_FONT
    ws.cell(row=row, column=5, value=end).font = NORMAL_FONT
    ws.cell(row=row, column=5).number_format = 'MM/DD/YYYY'
    ws.cell(row=row, column=6, value=dependency if dependency else "—").font = NORMAL_FONT
    ws.cell(row=row, column=7, value=status).font = NORMAL_FONT

    # Apply borders and alignment
    for col_idx in range(1, 8):
        cell = ws.cell(row=row, column=col_idx)
        cell.border = THIN_BORDER
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        if col_idx in (3, 5):
            cell.alignment = Alignment(horizontal="center", vertical="center")

    # Conditional formatting based on row type
    if is_milestone:
        for col_idx in range(1, 8):
            ws.cell(row=row, column=col_idx).fill = MILESTONE_FILL
            ws.cell(row=row, column=col_idx).font = MILESTONE_FONT
    elif task_name.startswith("  "):
        # Sub-task
        for col_idx in range(1, 8):
            ws.cell(row=row, column=col_idx).fill = bar_color
            ws.cell(row=row, column=col_idx).font = NORMAL_FONT
    elif task_name.startswith("Phase"):
        # Phase header
        for col_idx in range(1, 8):
            ws.cell(row=row, column=col_idx).fill = bar_color
            ws.cell(row=row, column=col_idx).font = Font(name="Calibri", bold=True, size=10, color="FFFFFF")
    else:
        # Alternating row colors for top-level tasks
        if (row - 3) % 2 == 0:
            for col_idx in range(1, 8):
                ws.cell(row=row, column=col_idx).fill = VERY_LIGHT_BLUE
        else:
            for col_idx in range(1, 8):
                ws.cell(row=row, column=col_idx).fill = WHITE_FILL

    row += 1

# --- Add Gantt Chart visualization using BarChart ---
# We'll create a separate section below the table for the visual Gantt chart
gantt_start_row = row + 2  # Leave a gap

# Headers for Gantt chart section
ws.cell(row=gantt_start_row, column=1, value="GANTT CHART VISUALIZATION").font = Font(name="Calibri", bold=True, color="1F3864", size=14)
ws.merge_cells(start_row=gantt_start_row, start_column=1, end_row=gantt_start_row, end_column=7)

# Timeline header row
timeline_start = gantt_start_row + 2
ws.cell(row=timeline_start, column=1, value="Task").font = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
ws.cell(row=timeline_start, column=1).fill = DARK_BLUE
ws.cell(row=timeline_start, column=1).alignment = Alignment(horizontal="center", vertical="center")

# Add month headers for the Gantt chart
months = ["Apr 2025", "May 2025", "Jun 2025", "Jul 2025", "Aug 2025"]
month_start_col = 2
for i, month in enumerate(months):
    ws.cell(row=timeline_start, column=month_start_col + i * 4, value=month).font = HEADER_FONT
    ws.cell(row=timeline_start, column=month_start_col + i * 4).fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
    ws.cell(row=timeline_start, column=month_start_col + i * 4).alignment = Alignment(horizontal="center", vertical="center")
    ws.merge_cells(
        start_row=timeline_start, start_column=month_start_col + i * 4,
        end_row=timeline_start, end_column=month_start_col + i * 4 + 3
    )

# Day-of-month sub-headers
sub_header_row = timeline_start + 1
ws.cell(row=sub_header_row, column=1, value="").font = NORMAL_FONT
ws.cell(row=sub_header_row, column=1).fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
for i, month in enumerate(months):
    if "Apr" in month:
        days_in_month = 30
        start_day = 6  # April 6
    elif "May" in month:
        days_in_month = 31
        start_day = 1
    elif "Jun" in month:
        days_in_month = 30
        start_day = 1
    elif "Jul" in month:
        days_in_month = 31
        start_day = 1
    else:
        days_in_month = 31
        start_day = 1

    for d in range(start_day, min(start_day + 28, days_in_month + 1)):
        col = month_start_col + i * 4 + ((d - start_day) % 4)
        ws.cell(row=sub_header_row, column=col, value=d).font = Font(name="Calibri", size=8, color="FFFFFF")
        ws.cell(row=sub_header_row, column=col).fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        ws.cell(row=sub_header_row, column=col).alignment = Alignment(horizontal="center", vertical="center")

# Write Gantt bars
gantt_data_start = sub_header_row + 2
gantt_row = gantt_data_start

# For the Gantt chart, show key tasks with colored bars
gantt_tasks = []
for task in workstreams:
    task_name, start, duration, category, dependency, is_milestone, bar_color = task
    if is_milestone:
        gantt_tasks.append((task_name, start, 1, bar_color, True))
    elif duration > 0:
        gantt_tasks.append((task_name, start, duration, bar_color, False))

# Calculate the chart area
chart_start_date = config["start_date"]
chart_end_date = config["end_date"]
total_days = (chart_end_date - chart_start_date).days

for task_name, start, duration, bar_color, is_milestone in gantt_tasks:
    # Calculate position
    days_from_start = (start - chart_start_date).days
    if days_from_start < 0:
        days_from_start = 0
    if days_from_start >= total_days:
        continue

    # Map to column (column 2 = day 0, column 2 + total_days = end)
    bar_col = 2 + days_from_start
    bar_width = min(duration, total_days - days_from_start)

    # Task name in column 1
    ws.cell(row=gantt_row, column=1, value=task_name).font = Font(name="Calibri", size=8)
    ws.cell(row=gantt_row, column=1).alignment = Alignment(horizontal="left", vertical="center")

    # Draw the bar
    if bar_width > 0 and bar_col <= 2 + total_days:
        actual_width = min(bar_width, 2 + total_days - bar_col)
        for c in range(bar_col, bar_col + actual_width):
            cell = ws.cell(row=gantt_row, column=c)
            cell.fill = bar_color
            cell.border = THIN_BORDER

    # Set row height for readability
    ws.row_dimensions[gantt_row].height = 8

    gantt_row += 1

# Set column widths for Gantt chart
ws.column_dimensions["A"].width = 42
for c in range(2, 2 + total_days + 1):
    ws.column_dimensions[get_column_letter(c)].width = 1.0

# --- Freeze panes ---
ws.freeze_panes = "B3"

# --- Print setup ---
ws.sheet_properties.pageSetUpPr = openpyxl.worksheet.properties.PageSetupProperties(fitToPage=True)
ws.page_setup.fitToWidth = 1
ws.page_setup.fitToHeight = 0
ws.page_setup.orientation = "landscape"

# ============================================================
# SHEET 2: KEY DATES & MILESTONES
# ============================================================
ws2 = wb.create_sheet("Key Dates & Milestones")

ws2.cell(row=1, column=1, value="KEY DATES & MILESTONES").font = Font(name="Calibri", bold=True, color="1F3864", size=16)
ws2.merge_cells("A1:F1")

ws2.cell(row=2, column=1, value="Critical path dates for the M&A transaction process.").font = Font(name="Calibri", italic=True, size=10, color="666666")
ws2.merge_cells("A2:F2")

milestone_headers = ["Milestone", "Date", "Days Until", "Phase", "Description", "Status"]
mh_widths = [35, 14, 14, 22, 55, 15]

for col_idx, (header, width) in enumerate(zip(milestone_headers, mh_widths), 1):
    cell = ws2.cell(row=4, column=col_idx, value=header)
    cell.font = HEADER_FONT
    cell.fill = DARK_BLUE
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = THIN_BORDER
    ws2.column_dimensions[get_column_letter(col_idx)].width = width

milestones_data = [
    ("CIM & Datapack Release", datetime(2025, 4, 6), "Phase 1: Pre-IOI", "Official release of Confidential Information Memorandum and data pack to targeted buyers"),
    ("IOI Submission Deadline", datetime(2025, 4, 24), "Phase 2: IOI - LOI", "Non-binding expression of interest due from all 8 targeted parties"),
    ("Partial Data Room Access", datetime(2025, 4, 27), "Phase 2: IOI - LOI", "Qualified buyers receive access to partial data room with key financials"),
    ("LOI Submission Deadline", datetime(2025, 5, 8), "Phase 3: LOI - Signing", "Formal letter of interest from shortlisted buyers (5-8 parties)"),
    ("Full Data Room Access", datetime(2025, 5, 11), "Phase 3: LOI - Signing", "Selected buyer(s) granted full data room access for comprehensive diligence"),
    ("Definitive Agreement Signing", datetime(2025, 6, 5), "Phase 3: LOI - Signing", "Execution of Purchase Agreement and definitive transaction documents"),
    ("Exclusivity Period Begins", datetime(2025, 6, 5), "Phase 4: Exclusivity", "4-week exclusive negotiation and complete diligence period"),
    ("Expected Closing", datetime(2025, 8, 1), "Phase 4: Exclusivity", "Target closing date following completion of all diligence and conditions"),
]

for i, (name, date, phase, desc) in enumerate(milestones_data):
    r = 5 + i
    days_until = (date - datetime(2025, 4, 6)).days  # relative to kickoff

    ws2.cell(row=r, column=1, value=name).font = Font(name="Calibri", bold=True, size=10)
    ws2.cell(row=r, column=2, value=date).font = NORMAL_FONT
    ws2.cell(row=r, column=2).number_format = 'MM/DD/YYYY'
    ws2.cell(row=r, column=3, value=days_until).font = NORMAL_FONT
    ws2.cell(row=r, column=3).alignment = Alignment(horizontal="center")
    ws2.cell(row=r, column=4, value=phase).font = NORMAL_FONT
    ws2.cell(row=r, column=5, value=desc).font = Font(name="Calibri", size=10)
    ws2.cell(row=r, column=6, value="Pending").font = NORMAL_FONT

    for col_idx in range(1, 7):
        cell = ws2.cell(row=r, column=col_idx)
        cell.border = THIN_BORDER
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        if col_idx in (2, 3):
            cell.alignment = Alignment(horizontal="center", vertical="center")

    # Color code by phase
    if "Pre-IOI" in phase:
        fill = LIGHT_BLUE
    elif "IOI" in phase:
        fill = TEAL
    elif "LOI" in phase or "Signing" in phase:
        fill = ORANGE
    else:
        fill = RED_FILL

    for col_idx in range(1, 7):
        ws2.cell(row=r, column=col_idx).fill = fill

    # Highlight the signing milestone
    if "Signing" in name:
        for col_idx in range(1, 7):
            ws2.cell(row=r, column=col_idx).fill = MILESTONE_FILL
            ws2.cell(row=r, column=col_idx).font = MILESTONE_FONT

# Freeze panes
ws2.freeze_panes = "A5"

# ============================================================
# SHEET 3: WORKSTREAM DETAILS
# ============================================================
ws3 = wb.create_sheet("Workstream Details")

ws3.cell(row=1, column=1, value="WORKSTREAM DETAILS & DILIGENCE ITEMS").font = Font(name="Calibri", bold=True, color="1F3864", size=16)
ws3.merge_cells("A1:G1")

ws3.cell(row=2, column=1, value="Detailed breakdown of each workstream, key deliverables, and responsible parties.").font = Font(name="Calibri", italic=True, size=10, color="666666")
ws3.merge_cells("A2:G2")

detail_headers = ["Workstream", "Phase", "Timeline", "Key Deliverables / Diligence Items", "Pre-IOI", "Pre-LOI", "Exclusivity"]
dw_widths = [30, 20, 20, 55, 18, 18, 18]

for col_idx, (header, width) in enumerate(zip(detail_headers, dw_widths), 1):
    cell = ws3.cell(row=4, column=col_idx, value=header)
    cell.font = HEADER_FONT
    cell.fill = DARK_BLUE
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = THIN_BORDER
    ws3.column_dimensions[get_column_letter(col_idx)].width = width

workstream_details = [
    ("Financial", "All Phases", "Apr 6 – Aug 1",
     "High-level model guidance → QoE financials, Q1 actuals, 2025 budget drivers → Complete QoE, working capital analysis",
     "✓ High-level model guidance", "✓ QoE financials\n✓ Q1 Actuals\n✓ 2025 Budget Drivers", "✓ Complete QoE\n✓ Working Capital\n✓ Normalization Adjustments"),
    ("Tax", "LOI – Signing", "May 11 – Aug 1",
     "Tax return review, NOL analysis, state/federal compliance, transfer pricing",
     "—", "—", "✓ Full tax diligence\n✓ NOL Analysis\n✓ Transfer Pricing"),
    ("Legal", "All Phases", "Apr 6 – Aug 1",
     "Off-the-shelf docs → Contract review, litigation assessment, IP review → Definitive agreement negotiation",
     "✓ Off-the-shelf legal docs", "✓ Contract review\n✓ Litigation assessment", "✓ Full legal diligence\n✓ IP Review\n✓ Definitive Agreements"),
    ("Commercial / Customer", "IOI – LOI", "Apr 27 – May 11",
     "Raw customer data, contract renewal statuses, customer concentration analysis",
     "—", "✓ Raw customer data\n✓ Contract renewal statuses", "✓ Customer satisfaction\n✓ Pipeline review"),
    ("Census / HR / Org", "IOI – LOI", "Apr 27 – May 11",
     "Census data compilation, org structure, compensation analysis, key employee retention",
     "—", "✓ Census data\n✓ Org structure\n✓ Compensation analysis", "✓ Key employee retention\n✓ Benefits review"),
    ("Capex", "LOI – Signing", "May 11 – Aug 1",
     "Capex analysis, maintenance vs growth capex split, future capex requirements",
     "—", "—", "✓ Full capex review\n✓ Maintenance capex\n✓ Growth capex forecast"),
    ("Regulatory", "LOI – Signing", "May 11 – Aug 1",
     "Regulatory compliance review, licenses & permits, industry-specific requirements",
     "—", "—", "✓ Full regulatory review\n✓ License verification"),
    ("Capital Structure", "LOI – Signing", "May 11 – Jul 5",
     "Cap table analysis, existing debt obligations, equity structure, option pool",
     "—", "—", "✓ Cap table finalization\n✓ Debt analysis\n✓ Option pool review"),
    ("IT / Cybersecurity", "LOI – Signing", "May 11 – Aug 1",
     "IT systems assessment, cybersecurity review, data privacy, tech stack evaluation",
     "—", "—", "✓ Full IT diligence\n✓ Cybersecurity audit\n✓ Data privacy review"),
    ("ESG", "LOI – Signing", "May 11 – Aug 1",
     "ESG assessment, sustainability metrics, environmental compliance, governance review",
     "—", "—", "✓ Full ESG assessment\n✓ Sustainability report"),
]

for i, (ws_name, phase, timeline, deliverables, pre_ioi, pre_loi, exclusivity) in enumerate(workstream_details):
    r = 5 + i
    ws3.cell(row=r, column=1, value=ws_name).font = Font(name="Calibri", bold=True, size=10)
    ws3.cell(row=r, column=2, value=phase).font = NORMAL_FONT
    ws3.cell(row=r, column=3, value=timeline).font = NORMAL_FONT
    ws3.cell(row=r, column=4, value=deliverables).font = Font(name="Calibri", size=10)
    ws3.cell(row=r, column=5, value=pre_ioi).font = NORMAL_FONT
    ws3.cell(row=r, column=6, value=pre_loi).font = NORMAL_FONT
    ws3.cell(row=r, column=7, value=exclusivity).font = NORMAL_FONT

    for col_idx in range(1, 8):
        cell = ws3.cell(row=r, column=col_idx)
        cell.border = THIN_BORDER
        cell.alignment = Alignment(vertical="center", wrap_text=True)

    # Color by phase
    if "Pre-IOI" in phase or "All" in phase:
        fill = LIGHT_BLUE
    elif "IOI" in phase:
        fill = TEAL
    else:
        fill = ORANGE
    for col_idx in range(1, 8):
        ws3.cell(row=r, column=col_idx).fill = fill

ws3.freeze_panes = "A5"

# ============================================================
# Save
# ============================================================
output_path = "/home/agent/workspace/banker_workspace/deliverables/MA_Deal_Timeline.xlsx"
wb.save(output_path)
print(f"Saved to {output_path}")
print("Done!")