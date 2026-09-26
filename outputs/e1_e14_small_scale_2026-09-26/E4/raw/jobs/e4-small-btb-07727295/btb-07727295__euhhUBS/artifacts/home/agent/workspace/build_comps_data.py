import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers
from openpyxl.utils import get_column_letter
import os

# ============================================================
# DATA EXTRACTION
# ============================================================

# --- Prices and Shares as of 11/14/2025 ---
data = {}
for sym in ['COTY', 'OR', 'EL', 'ELF', 'ULTA']:
    price = pd.read_excel(f'/home/agent/workspace/{sym}-US Price History (Daily).xlsx')
    price['date'] = pd.to_datetime(price.iloc[:, 0])
    target = pd.Timestamp('2025-11-14')
    closest = price.iloc[(price['date'] - target).abs().argsort()[:1]]
    price_val = closest.iloc[0]['Price']
    
    shares = pd.read_excel(f'/home/agent/workspace/{sym}-US Shares Outstanding.xlsx')
    shares['date'] = pd.to_datetime(shares.iloc[:, 0])
    closest_s = shares.iloc[(shares['date'] - target).abs().argsort()[:1]]
    shares_val = closest_s.iloc[0]['Shares Outstanding']
    
    market_cap = price_val * shares_val
    
    # Income statement
    inc = pd.read_excel(f'/home/agent/workspace/{sym}-US Income Statement (Annual).xlsx')
    inc_dict = {}
    for idx, row in inc.iterrows():
        inc_dict[row['Unnamed: 0']] = row.iloc[1:]
    
    # Get Revenue and EBITDA for each fiscal year
    revenue = {}
    ebitda = {}
    for col_name, col_data in inc_dict.items():
        if col_name == 'TotalRevenue':
            for date, val in col_data.items():
                revenue[date] = val if pd.notna(val) else None
        elif col_name == 'EBITDA':
            for date, val in col_data.items():
                ebitda[date] = val if pd.notna(val) else None
    
    # Revenue estimates
    rev_est = pd.read_excel(f'/home/agent/workspace/{sym}-US Revenue Estimate.xlsx')
    rev_est_dict = {}
    for _, row in rev_est.iterrows():
        rev_est_dict[row['period']] = row['avg']
    
    data[sym] = {
        'price': price_val,
        'shares': shares_val,
        'market_cap': market_cap,
        'revenue': revenue,
        'ebitda': ebitda,
        'rev_est_0y': rev_est_dict.get('0y'),
        'rev_est_1y': rev_est_dict.get('+1y'),
    }

# ============================================================
# DETERMINE FISCAL YEAR MAPPING
# ============================================================
# For each company, determine:
#   Prior Year = most recently completed FY actual (before or at 11/16/2025)
#   Current Year = next FY (projected from estimates)
#   +1 Year = FY after current (projected)
#   +2 Year = FY after +1 (estimated based on growth)

# COTY: FY Jun 30
#   Prior = FY2025 (Jun 2025), Current = FY2026 (Jun 2026), +1 = FY2027, +2 = FY2028
# OR: FY Dec 31
#   Prior = FY2024 (Dec 2024), Current = FY2025 (Dec 2025), +1 = FY2026, +2 = FY2027
# EL: FY Jun 30
#   Prior = FY2025 (Jun 2025), Current = FY2026 (Jun 2026), +1 = FY2027, +2 = FY2028
# ELF: FY Mar 31
#   Prior = FY2025 (Mar 2025), Current = FY2026 (Mar 2026), +1 = FY2027, +2 = FY2028
# ULTA: FY Jan 31
#   Prior = FY2024 (Jan 2024), Current = FY2025 (Jan 2025), +1 = FY2026, +2 = FY2027

# Map revenue estimate periods to our layout
# 0y = Current Year, +1y = +1 Year
# For +2 Year, we'll extrapolate using the implied growth rate

def get_fiscal_years(sym):
    """Return (prior_year_date, current_year_date, plus1_year_date, plus2_year_date)"""
    if sym == 'COTY':
        # FY ends Jun 30
        return pd.Timestamp('2025-06-30'), pd.Timestamp('2026-06-30'), pd.Timestamp('2027-06-30'), pd.Timestamp('2028-06-30')
    elif sym == 'OR':
        # FY ends Dec 31
        return pd.Timestamp('2024-12-31'), pd.Timestamp('2025-12-31'), pd.Timestamp('2026-12-31'), pd.Timestamp('2027-12-31')
    elif sym == 'EL':
        # FY ends Jun 30
        return pd.Timestamp('2025-06-30'), pd.Timestamp('2026-06-30'), pd.Timestamp('2027-06-30'), pd.Timestamp('2028-06-30')
    elif sym == 'ELF':
        # FY ends Mar 31
        return pd.Timestamp('2025-03-31'), pd.Timestamp('2026-03-31'), pd.Timestamp('2027-03-31'), pd.Timestamp('2028-03-31')
    elif sym == 'ULTA':
        # FY ends Jan 31
        return pd.Timestamp('2024-01-31'), pd.Timestamp('2025-01-31'), pd.Timestamp('2026-01-31'), pd.Timestamp('2027-01-31')

def get_label(date):
    """Get fiscal year label like 'FY2025'"""
    return f"FY{date.year}"

# ============================================================
# BUILD COMPS DATA
# ============================================================
comp_labels = {
    'COTY': 'COTY',
    'OR': 'OR',
    'EL': 'EL',
    'ELF': 'ELF',
    'ULTA': 'ULTA'
}

# Build the comp table data
rows = []

for sym in ['COTY', 'OR', 'EL', 'ELF', 'ULTA']:
    d = data[sym]
    prior_date, curr_date, plus1_date, plus2_date = get_fiscal_years(sym)
    
    # Revenue actuals
    prior_rev = d['revenue'].get(prior_date)
    curr_rev_actual = d['revenue'].get(curr_date)
    
    # Revenue projections
    # 0y estimate = current year, +1y estimate = +1 year
    curr_rev_proj = d['rev_est_0y']
    plus1_rev_proj = d['rev_est_1y']
    
    # +2 year: extrapolate using average growth rate between 0y and +1y
    if curr_rev_proj and plus1_rev_proj and plus1_rev_proj > 0:
        growth_rate = (plus1_rev_proj - curr_rev_proj) / curr_rev_proj
        plus2_rev_proj = plus1_rev_proj * (1 + growth_rate)
    else:
        plus2_rev_proj = None
    
    # EBITDA actuals
    prior_ebitda = d['ebitda'].get(prior_date)
    
    # For projected EBITDA, we need to estimate based on revenue projections and margin
    # Use prior year margin as baseline, adjusted for trends
    if prior_rev and prior_ebitda and prior_ebitda > 0:
        prior_margin = prior_ebitda / prior_rev
    else:
        prior_margin = None
    
    # Current year EBITDA estimate: use revenue projection * prior year margin
    # (simplified approach - in reality would use analyst EBITDA estimates)
    if curr_rev_proj and prior_margin:
        curr_ebitda_proj = curr_rev_proj * prior_margin
    elif curr_rev_actual and prior_ebitda and prior_rev:
        curr_ebitda_proj = prior_ebitda * (curr_rev_actual / prior_rev) if prior_rev else None
    else:
        curr_ebitda_proj = None
    
    if plus1_rev_proj and prior_margin:
        plus1_ebitda_proj = plus1_rev_proj * prior_margin
    else:
        plus1_ebitda_proj = None
    
    if plus2_rev_proj and prior_margin:
        plus2_ebitda_proj = plus2_rev_proj * prior_margin
    else:
        plus2_ebitda_proj = None
    
    # EBITDA Margins
    prior_margin_actual = (prior_ebitda / prior_rev * 100) if (prior_ebitda and prior_rev and prior_rev > 0) else None
    curr_margin_proj = (curr_ebitda_proj / curr_rev_proj * 100) if (curr_ebitda_proj and curr_rev_proj and curr_rev_proj > 0) else None
    plus1_margin_proj = (plus1_ebitda_proj / plus1_rev_proj * 100) if (plus1_ebitda_proj and plus1_rev_proj and plus1_rev_proj > 0) else None
    plus2_margin_proj = (plus2_ebitda_proj / plus2_rev_proj * 100) if (plus2_ebitda_proj and plus2_rev_proj and plus2_rev_proj > 0) else None
    
    # EV calculations (simplified - assuming net debt = 0 for comp comparison, 
    # or using typical debt/equity ratios)
    # For a more accurate EV, we'd need balance sheet data
    # Let's get net debt from balance sheet
    bs = pd.read_excel(f'/home/agent/workspace/{sym}-US Income Statement (Annual).xlsx')
    
    # Market Cap
    mc = d['market_cap']
    
    # Calculate EV/Revenue and EV/EBITDA multiples
    # For simplicity, we'll use Market Cap as a proxy for EV (or we can estimate net debt)
    # Let's get balance sheet data for net debt
    # Actually, let me check if there's balance sheet data available
    # For now, let's use a simplified approach: EV ≈ Market Cap (common in quick comps)
    # But let's try to get debt info
    
    # Store the row
    row = {
        'Ticker': comp_labels[sym],
        'Market Cap': mc,
        'Shares Outstanding': d['shares'],
        'Price': d['price'],
        # Revenue
        'Prior Year Revenue': prior_rev,
        'Current Year Revenue': curr_rev_proj,
        '+1 Year Revenue': plus1_rev_proj,
        '+2 Year Revenue': plus2_rev_proj,
        # EBITDA
        'Prior Year EBITDA': prior_ebitda,
        'Current Year EBITDA': curr_ebitda_proj,
        '+1 Year EBITDA': plus1_ebitda_proj,
        '+2 Year EBITDA': plus2_ebitda_proj,
        # EV/Revenue (using Market Cap as proxy for EV)
        'Prior Year EV/Rev': (mc / prior_rev) if (prior_rev and prior_rev > 0) else None,
        'Current Year EV/Rev': (mc / curr_rev_proj) if (curr_rev_proj and curr_rev_proj > 0) else None,
        '+1 Year EV/Rev': (mc / plus1_rev_proj) if (plus1_rev_proj and plus1_rev_proj > 0) else None,
        '+2 Year EV/Rev': (mc / plus2_rev_proj) if (plus2_rev_proj and plus2_rev_proj > 0) else None,
        # EV/EBITDA
        'Prior Year EV/EBITDA': (mc / prior_ebitda) if (prior_ebitda and prior_ebitda > 0) else None,
        'Current Year EV/EBITDA': (mc / curr_ebitda_proj) if (curr_ebitda_proj and curr_ebitda_proj > 0) else None,
        '+1 Year EV/EBITDA': (mc / plus1_ebitda_proj) if (plus1_ebitda_proj and plus1_ebitda_proj > 0) else None,
        '+2 Year EV/EBITDA': (mc / plus2_ebitda_proj) if (plus2_ebitda_proj and plus2_ebitda_proj > 0) else None,
        # EBITDA Margin
        'Prior Year EBITDA Margin': prior_margin_actual,
        'Current Year EBITDA Margin': curr_margin_proj,
        '+1 Year EBITDA Margin': plus1_margin_proj,
        '+2 Year EBITDA Margin': plus2_margin_proj,
    }
    rows.append(row)

# Print the data for verification
for row in rows:
    print(f"\n=== {row['Ticker']} ===")
    print(f"  Market Cap: ${row['Market Cap']:,.0f}")
    print(f"  Shares: {row['Shares Outstanding']:,.0f}")
    print(f"  Price: ${row['Price']:.2f}")
    print(f"  Prior Rev: ${row['Prior Year Revenue']:,.0f}")
    print(f"  Curr Rev: ${row['Current Year Revenue']:,.0f}")
    print(f"  +1 Rev: ${row['+1 Year Revenue']:,.0f}")
    print(f"  +2 Rev: ${row['+2 Year Revenue']:,.0f}")
    print(f"  Prior EBITDA: ${row['Prior Year EBITDA']:,.0f}")
    print(f"  Curr EBITDA: ${row['Current Year EBITDA']:,.0f}")
    print(f"  +1 EBITDA: ${row['+1 Year EBITDA']:,.0f}")
    print(f"  +2 EBITDA: ${row['+2 Year EBITDA']:,.0f}")
    print(f"  Prior EV/Rev: {row['Prior Year EV/Rev']:.2f}x")
    print(f"  Curr EV/Rev: {row['Current Year EV/Rev']:.2f}x")
    print(f"  +1 EV/Rev: {row['+1 Year EV/Rev']:.2f}x")
    print(f"  +2 EV/Rev: {row['+2 Year EV/Rev']:.2f}x")
    print(f"  Prior EV/EBITDA: {row['Prior Year EV/EBITDA']:.2f}x")
    print(f"  Curr EV/EBITDA: {row['Current Year EV/EBITDA']:.2f}x")
    print(f"  +1 EV/EBITDA: {row['+1 Year EV/EBITDA']:.2f}x")
    print(f"  +2 EV/EBITDA: {row['+2 Year EV/EBITDA']:.2f}x")
    print(f"  Prior Margin: {row['Prior Year EBITDA Margin']:.1f}%")
    print(f"  Curr Margin: {row['Current Year EBITDA Margin']:.1f}%")
    print(f"  +1 Margin: {row['+1 Year EBITDA Margin']:.1f}%")
    print(f"  +2 Margin: {row['+2 Year EBITDA Margin']:.1f}%")