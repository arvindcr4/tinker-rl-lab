import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers
from openpyxl.utils import get_column_letter
import os

# =============================================================================
# ADOBE (ADBE) LBO MODEL
# =============================================================================

# --- INPUT ASSUMPTIONS ---
PRICE_PER_SHARE = 331.11  # Nov 14, 2025 price
SHARES_OUTSTANDING = 418600000  # Most recent shares outstanding (Nov 2025)

# Transaction
TRANSACTION_DATE = "12/31/2025"
EXIT_DATE = "12/31/2030"
EXIT_YEAR = 2030
HOLDING_PERIOD = 5  # years

# Purchase Price / Enterprise Value
EQUITY_VALUE = PRICE_PER_SHARE * SHARES_OUTSTANDING

# Net Debt at closing (from FY2025 balance sheet - Nov 30, 2025)
TOTAL_DEBT = 6648000000  # Total debt (current + long-term)
CASH_AND_EQUIVALENTS = 5431000000  # Cash and cash equivalents
NET_DEBT = TOTAL_DEBT - CASH_AND_EQUIVALENTS  # 1,217,000,000

# Enterprise Value = Equity Value + Net Debt
ENTERPRISE_VALUE = EQUITY_VALUE + NET_DEBT

# Deal Sizing
TRANSACTION_FEES = 0.025  # 2.5% of enterprise value
FINANCING_FEES = 0.030  # 3.0% of enterprise value

# Debt Structures
# Senior Debt: 2.0x of Net Debt to fund the deal
# Term Loan B: 4.5x of Net Debt to fund the deal
# Actually, re-reading: "4.5x Term Loan B to fund the deal" and "2.0x of Senior Debt to fund the deal"
# These multiples are typically relative to EBITDA

# Let's use EBITDA for the multiples
EBITDA_LY = 9788000000  # FY2025 EBITDA (Nov 30, 2025)

# Term Loan B: 4.5x EBITDA
TLB_MULTIPLE = 4.5
TLB_AMOUNT = TLB_MULTIPLE * EBITDA_LY

# Senior Debt: 2.0x EBITDA
SD_MULTIPLE = 2.0
SD_AMOUNT = SD_MULTIPLE * EBITDA_LY

# Total Debt = TLB + Senior Debt
TOTAL_DEBT_PROVIDED = TLB_AMOUNT + SD_AMOUNT

# Sponsor Equity = Enterprise Value - Total Debt - Financing Fees - Transaction Fees
# Actually: Equity = EV - Debt - Fees (fees are paid from proceeds)
# The fees are typically added to the debt/equity raise
TOTAL_FEES = TRANSACTION_FEES * ENTERPRISE_VALUE + FINANCING_FEES * ENTERPRISE_VALUE

# Sponsor Equity = EV - Total Debt Raised - Total Fees
SPONSOR_EQUITY = ENTERPRISE_VALUE - TOTAL_DEBT_PROVIDED - TOTAL_FEES

# Debt Pricing
# Term Loan B: priced at S+650, 2.0% floor, 8.0% mandatory amortization
# S = SOFR (assume ~4.3% for modeling, but we'll use a standard rate)
# Actually, for LBO modeling, we need the all-in rate
# S+650 with 2.0% floor means: max(SOFR, 2.0%) + 650bps
# Let's assume SOFR ~4.3% (current rate environment), floor is 2.0%, so S = 4.3%
# TLB Rate = 4.3% + 6.5% = 10.8%
SOFR_RATE = 0.043  # Assume current SOFR
TLB_FLOOR = 0.020
TLB_SPREAD = 0.065
TLB_RATE = max(SOFR_RATE, TLB_FLOOR) + TLB_SPREAD  # ~10.8%
TLB_AMORTIZATION = 0.08  # 8% mandatory annual amortization

# Senior Debt: 9.5% with no amortization
SD_RATE = 0.095
SD_AMORTIZATION = 0.0  # No amortization during projection period

# Cash to Balance Sheet: 2.5% of revenue
CASH_TO_BS_PCT = 0.025

# Exit Multiple
EXIT_MULTIPLE = 20.0

# Tax Rate
TAX_RATE = 0.21  # US corporate tax rate

# =============================================================================
# FINANCIAL PROJECTIONS
# =============================================================================

# Revenue growth based on analyst estimates
# FY2025 (Nov 2025): $23.769B
# FY2026 (est): $26.064B (avg estimate) -> ~9.65% growth
# FY2027 (est): $28.357B (avg estimate) -> ~8.80% growth
# For years 3-5, assume linear decline to ~6%

revenue = {}
revenue[2026] = 26063561740  # Analyst consensus
revenue[2027] = 28356525680  # Analyst consensus
# Years 3-5: assume ~7%, 6.5%, 6% growth
revenue[2028] = revenue[2027] * 1.07
revenue[2029] = revenue[2028] * 1.065
revenue[2030] = revenue[2029] * 1.06

# EBITDA margin expansion (Adobe has been expanding margins)
# FY2025: EBITDA/Revenue = 9.788/23.769 = 41.2%
# Project modest margin expansion
ebitda_margin = {}
ebitda_margin[2026] = 0.415
ebitda_margin[2027] = 0.420
ebitda_margin[2028] = 0.425
ebitda_margin[2029] = 0.430
ebitda_margin[2030] = 0.435

ebitda = {yr: revenue[yr] * ebitda_margin[yr] for yr in range(2026, 2031)}

# D&A projection (based on historical ~$818M in FY2025, modest growth)
da = {}
da[2026] = 980000000
da[2027] = 1050000000
da[2028] = 1120000000
da[2029] = 1200000000
da[2030] = 1280000000

# CapEx (historical ~$179M in FY2025, assume ~1% of revenue)
capex = {}
for yr in range(2026, 2031):
    capex[yr] = revenue[yr] * 0.008

# SBC (Stock-Based Compensation, historical ~$1.94B)
sbc = {}
sbc[2026] = 2000000000
sbc[2027] = 2050000000
sbc[2028] = 2100000000
sbc[2029] = 2150000000
sbc[2030] = 2200000000

# =============================================================================
# LBO CALCULATIONS
# =============================================================================

# Initial Debt Balances
tlb_balance = TLB_AMOUNT
sd_balance = SD_AMOUNT

# Projections
projections = {}
for yr in range(2026, 2031):
    # EBITDA
    ebit = ebitda[yr] - da[yr]
    
    # Interest Expense
    tlb_interest = tlb_balance * TLB_RATE
    sd_interest = sd_balance * SD_RATE
    total_interest = tlb_interest + sd_interest
    
    # EBT
    ebt = ebit - total_interest
    
    # Taxes
    tax = max(0, ebt * TAX_RATE)
    
    # Net Income
    net_income = ebt - tax
    
    # Amortization
    tlb_amort = tlb_balance * TLB_AMORTIZATION
    sd_amort = 0  # No amortization on senior debt
    
    # Ending Balances
    tlb_balance = tlb_balance - tlb_amort
    sd_balance = sd_balance - sd_amort
    
    # Cash to Balance Sheet
    cash_to_bs = revenue[yr] * CASH_TO_BS_PCT
    
    # Free Cash Flow
    fcf = ebitda[yr] - capex[yr] - tlb_amort - sd_amort - cash_to_bs
    
    # Tax shield from interest
    tax_shield = total_interest * TAX_RATE
    
    projections[yr] = {
        'revenue': revenue[yr],
        'ebitda': ebitda[yr],
        'ebit': ebit,
        'da': da[yr],
        'capex': capex[yr],
        'sbc': sbc[yr],
        'tlb_interest': tlb_interest,
        'sd_interest': sd_interest,
        'total_interest': total_interest,
        'ebt': ebt,
        'tax': tax,
        'net_income': net_income,
        'tlb_amort': tlb_amort,
        'sd_amort': sd_amort,
        'tlb_balance_end': tlb_balance,
        'sd_balance_end': sd_balance,
        'total_debt_end': tlb_balance + sd_balance,
        'cash_to_bs': cash_to_bs,
        'fcf': fcf,
    }

# =============================================================================
# EXIT CALCULATIONS
# =============================================================================

exit_year = 2030
exit_ev = ebitda[exit_year] * EXIT_MULTIPLE

# Pay down debt with FCF and cash on balance sheet
# Cumulative FCF used for debt paydown
cumulative_fcf = sum(projections[yr]['fcf'] for yr in range(2026, 2031))

# Exit debt balances
tlb_exit = projections[exit_year]['tlb_balance_end']
sd_exit = projections[exit_year]['sd_balance_end']
total_debt_exit = tlb_exit + sd_exit

# Enterprise Value at Exit
# Exit EV = Exit EBITDA * Exit Multiple
exit_equity_value = exit_ev - total_debt_exit

# Exit Proceeds
# Add cash on balance sheet at exit (cumulative cash to balance sheet)
cash_on_bs = sum(projections[yr]['cash_to_bs'] for yr in range(2026, 2031))
exit_proceeds = exit_equity_value + cash_on_bs

# Less: Pay off remaining debt
net_proceeds = exit_proceeds - total_debt_exit

# Initial Equity Check
# Actually, the sponsor equity is what they put in
# But we need to check if the deal works

# =============================================================================
# RETURNS ANALYSIS
# =============================================================================

# Net Proceeds to Sponsor
# Exit Equity Value - Exit Debt + Cash on Balance Sheet
sponsor_return = net_proceeds

# IRR and MOIC
moic = sponsor_return / SPONSOR_EQUITY
irr = (sponsor_return / SPONSOR_EQUITY) ** (1 / HOLDING_PERIOD) - 1

# =============================================================================
# PRINT SUMMARY
# =============================================================================

print("=" * 80)
print("ADOBE (ADBE) - LBO ANALYSIS SUMMARY")
print("=" * 80)
print(f"\n--- TRANSACTION OVERVIEW ---")
print(f"Purchase Price per Share: ${PRICE_PER_SHARE:,.2f}")
print(f"Shares Outstanding: {SHARES_OUTSTANDING:,}")
print(f"Equity Value: ${EQUITY_VALUE/1e9:,.2f}B")
print(f"Total Debt (pre-acquisition): ${TOTAL_DEBT/1e9:,.2f}B")
print(f"Cash & Equivalents: ${CASH_AND_EQUIVALENTS/1e9:,.2f}B")
print(f"Net Debt (pre-acquisition): ${NET_DEBT/1e9:,.2f}B")
print(f"Enterprise Value: ${ENTERPRISE_VALUE/1e9:,.2f}B")

print(f"\n--- FINANCING STRUCTURE ---")
print(f"Term Loan B ({TLB_MULTIPLE}x EBITDA): ${TLB_AMOUNT/1e9:,.2f}B")
print(f"  Rate: {TLB_RATE*100:.2f}% (SOFR {SOFR_RATE*100:.1f}% floor {TLB_FLOOR*100:.1f}% + {TLB_SPREAD*100:.0f}bps)")
print(f"  Amortization: {TLB_AMORTIZATION*100:.0f}% annually")
print(f"Senior Debt ({SD_MULTIPLE}x EBITDA): ${SD_AMOUNT/1e9:,.2f}B")
print(f"  Rate: {SD_RATE*100:.2f}%")
print(f"  Amortization: {SD_AMORTIZATION*100:.0f}%")
print(f"Total Debt Raised: ${TOTAL_DEBT_PROVIDED/1e9:,.2f}B")
print(f"Sponsor Equity: ${SPONSOR_EQUITY/1e9:,.2f}B")
print(f"Transaction Fees (2.5%): ${TOTAL_FEES*0.025/0.055:,.0f}")
print(f"Financing Fees (3.0%): ${TOTAL_FEES*0.030/0.055:,.0f}")

print(f"\n--- KEY ASSUMPTIONS ---")
print(f"Transaction Date: {TRANSACTION_DATE}")
print(f"Exit Date: {EXIT_DATE}")
print(f"Holding Period: {HOLDING_PERIOD} years")
print(f"Exit Multiple: {EXIT_MULTIPLE}x EBITDA")
print(f"Tax Rate: {TAX_RATE*100:.0f}%")
print(f"Cash to Balance Sheet: {CASH_TO_BS_PCT*100:.1f}% of Revenue")

print(f"\n--- PROJECTIONS ---")
print(f"{'Year':<8} {'Revenue':>14} {'EBITDA':>14} {'FCF':>14} {'TLB Bal':>14} {'SD Bal':>14}")
for yr in range(2026, 2031):
    p = projections[yr]
    print(f"{yr:<8} ${p['revenue']/1e9:>12.2f}B ${p['ebitda']/1e9:>12.2f}B ${p['fcf']/1e9:>12.2f}B ${p['tlb_balance_end']/1e9:>12.2f}B ${p['sd_balance_end']/1e9:>12.2f}B")

print(f"\n--- EXIT ANALYSIS ---")
print(f"Exit EBITDA ({exit_year}): ${ebitda[exit_year]/1e9:,.2f}B")
print(f"Exit Multiple: {EXIT_MULTIPLE}x")
print(f"Exit Enterprise Value: ${exit_ev/1e9:,.2f}B")
print(f"Exit Total Debt: ${total_debt_exit/1e9:,.2f}B")
print(f"Exit Equity Value: ${exit_equity_value/1e9:,.2f}B")
print(f"Cash on Balance Sheet: ${cash_on_bs/1e9:,.2f}B")
print(f"Total Exit Proceeds: ${exit_proceeds/1e9:,.2f}B")
print(f"Net Proceeds to Sponsor: ${sponsor_return/1e9:,.2f}B")

print(f"\n--- INVESTOR RETURNS ---")
print(f"Sponsor Equity Invested: ${SPONSOR_EQUITY/1e9:,.2f}B")
print(f"Net Proceeds: ${sponsor_return/1e9:,.2f}B")
print(f"MOIC: {moic:.2f}x")
print(f"IRR: {irr*100:.1f}%")

# Save key values for Excel
import json
lbo_data = {
    'PRICE_PER_SHARE': PRICE_PER_SHARE,
    'SHARES_OUTSTANDING': SHARES_OUTSTANDING,
    'EQUITY_VALUE': EQUITY_VALUE,
    'TOTAL_DEBT_PRE': TOTAL_DEBT,
    'CASH_PRE': CASH_AND_EQUIVALENTS,
    'NET_DEBT_PRE': NET_DEBT,
    'ENTERPRISE_VALUE': ENTERPRISE_VALUE,
    'TLB_AMOUNT': TLB_AMOUNT,
    'SD_AMOUNT': SD_AMOUNT,
    'TOTAL_DEBT_PROVIDED': TOTAL_DEBT_PROVIDED,
    'SPONSOR_EQUITY': SPONSOR_EQUITY,
    'TOTAL_FEES': TOTAL_FEES,
    'TLB_RATE': TLB_RATE,
    'SD_RATE': SD_RATE,
    'TLB_AMORTIZATION': TLB_AMORTIZATION,
    'EXIT_MULTIPLE': EXIT_MULTIPLE,
    'TAX_RATE': TAX_RATE,
    'CASH_TO_BS_PCT': CASH_TO_BS_PCT,
    'HOLDING_PERIOD': HOLDING_PERIOD,
    'EXIT_EV': exit_ev,
    'TOTAL_DEBT_EXIT': total_debt_exit,
    'EXIT_EQUITY_VALUE': exit_equity_value,
    'CASH_ON_BS': cash_on_bs,
    'EXIT_PROCEEDS': exit_proceeds,
    'SPONSOR_RETURN': sponsor_return,
    'MOIC': moic,
    'IRR': irr,
    'projections': {str(k): v for k, v in projections.items()},
    'revenue': {str(k): v for k, v in revenue.items()},
    'ebitda': {str(k): v for k, v in ebitda.items()},
}

with open('/home/agent/workspace/lbo_data.json', 'w') as f:
    json.dump(lbo_data, f)

print("\n\nLBO data saved to JSON for Excel model creation.")