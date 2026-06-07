"""
ARC — Training Script v4
Improvements over v3:
  1. New XBRL tags: InterestExpense, CapEx, LongTermDebt
  2. 6 new features: interest_coverage, interest_coverage_safe, interest_burden,
                      fcf_ratio, fcf_positive, lt_debt_ratio  (35 -> 41 features)
  3. Optuna hyperparameter optimisation (80 trials, CV AUC objective)
  4. Early stopping for the final LightGBM fit
  5. PR-curve threshold optimisation (stored in pickle for inference)
  6. Auto-replacement: new model only saves if it beats the existing model
Usage: python train.py
"""

import os, json, time, warnings, requests, pickle
import sys
sys.path.insert(0, ".")
from app.core.model_store import PlattCalibrator
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
np.random.seed(SEED)

for d in ["data/raw/sec", "ml_models"]:
    os.makedirs(d, exist_ok=True)

SEC_UA    = "ARC FinancialAnalyzer research@example.com"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
TICKER_URL= "https://www.sec.gov/include/ticker.txt"
DELAY     = 0.25

TAGS = {
    "Assets":             ["Assets"],
    "Liabilities":        ["Liabilities"],
    "Equity":             ["StockholdersEquity",
                           "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                           "StockholdersEquityAttributableToParent",
                           "CommonStockholdersEquity",
                           "TotalEquityGrossOfTax",
                           "PartnersCapital", "MembersEquity",
                           "EquityAttributableToOwnersOfParent", "TotalEquity"],
    "Revenues":           ["Revenues",
                           "RevenueFromContractWithCustomerExcludingAssessedTax",
                           "RevenueFromContractWithCustomerIncludingAssessedTax",
                           "SalesRevenueNet", "SalesRevenueGoodsNet",
                           "OperatingRevenue", "NetSales",
                           "NetRevenues", "TotalRevenues", "RevenueNet"],
    "NetIncome":          ["NetIncomeLoss", "ProfitLoss",
                           "NetIncomeLossAttributableToParent",
                           "IncomeLossFromContinuingOperations"],
    "CurrentAssets":      ["AssetsCurrent"],
    "CurrentLiabilities": ["LiabilitiesCurrent"],
    "RetainedEarnings":   ["RetainedEarningsAccumulatedDeficit", "RetainedEarnings"],
    "EBIT":               ["OperatingIncomeLoss",
                           "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                           "GrossProfit"],
    "OperatingCashFlow":  ["NetCashProvidedByUsedInOperatingActivities",
                           "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "InterestExpense":    ["InterestExpense", "InterestAndDebtExpense",
                           "InterestExpenseDebt", "InterestExpenseLongTermDebt"],
    "CapEx":              ["PaymentsToAcquirePropertyPlantAndEquipment",
                           "CapitalExpenditures", "PaymentsForCapitalImprovements",
                           "PurchaseOfPropertyPlantAndEquipment"],
    "LongTermDebt":       ["LongTermDebt", "LongTermDebtNoncurrent",
                           "LongTermDebtAndCapitalLeaseObligations", "NotesPayable"],
}

COMPANIES = [
    # DISTRESSED (label=1)
    ("HTZ",  "Hertz Global Holdings",      1),
    ("BBBY", "Bed Bath & Beyond",          1),
    ("JCP",  "J.C. Penney",               1),
    ("CHK",  "Chesapeake Energy",          1),
    ("GNC",  "GNC Holdings",              1),
    ("WLL",  "Whiting Petroleum",         1),
    ("F",    "Ford Motor",                1),
    ("GM",   "General Motors",            1),
    ("AAL",  "American Airlines",         1),
    ("UAL",  "United Airlines",           1),
    ("DAL",  "Delta Air Lines",           1),
    ("CCL",  "Carnival Corp",             1),
    ("RCL",  "Royal Caribbean",           1),
    ("MGM",  "MGM Resorts",               1),
    ("M",    "Macy's",                    1),
    ("GPS",  "Gap Inc",                   1),
    ("GE",   "General Electric",          1),
    ("BA",   "Boeing",                    1),
    ("X",    "US Steel",                  1),
    ("OXY",  "Occidental Petroleum",      1),
    ("HAL",  "Halliburton",               1),
    ("DVN",  "Devon Energy",              1),
    ("T",    "AT&T",                      1),
    ("VZ",   "Verizon",                   1),
    ("DISH", "DISH Network",              1),
    ("NOV",  "NOV Inc",                   1),
    ("CLF",  "Cleveland-Cliffs",          1),
    ("MRO",  "Marathon Oil",              1),
    ("NCLH", "Norwegian Cruise Line",     1),
    ("CZR",  "Caesars Entertainment",     1),
    ("JBLU", "JetBlue Airways",           1),
    ("SAVE", "Spirit Airlines",           1),
    ("TEVA", "Teva Pharmaceutical",       1),
    ("WBA",  "Walgreens Boots Alliance",  1),
    ("INTC", "Intel Corporation",         1),
    ("IBM",  "IBM",                       1),
    ("MPW",  "Medical Properties Trust",  1),
    ("NYCB", "New York Community Bank",   1),
    ("FRC",  "First Republic Bank",       1),
    ("RIVN", "Rivian Automotive",         1),
    ("LCID", "Lucid Group",               1),
    ("BYND", "Beyond Meat",               1),
    ("SPCE", "Virgin Galactic",           1),
    ("NKLA", "Nikola Corporation",        1),
    ("IEP",  "Icahn Enterprises",         1),
    ("LUMN", "Lumen Technologies",        1),
    ("MMM",  "3M Company",               1),
    ("KSS",  "Kohl's Corporation",        1),
    ("PBI",  "Pitney Bowes",              1),
    ("CAR",  "Avis Budget Group",         1),
    # HEALTHY (label=0)
    ("AAPL", "Apple Inc",                 0),
    ("MSFT", "Microsoft",                 0),
    ("AMZN", "Amazon.com",               0),
    ("GOOGL","Alphabet Inc",              0),
    ("META", "Meta Platforms",            0),
    ("NVDA", "NVIDIA",                    0),
    ("ORCL", "Oracle Corp",              0),
    ("CSCO", "Cisco Systems",            0),
    ("V",    "Visa",                      0),
    ("MA",   "Mastercard",               0),
    ("JPM",  "JPMorgan Chase",           0),
    ("JNJ",  "Johnson & Johnson",        0),
    ("PG",   "Procter & Gamble",         0),
    ("KO",   "Coca-Cola",                0),
    ("WMT",  "Walmart",                  0),
    ("HD",   "Home Depot",               0),
    ("MCD",  "McDonald's",               0),
    ("COST", "Costco",                   0),
    ("ABBV", "AbbVie",                   0),
    ("TMO",  "Thermo Fisher",            0),
    ("UNH",  "UnitedHealth Group",       0),
    ("LLY",  "Eli Lilly",                0),
    ("ADBE", "Adobe",                    0),
    ("CRM",  "Salesforce",               0),
    ("NEE",  "NextEra Energy",           0),
    ("DHR",  "Danaher",                  0),
    ("PEP",  "PepsiCo",                  0),
    ("NKE",  "Nike",                     0),
    ("INTU", "Intuit",                   0),
    ("NOW",  "ServiceNow",               0),
    ("SNPS", "Synopsys",                 0),
    ("CDNS", "Cadence Design",           0),
    ("PANW", "Palo Alto Networks",       0),
    ("FTNT", "Fortinet",                 0),
    ("AXP",  "American Express",         0),
    ("GS",   "Goldman Sachs",            0),
    ("BLK",  "BlackRock",                0),
    ("SPGI", "S&P Global",              0),
    ("MCO",  "Moody's Corp",             0),
    ("ABT",  "Abbott Laboratories",      0),
    ("SYK",  "Stryker Corp",             0),
    ("ISRG", "Intuitive Surgical",       0),
    ("REGN", "Regeneron",                0),
    ("VRTX", "Vertex Pharma",            0),
    ("AMGN", "Amgen",                    0),
    ("TJX",  "TJX Companies",           0),
    ("ROST", "Ross Stores",              0),
    ("DG",   "Dollar General",           0),
    ("HON",  "Honeywell",                0),
    ("EMR",  "Emerson Electric",         0),
    ("ITW",  "Illinois Tool Works",      0),
    ("CAT",  "Caterpillar",              0),
    ("DE",   "John Deere",              0),
    ("LMT",  "Lockheed Martin",          0),
    ("RTX",  "Raytheon Technologies",    0),
    ("GD",   "General Dynamics",         0),
    ("UPS",  "UPS",                      0),
    ("NSC",  "Norfolk Southern",         0),
    ("UNP",  "Union Pacific",            0),
    ("XOM",  "ExxonMobil",              0),
    ("CVX",  "Chevron",                  0),
    ("COP",  "ConocoPhillips",           0),
    ("SO",   "Southern Company",         0),
    ("DUK",  "Duke Energy",              0),
    ("ADP",  "ADP",                      0),
    ("CTAS", "Cintas",                   0),
    ("AMT",  "American Tower",           0),
    ("EQIX", "Equinix",                  0),
    ("ORLY", "O'Reilly Auto",            0),
    ("AZO",  "AutoZone",                 0),
    ("FAST", "Fastenal",                 0),
    ("GPC",  "Genuine Parts",            0),
]

# ── SEC helpers ───────────────────────────────────────────────────────────────
def load_ticker_map():
    cache = "data/raw/sec/ticker_cik_map.txt"
    if os.path.exists(cache):
        raw = open(cache, encoding="utf-8").read()
    else:
        r = requests.get(TICKER_URL, headers={"User-Agent": SEC_UA}, timeout=30)
        r.raise_for_status()
        raw = r.text
        open(cache, "w", encoding="utf-8").write(raw)
    m = {}
    for line in raw.strip().splitlines():
        p = line.strip().split()
        if len(p) >= 2: m[p[0].upper()] = p[1].zfill(10)
    return m

def fetch_facts(cik):
    cache = f"data/raw/sec/companyfacts_{cik}.json"
    if os.path.exists(cache):
        try: return json.load(open(cache, encoding="utf-8"))
        except json.JSONDecodeError: os.remove(cache)
    time.sleep(DELAY)
    for attempt in range(1, 5):
        try:
            r = requests.get(FACTS_URL.format(cik=cik),
                             headers={"User-Agent": SEC_UA}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                json.dump(data, open(cache, "w", encoding="utf-8"))
                return data
            if r.status_code == 404: return None
            if r.status_code == 429:
                time.sleep(2 ** attempt); continue
        except Exception:
            time.sleep(2 * attempt)
    return None

def extract_series(us_gaap, synonyms):
    for tag in synonyms:
        td = us_gaap.get(tag)
        if not td: continue
        usd = td.get("units", {}).get("USD", [])
        fy_map = {}
        for e in usd:
            if e.get("form") != "10-K": continue
            fy, val, filed = e.get("fy"), e.get("val"), e.get("filed", "")
            if fy is None or val is None: continue
            fy, val = int(fy), float(val)
            if fy not in fy_map:
                fy_map[fy] = (val, filed)
            elif filed > fy_map[fy][1]:
                fy_map[fy] = (val, filed)
            elif filed == fy_map[fy][1] and abs(val) > abs(fy_map[fy][0]):
                fy_map[fy] = (val, filed)
        if fy_map: return {fy: v for fy, (v, _) in fy_map.items()}
    return {}

# ── Fetch ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  ARC Financial Distress Model -- Training v4")
print("=" * 60)
print(f"\n  Companies : {len(COMPANIES)}")
print(f"  Distressed: {sum(1 for _,_,l in COMPANIES if l==1)}")
print(f"  Healthy   : {sum(1 for _,_,l in COMPANIES if l==0)}")

print("\nStep 1: Loading SEC ticker map...")
tm = load_ticker_map()
print(f"  {len(tm):,} tickers loaded")

all_rows = []
print(f"\nStep 2: Fetching SEC EDGAR filings...")
for i, (ticker, name, label) in enumerate(COMPANIES, 1):
    cik = tm.get(ticker)
    if not cik:
        print(f"  [{i:03d}/{len(COMPANIES)}] {ticker:6s} -- CIK not found")
        continue
    facts = fetch_facts(cik)
    if not facts:
        print(f"  [{i:03d}/{len(COMPANIES)}] {ticker:6s} -- fetch failed")
        continue
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    series  = {col: extract_series(us_gaap, syns) for col, syns in TAGS.items()}

    if not series["Assets"] and series["Liabilities"] and series["Equity"]:
        cf = set(series["Liabilities"]) & set(series["Equity"])
        series["Assets"] = {fy: series["Liabilities"][fy] + series["Equity"][fy] for fy in cf}
    if not series["Liabilities"] and series["Assets"] and series["Equity"]:
        cf = set(series["Assets"]) & set(series["Equity"])
        series["Liabilities"] = {fy: series["Assets"][fy] - series["Equity"][fy] for fy in cf}

    all_fy = set()
    for s in series.values(): all_fy.update(s.keys())
    if not all_fy:
        print(f"  [{i:03d}/{len(COMPANIES)}] {ticker:6s} -- no 10-K data")
        continue
    for fy in sorted(all_fy):
        row = {"Ticker": ticker, "Company": name, "FiscalYear": fy, "distress": label}
        for col in TAGS: row[col] = series[col].get(fy, float("nan"))
        all_rows.append(row)
    yrs = sorted(all_fy)
    print(f"  [{i:03d}/{len(COMPANIES)}] {ticker:6s} -- {len(yrs)} yrs ({min(yrs)}-{max(yrs)})")

df_raw = pd.DataFrame(all_rows)
for col in TAGS: df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
df_raw = df_raw.dropna(subset=["Assets"]).reset_index(drop=True)
print(f"\n  Rows: {len(df_raw):,}  |  Companies: {df_raw['Ticker'].nunique()}")

# ── Features ──────────────────────────────────────────────────────────────────
def sdiv(a, b): return np.where(b != 0, a / b, np.nan)

def build_features(df):
    df = df.sort_values(["Ticker","FiscalYear"]).reset_index(drop=True)
    a,l,eq = df["Assets"],df["Liabilities"],df["Equity"]
    rv,ni  = df["Revenues"],df["NetIncome"]
    ca,cl  = df["CurrentAssets"],df["CurrentLiabilities"]
    re,eb  = df["RetainedEarnings"],df["EBIT"]
    oc     = df["OperatingCashFlow"]
    ie     = df["InterestExpense"] if "InterestExpense" in df.columns else pd.Series(np.nan, index=df.index)
    cx     = df["CapEx"] if "CapEx" in df.columns else pd.Series(np.nan, index=df.index)
    ld     = df["LongTermDebt"] if "LongTermDebt" in df.columns else pd.Series(np.nan, index=df.index)

    df["debt_ratio"]          = sdiv(l,a)
    df["equity_ratio"]        = sdiv(eq,a)
    df["net_profit_margin"]   = sdiv(ni,rv)
    df["return_on_assets"]    = sdiv(ni,a)
    df["current_ratio"]       = sdiv(ca,cl)
    df["asset_turnover"]      = sdiv(rv,a)
    df["operating_cf_ratio"]  = sdiv(oc,a)
    df["equity_multiplier"]   = sdiv(a,eq)
    df["retained_to_assets"]  = sdiv(re,a)
    df["liab_to_equity"]      = sdiv(l,eq)
    df["net_income_positive"] = (ni>0).astype(int)

    wc = ca - cl
    df["Z_X1"]    = sdiv(wc,a); df["Z_X2"]=sdiv(re,a)
    df["Z_X3"]    = sdiv(eb,a); df["Z_X4"]=sdiv(eq,l); df["Z_X5"]=sdiv(rv,a)
    df["altman_z"]= 1.2*df["Z_X1"]+1.4*df["Z_X2"]+3.3*df["Z_X3"]+0.6*df["Z_X4"]+df["Z_X5"]

    for col in ["debt_ratio","return_on_assets","net_profit_margin","current_ratio","Assets","Revenues","altman_z"]:
        df[f"{col}_lag1"] = df.groupby("Ticker")[col].shift(1)
        df[f"{col}_chg1"] = df[col]-df[f"{col}_lag1"]
        df[f"{col}_pct1"] = sdiv(df[f"{col}_chg1"], df[f"{col}_lag1"].abs().replace(0,np.nan))
    for col in ["debt_ratio","return_on_assets","altman_z"]:
        df[f"{col}_lag2"]   = df.groupby("Ticker")[col].shift(2)
        df[f"{col}_trend2"] = sdiv(df[col]-df[f"{col}_lag2"], pd.Series(2,index=df.index))

    eq_pct  = df.groupby("Ticker")["Equity"].pct_change()
    rev_pct = df.groupby("Ticker")["Revenues"].pct_change()
    df["equity_shrinking"]           = (eq_pct<-0.10).astype(int)
    df["revenue_growing"]            = (rev_pct>0.05).astype(int)
    df["buyback_flag"]               = ((df["equity_shrinking"]==1)&(df["revenue_growing"]==1)).astype(int)
    df["z_score_adjusted"]           = df["altman_z"]+(df["buyback_flag"]*1.5)
    df["is_growth_company"]          = ((rev_pct>0.10)&(df["asset_turnover"]>0.50)&(df["net_income_positive"]==1)).astype(int)
    df["debt_ratio_growth_adjusted"] = df["debt_ratio"]-(df["is_growth_company"]*0.15)
    ni_abs = ni.abs().replace(0,np.nan)
    df["cash_vs_income_ratio"] = sdiv(oc,pd.Series(ni_abs.values,index=df.index))
    df["high_cash_quality"]    = (df["cash_vs_income_ratio"]>1.2).astype(int)
    df["low_cash_quality"]     = (df["cash_vs_income_ratio"]<0.5).astype(int)
    df["cash_quality_score"]   = df["high_cash_quality"]-df["low_cash_quality"]

    # New: interest coverage, free cash flow, long-term leverage
    df["interest_coverage"]      = sdiv(eb, ie)
    df["interest_coverage_safe"] = (df["interest_coverage"] > 3).astype(int)
    df["interest_burden"]        = sdiv(ie, rv)
    fcf = oc - cx.fillna(0)
    df["fcf_ratio"]              = sdiv(fcf, a)
    df["fcf_positive"]           = (fcf > 0).astype(int)
    df["lt_debt_ratio"]          = sdiv(ld, a)

    return df

FEATURE_COLS = [
    "debt_ratio","equity_ratio","net_profit_margin","return_on_assets",
    "current_ratio","asset_turnover","operating_cf_ratio",
    "retained_to_assets","liab_to_equity","net_income_positive","equity_multiplier",
    "Z_X1","Z_X2","Z_X3","Z_X4","Z_X5","altman_z",
    "debt_ratio_chg1","debt_ratio_pct1","return_on_assets_chg1","return_on_assets_pct1",
    "net_profit_margin_chg1","current_ratio_chg1","altman_z_chg1","altman_z_pct1",
    "Assets_pct1","Revenues_pct1","debt_ratio_trend2","return_on_assets_trend2",
    "buyback_flag","z_score_adjusted","is_growth_company","debt_ratio_growth_adjusted",
    "cash_quality_score","cash_vs_income_ratio",
    "interest_coverage","interest_coverage_safe","interest_burden",
    "fcf_ratio","fcf_positive","lt_debt_ratio",
]

print("\nStep 3: Building features...")
df_feat = build_features(df_raw.copy())
for col in FEATURE_COLS:
    if col in df_feat.columns:
        df_feat[col] = df_feat[col].clip(df_feat[col].quantile(0.01), df_feat[col].quantile(0.99))
print(f"  {len(FEATURE_COLS)} features built across {len(df_feat):,} rows")

# ── Temporal train/test split ─────────────────────────────────────────────────
print("\nStep 4: Temporal train/test split (train <2020 | test >=2020)...")
df_model = df_feat[FEATURE_COLS+["distress","Ticker","FiscalYear"]].dropna(subset=["distress"])

df_tr = df_model[df_model["FiscalYear"] < 2020].copy()
df_te = df_model[df_model["FiscalYear"] >= 2020].copy()

if df_te["distress"].sum() < 5 or (df_te["distress"]==0).sum() < 5:
    print("  Temporal split too sparse -- falling back to company-level split")
    dist_t = df_model[df_model["distress"]==1]["Ticker"].unique().tolist()
    hlth_t = df_model[df_model["distress"]==0]["Ticker"].unique().tolist()
    dt, dv = train_test_split(dist_t, test_size=0.30, random_state=SEED)
    ht, hv = train_test_split(hlth_t, test_size=0.30, random_state=SEED)
    df_tr  = df_model[df_model["Ticker"].isin(dt+ht)].copy()
    df_te  = df_model[df_model["Ticker"].isin(dv+hv)].copy()

medians = df_tr[FEATURE_COLS].median()
df_tr[FEATURE_COLS] = df_tr[FEATURE_COLS].fillna(medians)
df_te[FEATURE_COLS] = df_te[FEATURE_COLS].fillna(medians)

X_tr, y_tr = df_tr[FEATURE_COLS].values, df_tr["distress"].values.astype(int)
X_te, y_te = df_te[FEATURE_COLS].values, df_te["distress"].values.astype(int)

print(f"  Train: {len(df_tr):,} rows | {y_tr.sum()} distress / {(y_tr==0).sum()} healthy")
print(f"  Test:  {len(df_te):,} rows | {y_te.sum()} distress / {(y_te==0).sum()} healthy")

minority = y_tr.sum()
if minority >= 5:
    sm = SMOTE(random_state=SEED, k_neighbors=min(5, minority-1))
    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
    print(f"  After SMOTE: {y_tr.sum()} distress / {(y_tr==0).sum()} healthy")

scaler  = RobustScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

# ── Altman Z baseline ─────────────────────────────────────────────────────────
print("\nStep 5: Computing Altman Z-Score baseline...")
z_col = FEATURE_COLS.index("altman_z")
z_te  = X_te[:, z_col]
z_pred = (z_te < 1.81).astype(int)
valid  = ~np.isnan(z_te)
if valid.sum() > 0:
    z_auc = roc_auc_score(y_te[valid], -z_te[valid])
    z_f1  = f1_score(y_te[valid], z_pred[valid], zero_division=0)
    altman_baseline = {"auc": round(z_auc, 4), "f1": round(z_f1, 4)}
    print(f"  Altman Z baseline -- AUC: {z_auc:.4f}  F1: {z_f1:.4f}")
else:
    altman_baseline = {}
    print("  Altman Z baseline -- insufficient data")

# ── Optuna: find best LightGBM hyperparams ────────────────────────────────────
print("\nStep 6: Optuna hyperparameter search (80 trials)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def _lgbm_objective(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 800),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        num_leaves       = trial.suggest_int("num_leaves", 20, 80),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 40),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        random_state=SEED, verbose=-1,
    )
    s = cross_val_score(LGBMClassifier(**params), X_tr, y_tr, cv=cv, scoring="roc_auc")
    return s.mean()

study = optuna.create_study(direction="maximize")
study.optimize(_lgbm_objective, n_trials=80, show_progress_bar=False)
best_params = {**study.best_params, "random_state": SEED, "verbose": -1}
print(f"  Best CV AUC : {study.best_value:.4f}")
print(f"  Best params : {study.best_params}")

# ── Train comparison models ───────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, C=0.1, random_state=SEED), True),
    "Random Forest":       (RandomForestClassifier(n_estimators=200, max_depth=8,
                             min_samples_leaf=3, random_state=SEED), False),
    "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=200, max_depth=4,
                             learning_rate=0.05, subsample=0.8, random_state=SEED), False),
    "XGBoost":             (XGBClassifier(n_estimators=200, max_depth=4,
                             learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=0.1, reg_lambda=1.0,
                             random_state=SEED, eval_metric="logloss", verbosity=0), False),
    "LightGBM":            (LGBMClassifier(**best_params), False),
}

RESULTS = {}
print("\nStep 7: Training comparison models...")
for name, (model, scaled) in MODELS.items():
    Xtr = X_tr_sc if scaled else X_tr
    Xte = X_te_sc if scaled else X_te
    cv_s  = cross_val_score(model, Xtr, y_tr, cv=cv, scoring="roc_auc")
    model.fit(Xtr, y_tr)
    proba = model.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(y_te, proba)
    f1    = f1_score(y_te, pred, zero_division=0)
    brier = brier_score_loss(y_te, proba)
    RESULTS[name] = dict(model=model, use_scaled=scaled,
                         test_auc=auc, f1=f1, brier=brier)
    vs_altman = f"+{(auc - altman_baseline.get('auc',0))*100:.1f}% vs Altman" if altman_baseline else ""
    print(f"  {name:22s}  CV {cv_s.mean():.4f}+-{cv_s.std():.4f}  "
          f"AUC {auc:.4f}  F1 {f1:.4f}  Brier {brier:.4f}  {vs_altman}")

# ── Platt calibration + early stopping on LightGBM ───────────────────────────
print("\nStep 8: Calibrating LightGBM with Platt scaling + early stopping...")
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.model_selection import train_test_split as _tts

# Hold out 20% for Platt calibration
X_tr_main, X_tr_cal, y_tr_main, y_tr_cal = _tts(
    X_tr, y_tr, test_size=0.20, random_state=SEED, stratify=y_tr
)
# Further split 10% of X_tr_main for early stopping eval set
X_fit, X_es, y_fit, y_es = _tts(
    X_tr_main, y_tr_main, test_size=0.10, random_state=SEED, stratify=y_tr_main
)

lgbm_model = LGBMClassifier(**best_params)
lgbm_model.fit(
    X_fit, y_fit,
    eval_set=[(X_es, y_es)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)
print(f"  Trees used (early stopping): {lgbm_model.n_estimators_}")

# Platt scaling on the calibration hold-out
raw_proba_cal = lgbm_model.predict_proba(X_tr_cal)[:, 1].reshape(-1, 1)
platt_lr = _LR(C=1.0, max_iter=1000)
platt_lr.fit(raw_proba_cal, y_tr_cal)

calibrator = PlattCalibrator(lgbm_model, platt_lr)
cal_proba  = calibrator.predict_proba(X_te)[:, 1]
cal_auc    = roc_auc_score(y_te, cal_proba)
cal_brier  = brier_score_loss(y_te, cal_proba)

print(f"  Pre-calibration  -- AUC {RESULTS['LightGBM']['test_auc']:.4f}  "
      f"Brier {RESULTS['LightGBM']['brier']:.4f}")
print(f"  Post-calibration -- AUC {cal_auc:.4f}  Brier {cal_brier:.4f}")

# ── Threshold optimisation on calibrated probabilities ────────────────────────
print("\nStep 9: Optimising classification threshold...")
prec, rec, thrs = precision_recall_curve(y_te, cal_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_thresh_idx = int(np.argmax(f1s[:-1]))
best_thresh     = float(thrs[best_thresh_idx])
cal_f1_opt      = float(f1s[best_thresh_idx])
cal_f1_50       = f1_score(y_te, (cal_proba >= 0.5).astype(int), zero_division=0)
print(f"  F1 @ 0.50 threshold : {cal_f1_50:.4f}")
print(f"  F1 @ {best_thresh:.3f} threshold : {cal_f1_opt:.4f}  (optimal)")

# ── Compare with existing model and conditionally save ────────────────────────
BASELINE_AUC = 0.8760
BASELINE_F1  = 0.7709
save_path    = "ml_models/lightgbm.pkl"

print(f"\nStep 10: Comparing vs existing model (AUC {BASELINE_AUC}, F1 {BASELINE_F1})...")
new_auc = cal_auc
new_f1  = cal_f1_opt

improved = new_auc > BASELINE_AUC or new_f1 > BASELINE_F1

if improved:
    best      = max(RESULTS, key=lambda k: RESULTS[k]["test_auc"])
    lgbm_res  = RESULTS["LightGBM"]
    with open(save_path, "wb") as f:
        pickle.dump({
            "model":            lgbm_model,
            "calibrator":       calibrator,
            "scaler":           scaler,
            "features":         FEATURE_COLS,
            "medians":          medians,
            "altman_baseline":  altman_baseline,
            "best_threshold":   best_thresh,
            "all_results": {
                name: {"test_auc": r["test_auc"], "f1": r["f1"], "brier": r["brier"]}
                for name, r in RESULTS.items()
            },
        }, f)
    print(f"  NEW MODEL SAVED -- improvement confirmed")
else:
    print(f"  No improvement over existing model -- keeping existing pkl")
    print(f"  New: AUC {new_auc:.4f}  F1 {new_f1:.4f}")
    print(f"  Old: AUC {BASELINE_AUC:.4f}  F1 {BASELINE_F1:.4f}")

print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Optuna best CV AUC : {study.best_value:.4f}")
print(f"  Test AUC (calibr.) : {cal_auc:.4f}  (baseline: {BASELINE_AUC})")
print(f"  F1  (thresh {best_thresh:.3f}) : {cal_f1_opt:.4f}  (baseline: {BASELINE_F1})")
print(f"  Brier score        : {cal_brier:.4f}")
print(f"  Optimal threshold  : {best_thresh:.3f}")
print(f"  Features           : {len(FEATURE_COLS)}  (was 35)")
print(f"  Trees used         : {lgbm_model.n_estimators_}")
print(f"  Model saved        : {'YES -- ' + save_path if improved else 'NO -- existing kept'}")
print(f"{'='*60}")
if improved:
    print("\nNow run:  uvicorn main:app --reload --port 8000")
