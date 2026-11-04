"""Calculation and charting helpers for the Superstore analytics report.

Every number in the notebook and in the Word report comes from the functions in this file,
so the two documents cannot disagree with each other.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- colours (validated reference palette: blue/red diverging pair, neutral greys) ----
BLUE, RED, GREY, LGREY = "#2a78d6", "#d03b3b", "#9a9892", "#d9d8d3"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

HIGH_DISC = 0.20                      # a line is "heavily discounted" above this discount
BAND_LABELS = ["No discount", "1-20%", "21-30%", "31-50%", "Over 50%"]


def style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "axes.edgecolor": LGREY, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": .8, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "legend.frameon": False,
    })


# ------------------------------------------------------------------ loading and cleaning
def load_raw(path):
    return pd.read_csv(path)


def split_raw(raw):
    """The CSV holds three tables stacked on top of each other: order lines, a managers list, a returns list."""
    is_line = raw["Order Date"].notna()
    lines, tail = raw[is_line].copy(), raw[~is_line].copy()
    returns = tail.loc[tail["Row ID"] == "Yes", "Order ID"]
    managers = (tail[~tail["Row ID"].isin(["Yes", "Returned", "Person"])][["Row ID", "Order ID"]]
                .rename(columns={"Row ID": "manager", "Order ID": "region"}))
    return lines, returns, managers


def clean(lines, returns):
    d = lines.copy()
    d.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in d.columns]
    d["row_id"] = d["row_id"].astype(int)
    d["quantity"] = d["quantity"].astype(int)
    d["order_date"] = pd.to_datetime(d["order_date"], format="%m/%d/%Y")
    d["ship_date"] = pd.to_datetime(d["ship_date"], format="%m/%d/%Y")
    d["year"] = d["order_date"].dt.year
    d["month"] = d["order_date"].dt.month
    d["quarter"] = d["order_date"].dt.quarter
    d["ship_days"] = (d["ship_date"] - d["order_date"]).dt.days
    d["returned"] = d["order_id"].isin(set(returns))
    d["high_disc"] = d["discount"] > HIGH_DISC
    d["disc_band"] = pd.cut(d["discount"], [-0.001, 0, 0.2, 0.3, 0.5, 1.0], labels=BAND_LABELS)
    d["loss_line"] = d["profit"] < 0
    return d


def validation_checks(d):
    """Business-rule checks used in the data-quality log."""
    return {
        "Row IDs unique": d["row_id"].is_unique,
        "Sales > 0": bool((d["sales"] > 0).all()),
        "Quantity >= 1": bool((d["quantity"] >= 1).all()),
        "Discount between 0 and 1": bool(d["discount"].between(0, 1).all()),
        "Ship date on/after order date": bool((d["ship_days"] >= 0).all()),
        "Order-ID year matches order date": bool((d["order_id"].str[3:7].astype(int) == d["year"]).all()),
        "Each order has one customer": bool((d.groupby("order_id")["customer_id"].nunique() == 1).all()),
        "Each order has one region": bool((d.groupby("order_id")["region"].nunique() == 1).all()),
    }


# ------------------------------------------------------------------ calculations
def _agg(g):
    out = g.agg(sales=("sales", "sum"), profit=("profit", "sum"), lines=("sales", "size"),
                avg_discount=("discount", "mean"), orders=("order_id", "nunique"))
    out["margin"] = out["profit"] / out["sales"]
    return out


def tables(d):
    T = {}
    T["kpi"] = dict(sales=d["sales"].sum(), profit=d["profit"].sum(), margin=d["profit"].sum() / d["sales"].sum(),
                    orders=d["order_id"].nunique(), customers=d["customer_id"].nunique(), lines=len(d),
                    first=d["order_date"].min(), last=d["order_date"].max())
    T["kpi"]["aov"] = T["kpi"]["sales"] / T["kpi"]["orders"]

    y = _agg(d.groupby("year"))
    y["customers"] = d.groupby("year")["customer_id"].nunique()
    y["sales_growth"] = y["sales"].pct_change()
    y["profit_growth"] = y["profit"].pct_change()
    hd = d[d["high_disc"]].groupby("year")
    y["hd_lines"] = hd.size()
    y["hd_lines_share"] = hd.size() / d.groupby("year").size()
    y["hd_sales_share"] = hd["sales"].sum() / y["sales"]
    y["hd_profit"] = hd["profit"].sum()
    y["profit_ex_hd"] = y["profit"] - y["hd_profit"]
    T["yearly"] = y

    T["category"] = _agg(d.groupby("category")).sort_values("profit")
    T["subcat"] = _agg(d.groupby("sub_category")).sort_values("profit")
    T["region"] = _agg(d.groupby("region")).sort_values("margin")
    T["region"]["hd_lines_share"] = d.groupby("region")["high_disc"].mean()
    T["state"] = _agg(d.groupby("state")).sort_values("profit")
    T["state_corr"] = T["state"][["avg_discount", "margin"]].corr().iloc[0, 1]
    T["segment"] = _agg(d.groupby("segment")).sort_values("margin")
    T["ship"] = _agg(d.groupby("ship_mode"))
    T["ship"]["avg_days"] = d.groupby("ship_mode")["ship_days"].mean()
    T["ship"]["share_lines"] = T["ship"]["lines"] / len(d)

    b = _agg(d.groupby("disc_band", observed=True))
    T["band"] = b
    T["hd_total"] = dict(lines=int(d["high_disc"].sum()), lines_share=d["high_disc"].mean(),
                         sales=d.loc[d["high_disc"], "sales"].sum(), sales_share=d.loc[d["high_disc"], "sales"].sum() / d["sales"].sum(),
                         profit=d.loc[d["high_disc"], "profit"].sum(), profit_low=d.loc[~d["high_disc"], "profit"].sum())
    T["loss_lines"] = dict(n=int(d["loss_line"].sum()), share=d["loss_line"].mean(), profit=d.loc[d["loss_line"], "profit"].sum(),
                           share_from_high_disc=d.loc[d["loss_line"] & d["high_disc"], "profit"].sum() / d.loc[d["loss_line"], "profit"].sum())

    # sub-categories: profit when discount is high or not
    sc = d.groupby(["sub_category", "high_disc"])["profit"].sum().unstack()
    sc.columns = ["profit_normal", "profit_high_disc"]
    sc["hd_lines_share"] = d.groupby("sub_category")["high_disc"].mean()
    T["subcat_disc"] = sc

    # profit bridge between the last two years
    y1, y2 = int(y.index[-2]), int(y.index[-1])
    a = d[d["year"] == y1].groupby("category").agg(s=("sales", "sum"), p=("profit", "sum"))
    c = d[d["year"] == y2].groupby("category").agg(s=("sales", "sum"), p=("profit", "sum"))
    br = pd.DataFrame({"volume": (c["s"] - a["s"]) * (a["p"] / a["s"]), "rate": c["s"] * (c["p"] / c["s"] - a["p"] / a["s"])})
    T["bridge"] = dict(y1=y1, y2=y2, start=a["p"].sum(), end=c["p"].sum(), volume=br["volume"].sum(), rate=br["rate"], table=br)

    # customers
    cu = d.groupby("customer_id").agg(sales=("sales", "sum"), profit=("profit", "sum"), orders=("order_id", "nunique"))
    cu = cu.sort_values("profit", ascending=False)
    cu["decile"] = pd.qcut(np.arange(len(cu)), 10, labels=range(1, 11))
    T["cust"] = cu
    top = cu.sort_values("sales", ascending=False).head(len(cu) // 10)
    T["cust_stats"] = dict(n=len(cu), neg_n=int((cu["profit"] < 0).sum()), neg_share=(cu["profit"] < 0).mean(),
                           neg_profit=cu.loc[cu["profit"] < 0, "profit"].sum(), pos_profit=cu.loc[cu["profit"] > 0, "profit"].sum(),
                           top10_sales_share=top["sales"].sum() / cu["sales"].sum(), top10_profit_share=top["profit"].sum() / cu["profit"].sum())
    T["decile"] = cu.groupby("decile", observed=True)["profit"].sum()

    # seasonality
    m = d.groupby(["year", "month"])["sales"].sum().unstack(0)
    T["monthly"] = m
    pooled = d.groupby("month")["sales"].sum()
    T["season"] = dict(q4=pooled.loc[[10, 11, 12]].sum() / pooled.sum(), sepdec=pooled.loc[[9, 10, 11, 12]].sum() / pooled.sum(),
                       janfeb=pooled.loc[[1, 2]].sum() / pooled.sum(), peak_month=int(pooled.idxmax()), low_month=int(pooled.idxmin()))

    # returns (order level)
    o = d.groupby("order_id").agg(region=("region", "first"), year=("year", "first"), returned=("returned", "max"), sales=("sales", "sum"))
    T["ret_region"] = o.groupby("region").agg(orders=("returned", "size"), returned=("returned", "sum"), rate=("returned", "mean")).sort_values("rate")
    T["ret_year"] = o.groupby("year").agg(orders=("returned", "size"), returned=("returned", "sum"), rate=("returned", "mean"))
    T["ret_total"] = dict(orders=int(o["returned"].sum()), rate=o["returned"].mean(), lines=int(d["returned"].sum()),
                          sales=d.loc[d["returned"], "sales"].sum(), sales_share=d.loc[d["returned"], "sales"].sum() / d["sales"].sum())
    T["ret_cat"] = d.groupby("category")["returned"].mean()

    # scenario on the latest year
    last = d[d["year"] == y2]
    loss = last.loc[last["high_disc"], "profit"].sum()
    base_p, base_s = last["profit"].sum(), last["sales"].sum()
    rows = []
    for rec in (0.25, 0.50, 1.00):
        gain = -loss * rec
        rows.append(dict(recovery=rec, gain=gain, profit=base_p + gain, profit_uplift=gain / base_p, margin=(base_p + gain) / base_s))
    T["scenario"] = pd.DataFrame(rows).set_index("recovery")
    T["scenario_base"] = dict(year=y2, loss=loss, profit=base_p, sales=base_s, margin=base_p / base_s)
    return T


# ------------------------------------------------------------------ formatting helpers
def money(x, dec=0):
    s = f"{abs(x):,.{dec}f}"
    return f"-${s}" if x < 0 else f"${s}"


def k(x, dec=1):
    s = f"{abs(x) / 1000:,.{dec}f}k"
    return f"-${s}" if x < 0 else f"${s}"


def pct(x, dec=1):
    return f"{x * 100:.{dec}f}%"


# ------------------------------------------------------------------ figures
def _fmt_k(ax, axis="y"):
    f = mpl.ticker.FuncFormatter(lambda v, _: (f"-${abs(v) / 1000:,.0f}k" if v < 0 else f"${v / 1000:,.0f}k"))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def fig_dashboard(T):
    y = T["yearly"]; kp = T["kpi"]
    fig = plt.figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.8, 2.2], hspace=0.45, wspace=0.35)
    tiles = [("Sales", money(kp["sales"] / 1e6, 2) + "M", f"{int(y.index[0])}-{int(y.index[-1])}"),
             ("Profit", k(kp["profit"], 0), f"margin {pct(kp['margin'])}"),
             ("Orders", f"{kp['orders']:,}", f"{kp['customers']:,} customers"),
             ("Avg order value", money(kp["aov"]), "sales per order")]
    for i, (lab, val, sub) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, i]); ax.axis("off")
        ax.text(0, 0.78, lab, fontsize=10, color=INK2); ax.text(0, 0.28, val, fontsize=24, fontweight="bold", color=INK)
        ax.text(0, -0.05, sub, fontsize=9, color=INK2)
    yrs = [str(int(v)) for v in y.index]
    last = len(yrs) - 1
    ax1 = fig.add_subplot(gs[1, 0:2]); ax2 = fig.add_subplot(gs[1, 2]); ax3 = fig.add_subplot(gs[1, 3])
    cols = [LGREY] * last + [BLUE]
    ax1.bar(yrs, y["sales"], color=cols, width=.55)
    for xi, v in enumerate(y["sales"]): ax1.text(xi, v + 8000, k(v, 0), ha="center", fontsize=9, color=INK)
    ax1.set_title("Sales by year"); _fmt_k(ax1); ax1.set_ylim(0, y["sales"].max() * 1.15)
    ax2.bar(yrs, y["profit"], color=cols, width=.55)
    for xi, v in enumerate(y["profit"]): ax2.text(xi, v + 1500, k(v, 0), ha="center", fontsize=8.5, color=INK)
    ax2.set_title("Profit by year"); _fmt_k(ax2); ax2.set_ylim(0, y["profit"].max() * 1.2); ax2.tick_params(axis="x", labelsize=8)
    ax3.plot(yrs, y["margin"] * 100, color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=1.5)
    for xi, v in enumerate(y["margin"] * 100): ax3.text(xi, v + .5, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax3.set_title("Profit margin"); ax3.set_ylim(8, 15); ax3.set_xlim(-0.5, len(yrs) - 0.5); ax3.tick_params(axis="x", labelsize=8)
    ax3.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.suptitle("Performance dashboard", x=0.02, y=0.99, ha="left", fontsize=13, fontweight="bold")
    return fig


def fig_discount(T):
    b = T["band"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, col, ttl in ((axes[0], "profit", "Profit by discount level"), (axes[1], "margin", "Profit margin by discount level")):
        vals = b[col] * (100 if col == "margin" else 1)
        colors = [BLUE if v >= 0 else RED for v in vals]
        ax.bar(range(len(b)), vals, color=colors, width=.6)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(b))); ax.set_xticklabels([f"{i}\n({int(n):,} lines)" for i, n in zip(b.index, b["lines"])], fontsize=8.5)
        ax.set_title(ttl)
        span = vals.max() - vals.min()
        for xi, v in enumerate(vals):
            lab = k(v, 0) if col == "profit" else f"{v:.0f}%"
            ax.text(xi, v + (span * .03 if v >= 0 else -span * .03), lab, ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.margins(y=.18)
    _fmt_k(axes[0]); axes[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    return fig


def fig_subcat(T):
    s = T["subcat"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = [BLUE if v >= 0 else RED for v in s["profit"]]
    ax.barh(s.index, s["profit"], color=colors, height=.62)
    ax.axvline(0, color=INK2, lw=1)
    for i, (p, m) in enumerate(zip(s["profit"], s["margin"])):
        ax.text(max(p, 0) + 900, i, f"{k(p)}  ({m * 100:.0f}% margin)", va="center", ha="left", fontsize=8.5)
    ax.set_title("Profit by sub-category (margin in brackets)"); _fmt_k(ax, "x"); ax.grid(axis="y", visible=False)
    ax.set_xlim(-22000, 82000)
    fig.tight_layout()
    return fig


def fig_geo(T):
    st, rg = T["state"], T["region"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    x, yv = st["avg_discount"] * 100, st["margin"] * 100
    ax.scatter(x, yv, s=np.sqrt(st["sales"]) / 3.2, color=BLUE, alpha=.65, edgecolor=SURF, linewidth=1)
    neg = st[st["profit"] < -6000]
    ax.scatter(neg["avg_discount"] * 100, neg["margin"] * 100, s=np.sqrt(neg["sales"]) / 3.2, color=RED, alpha=.8, edgecolor=SURF, linewidth=1)
    offsets = {"Texas": (8, -13, "left"), "Illinois": (9, 4, "left"), "Pennsylvania": (-8, 8, "right"), "Ohio": (-8, -12, "right"),
               "California": (8, 4, "left"), "New York": (8, 4, "left"), "Michigan": (8, 4, "left")}
    for name, (dx, dy, ha) in offsets.items():
        r = st.loc[name]; ax.annotate(name, (r["avg_discount"] * 100, r["margin"] * 100), xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha)
    z = np.polyfit(x, yv, 1); xs = np.linspace(x.min(), x.max(), 20); ax.plot(xs, np.polyval(z, xs), color=INK2, lw=1.2, ls="--")
    ax.axhline(0, color=INK2, lw=.8); ax.set_xlim(-2, 47)
    ax.set_xlabel("Average discount in state (%)   |   bubble size = sales"); ax.set_ylabel("Profit margin (%)")
    ax.set_title(f"States: higher discounts, lower margin (r = {T['state_corr']:.2f})")
    ax = axes[1]
    cols = [RED if r == "Central" else GREY for r in rg.index]
    ax.barh(rg.index, rg["margin"] * 100, color=cols, height=.55)
    for i, (m, h) in enumerate(zip(rg["margin"], rg["hd_lines_share"])):
        ax.text(m * 100 + .3, i, f"{m * 100:.1f}%  |  {h * 100:.0f}% of lines >20% off", va="center", fontsize=8.5)
    ax.set_xlim(0, 28); ax.set_title("Region margin"); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    return fig


def fig_bridge(T):
    br = T["bridge"]
    steps = [(f"{br['y1']} profit", br["start"], "total"), ("Higher sales volume", br["volume"], "delta")]
    for cat, v in br["rate"].items(): steps.append((f"{cat}\nmargin change", v, "delta"))
    steps.append((f"{br['y2']} profit", br["end"], "total"))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    run = 0
    for i, (lab, v, kind) in enumerate(steps):
        if kind == "total":
            ax.bar(i, v, color=GREY, width=.55); run = v; top = v
            ax.text(i, v + 1800, k(v), ha="center", fontsize=9.5, fontweight="bold")
        else:
            ax.bar(i, v, bottom=run, color=BLUE if v >= 0 else RED, width=.55)
            ax.text(i, max(run, run + v) + 1800, f"{'+' if v >= 0 else '-'}{k(abs(v))}", ha="center", fontsize=9.5)
            run += v
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[0] for s in steps], fontsize=8.5)
    ax.set_ylim(0, max(br["start"], br["end"]) * 1.35); _fmt_k(ax); ax.grid(axis="x", visible=False)
    ax.set_title(f"What changed between {br['y1']} and {br['y2']}: profit bridge")
    fig.tight_layout()
    return fig


def fig_hd_trend(T):
    y = T["yearly"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    yrs = [str(int(v)) for v in y.index]
    ax = axes[0]
    ax.bar(yrs, y["profit_ex_hd"], color=BLUE, width=.55)
    for xi, v in enumerate(y["profit_ex_hd"]): ax.text(xi, v + 2000, k(v, 0), ha="center", fontsize=9)
    ax.set_title("Profit from lines discounted 20% or less"); _fmt_k(ax); ax.set_ylim(0, y["profit_ex_hd"].max() * 1.18)
    ax = axes[1]
    ax.bar(yrs, y["hd_profit"], color=RED, width=.55); ax.axhline(0, color=INK2, lw=1)
    for xi, v in enumerate(y["hd_profit"]): ax.text(xi, v - 2500, k(v, 0), ha="center", va="top", fontsize=9)
    ax.set_title("Result on lines discounted over 20%"); _fmt_k(ax); ax.set_ylim(y["hd_profit"].min() * 1.35, 5000)
    fig.tight_layout()
    return fig


def fig_season(T):
    m = T["monthly"]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    mons = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = list(m.columns)
    for yr in years[:-1]:
        ax.plot(mons, m[yr], color=LGREY, lw=1.6); ax.text(11.15, m[yr].iloc[-1], str(yr), color=INK2, fontsize=8.5, va="center")
    ax.plot(mons, m[years[-1]], color=BLUE, lw=2.4, marker="o", ms=5, mec=SURF, mew=1.2)
    ax.text(11.15, m[years[-1]].iloc[-1], str(years[-1]), color=INK, fontsize=9, fontweight="bold", va="center")
    ax.set_xlim(-0.2, 12); _fmt_k(ax); ax.set_title(f"Monthly sales: the same autumn peak every year ({years[-1]} highlighted)")
    fig.tight_layout()
    return fig


def fig_returns(T):
    rr, ry = T["ret_region"], T["ret_year"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    cols = [RED if r == "West" else GREY for r in rr.index]
    ax.barh(rr.index, rr["rate"] * 100, color=cols, height=.55)
    for i, (r, n, t) in enumerate(zip(rr["rate"], rr["returned"], rr["orders"])): ax.text(r * 100 + .3, i, f"{r * 100:.1f}%  ({int(n)} of {int(t):,})", va="center", fontsize=8.5)
    ax.set_xlim(0, 24); ax.set_title("Share of orders returned, by region"); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax = axes[1]
    yrs = [str(int(v)) for v in ry.index]
    ax.bar(yrs, ry["rate"] * 100, color=BLUE, width=.55)
    for xi, v in enumerate(ry["rate"] * 100): ax.text(xi, v + .15, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, ry["rate"].max() * 100 * 1.3); ax.set_title("Share of orders returned, by year")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout()
    return fig


def fig_customers(T):
    sg, dec = T["segment"], T["decile"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.barh(sg.index, sg["margin"] * 100, color=BLUE, height=.5)
    for i, (m, s) in enumerate(zip(sg["margin"], sg["sales"] / sg["sales"].sum())): ax.text(m * 100 + .2, i, f"{m * 100:.1f}%  ({s * 100:.0f}% of sales)", va="center", fontsize=8.5)
    ax.set_xlim(0, 22); ax.set_title("Margin by customer segment"); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax = axes[1]
    colors = [BLUE if v >= 0 else RED for v in dec.values]
    ax.bar([str(i) for i in dec.index], dec.values, color=colors, width=.65); ax.axhline(0, color=INK2, lw=1)
    for xi, v in enumerate(dec.values): ax.text(xi, v + (2000 if v >= 0 else -2000), k(v, 0), ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xlabel("Customer decile (1 = most profitable 10% of customers)"); ax.set_title("Lifetime profit by customer decile"); _fmt_k(ax); ax.margins(y=.15)
    fig.tight_layout()
    return fig
