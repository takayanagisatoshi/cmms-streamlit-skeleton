# app.py — 単一ファイル版 CMMS ダッシュボード（CSV取込つき）
import os, io, textwrap, re
from datetime import date
import pandas as pd
import duckdb as ddb
import streamlit as st
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="CMMS ダッシュボード（β）", layout="wide", page_icon="🛠️")

# ---- DB 初期化 -----------------------------------------------------------
DB_PATH = os.environ.get("CMMS_DUCKDB_PATH", "app.duckdb")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
con = ddb.connect(DB_PATH)
con.execute("""
CREATE TABLE IF NOT EXISTS buildings(tenant TEXT, id TEXT, name TEXT, PRIMARY KEY(tenant,id));
CREATE TABLE IF NOT EXISTS locations(tenant TEXT, id TEXT, building_id TEXT, name TEXT, PRIMARY KEY(tenant,id));
CREATE TABLE IF NOT EXISTS floors(tenant TEXT, id TEXT, location_id TEXT, name TEXT, PRIMARY KEY(tenant,id));
CREATE TABLE IF NOT EXISTS rooms(tenant TEXT, id TEXT, floor_id TEXT, name TEXT, PRIMARY KEY(tenant,id));
CREATE TABLE IF NOT EXISTS devices(tenant TEXT, id TEXT, building_id TEXT, location_id TEXT, floor_id TEXT, room_id TEXT,
  name TEXT, category_l TEXT, category_m TEXT, category_s TEXT, symbol TEXT, cmms_url_rule TEXT,
  PRIMARY KEY(tenant,id));
CREATE TABLE IF NOT EXISTS targets(tenant TEXT, id TEXT, device_id TEXT, name TEXT, input_type TEXT,
  unit TEXT, lower DOUBLE, upper DOUBLE, ord INT, PRIMARY KEY(tenant,id));
CREATE TABLE IF NOT EXISTS schedules(tenant TEXT, id TEXT, job_id TEXT, name TEXT, freq TEXT, PRIMARY KEY(tenant,id));
CREATE TABLE IF NOT EXISTS schedule_targets(tenant TEXT, schedule_id TEXT, target_id TEXT,
  PRIMARY KEY(tenant,schedule_id,target_id));
CREATE TABLE IF NOT EXISTS schedule_dates(tenant TEXT, schedule_id TEXT, date DATE, status TEXT,
  done INT, total INT, done_at TIMESTAMP, PRIMARY KEY(tenant,schedule_id,date));
CREATE TABLE IF NOT EXISTS results(tenant TEXT, schedule_id TEXT, date DATE, target_id TEXT,
  value_num DOUBLE, value_text TEXT, value_bool BOOLEAN, judged TEXT,
  PRIMARY KEY(tenant,schedule_id,date,target_id));
CREATE TABLE IF NOT EXISTS issues(tenant TEXT, id TEXT, device_id TEXT, reported_on DATE, due_on DATE,
  status TEXT, severity TEXT, category TEXT, summary TEXT, cmms_url_rule TEXT, PRIMARY KEY(tenant,id));
CREATE TABLE IF NOT EXISTS documents(tenant TEXT, id TEXT, title TEXT, category TEXT, tags TEXT,
  url TEXT, version TEXT, ocr_status TEXT, ai_summary TEXT, ai_actions TEXT,
  uploaded_at TIMESTAMP DEFAULT now(), PRIMARY KEY(tenant,id));
CREATE TABLE IF NOT EXISTS document_bindings(tenant TEXT, doc_id TEXT, entity_type TEXT, entity_id TEXT,
  PRIMARY KEY(tenant,doc_id,entity_type,entity_id));
CREATE TABLE IF NOT EXISTS daily_kpis(tenant TEXT, date DATE, planned INT, done INT, overdue INT, findings INT,
  PRIMARY KEY(tenant,date));
""")

TENANT = st.session_state.setdefault("tenant", "demo")

# ---- 共通: 簡易アップサート ---------------------------------------------
def upsert_df(table: str, df: pd.DataFrame, pk: list[str]):
    if df is None or df.empty:
        return
    df = df.copy()
    df["tenant"] = TENANT
    cols = list(df.columns)
    for _, r in df.iterrows():
        where = " AND ".join([f"{k} = ?" for k in ["tenant", *pk]])
        con.execute(f"DELETE FROM {table} WHERE {where}", [TENANT, *[r[k] for k in pk]])
        con.execute(f"INSERT INTO {table}({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", r.tolist())

# ---- 便利：階層パスで一意な device_id を作る ----------------------------
def _dev_id(b, l, f, r, name):
    b = b or "指定なし"; l = l or "指定なし"; f = f or "指定なし"; r = r or "指定なし"; name = name or "不明設備"
    return f"{b}|{l}|{f}|{r}|{name}"

# ---- 取込：マスタ（階層 + 設備 + target + schedule 紐付け） ------------
def import_master(df: pd.DataFrame):
    df = df.rename(columns=str.lower)
    need = ["building_name","location_name","room_name","device_name","target_id","target_name","target_type_id","schedule_id"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.error(f"必須列が不足: {miss}"); return
    if "floor_name" not in df.columns:
        df["floor_name"] = "指定なし"

    # buildings
    b = df[["building_name"]].drop_duplicates().rename(columns={"building_name":"id"})
    b["name"] = b["id"]
    upsert_df("buildings", b[["id","name"]], ["id"])

    # locations
    l = df[["building_name","location_name"]].drop_duplicates().rename(
        columns={"building_name":"building_id","location_name":"id"})
    l["name"] = l["id"]
    upsert_df("locations", l[["id","building_id","name"]], ["id"])

    # floors
    f = df[["location_name","floor_name"]].drop_duplicates().rename(
        columns={"location_name":"location_id","floor_name":"id"})
    f["name"] = f["id"]
    upsert_df("floors", f[["id","location_id","name"]], ["id"])

    # rooms
    r = df[["floor_name","room_name"]].drop_duplicates().rename(
        columns={"floor_name":"floor_id","room_name":"id"})
    r["name"] = r["id"]
    upsert_df("rooms", r[["id","floor_id","name"]], ["id"])

    # devices（階層パスID）
    d = df[["building_name","location_name","floor_name","room_name","device_name"]].drop_duplicates()
    d = d.rename(columns={"building_name":"building_id","location_name":"location_id","floor_name":"floor_id",
                          "room_name":"room_id","device_name":"name"})
    d["id"] = d.apply(lambda x: _dev_id(x["building_id"],x["location_id"],x["floor_id"],x["room_id"],x["name"]), axis=1)
    d["category_l"]=d["category_m"]=d["category_s"]=d["symbol"]=d["cmms_url_rule"]=None
    upsert_df("devices", d[["id","building_id","location_id","floor_id","room_id","name",
                            "category_l","category_m","category_s","symbol","cmms_url_rule"]], ["id"])

    # targets（device_id をパスIDにする）
    t = df[["building_name","location_name","floor_name","room_name","device_name","target_id",
            "target_name","target_type_id","unit","lower","upper","order_no"]].copy()
    t["device_id"] = t.apply(lambda x: _dev_id(x["building_name"],x["location_name"],x["floor_name"],x["room_name"],x["device_name"]), axis=1)
    t = t.rename(columns={"target_id":"id","target_name":"name","target_type_id":"input_type","order_no":"ord"})
    upsert_df("targets", t[["id","device_id","name","input_type","unit","lower","upper","ord"]].drop_duplicates(), ["id"])

    # schedules
    s = df[["schedule_id"]].drop_duplicates().rename(columns={"schedule_id":"id"})
    s["job_id"]=None; s["name"]=None; s["freq"]=None
    upsert_df("schedules", s[["id","job_id","name","freq"]], ["id"])

    # schedule_targets
    stgt = df[["schedule_id","target_id"]].drop_duplicates().rename(columns={"schedule_id":"schedule_id","target_id":"target_id"})
    upsert_df("schedule_targets", stgt, ["schedule_id","target_id"])

    st.success(f"マスタ取込: {len(df)} 行（設備/ターゲット/スケジュール紐付け）")

# ---- 取込：運用チケット（実施日・進捗） ---------------------------------
def import_tickets(df: pd.DataFrame):
    df = df.rename(columns=str.lower)
    need = ["date","schedule_id","status"]; miss=[c for c in need if c not in df.columns]
    if miss: st.error(f"必須列が不足: {miss}"); return

    def split_prog(x):
        try: a,b = str(x).split("/"); return int(a), int(b)
        except: return None, None

    prog = df.get("progress")
    done, total = zip(*[split_prog(x) for x in (prog if prog is not None else [])]) if len(df)>0 else ([],[])
    s = pd.DataFrame({
        "schedule_id": df["schedule_id"],
        "date": pd.to_datetime(df["date"], errors="coerce").dt.date,
        "status": df["status"],
        "done": done, "total": total,
        "done_at": pd.to_datetime(df.get("done_at"), errors="coerce")
    })
    s = s.dropna(subset=["date"])
    upsert_df("schedule_dates", s, ["schedule_id","date"])
    st.success(f"チケット取込: {len(s)} 行")

# ---- 取込：不具合（階層ヒントで device_id 自動解決） --------------------
def import_issues(df: pd.DataFrame):
    df = df.rename(columns=str.lower)

    # reported_on / due_on の別名に対応
    if "reported_on" not in df.columns and "発生日時" in df.columns:
        df.rename(columns={"発生日時":"reported_on"}, inplace=True)
    if "due_on" not in df.columns and "対応期限" in df.columns:
        df.rename(columns={"対応期限":"due_on"}, inplace=True)

    # 設備名などのヒント列を取り込む
    if "設備" in df.columns and "device_id" not in df.columns:
        df["_equip_hint"] = df["設備"]
    elif "device" in df.columns and "device_id" not in df.columns:
        df["_equip_hint"] = df["device"]
    elif "equipment" in df.columns and "device_id" not in df.columns:
        df["_equip_hint"] = df["equipment"]

    for col in [("部屋名","_room_hint"),("room_name","_room_hint"),
                ("フロア","_floor_hint"),("floor_name","_floor_hint"),
                ("棟","_loc_hint"),("location_name","_loc_hint"),
                ("物件","_bld_hint"),("building_name","_bld_hint")]:
        if col[0] in df.columns: df[col[1]] = df[col[0]]

    # devices 一覧（階層列あり）
    devs = con.execute("SELECT id,name,building_id,location_id,floor_id,room_id FROM devices WHERE tenant=?",
                       [TENANT]).df()

    def resolve_device(row):
        cand = devs.copy()
        if pd.notna(row.get("_equip_hint")): cand = cand[cand["name"]==row["_equip_hint"]]
        if pd.notna(row.get("_room_hint")):  cand = cand[cand["room_id"]==row["_room_hint"]]
        if pd.notna(row.get("_floor_hint")): cand = cand[cand["floor_id"]==row["_floor_hint"]]
        if pd.notna(row.get("_loc_hint")):   cand = cand[cand["location_id"]==row["_loc_hint"]]
        if pd.notna(row.get("_bld_hint")):   cand = cand[cand["building_id"]==row["_bld_hint"]]
        if len(cand)==1:
            return cand.iloc[0]["id"]
        cand2 = devs[devs["name"]==row.get("_equip_hint")]
        return cand2.iloc[0]["id"] if len(cand2)==1 else None

    if "device_id" not in df.columns:
        df["device_id"] = df.apply(resolve_device, axis=1)

    # 正規化して保存
    df["reported_on"] = pd.to_datetime(df.get("reported_on"), errors="coerce").dt.date
    if "due_on" in df.columns: df["due_on"] = pd.to_datetime(df["due_on"], errors="coerce").dt.date
    keep = ["id","device_id","reported_on","due_on","status","severity","category","summary","cmms_url_rule"]
    for k in keep:
        if k not in df.columns: df[k]=None
    out = df[keep].dropna(subset=["id","reported_on"])
    upsert_df("issues", out, ["id"])
    st.success(f"不具合取込: {len(out)} 行（device_id自動解決 {out['device_id'].notna().sum()}件）")

# ---- 取込：点検結果（横持ち→縦melt） -----------------------------------
def import_results_wide(df: pd.DataFrame):
    df = df.rename(columns=lambda c: str(c).strip())
    df = df.rename(columns=str.lower)

    # ヘッダーが日付形式のものだけ（YYYY-MM-DD / YYYY/MM/DD）
    date_cols = [c for c in df.columns
                 if re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}", c)
                 or re.fullmatch(r"\d{4}/\d{2}/\d{2}", c)]

    rows = []
    for _, r in df.iterrows():
        for dc in date_cols:
            val = r.get(dc)
            if pd.isna(val) or str(val) == "":
                continue
            d = pd.to_datetime(dc.replace("/", "-"), errors="coerce")
            if pd.isna(d):
                continue
            num = pd.to_numeric(val, errors="coerce")
            rows.append({
                "schedule_id": r.get("schedule_id"),
                "date": d.date(),
                "target_id": r.get("target_id"),
                "value_num": None if pd.isna(num) else float(num),
                "value_text": None if not pd.isna(num) else str(val),
                "value_bool": None,
                "judged": None
            })

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["schedule_id","target_id","date"])
    if not out.empty:
        upsert_df("results", out, ["schedule_id","date","target_id"])
    st.success(f"点検結果取込: {len(df)} 行（{len(date_cols)}本の日付列を処理、レコード{len(out)}件）")

# ---- KPI 再計算（超簡易） ------------------------------------------------
def recalc_daily_kpis():
    con.execute("DELETE FROM daily_kpis WHERE tenant = ?", [TENANT])
    con.execute("""
        INSERT INTO daily_kpis
        SELECT ?, sd.date,
               COUNT(*) AS planned,
               SUM(CASE WHEN COALESCE(status,'') IN ('完了','実施済') THEN 1 ELSE 0 END) AS done,
               SUM(CASE WHEN sd.date < today() AND COALESCE(status,'') NOT IN ('完了','実施済') THEN 1 ELSE 0 END) AS overdue,
               0 AS findings
        FROM schedule_dates sd WHERE tenant=? GROUP BY sd.date
    """, [TENANT, TENANT])

# ========================= UI =============================================
st.title("CMMS ダッシュボード（β）")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📅 日報","🛠️ 設備","📈 月報","📄 ドキュメント","📥 取込","🤖 AIβ"])

# 取込
with tab5:
    st.subheader("CSV 取込")
    f1 = st.file_uploader("マスタ（階層+設備+target+schedule）CSV", type=["csv"])
    if f1: import_master(pd.read_csv(f1))
    f2 = st.file_uploader("operation_tickets.csv（実施日/進捗）", type=["csv"])
    if f2: import_tickets(pd.read_csv(f2))
    f3 = st.file_uploader("issues.csv（不具合）", type=["csv"])
    if f3: import_issues(pd.read_csv(f3))
    f4 = st.file_uploader("点検結果（横持ち）CSV", type=["csv"])
    if f4: import_results_wide(pd.read_csv(f4))
    if st.button("KPI再計算"): recalc_daily_kpis(); st.success("daily_kpis 再計算")

# 日報
with tab1:
    st.subheader("本日の業務予定と進捗")
    target_date = st.date_input("対象日", value=date.today())
    kpi = con.execute(
        "SELECT planned, done, overdue FROM daily_kpis WHERE tenant=? AND date=?",
        [TENANT, target_date]).fetchone()
    planned, done, overdue = (kpi if kpi else (0,0,0))
    c1,c2,c3 = st.columns(3); c1.metric("予定", planned); c2.metric("完了", done); c3.metric("期限超過", overdue)
    df = con.execute("SELECT * FROM schedule_dates WHERE tenant=? AND date=?", [TENANT, target_date]).df()
    st.dataframe(df, use_container_width=True)

    st.subheader("本日発生の不具合")
    issues = con.execute("SELECT * FROM issues WHERE tenant=? AND reported_on=?", [TENANT, target_date]).df()
    st.dataframe(issues, use_container_width=True)

# 設備
with tab2:
    st.title("🛠️ 設備ページ")

    # 段階選択（物件→棟→フロア→部屋→設備）
    blds = con.execute("SELECT DISTINCT building_id FROM devices WHERE tenant=?", [TENANT]).df()["building_id"].tolist()
    if not blds:
        st.info("まずは『📥 取込』からマスタCSVを投入してください。"); st.stop()
    bld = st.selectbox("物件", blds)

    locs = con.execute(
        "SELECT DISTINCT location_id FROM devices WHERE tenant=? AND building_id=?",
        [TENANT, bld]).df()["location_id"].tolist()
    loc = st.selectbox("棟", locs) if locs else None

    flrs = con.execute(
        "SELECT DISTINCT floor_id FROM devices WHERE tenant=? AND building_id=? AND location_id=?",
        [TENANT, bld, loc]).df()["floor_id"].tolist() if loc else []
    flr = st.selectbox("フロア", flrs) if flrs else None

    rooms = con.execute(
        "SELECT DISTINCT room_id FROM devices WHERE tenant=? AND building_id=? AND location_id=? AND floor_id=?",
        [TENANT, bld, loc, flr]).df()["room_id"].tolist() if flr else []
    room = st.selectbox("部屋", rooms) if rooms else None

    devs = con.execute(
        """SELECT id,name FROM devices
           WHERE tenant=? AND building_id=? AND location_id=? AND floor_id=? AND room_id=?""",
        [TENANT, bld, loc, flr, room]).df() if room else pd.DataFrame(columns=["id","name"])
    dev = st.selectbox("設備", options=devs["id"] if not devs.empty else [])
    if not dev:
        st.stop()

    meta = con.execute("SELECT * FROM devices WHERE tenant=? AND id=?", [TENANT, dev]).df().iloc[0]
    st.markdown(f"### {meta['name']}　〔{bld} / {loc} / {flr} / {room}〕")
    if pd.notna(meta.get("cmms_url_rule")) and str(meta.get("cmms_url_rule")) not in ("", "None"):
        st.link_button("CMMSで開く", str(meta["cmms_url_rule"]))

    # 期間
    dr = st.date_input("期間", [])
    if len(dr) != 2:
        st.info("期間を選択すると、グラフと表が表示されます。"); st.stop()

    # 点検結果（targets と閾値を JOIN）
    q = """
      SELECT r.date, r.target_id, r.value_num, r.value_text,
             t.name AS target_name, t.unit, t.lower, t.upper
      FROM results r
      JOIN targets t ON t.tenant=r.tenant AND t.id=r.target_id
      WHERE r.tenant=? AND t.device_id=? AND r.date BETWEEN ? AND ?
      ORDER BY r.date
    """
    df = con.execute(q, [TENANT, dev, dr[0], dr[1]]).df()
    if df.empty:
        st.warning("期間内のデータがありません。"); st.stop()

    df["date"] = pd.to_datetime(df["date"])
    # 異常（閾値逸脱 or ×/NG）
    abnormal_mask = (
        (df["value_num"].notna() & (
            (df["lower"].notna() & (df["value_num"] < df["lower"])) |
            (df["upper"].notna() & (df["value_num"] > df["upper"]))
        )) |
        (df["value_num"].isna() & df["value_text"].isin(["×","NG"]))
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("対象項目", int(df["target_id"].nunique()))
    c2.metric("データ点数", int(df.shape[0]))
    c3.metric("異常件数", int(abnormal_mask.sum()))

    # 五感のスコア化
    label_to_score = {"○":0, "OK":0, "良":0, "△":1, "要確認":1, "注意":1, "×":2, "NG":2, "異常":2}
    num  = df.dropna(subset=["value_num"]).copy()
    qual = df[df["value_num"].isna()].copy()
    qual["score"] = qual["value_text"].map(label_to_score).fillna(np.nan)

    # ヒートマップ用データ
    if not qual.empty:
        heat = (qual.pivot_table(index="target_name", columns="date", values="score", aggfunc="max")
                .sort_index())
    else:
        heat = pd.DataFrame(index=[], columns=[])

    # 同一時間軸の複合図
    # ---- 数値ライン + ×/△ 背景シェード（同一時間軸、単位で左右軸） ----
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 分割
num  = df.dropna(subset=["value_num"]).copy()
qual = df[df["value_num"].isna()].copy()

# 五感スコア（0=OK, 1=△, 2=×） ※列名は sev に統一
sev_map = {"○":0, "OK":0, "良":0, "△":1, "要確認":1, "注意":1, "▲":1, "×":2, "NG":2, "異常":2}
qual["sev"]  = qual["value_text"].map(sev_map).astype("float")
qual["date"] = pd.to_datetime(qual["date"], errors="coerce")

# --- 背景シェードの対象を選ぶ UI ---
qual_targets_all = sorted(qual["target_name"].dropna().unique().tolist())
default_sel = sorted(qual.loc[qual["sev"]==2, "target_name"].dropna().unique().tolist()) or qual_targets_all
sel_targets = st.multiselect("背景色の対象（五感項目）", qual_targets_all, default=default_sel)
shade_x   = st.checkbox("×の発生日で背景色", value=True)
shade_tri = st.checkbox("△の発生日も背景色に含める", value=False)
shade_alpha = st.slider("背景の濃さ", 0.05, 0.4, 0.12, 0.01)

qual_sel = qual[qual["target_name"].isin(sel_targets)] if len(sel_targets) else qual.iloc[0:0]
bad_days  = qual_sel.loc[qual_sel["sev"]>=2, "date"].dt.normalize().dropna().unique() if shade_x else []
warn_days = qual_sel.loc[qual_sel["sev"]==1, "date"].dt.normalize().dropna().unique() if shade_tri else []

# --- 数値：単位ごとに左右軸を分ける（最大2軸） ---
num["unit"] = num["unit"].astype(str).replace({"nan":"値","None":"値"})
units_all = num["unit"].dropna().unique().tolist()
if len(units_all) <= 2:
    units_pick = units_all
else:
    units_pick = st.multiselect("表示する数値の単位（最大2つ）", units_all, default=units_all[:2], max_selections=2)
    if not units_pick: units_pick = units_all[:1]

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(specs=[[{"secondary_y": True}]])

unit_axis = {}
if len(units_pick) >= 1: unit_axis[units_pick[0]] = "y"   # 左
if len(units_pick) >= 2: unit_axis[units_pick[1]] = "y2"  # 右

# 数値トレース
for (tname, unit), g in num[num["unit"].isin(units_pick)].groupby(["target_name","unit"]):
    use_y2 = (unit_axis.get(unit) == "y2")
    fig.add_trace(
        go.Scatter(
            x=g["date"], y=g["value_num"], mode="lines+markers", name=f"{tname}",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y} "+(unit or "")
        ),
        secondary_y=use_y2
    )

# 背景シェード（△=黄, ×=赤）
for d in warn_days:
    fig.add_vrect(x0=d, x1=d, fillcolor="#FFD166", opacity=shade_alpha, line_width=0)
for d in bad_days:
    fig.add_vrect(x0=d, x1=d, fillcolor="#EF476F", opacity=min(shade_alpha+0.05, 0.5), line_width=0)

# 上部マーカー（記号で日付を示す）
if st.checkbox("×/△ を上部に記号表示", value=True):
    for d in bad_days:
        fig.add_annotation(x=d, y=1.02, xref="x", yref="paper", text="×", showarrow=False,
                           font=dict(color="#EF476F", size=14))
    for d in warn_days:
        fig.add_annotation(x=d, y=1.02, xref="x", yref="paper", text="▲", showarrow=False,
                           font=dict(color="#FFD166", size=12))

# 軸タイトル
left_label  = units_pick[0] if len(units_pick)>=1 else ""
right_label = units_pick[1] if len(units_pick)>=2 else ""
fig.update_yaxes(title_text=left_label, secondary_y=False)
if right_label:
    fig.update_yaxes(title_text=right_label, secondary_y=True)

fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
st.plotly_chart(fig, use_container_width=True)

# --- 下：イベント（五感）一覧 ---
if not qual.empty:
    show = qual_sel.copy() if len(sel_targets) else qual.copy()
    col1, col2 = st.columns(2)
    f_bad = col1.checkbox("×のみ表示", value=False)
    f_tri = col2.checkbox("△のみ表示", value=False)
    if f_bad: show = show[show["sev"]==2]
    if f_tri: show = show[show["sev"]==1]
    st.dataframe(
        show.sort_values(["date","target_name"])[["date","target_name","value_text"]]
            .rename(columns={"date":"日付","target_name":"項目","value_text":"判定"}),
        use_container_width=True
    )


    # 上段：数値ライン
    for name, g in num.groupby("target_name"):
        u = str(g["unit"].iloc[0] if "unit" in g else "")
        fig.add_trace(
            go.Scatter(x=g["date"], y=g["value_num"], mode="lines+markers",
                       name=str(name), hovertemplate="%{x|%Y-%m-%d}<br>%{y} "+u),
            row=1, col=1
        )
    # 異常日の縦帯
    if not qual.empty:
        bad_days = qual[qual["score"] >= 2]["date"].dt.normalize().unique()
        for d in bad_days:
            fig.add_vrect(x0=d, x1=d, row="all", col=1, fillcolor="red", opacity=0.08, line_width=0)
    # 下段：五感ヒートマップ
    if not heat.empty:
        colorscale = [[0.0, "#3CB371"], [0.5, "#FFD166"], [1.0, "#EF476F"]]  # 0=緑,1=黄,2=赤
        fig.add_trace(
            go.Heatmap(z=heat.values, x=heat.columns, y=heat.index,
                       zmin=0, zmax=2, colorscale=colorscale, colorbar=dict(title=""),
                       hovertemplate="%{y}<br>%{x|%Y-%m-%d}<br>状態=%{z}<extra></extra>"),
            row=2, col=1
        )
    fig.update_layout(height=700, margin=dict(l=10,r=10,t=40,b=10))
    fig.update_xaxes(matches="x", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # 数値要約表
    if not num.empty:
        latest = num.sort_values("date").groupby("target_name").tail(1).set_index("target_name")[["value_num","unit"]]
        stats = num.groupby("target_name")["value_num"].agg(最小="min", 最大="max", 平均="mean")
        summary = latest.join(stats, how="left").rename(columns={"value_num":"最新値"})
        st.markdown("**数値ターゲットの要約**")
        st.dataframe(summary.reset_index(), use_container_width=True)

    # 五感テーブル
    if not qual.empty:
        st.markdown("**五感/選択ターゲット（表）**")
        st.dataframe(
            qual.pivot_table(index="date", columns="target_name", values="value_text", aggfunc="first"),
            use_container_width=True
        )

    # この設備の不具合
    st.subheader("この設備に紐づく不具合")
    iss = con.execute(
        """SELECT id, reported_on, due_on, status, severity, category, summary
           FROM issues WHERE tenant=? AND device_id=?
           ORDER BY COALESCE(due_on, reported_on) DESC""",
        [TENANT, dev]).df()
    c1, c2, c3 = st.columns(3)
    if iss.empty:
        c1.metric("未完了", 0); c2.metric("期限超過", 0); c3.metric("今月新規", 0)
        st.info("紐づく不具合はありません。")
    else:
        not_done = ~iss["status"].isin(["完了","対応済"])
        overdue = pd.to_datetime(iss["due_on"], errors="coerce") < pd.Timestamp.today()
        this_month = pd.to_datetime(iss["reported_on"], errors="coerce").dt.to_period("M") == pd.Timestamp.today().to_period("M")
        c1.metric("未完了", int(iss[not_done].shape[0]))
        c2.metric("期限超過", int(iss[not_done & overdue].shape[0]))
        c3.metric("今月新規", int(iss[this_month].shape[0]))
        st.dataframe(iss, use_container_width=True)

    # ドキュメント
    st.subheader("ドキュメント")
    docs = con.execute(
        """SELECT d.id, d.title, d.category, d.tags, d.ai_summary
           FROM document_bindings b
           JOIN documents d ON d.tenant=b.tenant AND d.id=b.doc_id
           WHERE b.tenant=? AND b.entity_type='device' AND b.entity_id=?
           ORDER BY d.uploaded_at DESC""",
        [TENANT, dev]).df()
    if docs.empty:
        st.info("この設備に紐づくドキュメントは未登録です。")
    else:
        st.dataframe(docs, use_container_width=True)

    # ターゲット定義
    st.subheader("点検項目（Targets 定義）")
    tl = con.execute(
        """SELECT id AS target_id, name, input_type, unit, lower, upper, ord
           FROM targets WHERE tenant=? AND device_id=? ORDER BY ord""",
        [TENANT, dev]).df()
    st.dataframe(tl, use_container_width=True)

# 月報（プレースホルダ）
with tab3:
    st.subheader("指定月のサマリー（プレースホルダ）")
    st.info("CSV取込後、月次集計を実装します。")

# ドキュメント（プレースホルダ）
with tab4:
    st.subheader("ドキュメント一覧（プレースホルダ）")
    docs = con.execute("SELECT id,title,category,tags,ai_summary FROM documents WHERE tenant=?", [TENANT]).df()
    st.dataframe(docs, use_container_width=True)

# AI（プレースホルダ）
with tab6:
    st.subheader("AIサマリー（β）")
    st.info("OCR/AI連携は後で接続。まずはCSV→ダッシュボードの流れを固めます。")

st.caption("Theme: 管理ロイド風 / データは UTF-8 CSV を 取込タブから投入")
