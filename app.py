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
    if df.empty:
        return
    df = df.copy()
    df["tenant"] = TENANT

    # PK を正規化 & 欠損/空を除外
    for k in pk:
        df[k] = df[k].apply(_norm)
    mask_valid = df[pk].apply(lambda s: s.astype(str).str.len() > 0).all(axis=1)
    bad = len(df) - int(mask_valid.sum())
    if bad:
        st.warning(f"{table}: PK欠損で {bad} 行をスキップ（空/NaN/None）")
    df = df[mask_valid]

    # PK 重複を折りたたみ
    before = len(df)
    df = df.drop_duplicates(subset=pk, keep="last")
    dup = before - len(df)
    if dup:
        st.warning(f"{table}: PK重複で {dup} 行を折りたたみ")

    cols = list(df.columns)
    for _, r in df.iterrows():
        where = " AND ".join([f"{k} = ?" for k in ["tenant", *pk]])
        con.execute(f"DELETE FROM {table} WHERE {where}", [TENANT, *[r[k] for k in pk]])
        con.execute(
            f"INSERT INTO {table}({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})",
            r.tolist()
        )


# ---- 便利：階層パスで一意な device_id を作る ----------------------------
# --- ID生成ヘルパ（欠損/空/スペース/Noneを吸収）---
def _norm(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none") else s

def _dev_id(b, l, f, r, name):
    # すべて正規化してから連結
    return "|".join([_norm(b), _norm(l), _norm(f), _norm(r), _norm(name)])


# ---- 取込：マスタ（階層 + 設備 + target + schedule 紐付け） ------------
def import_master(df: pd.DataFrame):
    # ❶ 列名整形
    df = df.copy()
    df.columns = [str(c).strip().lower()for c in df.columns]

    # ❷ エイリアス（日本語/英語どちらでも）
    alias = {
        "building_name": ["building","building_id","物件","物件名"],
        "location_name": ["location","棟","棟名","ブロック"],
        "floor_name":    ["floor","階","フロア"],
        "room_name":     ["room","部屋","室","区画"],
        "device_name":   ["device","設備","設備名"],
        "target_id":     ["点検項目id","target"],
        "target_name":   ["点検項目名","target_label","項目名"],
        "target_type_id":["input_type","種別","入力型"],
        "schedule_id":   ["schedule","スケジュールid","スケジュール"],
        "unit":          ["単位"],
        "lower":         ["下限","min"],
        "upper":         ["上限","max"],
        "order_no":      ["ord","順序","order"],
    }
    for canon, alts in alias.items():
        if canon not in df.columns:
            for a in alts:
                if a in df.columns:
                    df.rename(columns={a: canon}, inplace=True)
                    break

    # ❸ 必須列チェック（ここでだけ落とす）
    need = ["building_name","location_name","room_name",
            "device_name","target_id","target_name","target_type_id","schedule_id"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.error(f"必須列が不足: {miss}")
        st.write("受け取った列:", list(df.columns))
        return

    # ❹ 任意列を補完
    defaults = {"floor_name":"", "unit":None, "lower":None, "upper":None, "order_no":0}
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v

    # 以降は既存ロジックそのまま -----------------------------
    # devices のID（階層パス）
    def _dev_id(b,l,f,r,name): return f"{b}|{l}|{f}|{r}|{name}"

       # --- devices：安全にID作成 → 重複/欠損を除去してUPSERT ---
    d = df[["building_name","location_name","floor_name","room_name","device_name"]].copy()
    d = d.rename(columns={"building_name":"building_id","location_name":"location_id",
                          "floor_name":"floor_id","room_name":"room_id","device_name":"name"})
    for c in ["building_id","location_id","floor_id","room_id","name"]:
        d[c] = d[c].apply(_norm)

    # 設備名が空の行は除外
    d = d[d["name"].str.len() > 0]

    # ID作成
    d["id"] = d.apply(lambda r: _dev_id(r["building_id"], r["location_id"],
                                        r["floor_id"], r["room_id"], r["name"]), axis=1)

    # ID重複があれば警告しつつ畳む
    if d.duplicated(subset=["id"]).any():
        cnt = int(d.duplicated(subset=["id"], keep=False).sum())
        st.warning(f"devices: 同一IDが {cnt} 行見つかりました（一部を折りたたみます）")
    d = d.drop_duplicates(subset=["id"], keep="last")

    d["category_l"]=d["category_m"]=d["category_s"]=d["symbol"]=d["cmms_url_rule"]=None
    upsert_df(
        "devices",
        d[["id","building_id","location_id","floor_id","room_id","name",
           "category_l","category_m","category_s","symbol","cmms_url_rule"]],
        ["id"]
    )

    # --- targets：device_id を安全生成してUPSERT ---
    t = df[["building_name","location_name","floor_name","room_name","device_name",
            "target_id","target_name","target_type_id","unit","lower","upper","order_no"]].copy()

    for c in ["building_name","location_name","floor_name","room_name","device_name"]:
        t[c] = t[c].apply(_norm)

    t["device_id"] = t.apply(lambda r: _dev_id(r["building_name"], r["location_name"],
                                               r["floor_name"], r["room_name"], r["device_name"]), axis=1)
    t = t.rename(columns={"target_id":"id","target_name":"name",
                          "target_type_id":"input_type","order_no":"ord"})
    t = t.drop_duplicates(subset=["id"], keep="last")

    upsert_df("targets", t[["id","device_id","name","input_type","unit","lower","upper","ord"]], ["id"])


    # targets
    t = df[["building_name","location_name","floor_name","room_name","device_name",
            "target_id","target_name","target_type_id","unit","lower","upper","order_no"]].copy()
    t["device_id"] = t.apply(lambda r: _dev_id(r["building_name"],r["location_name"],
                                               r["floor_name"],r["room_name"],r["device_name"]), axis=1)
    t = t.rename(columns={"target_id":"id","target_name":"name",
                          "target_type_id":"input_type","order_no":"ord"})
    upsert_df("targets", t[["id","device_id","name","input_type","unit","lower","upper","ord"]].drop_duplicates(), ["id"])

    # schedules / schedule_targets は既存処理のままでOK
    st.success(f"マスタ取込: {len(df)} 行（不足列は既定値で補完）")

def import_annual_plan(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]

    # 列名エイリアス（日本語/英語どちらでもOK）
    alias = {
        "schedule_id": ["schedule_id","スケジュールid","チケットid","job_id","業務id","id"],
        "name":        ["schedule_name","job_name","業務名","作業名","名称","name"],
        "freq":        ["freq","frequency","周期","頻度"],
        # どちらかの形で来る想定（列ごと or 一覧）
        "date":        ["date","予定日","実施日","日付"],
        "start":       ["start","開始日","from"],
        "end":         ["end","終了日","to"],
        "status":      ["status","ステータス","状態"]
    }
    for canon, alts in alias.items():
        if canon not in df.columns:
            for a in alts:
                if a in df.columns:
                    df.rename(columns={a: canon}, inplace=True)
                    break

    # schedules（ユニーク）を upsert
    sch = df[["schedule_id","name","freq"]].dropna(subset=["schedule_id"]).copy()
    sch["schedule_id"] = sch["schedule_id"].astype(str).str.strip()
    sch["name"] = sch["name"].astype(str).str.strip()
    sch = sch.drop_duplicates(subset=["schedule_id"])
    sch = sch.rename(columns={"schedule_id":"id"})
    upsert_df("schedules", sch[["id","name","freq"]], ["id"])

    # 予定日の作成
    dates = pd.DataFrame(columns=["schedule_id","date","status","done","total","done_at"])
    if "date" in df.columns:
        # 行ごとに予定日が入っているパターン
        dates = pd.DataFrame({
            "schedule_id": df["schedule_id"].astype(str).str.strip(),
            "date": pd.to_datetime(df["date"], errors="coerce").dt.date,
            "status": df.get("status"),
            "done": None, "total": None, "done_at": None
        }).dropna(subset=["schedule_id","date"])
    elif {"start","end"}.issubset(df.columns):
        # 期間 + freq から展開するパターン（freqは任意）
        tmp = []
        for r in df.itertuples(index=False):
            sid = str(getattr(r, "schedule_id")).strip()
            if not sid: 
                continue
            start = pd.to_datetime(getattr(r, "start"), errors="coerce")
            end   = pd.to_datetime(getattr(r, "end"), errors="coerce")
            if pd.isna(start) or pd.isna(end) or end < start:
                continue
            # 簡易展開：freq が 'monthly','weekly','daily' などを想定
            freq = (str(getattr(r, "freq") or "")).lower()
            rule = {"monthly":"MS", "week":"W", "weekly":"W", "day":"D", "daily":"D"}.get(freq, "W")
            for d in pd.date_range(start=start, end=end, freq=rule):
                tmp.append((sid, d.date(), None))
        if tmp:
            dates = pd.DataFrame(tmp, columns=["schedule_id","date","status"])
            dates["done"]=None; dates["total"]=None; dates["done_at"]=None

    if not dates.empty:
        upsert_df("schedule_dates", dates, ["schedule_id","date"])
        st.success(f"年間業務計画 取込: schedules {sch.shape[0]} 件 / 予定日 {dates.shape[0]} 件")
    else:
        st.success(f"年間業務計画 取込: schedules {sch.shape[0]} 件（予定日は列なし）")

# ---- 取込：運用チケット（実施日・進捗） ---------------------------------
def import_tickets(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]

    alias = {
        "date":        ["実施日","予定日","日付","date"],
        "schedule_id": ["schedule_id","スケジュールid","チケットid"],
        "job_id":      ["job_id","業務id","業務コード"],
        "name":        ["job_name","業務名","作業名","name"],
        "status":      ["status","ステータス","状態"],
        "progress":    ["progress","進捗","達成","完了/総数"]
    }
    for canon, alts in alias.items():
        if canon not in df.columns:
            for a in alts:
                if a in df.columns:
                    df.rename(columns={a: canon}, inplace=True)
                    break

    # schedule_id が無い場合は job_id または name から引く
    if "schedule_id" not in df.columns or df["schedule_id"].isna().all():
        sch = con.execute("SELECT id AS schedule_id, name FROM schedules WHERE tenant=?", [TENANT]).df()
        df["schedule_id"] = None
        if "job_id" in df.columns:
            # job_id = schedules.id と一致する想定
            df.loc[df["schedule_id"].isna(), "schedule_id"] = df.loc[df["schedule_id"].isna(), "job_id"]
        if "name" in df.columns and not sch.empty:
            # name をキーに解決（前後空白無視）
            m = df["schedule_id"].isna()
            df.loc[m, "schedule_id"] = df.loc[m, "name"].astype(str).str.strip().map(
                sch.set_index(sch["name"].astype(str).str.strip())["schedule_id"]
            )

    need = ["date","schedule_id","status"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.error(f"必須列が不足: {miss}")
        st.write("受け取った列:", list(df.columns))
        return

    s = pd.DataFrame()
    s["schedule_id"] = df["schedule_id"].astype(str).str.strip()
    s["date"]        = pd.to_datetime(df["date"], errors="coerce").dt.date
    s["status"]      = df["status"].astype(str).str.strip()

    # 進捗（59/61 等）を分解
    if "done" in df.columns and "total" in df.columns:
        s["done"]  = pd.to_numeric(df["done"], errors="coerce")
        s["total"] = pd.to_numeric(df["total"], errors="coerce")
    else:
        prog = df.get("progress")
        if prog is not None and len(df)>0:
            def split_prog(x):
                xs = str(x).replace(" ", "")
                if "/" in xs:
                    a,b = xs.split("/",1)
                    try:    return int(a or 0), int(b or 0)
                    except: return None, None
                return None, None
            d,t = zip(*[split_prog(v) for v in prog])
            s["done"], s["total"] = d, t

    s["done_at"] = pd.to_datetime(df.get("done_at") or df.get("完了日時"), errors="coerce")

    # schedule_id と date の欠損/空を除外
    s = s.dropna(subset=["schedule_id","date"])
    s = s[s["schedule_id"].astype(str).str.len() > 0]

    upsert_df("schedule_dates", s, ["schedule_id","date"])
    st.success(f"チケット取込: {len(s)} 行（schedule_id 自動解決あり）")


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
def render_analysis():
  st.title("分析・サマリー（β版）")
  tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📅 日報","🛠️ 設備","📈 月報","📄 ドキュメント","📥 取込","🤖 AIβ"])
  
  # 取込
with tab5:
    st.subheader("CSV 取込")
    st.caption("推奨順序：①マスタ → ②年間業務計画 → ③operation_tickets → ④issues → ⑤点検結果 → KPI再計算")

    # ① マスタ（階層+設備+targets）— 既存のまま
    f_master = st.file_uploader("マスタ（階層+設備+target）CSV", type=["csv"], key="upl_master")
    if f_master:
        import_master(pd.read_csv(f_master, encoding="utf-8-sig"))

    # ② 年間業務計画（スケジュール定義／予定日）
    f_plan = st.file_uploader("年間業務計画.csv（スケジュール定義/予定日）", type=["csv"], key="upl_plan")
    if f_plan:
        import_annual_plan(pd.read_csv(f_plan, encoding="utf-8-sig"))

    # ③ 実施チケット（実施日/進捗）— schedule_id が無くても job_id/業務名から自動解決
    f_tickets = st.file_uploader("operation_tickets.csv（実施日/進捗：schedule自動解決可）", type=["csv"], key="upl_tickets")
    if f_tickets:
        import_tickets(pd.read_csv(f_tickets, encoding="utf-8-sig"))

    # ④ 不具合
    f_issues = st.file_uploader("issues.csv（不具合）", type=["csv"], key="upl_issues")
    if f_issues:
        import_issues(pd.read_csv(f_issues, encoding="utf-8-sig"))

    # ⑤ 点検結果（横持ち）
    f_results = st.file_uploader("点検結果（横持ち）CSV", type=["csv"], key="upl_results")
    if f_results:
        import_results_wide(pd.read_csv(f_results, encoding="utf-8-sig"))

    # KPI 再計算
    if st.button("KPI再計算"):
        recalc_daily_kpis()
        st.success("daily_kpis 再計算")

  
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
     
    # ---- 数値ライン + ×/△ 背景シェード（同一時間軸・右軸をUIで選択） ----
      df["date"] = pd.to_datetime(df["date"], errors="coerce")
      num  = df.dropna(subset=["value_num"]).copy()
      qual = df[df["value_num"].isna()].copy()
      
      # 五感→sev(0=OK,1=△,2=×)
      sev_map = {"○":0, "OK":0, "良":0, "△":1, "要確認":1, "注意":1, "▲":1, "×":2, "NG":2, "異常":2}
      if "sev" not in qual.columns:
          qual["sev"] = qual["value_text"].map(sev_map).astype("float")
      qual["date"] = pd.to_datetime(qual["date"], errors="coerce")
      
      # 背景シェード対象の選択
      qual_targets_all = sorted(qual["target_name"].dropna().unique().tolist())
      default_sel = sorted(qual.loc[qual["sev"]==2, "target_name"].dropna().unique().tolist()) or qual_targets_all
      sel_targets = st.multiselect("背景色の対象（五感項目）", qual_targets_all, default=default_sel)
      shade_x     = st.checkbox("×の発生日で背景色", value=True)
      shade_tri   = st.checkbox("△の発生日も背景色に含める", value=False)
      shade_alpha = st.slider("背景の濃さ", 0.05, 0.4, 0.12, 0.01)
      
      qual_sel  = qual[qual["target_name"].isin(sel_targets)] if len(sel_targets) else qual.iloc[0:0]
      bad_days  = pd.to_datetime(qual_sel.loc[qual_sel["sev"]>=2, "date"], errors="coerce").dt.normalize().dropna().unique() if shade_x else []
      warn_days = pd.to_datetime(qual_sel.loc[qual_sel["sev"]==1, "date"], errors="coerce").dt.normalize().dropna().unique() if shade_tri else []
      
      # 右軸に出すシリーズを選択（単位欠損でもOK）
      series = (num.groupby("target_name")
                  .agg(unit=("unit", lambda s: next((u for u in s if pd.notna(u) and str(u) not in ["nan","None",""]), "")))
                  .reset_index())
      
      KW_RIGHT = ("圧力","差圧","圧","MPa","kPa","電圧","V")  # 初期候補
      suggest_right = [row.target_name for row in series.itertuples(index=False)
                       if any(k in row.target_name for k in KW_RIGHT)]
      right_targets = st.multiselect("右軸にする項目（圧力・電圧など）",
                                     series["target_name"].tolist(), default=suggest_right)
      
      from plotly.subplots import make_subplots
      import plotly.graph_objects as go
      fig = make_subplots(specs=[[{"secondary_y": True}]])
      
      # 数値トレース
      for tname, g in num.groupby("target_name"):
          unit_label = series.loc[series["target_name"]==tname, "unit"].iloc[0]
          fig.add_trace(
              go.Scatter(x=g["date"], y=g["value_num"], mode="lines+markers",
                         name=tname, hovertemplate="%{x|%Y-%m-%d}<br>%{y} "+(unit_label or "")),
              secondary_y=(tname in right_targets)
          )
      
      # 背景シェード
      for d in warn_days:
          fig.add_vrect(x0=d, x1=d, fillcolor="#FFD166", opacity=shade_alpha, line_width=0)
      for d in bad_days:
          fig.add_vrect(x0=d, x1=d, fillcolor="#EF476F", opacity=min(shade_alpha+0.05, 0.5), line_width=0)
      
      # 上部マーカー
      if st.checkbox("×/△ を上部に記号表示", value=True):
          for d in bad_days:
              fig.add_annotation(x=d, y=1.02, xref="x", yref="paper", text="×", showarrow=False,
                                 font=dict(color="#EF476F", size=14))
          for d in warn_days:
              fig.add_annotation(x=d, y=1.02, xref="x", yref="paper", text="▲", showarrow=False,
                                 font=dict(color="#FFD166", size=12))
      
      # 軸タイトル（←スクショで left_label が「右軸」になってたので直してね）
      left_units  = series.loc[~series["target_name"].isin(right_targets), "unit"].unique().tolist()
      right_units = series.loc[ series["target_name"].isin(right_targets), "unit"].unique().tolist()
      left_label  = " / ".join([u for u in left_units  if u]) or "左軸"
      right_label = " / ".join([u for u in right_units if u]) or "右軸"
      fig.update_yaxes(title_text=left_label,  secondary_y=False)
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
      # 五感スコア列を用意（なければ作成）
      if "sev" not in qual.columns:
          sev_map = {"○":0, "OK":0, "良":0, "△":1, "要確認":1, "注意":1, "▲":1, "×":2, "NG":2, "異常":2}
          qual["sev"] = qual["value_text"].map(sev_map).astype("float")
      
      # 異常日の縦帯（×/NGなど sev>=2）
      bad_days = (
          pd.to_datetime(qual.loc[qual["sev"] >= 2, "date"], errors="coerce")
            .dt.normalize().dropna().unique()
      )
      
      warn_days = (
          pd.to_datetime(qual.loc[qual["sev"] == 1, "date"], errors="coerce")
            .dt.normalize().dropna().unique()
      )
  for d in warn_days:
      fig.add_vrect(x0=d, x1=d, row="all", col=1, fillcolor="#FFD166", opacity=0.10, line_width=0)
  
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

import streamlit as st
import streamlit.components.v1 as components

# ---- 外部CMMSのベースURL（適宜変更）----
CMMS_BASE = st.secrets.get("CMMS_BASE", "https://your.cmms.example")
CMMS_LINKS = {
    "業務チケット":   f"{CMMS_BASE}/tickets",
    "不具合管理":     f"{CMMS_BASE}/issues",
    "モニタリング":   f"{CMMS_BASE}/monitoring",
    "報告書":         f"{CMMS_BASE}/reports",
    "写真報告":       f"{CMMS_BASE}/photos",
    "台帳管理":       f"{CMMS_BASE}/ledger",
    "年間業務計画":   f"{CMMS_BASE}/annual-plan",
}

def render_sidebar_and_route():
    with st.sidebar:
        st.markdown("### 管理ロイド")

        # 外部リンク（←ここを for の中で1段インデント）
        for label in ["業務チケット","不具合管理","モニタリング","報告書","写真報告","台帳管理","年間業務計画"]:
            st.link_button(f"・{label}", CMMS_LINKS[label], use_container_width=True, type="secondary")

        st.divider()

        # 年間業務計画の直下に β版（アプリ内ルート）
        go_analysis = st.button("・分析・サマリー（β版）", type="primary",
                                use_container_width=True, key="btn_analysis")

    # ルーティング（押したら保持）
    if go_analysis or st.session_state.get("route") == "analysis":
        st.session_state["route"] = "analysis"
        render_analysis()
    else:
        st.title("管理ロイド シェル")
        st.info("左メニューから各画面へ。『分析・サマリー（β版）』のみ本アプリ内で動作します。")


# ---- 起動 ----
render_sidebar_and_route()

