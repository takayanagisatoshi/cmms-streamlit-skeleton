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

    # PK 正規化（空/NaN/None を除外）
    for k in pk:
        df[k] = df[k].apply(_norm)
    mask_valid = df[pk].apply(lambda s: s.astype(str).str.len() > 0).all(axis=1)
    df = df[mask_valid].drop_duplicates(subset=pk, keep="last")

    # テーブル定義に存在する列だけに絞る（列順はこの順で固定）
    tbl_cols = con.execute(f"PRAGMA table_info('{table}')").df()["name"].tolist()
    cols = [c for c in df.columns if c in tbl_cols]

    # 先に DELETE（tenant + PK）をまとめて実行
    del_sql = f"DELETE FROM {table} WHERE " + " AND ".join([f"{k}=?" for k in ["tenant", *pk]])
    del_params = [(TENANT, *[r[k] for k in pk]) for _, r in df[pk].iterrows()]
    if del_params:
        con.executemany(del_sql, del_params)

    # 明示した列名で INSERT（不足列は DataFrame 側に作らずとも OK）
    ins_sql = f"INSERT INTO {table}({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
    ins_params = [tuple(r[c] for c in cols) for _, r in df[cols].iterrows()]
    if ins_params:
        con.executemany(ins_sql, ins_params)





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


# ---- 取込：マスタ（階層 + 設備 + targets + schedule_targets）------------
# ---- 取込：マスタ（階層 + 設備 + targets）---------------------------------
def import_master(df: pd.DataFrame):
    # 1) 列名を正規化（BOM/空白/大文字小文字）
    df = df.copy()
    df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]

    # 2) 列名エイリアス（日本語/英語どちらでもOK）
    alias = {
        "building_name": ["building", "building_id", "物件", "物件名"],
        "location_name": ["location", "棟", "棟名", "ブロック"],
        "floor_name":    ["floor", "階", "フロア"],
        "room_name":     ["room", "部屋", "室", "区画"],
        "device_name":   ["device", "設備", "設備名"],
        "target_id":     ["点検項目id", "target"],
        "target_name":   ["点検項目名", "target_label", "項目名"],
        "target_type_id":["input_type", "種別", "入力型"],
        "schedule_id":   ["schedule", "スケジュールid", "スケジュール"],
        "unit":          ["単位"],
        "lower":         ["下限", "min"],
        "upper":         ["上限", "max"],
        "order_no":      ["ord", "順序", "order"],
    }
    for canon, alts in alias.items():
        if canon not in df.columns:
            for a in alts:
                if a in df.columns:
                    df.rename(columns={a: canon}, inplace=True)
                    break

    # 3) 必須列チェック
    need = [
        "building_name","location_name","room_name",
        "device_name","target_id","target_name","target_type_id","schedule_id"
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.error(f"必須列が不足: {miss}")
        st.write("受け取った列:", list(df.columns))
        return

    # 4) 任意列のデフォルト
    defaults = {"floor_name": "", "unit": None, "lower": None, "upper": None, "order_no": 0}
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v

    # ---------- devices ----------
    d = df[["building_name","location_name","floor_name","room_name","device_name"]].copy()
    d = d.rename(columns={
        "building_name":"building_id",
        "location_name":"location_id",
        "floor_name":"floor_id",
        "room_name":"room_id",
        "device_name":"name"
    })
    for c in ["building_id","location_id","floor_id","room_id","name"]:
        d[c] = d[c].apply(_norm)

    # 設備名が空は除外
    d = d[d["name"].str.len() > 0]

    # 階層パスで device_id を生成
    d["id"] = d.apply(lambda r: _dev_id(
        r["building_id"], r["location_id"], r["floor_id"], r["room_id"], r["name"]
    ), axis=1)

    # 重複は畳む（警告も出す）
    if d.duplicated(subset=["id"]).any():
        cnt = int(d.duplicated(subset=["id"], keep=False).sum())
        st.warning(f"devices: 同一IDが {cnt} 行見つかりました（一部を折りたたみます）")
    d = d.drop_duplicates(subset=["id"], keep="last")

    # 追加属性（空でOK）
    d["category_l"]=d["category_m"]=d["category_s"]=d["symbol"]=d["cmms_url_rule"]=None

    upsert_df(
        "devices",
        d[[
            "id","building_id","location_id","floor_id","room_id","name",
            "category_l","category_m","category_s","symbol","cmms_url_rule"
        ]],
        ["id"]
    )

    # ---------- targets ----------
    t = df[[
        "building_name","location_name","floor_name","room_name","device_name",
        "target_id","target_name","target_type_id","unit","lower","upper","order_no"
    ]].copy()

    for c in ["building_name","location_name","floor_name","room_name","device_name","target_id"]:
        t[c] = t[c].apply(_norm)

    # device_id を階層パスから生成
    t["device_id"] = t.apply(lambda r: _dev_id(
        r["building_name"], r["location_name"], r["floor_name"], r["room_name"], r["device_name"]
    ), axis=1)

    # 整形
    t = t.rename(columns={
        "target_id":"id",
        "target_name":"name",
        "target_type_id":"input_type",
        "order_no":"ord"
    })

    # 空ID除外 + 重複折りたたみ
    t = t[t["id"].str.len() > 0]
    dup_cnt = int(t.duplicated(subset=["id"], keep=False).sum())
    if dup_cnt:
        st.warning(f"targets: 同一 target_id が {dup_cnt} 行見つかりました（最後の1件で上書き）")
    t = t.drop_duplicates(subset=["id"], keep="last")

    upsert_df(
        "targets",
        t[["id","device_id","name","input_type","unit","lower","upper","ord"]],
        ["id"]
    )

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

    # 1) schedule_id が無い/欠損の行だけ自動解決
    if "schedule_id" not in df.columns:
        df["schedule_id"] = None

    need_resolve = df["schedule_id"].isna() | (df["schedule_id"].astype(str).str.strip() == "")

    # 1-1) job_id を優先採用（job_id = schedules.id 想定）
    if "job_id" in df.columns:
        df.loc[need_resolve, "schedule_id"] = df.loc[need_resolve, "job_id"]

    # 1-2) name（業務名）→ schedules.name から引く（重複名は最後を採用）
    still = df["schedule_id"].isna() | (df["schedule_id"].astype(str).str.strip() == "")
    if still.any():
        sch = con.execute(
            "SELECT id AS schedule_id, name FROM schedules WHERE tenant=?",
            [TENANT]
        ).df()
        if not sch.empty and "name" in df.columns:
            sch_map = (
                sch.assign(key=sch["name"].astype(str).str.strip())
                   .dropna(subset=["key"])
                   .drop_duplicates(subset=["key"], keep="last")  # ★重複名を畳む
                   .set_index("key")["schedule_id"]
                   .to_dict()
            )
            df.loc[still, "schedule_id"] = (
                df.loc[still, "name"].astype(str).str.strip().map(sch_map)
            )

    # 以降は従来通り
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

    # 進捗（59/61 形式）
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

    # schedule_id / date の欠損は落とす
    before = len(s)
    s = s.dropna(subset=["schedule_id","date"])
    s = s[s["schedule_id"].astype(str).str.len() > 0]
    dropped = before - len(s)
    if dropped:
        st.warning(f"schedule_id または date 欠損で {dropped} 行をスキップ")

    upsert_df("schedule_dates", s, ["schedule_id","date"])

    # schedules に業務名を同期（年間計画を入れていない場合の保険）
    if "name" in df.columns:
        sch2 = (df[["schedule_id","name"]]
                .dropna(subset=["schedule_id"])
                .astype({"schedule_id": str}))
        sch2["name"] = sch2["name"].astype(str).str.strip()
        sch2 = sch2.drop_duplicates(subset=["schedule_id"])
        sch2 = sch2.rename(columns={"schedule_id": "id"})
        # freq は分からなければ入れないでOK
        upsert_df("schedules", sch2[["id","name"]], ["id"])


    st.success(f"チケット取込: {len(s)} 行（自動解決あり）")



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

    # ★タブをここで作成（この行と同じインデント階層で with tabX: を並べます）
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📅 日報","🛠️ 設備","📈 月報","📄 ドキュメント","📥 取込","🤖 AIβ"]
    )

    # ================= 取込（そのまま中身は流用してください） ================
    with tab5:
        st.subheader("CSV 取込")
        st.caption("推奨順序：①マスタ → ②年間業務計画 → ③operation_tickets → ④issues → ⑤点検結果 → KPI再計算")
        f_master  = st.file_uploader("マスタ（階層+設備+target）CSV", type=["csv"], key="upl_master")
        if f_master:  import_master(pd.read_csv(f_master, encoding="utf-8-sig"))

        f_plan = st.file_uploader("年間業務計画.csv（スケジュール定義/予定日）", type=["csv"], key="upl_plan")
        if f_plan:    import_annual_plan(pd.read_csv(f_plan, encoding="utf-8-sig"))

        f_tickets = st.file_uploader("operation_tickets.csv（実施日/進捗：schedule自動解決可）", type=["csv"], key="upl_tickets")
        if f_tickets: import_tickets(pd.read_csv(f_tickets, encoding="utf-8-sig"))

        f_issues = st.file_uploader("issues.csv（不具合）", type=["csv"], key="upl_issues")
        if f_issues:  import_issues(pd.read_csv(f_issues, encoding="utf-8-sig"))

        f_results = st.file_uploader("点検結果（横持ち）CSV", type=["csv"], key="upl_results")
        if f_results: import_results_wide(pd.read_csv(f_results, encoding="utf-8-sig"))

        if st.button("KPI再計算"): recalc_daily_kpis(); st.success("daily_kpis 再計算")

    # ================= 日報（業務名 JOIN 版） =================
    with tab1:
        st.subheader("本日の業務予定と進捗")
        target_date = st.date_input("対象日", value=date.today())

        q = """
        SELECT
            sd.schedule_id,
            COALESCE(s.name, CAST(sd.schedule_id AS VARCHAR)) AS job_name,
            sd.date, sd.status, sd.done, sd.total, sd.done_at
        FROM schedule_dates sd
        LEFT JOIN schedules s
          ON s.tenant = sd.tenant AND s.id = sd.schedule_id
        WHERE sd.tenant = ? AND sd.date = ?
        ORDER BY job_name
        """
        df = con.execute(q, [TENANT, target_date]).df()

        planned = int(df.shape[0])
        done    = int(df["status"].isin(["完了","実施済"]).sum()) if not df.empty else 0
        overdue = 0  # 当日ページなので0。必要ならロジック追加

        c1, c2, c3 = st.columns(3)
        c1.metric("予定", planned); c2.metric("完了", done); c3.metric("期限超過", overdue)

        if df.empty:
            st.info("対象日のデータがありません。")
        else:
            view = df.rename(columns={
                "schedule_id":"スケジュールID","job_name":"業務名","status":"ステータス",
                "done":"完了","total":"総数","done_at":"完了日時"
            })
            st.dataframe(view[["スケジュールID","業務名","ステータス","完了","総数","完了日時"]],
                         use_container_width=True)

        st.subheader("本日発生の不具合")
        issues = con.execute(
            "SELECT * FROM issues WHERE tenant=? AND reported_on=?",
            [TENANT, target_date]
        ).df()
        st.dataframe(issues, use_container_width=True)

    # ============ 以降、既存の tab2/tab3/tab4/tab6 をこの関数の中に続けて書く ============
    # with tab2: ...（設備）
    # with tab3: ...（月報）
    # with tab4: ...（ドキュメント）
    # with tab6: ...（AI）



    # ===== 設備 =========================================================
    with tab2:
        st.title("🛠️ 設備ページ")

        # 段階選択
        blds = con.execute("SELECT DISTINCT building_id FROM devices WHERE tenant=?", [TENANT]).df()["building_id"].tolist()
        if not blds:
            st.info("まずは『📥 取込』からマスタCSVを投入してください。")
            st.stop()
        bld = st.selectbox("物件", blds)

        locs = con.execute(
            "SELECT DISTINCT location_id FROM devices WHERE tenant=? AND building_id=?", [TENANT, bld]
        ).df()["location_id"].tolist()
        loc = st.selectbox("棟", locs) if locs else None

        flrs = con.execute(
            "SELECT DISTINCT floor_id FROM devices WHERE tenant=? AND building_id=? AND location_id=?",
            [TENANT, bld, loc]
        ).df()["floor_id"].tolist() if loc else []
        flr = st.selectbox("フロア", flrs) if flrs else None

        rooms = con.execute(
            "SELECT DISTINCT room_id FROM devices WHERE tenant=? AND building_id=? AND location_id=? AND floor_id=?",
            [TENANT, bld, loc, flr]
        ).df()["room_id"].tolist() if flr else []
        room = st.selectbox("部屋", rooms) if rooms else None

        devs = con.execute(
            """SELECT id,name FROM devices
               WHERE tenant=? AND building_id=? AND location_id=? AND floor_id=? AND room_id=?""",
            [TENANT, bld, loc, flr, room]
        ).df() if room else pd.DataFrame(columns=["id", "name"])
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
            st.info("期間を選択すると、グラフと表が表示されます。")
            st.stop()

        # データ取得
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
            st.warning("期間内のデータがありません。")
            st.stop()

        df["date"] = pd.to_datetime(df["date"])
        abnormal_mask = (
            (df["value_num"].notna() & (
                (df["lower"].notna() & (df["value_num"] < df["lower"])) |
                (df["upper"].notna() & (df["value_num"] > df["upper"]))
            )) |
            (df["value_num"].isna() & df["value_text"].isin(["×", "NG"]))
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("対象項目", int(df["target_id"].nunique()))
        c2.metric("データ点数", int(df.shape[0]))
        c3.metric("異常件数", int(abnormal_mask.sum()))

        # スコア化・系列/背景指定
        label_to_score = {"○": 0, "OK": 0, "良": 0, "△": 1, "要確認": 1, "注意": 1, "▲": 1, "×": 2, "NG": 2, "異常": 2}
        num  = df.dropna(subset=["value_num"]).copy()
        qual = df[df["value_num"].isna()].copy()
        qual["sev"] = qual["value_text"].map(label_to_score).astype("float")

        qual_targets_all = sorted(qual["target_name"].dropna().unique().tolist())
        default_sel = sorted(qual.loc[qual["sev"] == 2, "target_name"].dropna().unique().tolist()) or qual_targets_all
        sel_targets = st.multiselect("背景色の対象（五感項目）", qual_targets_all, default=default_sel)
        shade_x     = st.checkbox("×の発生日で背景色", value=True)
        shade_tri   = st.checkbox("△の発生日も背景色に含める", value=False)
        shade_alpha = st.slider("背景の濃さ", 0.05, 0.4, 0.12, 0.01)

        qual_sel  = qual[qual["target_name"].isin(sel_targets)] if len(sel_targets) else qual.iloc[0:0]
        bad_days  = pd.to_datetime(qual_sel.loc[qual_sel["sev"] >= 2, "date"], errors="coerce").dt.normalize().dropna().unique() if shade_x else []
        warn_days = pd.to_datetime(qual_sel.loc[qual_sel["sev"] == 1, "date"], errors="coerce").dt.normalize().dropna().unique() if shade_tri else []

        series = (num.groupby("target_name")
                    .agg(unit=("unit", lambda s: next((u for u in s if pd.notna(u) and str(u) not in ["nan", "None", ""]), "")))
                    .reset_index())

        KW_RIGHT = ("圧力", "差圧", "圧", "MPa", "kPa", "電圧", "V")
        suggest_right = [row.target_name for row in series.itertuples(index=False)
                         if any(k in row.target_name for k in KW_RIGHT)]
        right_targets = st.multiselect("右軸にする項目（圧力・電圧など）",
                                       series["target_name"].tolist(), default=suggest_right)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for tname, g in num.groupby("target_name"):
            unit_label = series.loc[series["target_name"] == tname, "unit"].iloc[0]
            fig.add_trace(
                go.Scatter(x=g["date"], y=g["value_num"], mode="lines+markers",
                           name=tname, hovertemplate="%{x|%Y-%m-%d}<br>%{y} " + (unit_label or "")),
                secondary_y=(tname in right_targets)
            )

        for d in warn_days:
            fig.add_vrect(x0=d, x1=d, fillcolor="#FFD166", opacity=shade_alpha, line_width=0)
        for d in bad_days:
            fig.add_vrect(x0=d, x1=d, fillcolor="#EF476F", opacity=min(shade_alpha + 0.05, 0.5), line_width=0)

        if st.checkbox("×/△ を上部に記号表示", value=True):
            for d in bad_days:
                fig.add_annotation(x=d, y=1.02, xref="x", yref="paper", text="×", showarrow=False,
                                   font=dict(color="#EF476F", size=14))
            for d in warn_days:
                fig.add_annotation(x=d, y=1.02, xref="x", yref="paper", text="▲", showarrow=False,
                                   font=dict(color="#FFD166", size=12))

        left_units  = series.loc[~series["target_name"].isin(right_targets), "unit"].unique().tolist()
        right_units = series.loc[ series["target_name"].isin(right_targets), "unit"].unique().tolist()
        left_label  = " / ".join([u for u in left_units  if u]) or "左軸"
        right_label = " / ".join([u for u in right_units if u]) or "右軸"
        fig.update_yaxes(title_text=left_label,  secondary_y=False)
        fig.update_yaxes(title_text=right_label, secondary_y=True)

        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)

        # 下：イベント（五感）一覧
        if not qual.empty:
            show = qual_sel.copy() if len(sel_targets) else qual.copy()
            col1, col2 = st.columns(2)
            f_bad = col1.checkbox("×のみ表示", value=False)
            f_tri = col2.checkbox("△のみ表示", value=False)
            if f_bad: show = show[show["sev"] == 2]
            if f_tri: show = show[show["sev"] == 1]
            st.dataframe(
                show.sort_values(["date", "target_name"])[["date", "target_name", "value_text"]]
                    .rename(columns={"date": "日付", "target_name": "項目", "value_text": "判定"}),
                use_container_width=True
            )

        # この設備の不具合
        st.subheader("この設備に紐づく不具合")
        iss = con.execute(
            """SELECT id, reported_on, due_on, status, severity, category, summary
               FROM issues WHERE tenant=? AND device_id=?
               ORDER BY COALESCE(due_on, reported_on) DESC""",
            [TENANT, dev]
        ).df()
        c1, c2, c3 = st.columns(3)
        if iss.empty:
            c1.metric("未完了", 0); c2.metric("期限超過", 0); c3.metric("今月新規", 0)
            st.info("紐づく不具合はありません。")
        else:
            not_done = ~iss["status"].isin(["完了", "対応済"])
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
            [TENANT, dev]
        ).df()
        if docs.empty:
            st.info("この設備に紐づくドキュメントは未登録です。")
        else:
            st.dataframe(docs, use_container_width=True)

        # ターゲット定義
        st.subheader("点検項目（Targets 定義）")
        tl = con.execute(
            """SELECT id AS target_id, name, input_type, unit, lower, upper, ord
               FROM targets WHERE tenant=? AND device_id=? ORDER BY ord""",
            [TENANT, dev]
        ).df()
        st.dataframe(tl, use_container_width=True)

    # ===== 月報/ドキュメント/AI（プレースホルダ） =========================
    with tab3:
        st.subheader("指定月のサマリー（プレースホルダ）")
        st.info("CSV取込後、月次集計を実装します。")

    with tab4:
        st.subheader("ドキュメント一覧（プレースホルダ）")
        docs = con.execute("SELECT id,title,category,tags,ai_summary FROM documents WHERE tenant=?", [TENANT]).df()
        st.dataframe(docs, use_container_width=True)

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

