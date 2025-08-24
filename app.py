# app.py — 単一ファイル版 CMMS ダッシュボード（CSV取込つき）
import os, io, textwrap, pandas as pd, duckdb as ddb
import streamlit as st
import plotly.express as px

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
    if df.empty: return
    df = df.copy()
    df["tenant"] = TENANT
    cols = list(df.columns)
    for _, r in df.iterrows():
        where = " AND ".join([f"{k} = ?" for k in ["tenant", *pk]])
        con.execute(f"DELETE FROM {table} WHERE {where}", [TENANT, *[r[k] for k in pk]])
        con.execute(f"INSERT INTO {table}({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", r.tolist())

# ---- 取込：マスタ（階層 + 設備 + target + schedule 紐付け） ------------
def import_master(df: pd.DataFrame):
    df = df.rename(columns=str.lower)
    need = ["building_name","location_name","room_name","device_name","target_id","target_name","target_type_id","schedule_id"]
    missing = [c for c in need if c not in df.columns]
    if missing: st.error(f"必須列が不足: {missing}"); return
    df["floor_name"] = df.get("floor_name","指定なし")
    # IDs（名前をそのままID扱いでOK）
    b = df[["building_name"]].drop_duplicates().rename(columns={"building_name":"name"}); b["id"]=b["name"]; upsert_df("buildings", b[["id","name"]], ["id"])
    l = df[["location_name","building_name"]].drop_duplicates().rename(columns={"location_name":"name","building_name":"building_id"})
    l["id"]=l["name"]; upsert_df("locations", l[["id","building_id","name"]], ["id"])
    f = df[["floor_name","location_name"]].drop_duplicates().rename(columns={"floor_name":"name","location_name":"location_id"})
    f["id"]=f["name"]; upsert_df("floors", f[["id","location_id","name"]], ["id"])
    r = df[["room_name","floor_name"]].drop_duplicates().rename(columns={"room_name":"name","floor_name":"floor_id"})
    r["id"]=r["name"]; upsert_df("rooms", r[["id","floor_id","name"]], ["id"])
    d = df[["device_name","building_name","location_name","floor_name","room_name"]].drop_duplicates().rename(
        columns={"device_name":"name","building_name":"building_id","location_name":"location_id","floor_name":"floor_id","room_name":"room_id"})
    d["id"]=d["name"]; d["category_l"]=d["category_m"]=d["category_s"]=d["symbol"]=d["cmms_url_rule"]=None
    upsert_df("devices", d[["id","building_id","location_id","floor_id","room_id","name","category_l","category_m","category_s","symbol","cmms_url_rule"]], ["id"])
    t = df[["target_id","device_name","target_name","target_type_id","unit","lower","upper","order_no"]].copy()
    t.rename(columns={"target_id":"id","device_name":"device_id","target_name":"name","target_type_id":"input_type","order_no":"ord"}, inplace=True)
    upsert_df("targets", t[["id","device_id","name","input_type","unit","lower","upper","ord"]], ["id"])
    s = df[["schedule_id"]].drop_duplicates().rename(columns={"schedule_id":"id"}); s["job_id"]=None; s["name"]=None; s["freq"]=None
    upsert_df("schedules", s[["id","job_id","name","freq"]], ["id"])
    stgt = df[["schedule_id","target_id"]].drop_duplicates().rename(columns={"schedule_id":"schedule_id","target_id":"target_id"})
    upsert_df("schedule_targets", stgt, ["schedule_id","target_id"])
    st.success(f"マスタ取込: {len(df)} 行")

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
        "date": pd.to_datetime(df["date"]).dt.date,
        "status": df["status"],
        "done": done, "total": total,
        "done_at": pd.to_datetime(df.get("done_at"), errors="coerce")
    })
    upsert_df("schedule_dates", s, ["schedule_id","date"])
    st.success(f"チケット取込: {len(df)} 行")

# ---- 取込：不具合 --------------------------------------------------------
def import_issues(df: pd.DataFrame):
    df = df.rename(columns=str.lower)
    need = ["id","reported_on"]; miss=[c for c in need if c not in df.columns]
    if miss: st.error(f"必須列が不足: {miss}"); return
    df["reported_on"] = pd.to_datetime(df["reported_on"], errors="coerce").dt.date
    if "due_on" in df.columns: df["due_on"] = pd.to_datetime(df["due_on"], errors="coerce").dt.date
    keep = ["id","device_id","reported_on","due_on","status","severity","category","summary","cmms_url_rule"]
    for k in keep:
        if k not in df.columns: df[k]=None
    upsert_df("issues", df[keep], ["id"])
    st.success(f"不具合取込: {len(df)} 行")

# ---- 取込：点検結果（横持ち→縦melt） -----------------------------------
def import_results_wide(df: pd.DataFrame):
    df = df.rename(columns=str.lower)
    fixed = ["schedule_id","device_id","target_id","target_name","target_type_id","unit"]
    date_cols = [c for c in df.columns if c not in fixed]
    rows = []
    for _, r in df.iterrows():
        for dc in date_cols:
            val = r[dc]
            if pd.isna(val) or str(val)=="":
                continue
            rows.append({
                "schedule_id": r["schedule_id"],
                "date": pd.to_datetime(dc).date(),
                "target_id": r["target_id"],
                "value_num": pd.to_numeric(val, errors="coerce"),
                "value_text": None if pd.to_numeric(val, errors="coerce")==pd.to_numeric(val, errors="coerce") else str(val),
                "value_bool": None,
                "judged": None
            })
    if rows:
        upsert_df("results", pd.DataFrame(rows), ["schedule_id","date","target_id"])
    st.success(f"点検結果取込: {len(df)} 行（空値は除外）")

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

with tab5:  # 取込
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

with tab1:  # 日報
    st.subheader("本日の業務予定と進捗")
    target_date = st.date_input("対象日", pd.Timestamp.today()).to_pydatetime().date()
    kpi = con.execute("SELECT planned, done, overdue FROM daily_kpis WHERE tenant=? AND date=?",
                      [TENANT, target_date]).fetchone()
    planned, done, overdue = (kpi if kpi else (0,0,0))
    c1,c2,c3 = st.columns(3); c1.metric("予定", planned); c2.metric("完了", done); c3.metric("期限超過", overdue)
    df = con.execute("SELECT * FROM schedule_dates WHERE tenant=? AND date=?", [TENANT, target_date]).df()
    st.dataframe(df, use_container_width=True)

    st.subheader("本日発生の不具合")
    issues = con.execute("SELECT * FROM issues WHERE tenant=? AND reported_on=?", [TENANT, target_date]).df()
    st.dataframe(issues, use_container_width=True)

with tab2:  # 設備
    st.subheader("設備の点検結果")
    devices = con.execute("SELECT id, name FROM devices WHERE tenant=?", [TENANT]).df()
    dev = st.selectbox("設備を選択", options=devices["id"] if not devices.empty else [])
    dr = st.date_input("期間", [])
    if dev and len(dr)==2:
        q = """
        SELECT r.date, r.target_id, r.value_num, r.value_text
        FROM results r JOIN targets t ON t.tenant=r.tenant AND t.id=r.target_id
        WHERE r.tenant=? AND t.device_id=? AND r.date BETWEEN ? AND ? ORDER BY r.date
        """
        df = con.execute(q, [TENANT, dev, dr[0], dr[1]]).df()
        num = df.dropna(subset=["value_num"])
        if not num.empty:
            st.plotly_chart(px.line(num, x="date", y="value_num", color="target_id"), use_container_width=True)
        qual = df[df["value_num"].isna()]
        if not qual.empty:
            st.dataframe(qual.pivot_table(index="date", columns="target_id", values="value_text", aggfunc="first"),
                         use_container_width=True)

with tab3:  # 月報（プレースホルダ）
    st.subheader("指定月のサマリー（プレースホルダ）")
    st.info("CSV取込後、月次集計を実装します。")

with tab4:  # ドキュメント（プレースホルダ）
    st.subheader("ドキュメント一覧（プレースホルダ）")
    docs = con.execute("SELECT id,title,category,tags,ai_summary FROM documents WHERE tenant=?", [TENANT]).df()
    st.dataframe(docs, use_container_width=True)

with tab6:  # AI（プレースホルダ）
    st.subheader("AIサマリー（β）")
    st.info("OCR/AI連携は後で接続。まずはCSV→ダッシュボードの流れを固めます。")

st.caption("Theme: 管理ロイド風 / データは UTF-8 CSV を 取込タブから投入")
