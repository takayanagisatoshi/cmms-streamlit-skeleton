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
    st.succe
