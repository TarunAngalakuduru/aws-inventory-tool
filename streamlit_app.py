"""
Web UI for the AWS Inventory AI tool   (run locally:  streamlit run streamlit_app.py)

- Sidebar: region to inventory (user chooses) + optional IAM Role ARN + optional own access keys
  Priority: Role ARN > user keys > server keys from Secrets
- Chat: short answer from Nova + full AWS data as a table (CSV/JSON download, copy answer)
- Bottom of page: Nova input/output tokens and estimated cost, per question and all-time.
  Usage is saved to disk (and optionally S3) so a browser refresh does NOT clear it.

Streamlit Cloud -> app Settings -> Secrets (all optional except the AWS keys):
    AWS_ACCESS_KEY_ID        = "..."
    AWS_SECRET_ACCESS_KEY    = "..."
    BEDROCK_REGION           = "ap-south-1"      # where Nova is called (NOT the inventory region)
    APP_PASSWORD             = "..."             # strongly recommended (app is public)
    NOVA_INPUT_PRICE_PER_1M  = "0.06"            # USD, override if your region differs
    NOVA_OUTPUT_PRICE_PER_1M = "0.24"
    USAGE_S3_BUCKET          = "my-bucket"       # optional: keeps usage even after app reboot
"""
import hmac
import json
import os
from datetime import datetime, timezone

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError

st.set_page_config(page_title="AWS Inventory AI", page_icon="☁️", layout="wide")

# ------------------------------------------------------------------ 0. SECRETS -> ENV
# Streamlit Cloud "Secrets" become environment variables (boto3 reads AWS keys from env).
# Must run BEFORE importing app.py, which reads its settings from env.
try:
    for _k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
               "BEDROCK_REGION", "BEDROCK_MODEL_ID", "ALLOWED_SERVICES", "APP_PASSWORD",
               "NOVA_INPUT_PRICE_PER_1M", "NOVA_OUTPUT_PRICE_PER_1M",
               "USAGE_S3_BUCKET", "USAGE_S3_KEY", "USAGE_FILE"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass                     # no secrets file (local run) -> use CLI credentials

import app as core          # all AWS + AI logic lives in app.py

# Nova Lite on-demand list price in USD per 1M tokens (override via Secrets).
PRICE_IN = float(os.getenv("NOVA_INPUT_PRICE_PER_1M", "0.06"))
PRICE_OUT = float(os.getenv("NOVA_OUTPUT_PRICE_PER_1M", "0.24"))

USAGE_FILE = os.getenv("USAGE_FILE", "usage_log.json")
USAGE_BUCKET = os.getenv("USAGE_S3_BUCKET", "").strip()
USAGE_KEY = os.getenv("USAGE_S3_KEY", "aws-inventory-ai/usage_log.json")
MAX_HISTORY = 500            # keep the last 500 questions in the history table


# ------------------------------------------------------------------ 1. USAGE STORAGE
def fresh_store():
    return {"totals": {"input": 0, "output": 0, "calls": 0, "questions": 0, "cost": 0.0},
            "since": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "history": []}


def usage_warn(msg):
    """Remember a storage problem so the usage panel can show it."""
    st.session_state.usage_warning = msg


def s3_client():
    return boto3.client("s3", region_name=core.BEDROCK_REGION)


def load_usage():
    """Read saved usage (S3 first if configured, then local file, else empty)."""
    st.session_state.pop("usage_warning", None)
    if USAGE_BUCKET:
        try:
            body = s3_client().get_object(Bucket=USAGE_BUCKET, Key=USAGE_KEY)["Body"].read()
            return json.loads(body)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code not in ("NoSuchKey", "404"):              # NoSuchKey = first run, that's normal
                usage_warn(f"Could not read usage from S3 ({code}). Check bucket name and s3:GetObject permission.")
        except Exception as e:
            usage_warn(f"Could not read usage from S3: {e}")
    try:
        with open(USAGE_FILE) as f:
            return json.load(f)
    except Exception:
        return fresh_store()


def save_usage(store):
    """Write usage to local file, and to S3 if configured."""
    try:
        with open(USAGE_FILE, "w") as f:
            json.dump(store, f)
    except Exception as e:
        core.log.warning("Usage file save failed: %s", e)
    if USAGE_BUCKET:
        try:
            s3_client().put_object(Bucket=USAGE_BUCKET, Key=USAGE_KEY,
                                   Body=json.dumps(store).encode(), ContentType="application/json")
        except ClientError as e:
            usage_warn(f"Could not save usage to S3 ({e.response['Error']['Code']}). Check s3:PutObject permission.")
        except Exception as e:
            usage_warn(f"Could not save usage to S3: {e}")


def cost_of(inp, out):
    return inp / 1_000_000 * PRICE_IN + out / 1_000_000 * PRICE_OUT


def record_usage(question, usage):
    """Add one question's tokens + cost to the saved usage and return the new store."""
    store = load_usage()                                  # reload so we never overwrite newer data
    inp, out, calls = usage.get("input", 0), usage.get("output", 0), usage.get("calls", 0)
    cost = cost_of(inp, out)
    t = store["totals"]
    t["input"] += inp
    t["output"] += out
    t["calls"] += calls
    t["questions"] += 1
    t["cost"] += cost
    store["history"].append({"time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                             "question": question[:80], "input": inp, "output": out,
                             "calls": calls, "cost": cost})
    store["history"] = store["history"][-MAX_HISTORY:]
    save_usage(store)
    return store


# ------------------------------------------------------------------ 2. TABLE BUILDER
def to_table(datasets):
    """Turn raw AWS responses of ONE operation into a single table (or None)."""
    frames = []
    for d in datasets:
        groups = {}                                       # path -> list of row dicts

        def walk(o, path):
            if isinstance(o, list):
                if o and all(isinstance(x, dict) for x in o):
                    groups.setdefault(path, []).extend(o)
                for x in o:
                    walk(x, path + "[]")
            elif isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{path}.{k}" if path else k)

        walk(d["data"], "")
        if groups:   # pick the list that looks like the "main" resources (most rows x columns)
            rows = max(groups.values(), key=lambda r: len(r) * len({k for x in r for k in x}))
        else:
            rows = [d["data"]] if isinstance(d["data"], dict) and d["data"] else []
        if not rows:
            continue
        df = pd.json_normalize(rows, sep=".")
        for c in df.columns:                              # nested lists/dicts -> JSON text
            df[c] = df[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)
        if len(datasets) > 1:                             # many calls (e.g. one per bucket)
            df.insert(0, "_request", json.dumps(d["params"], default=str))
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).dropna(axis=1, how="all")
    return out.fillna("").astype(str)


# ------------------------------------------------------------------ 3. SHOW FULL DATA
def render_data(captured, key):
    """Under each answer: one expander per AWS operation with table + downloads."""
    by_op = {}
    for d in captured:
        by_op.setdefault((d["service"], d["operation"]), []).append(d)
    for i, ((svc, op), items) in enumerate(by_op.items()):
        with st.expander(f"📋 Full data: {svc}.{op}"):
            table = to_table(items)
            if table is not None:
                st.dataframe(table, width="stretch")
                st.download_button("Download CSV", table.to_csv(index=False), f"{svc}_{op}.csv",
                                   key=f"{key}-{i}-csv")
            else:
                st.info("No rows returned.")
            st.download_button("Download JSON", json.dumps([x["data"] for x in items], default=str, indent=2),
                               f"{svc}_{op}.json", key=f"{key}-{i}-json")


def render_message(m, idx):
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m["role"] == "assistant":
            with st.expander("📄 Copy answer"):
                st.code(m["text"], language=None)         # code blocks have a copy button
            render_data(m.get("data", []), f"m{idx}")


# ------------------------------------------------------------------ 4. PASSWORD (optional)
def password_gate():
    pw = os.getenv("APP_PASSWORD")
    if not pw:
        st.warning("No APP_PASSWORD set - anyone with the link can see your AWS inventory.")
        return
    if st.session_state.get("authed"):
        return
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered, pw):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


# ------------------------------------------------------------------ 5. CONNECT / SIDEBAR
def connect(region, role, access_key="", secret_key="", token=""):
    """Log in to AWS and store the session for this browser tab only."""
    session, who = core.make_session(region, role, access_key, secret_key, token)
    st.session_state.update(
        session=session, who=who, region=region, history=[], chat=[],
        bedrock=core.make_bedrock_client())


def sidebar():
    with st.sidebar:
        st.header("Connect")
        region = st.text_input("AWS Region to inventory", placeholder="e.g. us-east-1").strip()
        role = st.text_input("IAM Role ARN (optional)",
                             placeholder="arn:aws:iam::123456789012:role/ReadOnly").strip()
        st.caption("If a Role ARN is given it is always used.")
        with st.expander("Or use your own access keys (optional)"):
            ak = st.text_input("Access Key ID", type="password").strip()
            sk = st.text_input("Secret Access Key", type="password").strip()
            tok = st.text_input("Session Token (if temporary)", type="password").strip()
            st.caption("Kept in this browser session only. Never saved or logged.")
        if st.button("Connect", type="primary"):
            if not region:
                st.session_state.connect_error = "Please enter an AWS region."
            else:
                try:
                    connect(region, role, ak, sk, tok)
                    st.session_state.pop("connect_error", None)
                except core.AppError as e:
                    st.session_state.connect_error = str(e)
        if st.session_state.get("connect_error"):
            st.error(st.session_state.connect_error)
        if "who" in st.session_state:
            st.success(f"Connected ({st.session_state.region})\n\n{st.session_state.who}")
            if st.button("Clear chat"):
                st.session_state.history, st.session_state.chat = [], []
                st.rerun()
        st.caption(f"Model: {core.MODEL_ID} (called from {core.BEDROCK_REGION})")


# ------------------------------------------------------------------ 6. TOKEN + COST PANEL (bottom)
def money(x):
    return f"${x:,.6f}" if x < 1 else f"${x:,.4f}"


def token_footer():
    store = st.session_state.usage_store
    hist, tot = store["history"], store["totals"]
    last = hist[-1] if hist else {"input": 0, "output": 0, "cost": 0.0, "question": "-"}
    st.divider()
    st.subheader("Nova token usage & cost")
    if st.session_state.get("usage_warning"):
        st.warning(st.session_state.usage_warning)
    elif not USAGE_BUCKET:
        st.info("Usage is saved on the app's temporary disk and is lost when Streamlit restarts, redeploys or "
                "sleeps the app. Add USAGE_S3_BUCKET in Secrets to keep it permanently.")
    st.caption(f"Last question: “{last['question']}”")
    a = st.columns(4)
    a[0].metric("Last - input tokens", f"{last['input']:,}")
    a[1].metric("Last - output tokens", f"{last['output']:,}")
    a[2].metric("Last - total tokens", f"{last['input'] + last['output']:,}")
    a[3].metric("Last - cost", money(last["cost"]))
    st.caption(f"All-time since {store.get('since', '-')}  ({tot['questions']:,} questions, {tot['calls']:,} model calls)")
    b = st.columns(4)
    b[0].metric("All-time - input tokens", f"{tot['input']:,}")
    b[1].metric("All-time - output tokens", f"{tot['output']:,}")
    b[2].metric("All-time - total tokens", f"{tot['input'] + tot['output']:,}")
    b[3].metric("All-time - cost", money(tot["cost"]))
    with st.expander("📊 Usage history (per question)"):
        if hist:
            df = pd.DataFrame(hist[::-1]).rename(columns={
                "time": "Time", "question": "Question", "input": "Input tokens",
                "output": "Output tokens", "calls": "Model calls", "cost": "Cost (USD)"})
            df["Cost (USD)"] = df["Cost (USD)"].map(lambda c: f"{c:.6f}")
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No questions yet.")
        if st.button("Reset usage history"):
            st.session_state.usage_store = fresh_store()
            save_usage(st.session_state.usage_store)
            st.rerun()
    st.caption(f"Estimate: ${PRICE_IN}/1M input + ${PRICE_OUT}/1M output tokens (Nova Lite on-demand list price; "
               "your region may differ - set NOVA_INPUT_PRICE_PER_1M / NOVA_OUTPUT_PRICE_PER_1M). "
               "Excludes Cost Explorer API charges.")


# ------------------------------------------------------------------ 7. MAIN
def main():
    st.title("☁️ AWS Inventory AI")
    password_gate()
    if "usage_store" not in st.session_state:             # loaded from disk/S3 -> survives refresh
        st.session_state.usage_store = load_usage()
    sidebar()
    if "session" not in st.session_state:
        st.info("Enter the AWS region (and optional Role ARN or access keys) in the sidebar, then click Connect.")
        token_footer()
        return

    for i, m in enumerate(st.session_state.chat):
        render_message(m, i)

    token_footer()
    question = st.chat_input("Ask about your AWS resources, e.g. 'show my EC2 instances'")
    if not question:
        return
    st.session_state.chat.append({"role": "user", "text": question})
    render_message(st.session_state.chat[-1], len(st.session_state.chat) - 1)

    captured, usage = [], {}
    with st.chat_message("assistant"), st.spinner("Working..."):
        try:
            text = core.ask_ai(question, st.session_state.history, st.session_state.bedrock,
                               core.tool_definitions(), st.session_state.session,
                               st.session_state.region, captured, usage)
        except Exception as e:                            # never show stack traces
            core.log.exception("UI error")
            text = f"Error: {core.friendly_error(e)}"
    st.session_state.chat.append({"role": "assistant", "text": text, "data": captured})
    st.session_state.usage_store = record_usage(question, usage)   # saved -> survives refresh
    st.rerun()                                            # redraw with table, copy box, new totals


if __name__ == "__main__":
    main()
