"""
AWS Inventory AI Tool  (read-only, no hardcoded APIs)

What it does
  1. Asks for AWS region + IAM Role ARN (or uses your CLI credentials).
  2. You ask questions in plain English ("show my EC2 instances").
  3. Amazon Nova Lite looks through boto3's own API list (the same list as
     https://docs.aws.amazon.com/boto3/latest/reference/services/),
     picks the right API, runs it, and explains the result.

Setup:   pip install boto3
Run:     python app.py
Optional environment variables:
  BEDROCK_MODEL_ID  default: apac.amazon.nova-lite-v1:0
  BEDROCK_REGION    default: ap-south-1  (where the Nova model is called; not your inventory region)
  ALLOWED_SERVICES  e.g. "s3,ec2,rds,lambda"  (default: every boto3 service)
"""
import json
import logging
import os
import re
from datetime import datetime, timezone

import boto3
import botocore.session
from botocore import xform_name
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

# =====================================================================
# 1. SETTINGS
# =====================================================================
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "apac.amazon.nova-lite-v1:0")
# Region where the Nova model is CALLED (separate from the region the user wants to inventory).
# "apac." model IDs must be called from an Asia Pacific region.
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "ap-south-1")

# Optional: restrict to some services. Empty = all services boto3 knows.
ALLOWED_SERVICES = [s.strip() for s in os.getenv("ALLOWED_SERVICES", "").split(",") if s.strip()]

# SAFETY: only operations whose names start with these are ever allowed (read-only).
READ_PREFIXES = ("list_", "describe_", "get_")
# A few "get_" operations that expose secrets/data are blocked anyway.
BLOCKED_OPS = {("s3", "get_object"), ("s3", "get_object_torrent"),
               ("ec2", "get_password_data"), ("ec2", "get_console_screenshot"),
               ("secretsmanager", "get_secret_value"), ("ssm", "get_parameter"),
               ("ssm", "get_parameters"), ("ssm", "get_parameters_by_path")}

MAX_ITEMS = 100      # max items fetched per paginated AWS call
MAX_CHARS = 40000    # max size of AWS data handed back to the AI
MAX_STEPS = 20       # max AWS calls the AI may make for ONE question

# =====================================================================
# 2. LOGGING  (everything goes to app.log for troubleshooting)
# =====================================================================
logging.basicConfig(filename="app.log", level=logging.DEBUG,
                    format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("botocore", "boto3", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)   # hide boto's noisy debug logs
log = logging.getLogger("inventory")


class AppError(Exception):
    """Startup problem shown to the user (bad region, bad role, ...)."""


class ToolError(Exception):
    """Problem sent back to the AI so it can fix its request or explain it."""


# =====================================================================
# 3. LOGIN: region + IAM role (or CLI credentials)
# =====================================================================
def make_session(region, role_arn="", access_key="", secret_key="", session_token=""):
    """Return (boto3 session, identity ARN).
    Priority: Role ARN (assumed) > user-supplied keys > server/CLI credentials.
    If a Role ARN is given it is assumed using the user's keys if supplied, else the server/CLI credentials."""
    if region not in boto3.Session().get_available_regions("ec2"):
        raise AppError(f"Invalid AWS region: {region}")
    if role_arn and (not role_arn.startswith("arn:aws") or ":role/" not in role_arn):
        raise AppError("Invalid IAM Role ARN format.")
    if bool(access_key) != bool(secret_key):
        raise AppError("Provide both Access Key ID and Secret Access Key (or neither).")
    try:
        if access_key:                                            # user's own keys (never logged/stored)
            base = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                                 aws_session_token=session_token or None, region_name=region)
        else:                                                     # server / CLI credentials
            base = boto3.Session(region_name=region)

        if role_arn:                                              # role takes priority
            c = base.client("sts").assume_role(
                RoleArn=role_arn, RoleSessionName="inventory-ai")["Credentials"]
            session = boto3.Session(aws_access_key_id=c["AccessKeyId"],
                                    aws_secret_access_key=c["SecretAccessKey"],
                                    aws_session_token=c["SessionToken"], region_name=region)
            log.info("Using assumed role %s", role_arn)
        else:
            session = base
            log.info("Using %s credentials", "user-supplied" if access_key else "server/CLI")
        who = session.client("sts").get_caller_identity()["Arn"]  # proves access works
        log.info("Logged in as %s", who)
        return session, who
    except (ClientError, BotoCoreError) as e:
        log.exception("Login failed")
        raise AppError(f"Access validation failed: {friendly_error(e)}")


def friendly_error(e):
    """Turn AWS errors into short readable messages (no stack traces)."""
    if isinstance(e, ClientError):
        code = e.response["Error"]["Code"]
        msg = e.response["Error"].get("Message", "")
        if code in ("AccessDenied", "AccessDeniedException", "UnauthorizedOperation", "403", "AuthFailure"):
            return f"Access denied. AWS said: {msg}"
        return f"AWS error {code}: {msg}"
    return str(e)


# =====================================================================
# 4. API DISCOVERY  (built from boto3's own API definitions - nothing hardcoded)
# =====================================================================
_botocore = botocore.session.get_session()
_op_cache = {}


def service_allowed(service):
    if service not in _botocore.get_available_services():
        raise ToolError(f"Unknown service '{service}'. Use list_aws_services to find the right name.")
    if ALLOWED_SERVICES and service not in ALLOWED_SERVICES:
        raise ToolError(f"Service '{service}' is not enabled. Enabled: {ALLOWED_SERVICES}")


def read_only_operations(service):
    """All read-only operations of a service, with their parameters and description."""
    if service not in _op_cache:
        model = _botocore.get_service_model(service)
        ops = {}
        for name in model.operation_names:
            snake = xform_name(name)                              # DescribeInstances -> describe_instances
            if snake.startswith(READ_PREFIXES) and (service, snake) not in BLOCKED_OPS:
                op = model.operation_model(name)
                members = list(op.input_shape.members) if op.input_shape else []
                required = list(op.input_shape.required_members) if op.input_shape else []
                doc = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", op.documentation or "")).strip()
                ops[snake] = {"required": required, "all_params": members[:20], "doc": doc[:160], "api_name": name}
        _op_cache[service] = ops
    return _op_cache[service]


_full_names = {}


def tool_list_services(query=""):
    """AI tool: find AWS service names (ec2, rds, lambda, ...) by name or full title."""
    if not _full_names:                                           # loaded once (~3 sec)
        for n in _botocore.get_available_services():
            _full_names[n] = _botocore.get_service_model(n).metadata.get("serviceFullName", "").lower()
    names = sorted(_full_names)
    if ALLOWED_SERVICES:
        names = [n for n in names if n in ALLOWED_SERVICES]
    tokens = [t.rstrip("s") for t in re.split(r"\W+", (query or "").lower()) if t]
    hits = [n for n in names if any(t in n or t in _full_names[n] for t in tokens)] if tokens else names
    return [f"{n} ({_full_names[n]})" for n in hits[:40]]


def tool_search_operations(service, query):
    """AI tool: find the best read-only API operations for a service by keyword."""
    service_allowed(service)
    tokens = [t.rstrip("s") for t in re.split(r"\W+", (query or "").lower()) if t]
    scored = []
    for name, info in read_only_operations(service).items():
        score = sum(3 for t in tokens if t in name) + sum(1 for t in tokens if t in info["doc"].lower())
        if score or not tokens:
            scored.append((score, name, info))
    scored.sort(key=lambda x: (-x[0], len(x[1])))              # best match first, shorter names first
    return [{"operation": n, "required_params": i["required"],
             "optional_params": [p for p in i["all_params"] if p not in i["required"]],
             "description": i["doc"]} for _, n, i in scored[:10]]


# =====================================================================
# 5. API EXECUTION  (safe, generic - runs whatever operation the AI chose)
# =====================================================================
def make_bedrock_client():
    """Bedrock client with automatic retries (handles throttling) and a longer timeout."""
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION,
                        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, read_timeout=120))


def coerce_params(service, operation, params):
    """The AI sometimes sends numbers/booleans/lists as text. Convert them to the type AWS expects."""
    api_name = read_only_operations(service)[operation]["api_name"]
    shape = _botocore.get_service_model(service).operation_model(api_name).input_shape
    if not shape:
        return params
    for k, v in list(params.items()):
        member = shape.members.get(k)
        if member is None or not isinstance(v, str):
            continue
        try:
            if member.type_name in ("integer", "long"):
                params[k] = int(v)
            elif member.type_name in ("float", "double"):
                params[k] = float(v)
            elif member.type_name == "boolean":
                params[k] = v.strip().lower() == "true"
            elif member.type_name in ("list", "structure", "map"):
                params[k] = json.loads(v)
        except ValueError:
            pass                                              # leave as is; AWS validation will explain
    return params


SENSITIVE_KEYS = {"UserData", "PasswordData", "MasterUserPassword", "Password",
                  "SecretString", "SecretAccessKey", "SessionToken"}


def redact(obj, parent=""):
    """Hide secrets in AWS responses (Lambda environment variables, EC2 user-data, passwords, ...)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in SENSITIVE_KEYS:
                out[k] = "***REDACTED***"
            elif parent == "Environment" and k == "Variables" and isinstance(v, dict):
                out[k] = {name: "***REDACTED***" for name in v}
            else:
                out[k] = redact(v, k)
        return out
    if isinstance(obj, list):
        return [redact(x, parent) for x in obj]
    return obj


def human_bytes(n):
    """Exact, readable size (1024-based, same as the S3 console)."""
    v, i = float(n), 0
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.2f} {units[i]}"


def enrich_response(service, operation, resp):
    """Generic post-processing so the AI never has to do maths or guess:
    CloudWatch datapoints -> add the latest value (with readable size), or a hint if empty."""
    if service == "cloudwatch" and operation == "get_metric_statistics":
        pts = sorted(resp.get("Datapoints", []), key=lambda d: d["Timestamp"])
        if not pts:
            resp["Note"] = ("No datapoints returned. S3 metrics exist only in the bucket's own region "
                            "(pass region = s3 get_bucket_location) and need a valid StorageType "
                            "(find them with cloudwatch list_metrics). BucketSizeBytes is reported once per day.")
        else:
            last = pts[-1]
            stat = next((k for k in ("Average", "Sum", "Maximum", "Minimum") if k in last), None)
            latest = {"Timestamp": str(last["Timestamp"]), "Value": last.get(stat), "Unit": last.get("Unit")}
            if last.get("Unit") == "Bytes" and stat:
                latest["HumanReadable"] = human_bytes(last[stat])
            resp["LatestDatapoint"] = latest
    return resp


def tool_call_operation(session, default_region, service, operation, params, region=None):
    service_allowed(service)
    if operation not in read_only_operations(service):            # read-only safety check
        raise ToolError(f"'{operation}' is not an allowed read-only operation for {service}. "
                        "Use search_aws_operations to get valid names.")
    if region and not re.fullmatch(r"[a-z]{2}(-[a-z]+)+-\d", region):
        raise ToolError(f"'{region}' is not a valid AWS region name (example: ap-south-1).")
    params = {k: v for k, v in (params or {}).items() if v not in (None, "")}  # never send Bucket=""
    params = coerce_params(service, operation, params)
    client = session.client(service, region_name=region or default_region)
    log.info("AWS service=%s operation=%s final params=%s", service, operation, params)

    if client.can_paginate(operation):                            # fetch all pages (up to MAX_ITEMS)
        resp = client.get_paginator(operation).paginate(
            PaginationConfig={"MaxItems": MAX_ITEMS}, **params).build_full_result()
    else:
        resp = getattr(client, operation)(**params)
    resp.pop("ResponseMetadata", None)
    resp = enrich_response(service, operation, resp)
    if service == "lambda" and isinstance(resp.get("Code"), dict):
        resp["Code"].pop("Location", None)                        # hide presigned code-download URL
    resp = redact(resp)                                           # hide secrets before AI / UI see them
    log.info("AWS API executed OK")
    return resp


def run_tool(name, args, session, region, capture=None):
    """Run one tool the AI asked for. Always returns (text, status) for the AI to read."""
    try:
        if name == "list_aws_services":
            out = tool_list_services(args.get("query", ""))
        elif name == "search_aws_operations":
            out = tool_search_operations(args.get("service", ""), args.get("query", ""))
        elif name == "call_aws_operation":
            raw = args.get("parameters_json") or "{}"
            try:
                params = raw if isinstance(raw, dict) else json.loads(raw)
            except json.JSONDecodeError:
                raise ToolError("parameters_json is not valid JSON.")
            out = tool_call_operation(session, region, args.get("service", ""),
                                      args.get("operation", ""), params, args.get("region"))
            if capture is not None:                                   # UI: keep FULL data for tables
                capture.append({"service": args.get("service", ""), "operation": args.get("operation", ""),
                                "params": params, "data": out})
        else:
            raise ToolError(f"Unknown tool '{name}'.")
        text = json.dumps(out, default=str, separators=(",", ":"))
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "...[TRUNCATED - tell the user the result was cut]"
        return text, "success"
    except ToolError as e:
        log.warning("Tool error: %s", e)
        return str(e), "error"
    except (ClientError, BotoCoreError) as e:                     # AWS errors go back to the AI to explain
        log.error("AWS API error: %s", e)
        return friendly_error(e), "error"
    except Exception as e:                                        # unexpected bug: tell the AI, don't crash
        log.exception("Unexpected tool error")
        return f"Unexpected error while running {name}: {e}", "error"


# =====================================================================
# 6. THE AI (Amazon Nova Lite on Bedrock)
# =====================================================================
def system_prompt():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""You are a read-only AWS inventory assistant. Current time: {now}.
How to work:
1. Find the API: call list_aws_services if unsure of the service name, then search_aws_operations, then call_aws_operation.
2. If the question is about ONE resource property (a bucket's size, type, versioning, encryption; an instance's details...)
   and the user did NOT name the resource, STOP and ask in plain text (for S3: "Enter S3 bucket name:").
   Do NOT list resources or loop through all of them to guess. Never guess values.
3. Only when the user says "all", "every" or "each" (e.g. "all buckets"): first list them, then call the detail API for each one.
4. Use only data returned by tools. NEVER estimate, calculate or invent prices/costs from your own knowledge.
   For ANY cost, billing, spend or invoice question you MUST call the Cost Explorer API (service ce) first. If nothing is found, say clearly that no resources were found.
5. If access is denied, state exactly which permission/action is missing.
6. Present results as clean readable text, for example:
   S3 Buckets
   ----------
   Bucket : name
   Created: 2026-01-15
   Never show raw JSON. Never output <thinking> content.
7. Keep wording brief (max ~15 lines) - the app shows the complete data as a table under your answer.
   But ALWAYS give exact values from the tools (names, sizes, counts, dates). Never round, estimate or omit numbers.
   If a tool result contains NextToken, the list is partial (first 100 items): say so.
Hints:
- Costs: ce get_cost_and_usage with TimePeriod {{"Start":"YYYY-MM-DD","End":"YYYY-MM-DD"}} (End is exclusive),
  Granularity MONTHLY, Metrics ["UnblendedCost"]; for per-service use GroupBy [{{"Type":"DIMENSION","Key":"SERVICE"}}].
  "This month" = first day of current month to tomorrow. Budgets: service budgets, operation describe_budgets (needs AccountId).
- Bucket size / storage used (needs a bucket name): (1) s3 get_bucket_location -> bucket region (empty = us-east-1).
  (2) cloudwatch list_metrics with region = bucket region, Namespace AWS/S3, MetricName BucketSizeBytes,
  Dimensions [{{"Name":"BucketName","Value":"<bucket>"}}] -> shows which StorageTypes the bucket has.
  (3) For EACH StorageType call cloudwatch get_metric_statistics with region = bucket region, Namespace AWS/S3,
  MetricName BucketSizeBytes, Dimensions BucketName + StorageType, Period 86400, Statistics ["Average"],
  StartTime = 7 days ago, EndTime = now (ISO strings). Use each result's LatestDatapoint.HumanReadable.
  (4) Report each StorageType and the total (add the byte values). If no StorageType is listed, the bucket is empty or has no metrics yet.
- Storage class / storage type / "bucket type" (needs a bucket name): s3 list_objects_v2 returns StorageClass per object for that ONE bucket (size is NOT storage class).
- Bucket versioning with no Status means versioning was never enabled."""


def tool_definitions():
    """Tells Nova which 3 generic tools it can use."""
    def spec(name, desc, props, required):
        return {"toolSpec": {"name": name, "description": desc, "inputSchema": {"json": {
            "type": "object", "properties": props, "required": required}}}}
    return {"toolChoice": {"auto": {}}, "tools": [
        spec("list_aws_services", "Find AWS service names, e.g. query='database' -> rds, dynamodb.",
             {"query": {"type": "string", "description": "keyword"}}, ["query"]),
        spec("search_aws_operations",
             "Find read-only API operations of a service by keyword, e.g. service=ec2 query='instances'.",
             {"service": {"type": "string", "description": "boto3 service name, e.g. ec2"},
              "query": {"type": "string", "description": "keywords, e.g. 'security groups'"}},
             ["service", "query"]),
        spec("call_aws_operation", "Run one read-only AWS operation found via search_aws_operations.",
             {"service": {"type": "string", "description": "boto3 service name"},
              "operation": {"type": "string", "description": "snake_case name, e.g. describe_instances"},
              "parameters_json": {"type": "string",
                                  "description": 'JSON object string in boto3 PascalCase, e.g. {"Bucket":"x"}. Use {} if none.'},
              "region": {"type": "string", "description": "optional region override"}},
             ["service", "operation"]),
    ]}


def trim_history(history, keep=40):
    """Keep the conversation short. Must start on a normal user message."""
    if len(history) <= keep:
        return
    del history[:len(history) - keep]
    while history and not (history[0]["role"] == "user" and "text" in history[0]["content"][0]):
        history.pop(0)


def ask_ai(question, history, bedrock, tools, session, region, capture=None, usage=None):
    """One user question: the AI can call tools repeatedly until it can answer."""
    log.info("User request: %s", question)
    trim_history(history)
    start = len(history)
    history.append({"role": "user", "content": [{"text": question}]})
    try:
        for _ in range(MAX_STEPS):
            resp = bedrock.converse(modelId=MODEL_ID, system=[{"text": system_prompt()}],
                                    messages=history, toolConfig=tools,
                                    inferenceConfig={"temperature": 0, "maxTokens": 3000})
            if usage is not None:                                     # count Nova tokens for the UI
                u = resp.get("usage", {})
                usage["input"] = usage.get("input", 0) + u.get("inputTokens", 0)
                usage["output"] = usage.get("output", 0) + u.get("outputTokens", 0)
                usage["calls"] = usage.get("calls", 0) + 1
                log.info("Nova tokens: input=%s output=%s", u.get("inputTokens"), u.get("outputTokens"))
            msg = resp["output"]["message"]
            history.append(msg)
            calls = [b["toolUse"] for b in msg["content"] if "toolUse" in b]

            if resp["stopReason"] != "tool_use" or not calls:      # AI is done -> final answer
                text = "".join(b.get("text", "") for b in msg["content"])
                text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.S).strip()
                log.info("Final response: %s", text)
                return text or "No answer produced."

            results = []                                          # AI wants tools -> run them
            for c in calls:
                log.info("Nova tool selection: %s %s", c["name"], c.get("input"))
                text, status = run_tool(c["name"], c.get("input", {}), session, region, capture)
                results.append({"toolResult": {"toolUseId": c["toolUseId"],
                                               "content": [{"text": text}], "status": status}})
            history.append({"role": "user", "content": results})
        return "I couldn't finish within the step limit. Please ask a more specific question."
    except Exception:
        del history[start:]                                       # keep history valid after a failure
        raise


# =====================================================================
# 7. MAIN PROGRAM  (login, then chat loop)
# =====================================================================
def main():
    print("AWS Inventory AI Tool (read-only)\n")
    region = input("AWS Region: ").strip()
    role_arn = input("IAM Role ARN (leave blank to use CLI credentials): ").strip()
    try:
        session, who = make_session(region, role_arn)
        bedrock = make_bedrock_client()
    except AppError as e:
        print(f"Error: {e}")
        return

    print(f"\nAccess validated as: {who}")
    print("Ask about your AWS resources (type 'exit' to quit).\n")
    tools, history = tool_definitions(), []
    while True:
        question = input("> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        try:
            print("\n" + ask_ai(question, history, bedrock, tools, session, region) + "\n")
        except (ClientError, BotoCoreError) as e:                 # e.g. Bedrock model/access problems
            log.exception("Bedrock error")
            print(f"\nError calling Bedrock: {friendly_error(e)}\n")
        except Exception:
            log.exception("Unexpected error")
            print("\nUnexpected error. See app.log for details.\n")


if __name__ == "__main__":
    main()
