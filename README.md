# ☁️ AWS Inventory AI

Ask questions about your AWS resources in plain English and get readable answers.
An LLM (Amazon Nova Lite on Bedrock) finds the right AWS API on its own, runs it **read-only**, and explains the result. Nothing is hardcoded per question.

> "Show my EC2 instances" · "Is versioning enabled on my-bucket?" · "List Lambda functions and runtimes" · "What was my cost last month by service?"

## What it does

- **Natural-language inventory** for any AWS service boto3 supports (EC2, S3, RDS, Lambda, EKS, IAM, ...). Restrict to some services if you like.
- **Automatic API discovery.** The AI searches boto3's own API definitions, picks an operation, fills the parameters, calls it, and can chain calls (e.g. list all buckets, then check each one).
- **Asks when info is missing** (bucket name, instance ID, ...) instead of guessing.
- **Short answer + full data.** A brief summary in chat, with the complete AWS response as a table you can download (CSV/JSON) and a copy button.
- **Token and cost tracking.** Nova input/output tokens and estimated cost per question and all-time, saved so a browser refresh doesn't clear it.
- **Bring your own access.** The user enters the region to inventory, then optionally a Role ARN and/or their own access keys.
- **Readable errors.** Access denied, missing permissions, bad names, and so on show as plain messages, not stack traces.

## How it works

```
Your question -> Nova Lite (Bedrock) -> searches boto3 API list -> picks operation
              -> app runs it (read-only) -> AWS response -> Nova summarizes + table shown
```

The AI has three generic tools: `list_aws_services`, `search_aws_operations`, `call_aws_operation`. Everything else is discovered at run time.

## Safety

- **Read-only in code:** only operations starting with `list_`, `describe_` or `get_` are allowed. Sensitive ones are blocked (e.g. `s3 get_object`, `ec2 get_password_data`, secret and parameter values).
- **Use a read-only IAM role/user.** The IAM permissions are your real safety layer.
- **Limits:** 100 items per API call, 20 model calls per question, capped result size sent to the model.
- **Password gate:** set `APP_PASSWORD`. Streamlit Community Cloud apps are public by default.
- **User keys** are held in the browser session only, never saved or logged.

## Credentials (priority order)

1. **Role ARN** (optional): if given, it is always used. The app assumes it with the user's keys if provided, otherwise the server's keys.
2. **User's own access keys** (optional).
3. **Server keys** from Secrets / CLI.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Web UI: login sidebar, chat, tables, token/cost panel |
| `app.py` | Core logic (AWS access, API discovery, AI loop). Also runs as a CLI: `python app.py` |
| `requirements.txt` | Dependencies |
| `.gitignore` | Keeps secrets, logs and usage data out of git |

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Uses your AWS CLI credentials unless the user enters a Role ARN or keys.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (private is fine).
2. In Streamlit Community Cloud, create an app with main file `streamlit_app.py`.
3. Add to **Settings -> Secrets** (GitHub secrets are not visible to Streamlit):

```toml
AWS_ACCESS_KEY_ID = "..."
AWS_SECRET_ACCESS_KEY = "..."
BEDROCK_REGION = "ap-south-1"
APP_PASSWORD = "choose-a-strong-password"
```

Optional settings:

| Key | Default | Meaning |
|---|---|---|
| `BEDROCK_MODEL_ID` | `apac.amazon.nova-lite-v1:0` | Model used |
| `BEDROCK_REGION` | `ap-south-1` | Region the model is **called** from (must suit the model ID; not the inventory region) |
| `ALLOWED_SERVICES` | all | e.g. `s3,ec2,rds,lambda` |
| `NOVA_INPUT_PRICE_PER_1M` / `NOVA_OUTPUT_PRICE_PER_1M` | `0.06` / `0.24` | USD per 1M tokens, for the cost estimate |
| `USAGE_S3_BUCKET` / `USAGE_S3_KEY` | none | Keep usage history across app reboots |

## IAM permissions

- Inventory: a read-only policy (e.g. AWS-managed `ReadOnlyAccess`), plus `ce:GetCostAndUsage` for cost questions.
- Model: `bedrock:InvokeModel`, and Nova Lite access enabled in the Bedrock console.
- Assuming roles: `sts:AssumeRole` for the identity doing it, and a matching trust policy on the role.
- Optional usage history in S3: `s3:GetObject` and `s3:PutObject` on that bucket.

## Notes

- Token cost is an estimate from Nova Lite list prices and excludes Cost Explorer API charges (about $0.01 per call).
- Without S3, usage history lives on the app's disk and resets if Streamlit reboots or redeploys the app.
- Logs are written to `app.log`.
