"""Send Feishu direct message notification."""
import argparse
import json
import os
import urllib.request

APP_ID = os.environ.get("FEISHU_APP_ID", "")
APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
OPEN_ID = os.environ.get("FEISHU_OPEN_ID", "")
BASE = "https://open.feishu.cn/open-apis"


def _get_token() -> str:
    url = f"{BASE}/auth/v3/tenant_access_token/internal"
    body = json.dumps({"app_id": APP_ID, "app_secret": APP_SECRET}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["tenant_access_token"]


def send(text: str) -> dict:
    token = _get_token()
    url = f"{BASE}/im/v1/messages?receive_id_type=open_id"
    body = json.dumps({
        "receive_id": OPEN_ID,
        "msg_type": "text",
        "content": json.dumps({"text": text}),
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Message to send")
    args = parser.parse_args()
    result = send(args.text)
    if result.get("code") == 0:
        print("sent OK")
    else:
        print(f"FAILED: {result}")
        raise SystemExit(1)
