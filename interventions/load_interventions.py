# load_interventions.py  ────────────────────────────────────────────────
import os, json, re, textwrap
from typing import Optional, Dict, List

import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "interventions-test.csv"
df = pd.read_csv(CSV_PATH).fillna("")
INTERVENTIONS: List[Dict] = df.to_dict("records")

# ╭──────────────────────────────────────────────────────────────────╮
# │  1)  ROUTER  → chooses which intervention to start              │
# ╰──────────────────────────────────────────────────────────────────╯
_client = AzureOpenAI(
    api_key        = os.getenv("OPENAI_API_KEY"),
    api_version    = os.getenv("OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("OPENAI_API_BASE"),
)
DEPLOY = os.getenv("AZURE_DEPLOYMENT_NAME")

_ROUTER_SYS = (
    "You are a routing assistant.  Pick the single most relevant "
    "Intervention name for the USER_TEXT or output 'none'. "
    "Respond ONLY with {\"choice\":\"<Name|none>\"}"
)
_BULLETS = "\n".join(
    f"- {row['name']}: {row['description'][:150]}…" for row in INTERVENTIONS
)

def pick_intervention(user_text: str) -> Optional[Dict]:
    prompt = f"USER_TEXT:\n{user_text}\n\nINTERVENTIONS:\n{_BULLETS}"
    resp = _client.chat.completions.create(
        model=DEPLOY,
        messages=[{"role": "system", "content": _ROUTER_SYS},
                  {"role": "user",   "content": prompt}],
        temperature=0, max_tokens=20
    ).choices[0].message.content
    try:
        choice = json.loads(re.search(r"\{.*}", resp).group(0))["choice"]
    except Exception:
        choice = "none"
    return next((r for r in INTERVENTIONS if r["name"] == choice), None)


# ╭──────────────────────────────────────────────────────────────────╮
# │  2)  REFEREE  → decides whether to close the intervention       │
# ╰──────────────────────────────────────────────────────────────────╯
_REFEREE_SYS = textwrap.dedent("""
    You are Conversation-Referee-GPT.
    You receive a SCORECARD (JSON) that summarises the current intervention.

    End the intervention if **either** of these is true:
      • assistant already wrapped-up (assistant.wrap_up ≥ 1)
        AND user accepted / appreciates OR asked to stop (user.accepts ≥ 1 OR user.bails ≥ 1)
      • turns > 9  (failsafe)

    Respond ONLY with {"decision":"close"}  or  {"decision":"continue"}.
""").strip()

def should_close_intervention(scorecard_json: str) -> bool:
    resp = _client.chat.completions.create(
        model=DEPLOY,
        messages=[
            {"role": "system", "content": _REFEREE_SYS},
            {"role": "user",   "content": scorecard_json}
        ],
        temperature=0, max_tokens=10
    ).choices[0].message.content
    return '"close"' in resp.lower()
