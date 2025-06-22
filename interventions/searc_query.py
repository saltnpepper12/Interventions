# mem0_search.py
import sys, os, json
from mem0 import MemoryClient

# ----- config ---------------------------------------------------
API_KEY  = os.getenv("MEM0_API_KEY")             # export beforehand
USER_ID  = "trial1"                        # ← your target user
TOP_K    = 3                                     # how many hits back
# ----------------------------------------------------------------

if len(sys.argv) < 2:
    print("usage: python mem0_search.py \"Earliest money memory from childhood\"")
    sys.exit(1)

query = sys.argv[1]

client = MemoryClient(api_key="m0-rlx1MmNof0FAIljHn2J9Jt045kGWhMfUqJTYoJsB")

results = client.search(
    query=query,
    top_k=TOP_K,
    user_id=USER_ID,
    # optional metadata filters:
    # metadata_filters={"topic": "origin_story"}
)

print(f"\n🔍 Search: {query!r} (top {TOP_K})\n")
for i, hit in enumerate(results, start=1):
    print(f"{i}. {hit['memory']}")
    print("   score:", round(hit["score"], 3))
    print("   metadata:", json.dumps(hit.get("metadata", {}), indent=2))
    print()
