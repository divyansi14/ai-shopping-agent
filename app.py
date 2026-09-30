"""AI shopping agent: Gemini plans, Tavily searches, Python compares."""
import json
import os
import re
import sqlite3
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from google import genai
from google.genai import types

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "shopping.db"
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TAVILY_URL = "https://api.tavily.com/search"
MAX_AGENT_STEPS = 8

# --------------------------------------------------------------------------- db

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


with db() as _c:
    _c.execute(
        """CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            answer TEXT,
            products_json TEXT,
            trace_json TEXT,
            created_at REAL NOT NULL
        )"""
    )

# ------------------------------------------------------------------------ tools


def tavily_search(query: str, max_results: int = 6) -> list[dict]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise HTTPException(500, "TAVILY_API_KEY not set")
    r = httpx.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        },
        timeout=45,
    )
    if r.status_code >= 400:
        raise HTTPException(502, f"Tavily {r.status_code}: {r.text[:200]}")
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("content", "")}
        for x in r.json().get("results", [])
    ]


PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full product name incl. brand and model"},
                    "price_inr": {"type": "integer", "description": "Price in INR, 0 if not stated"},
                    "key_specs": {"type": "array", "items": {"type": "string"},
                                  "description": "Short spec strings e.g. '16GB RAM', 'RTX 4060 8GB'"},
                    "source_url": {"type": "string"},
                    "source_title": {"type": "string"},
                },
                "required": ["name", "price_inr", "key_specs", "source_url", "source_title"],
            },
        }
    },
    "required": ["products"],
}


def llm_extract_products(client: genai.Client, snippets: list[dict]) -> list[dict]:
    """Second LLM pass: turn messy search text into structured product rows."""
    if not snippets:
        return []
    blob = "\n\n".join(
        f"URL: {s['url']}\nTITLE: {s['title']}\nTEXT: {s['content'][:1500]}" for s in snippets
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=(
            "Extract every concrete purchasable product mentioned below. "
            "One entry per distinct model. Prices are Indian Rupees; strip commas and symbols. "
            "Use 0 when no price is stated. source_url must be one of the URLs given.\n\n" + blob
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PRODUCT_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    try:
        return json.loads(resp.text).get("products", [])
    except (json.JSONDecodeError, TypeError):
        return []


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def dedupe(products: list[dict]) -> list[dict]:
    """Keep first occurrence per normalised name; prefer the row that has a price."""
    out: dict[str, dict] = {}
    for p in products:
        k = norm_name(p.get("name", ""))
        if not k:
            continue
        prev = out.get(k)
        if prev is None or (not prev.get("price_inr") and p.get("price_inr")):
            out[k] = p
    return list(out.values())


def score_product(p: dict, budget: int, must_have: list[str]) -> float:
    """Higher is better. Over-budget is heavily penalised but still ranked."""
    score = 0.0
    price = p.get("price_inr") or 0
    if price and budget:
        if price > budget:
            score -= 100 + 30 * (price - budget) / budget
        else:
            # under budget: spending more of it usually buys better hardware
            score += 50 + 20 * (price / budget)
    elif price:
        score += 30
    text = (p.get("name", "") + " " + " ".join(p.get("key_specs", []))).lower()
    score += 15 * sum(1 for k in must_have if k.lower() in text)
    return round(score, 2)


def compare(products: list[dict], budget: int, must_have: list[str]) -> list[dict]:
    ranked = []
    for p in dedupe(products):
        q = dict(p)
        q["score"] = score_product(p, budget, must_have)
        q["over_budget"] = bool(budget and (p.get("price_inr") or 0) > budget)
        ranked.append(q)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ------------------------------------------------------------------- agent loop

TOOL_DECLS = [
    {
        "name": "web_search",
        "description": "Search the public web for products, prices and reviews. Call it several times with different phrasings to widen coverage.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'best laptop for machine learning under 70000 India 2026'"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "extract_products",
        "description": "Parse every product collected by web_search so far into structured rows (name, price, specs, source link). Call after searching.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "compare_products",
        "description": "Rank the extracted products against the user's budget and required specs. Call after extract_products.",
        "parameters": {
            "type": "object",
            "properties": {
                "budget_inr": {"type": "integer", "description": "Maximum price in INR, 0 if the user gave none"},
                "must_have_specs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lowercase keywords the ideal product should match, e.g. ['rtx', '16gb', 'nvidia']",
                },
            },
            "required": ["budget_inr", "must_have_specs"],
        },
    },
]

SYSTEM = """You are a shopping research agent for Indian buyers (prices in INR).

Workflow, in order:
1. Read the request and decide the budget and the specs that actually matter for the stated use case.
2. Call web_search 2-4 times with different angles (buying guides, specific models, retailer listings).
3. Call extract_products once searching is done.
4. Call compare_products with the budget and the must-have spec keywords.
5. Write the final answer: top 3 picks, each with price, why it fits the use case, and its source link.
   Then one line on what to avoid. Be concrete, no filler. Use markdown.

If a search returns nothing useful, search again with different words before giving up."""


def run_agent(query: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    snippets: list[dict] = []
    extracted: list[dict] = []
    ranked: list[dict] = []
    trace: list[dict] = []

    def do_web_search(query: str):
        hits = tavily_search(query)
        snippets.extend(hits)
        return {"results": hits, "total_collected": len(snippets)}

    def do_extract_products():
        extracted.clear()
        extracted.extend(llm_extract_products(client, snippets))
        return {"products": extracted, "count": len(extracted)}

    def do_compare_products(budget_inr: int = 0, must_have_specs: list[str] | None = None):
        ranked.clear()
        ranked.extend(compare(extracted, budget_inr, must_have_specs or []))
        return {"ranked": ranked[:10]}

    impls = {
        "web_search": do_web_search,
        "extract_products": do_extract_products,
        "compare_products": do_compare_products,
    }

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[types.Tool(function_declarations=TOOL_DECLS)],
        # ponytail: low budget keeps planning sane without the 2min think-tax; raise if plans get dumb
        thinking_config=types.ThinkingConfig(thinking_budget=512),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=query)])]
    answer = ""

    for _ in range(MAX_AGENT_STEPS):
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
        content = resp.candidates[0].content
        contents.append(content)
        calls = [p.function_call for p in (content.parts or []) if p.function_call]
        if not calls:
            answer = resp.text or ""
            break
        replies = []
        for call in calls:
            args = dict(call.args or {})
            try:
                result = impls[call.name](**args)
                summary = {"products": len(result.get("products", []))} if "products" in result else \
                          {"results": len(result.get("results", []))} if "results" in result else \
                          {"ranked": len(result.get("ranked", []))}
            except Exception as exc:  # surface tool failure to the model, don't kill the run
                result, summary = {"error": str(exc)}, {"error": str(exc)[:120]}
            trace.append({"tool": call.name, "args": args, "summary": summary})
            replies.append(types.Part.from_function_response(name=call.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=replies))
    else:
        answer = "Agent hit the step limit before finishing. Partial results below."

    products = ranked or dedupe(extracted)
    return {"answer": answer, "products": products[:10], "trace": trace}


# --------------------------------------------------------------------------- api

app = FastAPI(title="AI Shopping Agent")


class SearchIn(BaseModel):
    query: str


@app.post("/api/search")
def search(body: SearchIn):
    q = body.query.strip()
    if not q:
        raise HTTPException(400, "query is empty")
    out = run_agent(q)
    with db() as c:
        cur = c.execute(
            "INSERT INTO searches (query, answer, products_json, trace_json, created_at) VALUES (?,?,?,?,?)",
            (q, out["answer"], json.dumps(out["products"]), json.dumps(out["trace"]), time.time()),
        )
    return {"id": cur.lastrowid, **out}


@app.get("/api/history")
def history():
    with db() as c:
        rows = c.execute(
            "SELECT id, query, created_at FROM searches ORDER BY id DESC LIMIT 25"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/history/{sid}")
def history_item(sid: int):
    with db() as c:
        row = c.execute("SELECT * FROM searches WHERE id=?", (sid,)).fetchone()
    if not row:
        raise HTTPException(404, "not found")
    return {
        "id": row["id"],
        "query": row["query"],
        "answer": row["answer"],
        "products": json.loads(row["products_json"] or "[]"),
        "trace": json.loads(row["trace_json"] or "[]"),
        "created_at": row["created_at"],
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")