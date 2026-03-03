"""
UAI Weekly AI Digest Generator
Brought To You By The Foundry

Uses Google Gemini API with Google Search grounding to generate
a comprehensive weekly AI news digest across 8 categories.
Outputs formatted Markdown.

Free tier: 15 RPM, 1M tokens/day on gemini-2.0-flash
"""

import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import requests
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

# ── Config ──────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_ID = "models/gemini-2.5-flash-lite"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "digests")
RESOLVE_CANONICAL = os.environ.get("RESOLVE_CANONICAL_URLS", "true").lower() in ("1", "true", "yes")

# Paywalled sources to exclude
PAYWALLED = [
    "The Information", "MIT Technology Review", "Financial Times",
    "Wall Street Journal", "Bloomberg", "New York Times",
    "The Atlantic", "Wired", "The Economist"
]

PAYWALL_INSTRUCTION = (
    "Do NOT cite or summarize content from these paywalled publications: "
    + ", ".join(PAYWALLED)
    + ". Only use freely accessible sources where the full article can be read without a subscription."
)

# ── Categories ──────────────────────────────────────────────────────
CATEGORIES = [
    {
        "id": "top_stories",
        "label": "Top Stories",
        "emoji": "⚡",
        "prompt": "What are the biggest AI news stories from the past 14 days? Major announcements, launches, controversies, or breakthroughs that had significant industry impact.",
    },
    {
        "id": "models",
        "label": "Models & Benchmarks",
        "emoji": "🧠",
        "prompt": "Were any new AI models released or benchmarked in the past 14 days? Include open-source and commercial models. Cover performance comparisons, architecture innovations, or notable fine-tunes.",
    },
    {
        "id": "research",
        "label": "Research Spotlight",
        "emoji": "📄",
        "prompt": "What notable AI or machine learning research papers were published or went viral in the past 14 days? Focus on arXiv preprints, conference acceptances, or papers getting significant community attention.",
    },
    {
        "id": "industry",
        "label": "Industry & Funding",
        "emoji": "💰",
        "prompt": "What AI startup funding rounds, acquisitions, partnerships, or major business moves happened in the past 14 days? Include valuations and deal sizes where available.",
    },
    {
        "id": "policy",
        "label": "Policy & Regulation",
        "emoji": "⚖️",
        "prompt": "What happened in AI regulation, policy, or government action in the past 14 days? Include executive orders, legislation, international agreements, and regulatory body announcements.",
    },
    {
        "id": "opensource",
        "label": "Open Source & Tools",
        "emoji": "🔧",
        "prompt": "What's trending in open-source AI the past two weeks? New tools, libraries, frameworks, Hugging Face models, GitHub repos, or developer tools released in the past 14 days.",
    },
    {
        "id": "safety",
        "label": "Safety & Alignment",
        "emoji": "🛡️",
        "prompt": "Any developments in AI safety, alignment, interpretability, or AI ethics in the past 14 days? Include research, organizational announcements, incidents, or public discourse.",
    },
    {
        "id": "products",
        "label": "Product Updates",
        "emoji": "🚀",
        "prompt": "What AI product updates, new features, integrations, or consumer-facing AI launches were announced in the past 14 days? Focus on tools people can actually use.",
    },
]

SYSTEM_PROMPT = f"""You are a research assistant generating a weekly AI news digest for a university AI club called UAI.
Your audience is NOT all technical — write so anyone can follow.

RULES:
1. Return ONLY valid JSON. No markdown, no backticks, no preamble, no explanation.
2. {PAYWALL_INSTRUCTION}
3. Only include items from the past 14 days.
4. Each summary should be 2-3 sentences in plain language. Avoid jargon — if you must use a technical term, briefly explain it.
5. Be specific. Include names, numbers, dates.
6. CRITICAL — Source URL accuracy:
   - source_url MUST be the EXACT URL of the article you summarized.
   - Use the CANONICAL or PRIMARY URL (where the content originated), not syndicated copies.
   - If the same article appears on multiple sites, cite the original publisher (e.g. blog.domain.com over syndication partners).
   - source_name should match the site/publication at that URL.
   - Verify each URL leads to the article you described — do not mix up similar articles.

Return this exact JSON structure:
{{"items": [{{"title": "Short headline", "summary": "2-3 sentence accessible summary", "source_name": "Publication name", "source_url": "Full URL", "date": "YYYY-MM-DD"}}]}}

Return up to 5 items ranked by significance. If nothing notable happened, return {{"items": []}}."""


# ── URL correction from grounding ────────────────────────────────────
def _domain(url):
    """Extract domain from URL for comparison."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path
        return host.lower().replace("www.", "")
    except Exception:
        return ""


def _normalize_for_match(s):
    """Lowercase, strip, keep alphanumeric and spaces for fuzzy matching."""
    return "".join(c for c in s.lower() if c.isalnum() or c.isspace())


def _apply_grounding_support_urls(items, full_text, grounding):
    """
    Use grounding_supports to map each digest item to the actual URLs the model
    selected from search results. The API exposes which grounding chunks support
    which segments of the response — we use that instead of the LLM-generated URLs.
    """
    if not grounding or not items:
        return items

    try:
        chunks = getattr(grounding, "grounding_chunks", None) or []
        supports = getattr(grounding, "grounding_supports", None) or []
    except Exception:
        return items

    if not chunks or not supports:
        return items

    # Build chunk URL list (only web chunks have URLs)
    chunk_urls = []
    for chunk in chunks:
        url = ""
        if hasattr(chunk, "web") and chunk.web:
            url = getattr(chunk.web, "uri", "") or ""
        chunk_urls.append(url)

    # Search from this position so duplicate titles map to correct items in order
    search_start = 0

    for item in items:
        title = item.get("title", "")
        if not title:
            continue

        # Find where this title appears (advance search_start so duplicates map correctly)
        # Try literal first; fallback to JSON-encoded form for titles with " or \
        search_str = title
        pos = full_text.find(search_str, search_start)
        if pos < 0 and ('"' in title or '\\' in title):
            try:
                search_str = json.dumps(title)[1:-1]  # Inner content without outer quotes
                pos = full_text.find(search_str, search_start)
            except Exception:
                pass
        if pos < 0:
            continue
        search_start = pos + len(search_str)

        item_start, item_end = pos, pos + len(search_str)

        # Find which grounding support segment overlaps with this position
        best_url = None
        best_overlap = 0

        for support in supports:
            segment = getattr(support, "segment", None)
            if not segment:
                continue
            start = getattr(segment, "start_index", None) or getattr(segment, "startIndex", 0) or 0
            end = getattr(segment, "end_index", None) or getattr(segment, "endIndex", 0) or 0

            # Check overlap (segment supports this part of the response)
            overlap_start = max(item_start, start)
            overlap_end = min(item_end, end)
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    indices = getattr(support, "grounding_chunk_indices", None) or getattr(support, "groundingChunkIndices", []) or []
                    if indices:
                        chunk_idx = indices[0]
                        if 0 <= chunk_idx < len(chunk_urls) and chunk_urls[chunk_idx]:
                            best_overlap = overlap
                            best_url = chunk_urls[chunk_idx]

        if best_url:
            old_url = item.get("source_url", "")
            if best_url != old_url:
                item["source_url"] = best_url
                print(f"    URL from grounding: {_domain(old_url)} -> {_domain(best_url)}")

    return items


def _correct_urls_from_grounding(items, grounding_sources):
    """
    Fallback: when grounding_supports is unavailable, use fuzzy matching against
    grounding_chunks. Less reliable than segment-based mapping.
    """
    if not grounding_sources:
        return items

    for item in items:
        item_title = _normalize_for_match(item.get("title", ""))
        item_source = _normalize_for_match(item.get("source_name", ""))
        item_url = item.get("source_url", "")

        best_match = None
        best_score = 0

        for g in grounding_sources:
            g_title = _normalize_for_match(g.get("title", ""))
            g_url = g.get("url", "")

            if not g_url or not g_title:
                continue

            score = 0
            title_words = set(w for w in item_title.split() if len(w) > 2)
            g_words = set(g_title.split())
            overlap = len(title_words & g_words) / max(len(title_words), 1)
            score += overlap * 2

            if item_source and item_source in g_title:
                score += 1
            key_terms = {"ai", "model", "2026", "2025", "march", "february", "release"}
            shared = (title_words | set(item_source.split())) & g_words & key_terms
            score += len(shared) * 0.3

            if score > best_score and score >= 0.5:
                best_score = score
                best_match = g_url

        if best_match and best_match != item_url and best_score >= 1.0:
            if _domain(item_url) != _domain(best_match):
                item["source_url"] = best_match
                print(f"    URL corrected (fallback): {_domain(item_url)} -> {_domain(best_match)}")

    return items


# ── Canonical URL extraction ────────────────────────────────────────
CANONICAL_RE = re.compile(
    r'<link[^>]*\srel\s*=\s*["\']canonical["\'][^>]*\shref\s*=\s*["\']([^"\']+)["\']',
    re.I
)
CANONICAL_RE_ALT = re.compile(
    r'<link[^>]*\shref\s*=\s*["\']([^"\']+)["\'][^>]*\srel\s*=\s*["\']canonical["\']',
    re.I
)


def _get_canonical_url(url, timeout=3):
    """
    Fetch page and extract canonical URL if present.
    Many syndicated pages have <link rel="canonical" href="..."> pointing to the original.
    """
    if not url or not url.startswith(("http://", "https://")):
        return url
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; UAI-Digest/1.0)"},
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return url
        html = resp.text[:100000]  # Limit parse size
        m = CANONICAL_RE.search(html) or CANONICAL_RE_ALT.search(html)
        if m:
            canonical = m.group(1).strip()
            if canonical:
                canonical = urljoin(url, canonical)
                if canonical != url:
                    return canonical
    except Exception:
        pass
    return url


def _resolve_canonical_urls(items, max_workers=5):
    """Resolve canonical URLs for all items in parallel."""
    def resolve_one(item):
        url = item.get("source_url", "")
        if url:
            canonical = _get_canonical_url(url)
            if canonical != url:
                item["source_url"] = canonical
                print(f"    Canonical: {_domain(url)} -> {_domain(canonical)}")
                return True
        return False

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(resolve_one, items))


# ── Gemini Client ───────────────────────────────────────────────────
def init_client():
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return genai.Client(api_key=GEMINI_API_KEY)


def fetch_category(client, category):
    """Call Gemini with Google Search grounding for one category."""
    today = datetime.now().strftime("%Y-%m-%d")
    google_search_tool = Tool(google_search=GoogleSearch())

    user_prompt = f"Today's date is {today}. Search the web and answer: {category['prompt']}"

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_prompt,
            config=GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[google_search_tool],
                response_modalities=["TEXT"],
                temperature=0.3,
            ),
        )

        if not response.candidates:
            print(f"  [{category['id']}] No candidates in response")
            return {"id": category["id"], "items": [], "sources": []}

        candidate = response.candidates[0]
        text = ""
        for part in (candidate.content.parts or []):
            if getattr(part, "text", None):
                text += part.text

        if not text.strip():
            print(f"  [{category['id']}] No text in response")
            return {"id": category["id"], "items": [], "sources": []}

        # Extract grounding metadata
        sources = []
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding and hasattr(grounding, "grounding_chunks"):
            for chunk in (grounding.grounding_chunks or []):
                if hasattr(chunk, "web") and chunk.web:
                    sources.append({
                        "title": getattr(chunk.web, "title", ""),
                        "url": getattr(chunk.web, "uri", ""),
                    })

        # Parse JSON from response
        clean = text.replace("```json", "").replace("```", "").strip()
        # Find JSON object
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(clean[start:end])
            items = parsed.get("items", [])

            # Use grounding_supports to get the actual URLs the model selected
            if grounding and items:
                supports = getattr(grounding, "grounding_supports", None) or []
                if supports:
                    items = _apply_grounding_support_urls(items, text, grounding)
                elif sources:
                    items = _correct_urls_from_grounding(items, sources)

            print(f"  [{category['id']}] Found {len(items)} items")
            return {"id": category["id"], "items": items, "sources": sources}
        else:
            print(f"  [{category['id']}] No JSON found in response")
            return {"id": category["id"], "items": [], "sources": sources}

    except Exception as e:
        print(f"  [{category['id']}] Error: {e}")
        return {"id": category["id"], "items": [], "sources": []}


# ── Markdown Formatter ──────────────────────────────────────────────
def format_digest(results):
    now = datetime.now()
    two_weeks_ago = now - timedelta(days=13)

    fmt = lambda d: d.strftime("%b %d")
    date_range = f"{fmt(two_weeks_ago)} – {fmt(now)}, {now.year}"

    lines = []
    lines.append("# UAI WEEKLY AI DIGEST")
    lines.append("### Brought To You By The Foundry")
    lines.append(f"**{date_range}**")
    lines.append("")
    lines.append("---")
    lines.append("")

    all_sources = []
    total_items = 0

    for cat in CATEGORIES:
        cat_data = results.get(cat["id"], {})
        items = cat_data.get("items", [])
        if not items:
            continue

        total_items += len(items)
        lines.append(f"## {cat['emoji']} {cat['label']}")
        lines.append("")

        for i, item in enumerate(items, 1):
            title = item.get("title", "Untitled")
            summary = item.get("summary", "")
            source_name = item.get("source_name", "Source")
            source_url = item.get("source_url", "")
            date = item.get("date", "")

            lines.append(f"**{i}. {title}**")
            lines.append(f"{summary}")
            if source_url:
                lines.append(f"[{source_name}]({source_url}){' · ' + date if date else ''}")
                all_sources.append({"name": source_name, "url": source_url})
            lines.append("")

        lines.append("---")
        lines.append("")

    # Sources footer
    if all_sources:
        lines.append("## 📚 All Sources Cited")
        lines.append("")
        seen = set()
        for i, s in enumerate(all_sources, 1):
            if s["url"] not in seen:
                lines.append(f"{i}. [{s['name']}]({s['url']})")
                seen.add(s["url"])
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated {now.strftime('%B %d, %Y at %I:%M %p')} · UAI × The Foundry*")
    lines.append(f"*{total_items} items across {sum(1 for c in CATEGORIES if results.get(c['id'], {}).get('items'))} categories*")

    return "\n".join(lines)


# ── Plain Text (for Discord) ───────────────────────────────────────
def format_discord(results):
    """Format a shorter version for Discord webhooks (2000 char limit per message)."""
    now = datetime.now()
    two_weeks_ago = now - timedelta(days=13)
    fmt = lambda d: d.strftime("%b %d")
    date_range = f"{fmt(two_weeks_ago)} – {fmt(now)}, {now.year}"

    lines = []
    lines.append(f"# UAI WEEKLY AI DIGEST")
    lines.append(f"### Brought To You By The Foundry · {date_range}")
    lines.append("")

    for cat in CATEGORIES:
        cat_data = results.get(cat["id"], {})
        items = cat_data.get("items", [])
        if not items:
            continue

        lines.append(f"## {cat['emoji']} {cat['label']}")
        for item in items[:3]:  # Limit to top 3 per category for Discord
            title = item.get("title", "")
            source_url = item.get("source_url", "")
            lines.append(f"- **{title}** — {item.get('summary', '')[:120]}... [Link]({source_url})")
        lines.append("")

    return "\n".join(lines)


def send_discord(content):
    """Send digest to Discord via webhook. Splits into multiple messages if needed."""
    if not DISCORD_WEBHOOK_URL:
        print("No DISCORD_WEBHOOK_URL set, skipping Discord post")
        return

    # Discord has 2000 char limit per message
    chunks = []
    current = ""
    for line in content.split("\n"):
        if len(current) + len(line) + 1 > 1900:
            chunks.append(current)
            current = line + "\n"
        else:
            current += line + "\n"
    if current:
        chunks.append(current)

    for i, chunk in enumerate(chunks):
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": chunk},
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code in (200, 204):
            print(f"  Discord message {i+1}/{len(chunks)} sent")
        else:
            print(f"  Discord error: {resp.status_code} {resp.text[:200]}")
        time.sleep(1)  # Rate limit


# ── Main ────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("UAI WEEKLY AI DIGEST GENERATOR")
    print("Brought To You By The Foundry")
    print("=" * 50)
    print()

    client = init_client()
    results = {}

    for i, cat in enumerate(CATEGORIES):
        print(f"[{i+1}/{len(CATEGORIES)}] Scanning: {cat['label']}...")
        result = fetch_category(client, cat)
        results[cat["id"]] = result

        # Rate limit: 5 req/min — after first 5, wait 60s before continuing
        if i < len(CATEGORIES) - 1:
            if (i + 1) % 5 == 0:
                print("  Rate limit: waiting 60s...")
                time.sleep(60)
            else:
                time.sleep(5)

    # Resolve canonical URLs (fixes syndication — e.g. female-entrepreneurs.com -> blog.mean.ceo)
    if RESOLVE_CANONICAL:
        all_items = []
        for cat_data in results.values():
            all_items.extend(cat_data.get("items", []))
        if all_items:
            print("\nResolving canonical URLs...")
            _resolve_canonical_urls(all_items)

    # Generate markdown
    digest_md = format_digest(results)

    # Save to file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(OUTPUT_DIR, f"digest-{date_str}.md")
    with open(filepath, "w") as f:
        f.write(digest_md)
    print(f"\nDigest saved to {filepath}")

    # Also save as latest.md for easy access
    latest_path = os.path.join(OUTPUT_DIR, "latest.md")
    with open(latest_path, "w") as f:
        f.write(digest_md)
    print(f"Latest copy saved to {latest_path}")

    # Post to Discord if configured
    if DISCORD_WEBHOOK_URL:
        print("\nPosting to Discord...")
        discord_content = format_discord(results)
        send_discord(discord_content)

    total = sum(len(results.get(c["id"], {}).get("items", [])) for c in CATEGORIES)
    active = sum(1 for c in CATEGORIES if results.get(c["id"], {}).get("items"))
    print(f"\nDone. {total} items across {active} categories.")


if __name__ == "__main__":
    main()
