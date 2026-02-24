"""
fallback.py — Deterministic v1.5-style pipeline.

Called automatically by agent.py if the agent fails.
Can also be run directly:
    python fallback.py [--dry-run]

This is a self-contained script: no Agent SDK dependency.
"""

import os
import sys
import re
import json
import imaplib
import email
import datetime
import argparse
import time
import feedparser
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
RESEND_API_KEY    = os.environ["RESEND_API_KEY"]
GMAIL_ADDRESS     = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASS    = os.environ.get("GMAIL_APP_PASS", "")
DIGEST_TO         = os.environ.get("DIGEST_TO", GMAIL_ADDRESS)
ANTHROPIC_MODEL   = "claude-haiku-4-5"

SOURCES = [
    # RSS sources
    ("samkuehnle",   "rss",  "https://www.samkuehnle.com/feed",                            10),
    ("techcrunch",   "rss",  "https://techcrunch.com/feed/",                               10),
    ("producthunt",  "rss",  "https://www.producthunt.com/feed",                           20),
    ("martech",      "rss",  "https://martech.org/feed/",                                  10),
    ("techinasia",   "rss",  "https://feeds.feedburner.com/techinasia",                    10),
    # IMAP sources
    ("tldr",         "imap", ("dan@tldrnewsletter.com",     "TLDR"),                       None),
    ("tldrmarketing","imap", ("dan@tldrnewsletter.com",     "TLDR Marketing"),              None),
    ("lenny",        "imap", ("lenny@lennysnewsletter.com", "Lenny"),                      None),
    ("elena",        "imap", ("elenaverna@substack.com",    "Elena's Growth Scoop"),        None),
    ("danhock",      "imap", ("danhock@substack.com",       "Dan Hock"),                   None),
    ("theskip",      "imap", ("theskip@substack.com",       "The Skip"),                   None),
    ("athletic",     "imap", ("TheAthletic@e1.theathletic.com", "The Briefing"),           None),
]

SECTION_LABELS = {
    "samkuehnle":    "Marketing: Sam Kuehnle's Blog",
    "techcrunch":    "Tech News: TechCrunch",
    "producthunt":   "New Products: Product Hunt",
    "martech":       "Martech: Martech.org",
    "techinasia":    "Asia Tech: Tech in Asia",
    "tldr":          "AI & Tech Headlines: TLDR",
    "tldrmarketing": "Marketing Headlines: TLDR Marketing",
    "lenny":         "Product & Growth: Lenny's Newsletter",
    "elena":         "Growth: Elena's Growth Scoop",
    "danhock":       "Essays: Dan Hock",
    "theskip":       "Leadership: The Skip",
    "athletic":      "Sports: The Athletic Briefing",
}

ICONS = {
    "samkuehnle":    "✍️",
    "techcrunch":    "💻",
    "producthunt":   "🚀",
    "martech":       "📊",
    "techinasia":    "🌏",
    "tldr":          "📰",
    "tldrmarketing": "📣",
    "lenny":         "💡",
    "elena":         "📈",
    "danhock":       "🖊️",
    "theskip":       "⏭️",
    "athletic":      "⚽",
}

# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_rss(url: str, limit: int) -> str:
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:limit]:
        summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:200]
        items.append(f"- {entry.get('title', '')}: {entry.get('link', '')}\n  {summary}")
    return "\n".join(items)


def fetch_imap(sender_kw: str, subject_kw: str) -> str:
    if not GMAIL_ADDRESS or not GMAIL_APP_PASS:
        return "[Gmail credentials not configured]"
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
        mail.select("inbox")
        for criteria in [f'FROM "{sender_kw}"', f'SUBJECT "{subject_kw}"']:
            _, data = mail.search(None, criteria)
            ids = data[0].split()
            if ids:
                _, msg_data = mail.fetch(ids[-1], "(RFC822)")
                msg = email.message_from_bytes(msg_data[0][1])
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        ct = part.get_content_type()
                        if ct == "text/plain":
                            body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            break
                        elif ct == "text/html" and not body:
                            html = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            body = BeautifulSoup(html, "html.parser").get_text(separator="\n")
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                mail.logout()
                return body[:6000]
        mail.logout()
        return ""
    except Exception as e:
        return f"[Email fetch failed: {e}]"


# ---------------------------------------------------------------------------
# LLM summarisation
# ---------------------------------------------------------------------------

SYSTEM = ("You are a concise, friendly assistant writing a personal morning digest. "
          "Write in plain English. No hype, no filler. Be direct and specific. "
          "Use bullet points. Do not exceed the requested length.")


def llm_call(user_prompt: str, max_tokens: int = 350) -> str:
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        return f"[Summary unavailable: {e}]"


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _apply_inline(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r'(?<!["\'])(https?://[^\s<>"\']+)', r'<a href="\1">\1</a>', s)
    return s


def md_to_html(text: str) -> str:
    lines = text.split("\n")
    html_lines = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"  <li>{_apply_inline(stripped[2:])}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if stripped:
                html_lines.append(f"<p>{_apply_inline(stripped)}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def build_html(sections: dict, fallback_note: bool = True) -> str:
    today = datetime.date.today().strftime("%A, %B %-d, %Y")
    body_style  = ("font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
                   "font-size: 15px; line-height: 1.6; color: #1f2937; "
                   "max-width: 640px; margin: 0 auto; padding: 24px 16px;")
    sec_style   = "margin-bottom: 32px;"
    hdr_style   = ("font-size: 13px; font-weight: 700; letter-spacing: 0.08em; "
                   "text-transform: uppercase; color: #6b7280; border-bottom: 1px solid #e5e7eb; "
                   "padding-bottom: 6px; margin-bottom: 12px;")

    section_blocks = ""
    for key, body_html in sections.items():
        label = SECTION_LABELS.get(key, key)
        icon  = ICONS.get(key, "•")
        section_blocks += f"""
        <div style="{sec_style}">
            <div style="{hdr_style}">{icon} {label}</div>
            {body_html}
        </div>"""

    fallback_banner = ""
    if fallback_note:
        fallback_banner = """
        <div style="background:#fef3c7; border:1px solid #f59e0b; border-radius:6px; padding:12px 16px; margin-bottom:24px; font-size:13px; color:#92400e;">
            Note: Today's digest was generated using the fallback pipeline (agent encountered an error).
        </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body>
  <div style="{body_style}">
    <h1 style="font-size:22px; font-weight:700; margin-bottom:4px;">Good morning, Abraham ☀️</h1>
    <p style="color:#6b7280; margin-top:0; margin-bottom:24px;">Your daily digest for {today}</p>
    {fallback_banner}
    {section_blocks}
    <p style="color:#9ca3af; font-size:12px; margin-top:40px; border-top:1px solid #e5e7eb; padding-top:16px;">
      Generated by Daily Digest v2 (fallback) · {today}
    </p>
  </div>
</body></html>"""


def send_email(subject: str, html: str) -> None:
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
        json={"from": "Daily Digest <onboarding@resend.dev>", "to": [DIGEST_TO],
              "subject": subject, "html": html},
        timeout=15,
    )
    resp.raise_for_status()
    print(f"  Fallback email sent — status {resp.status_code}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    today = datetime.date.today().strftime("%A, %B %-d, %Y")
    print(f"  Fallback pipeline running for {today}...")

    raw: dict[str, str] = {}
    for key, source_type, source_arg, limit in SOURCES:
        print(f"    Fetching {key}...")
        if source_type == "rss":
            raw[key] = fetch_rss(source_arg, limit)
        else:
            raw[key] = fetch_imap(*source_arg)
        print(f"    [{key}] {len(raw[key])} chars")

    summaries: dict[str, str] = {}
    prompts = {
        "samkuehnle":    f"Summarise the 2-3 most insightful marketing reflections or ideas. Include title and URL.\n\n{raw['samkuehnle'] or 'No content.'}",
        "techcrunch":    f"Pick 3-4 most notable AI/ML or tech product stories. Skip pure funding news. Include company and URL.\n\n{raw['techcrunch'] or 'No content.'}",
        "producthunt":   f"Pick top 5 most interesting new products, especially any relevant to marketing or AI. Name, what it does, URL.\n\n{raw['producthunt'] or 'No content.'}",
        "martech":       f"Summarise 3-4 most relevant martech or marketing automation stories. Include key takeaway and URL.\n\n{raw['martech'] or 'No content.'}",
        "techinasia":    f"Summarise 3-4 most relevant Southeast Asia or Singapore tech stories. Include company and URL.\n\n{raw['techinasia'] or 'No content.'}",
        "tldr":          f"Extract 4-5 most important AI/tech stories. One bullet, one sentence each.\n\n{raw['tldr'][:3000] or 'No email.'}",
        "tldrmarketing": f"Extract 4-5 most important marketing stories. One bullet, one sentence each.\n\n{raw['tldrmarketing'][:3000] or 'No email.'}",
        "lenny":         f"Summarise key ideas from this Lenny's Newsletter in 4-5 bullets.\n\n{raw['lenny'][:3000] or 'No email.'}",
        "elena":         f"Summarise the main growth or marketing insight in 3-4 bullets. Note if paywalled.\n\n{raw['elena'][:3000] or 'No email.'}",
        "danhock":       f"Summarise the main argument or idea from this essay in 3-4 bullets. Note if paywalled.\n\n{raw['danhock'][:3000] or 'No email.'}",
        "theskip":       f"Summarise the key leadership or career insight in 3-4 bullets. Note if paywalled.\n\n{raw['theskip'][:3000] or 'No email.'}",
        "athletic":      f"Summarise top 3-4 sports stories, prioritising Premier League and Formula 1. Include key result or development.\n\n{raw['athletic'][:3000] or 'No email.'}",
    }

    for key, prompt in prompts.items():
        print(f"    Summarising {key}...")
        summaries[key] = llm_call(prompt)
        time.sleep(3)

    sections = {k: md_to_html(v) for k, v in summaries.items()}
    html = build_html(sections, fallback_note=True)

    if dry_run:
        out = Path("/tmp/fallback-digest.html")
        out.write_text(html)
        print(f"  [DRY RUN] HTML saved to {out}")
    else:
        send_email(f"Your Daily Digest — {today}", html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
