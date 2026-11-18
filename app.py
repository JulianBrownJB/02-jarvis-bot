"""Jarvis AI Assistant — a polite, proactive AI assistant with system tools.

Runs locally in Pinokio (or anywhere Streamlit runs) on Windows. Two modes:

* Local Mode (default): works immediately with no API keys. A built-in
  responder handles tool commands and general questions, and always says
  clearly that it is running locally without a live model.
* Live Mode: with an OpenAI or Anthropic API key (entered in the sidebar and
  held only in st.session_state — never written to disk), Jarvis calls the
  live model. Any API failure is shown plainly and the turn falls back to
  Local Mode.

System tools: Summarize Clipboard, Draft Follow-Up Email, System Health
Check (CPU / RAM / disk via psutil).
"""

import os
import re
from datetime import datetime

import streamlit as st

try:
    import pyperclip
except Exception:  # pragma: no cover - pyperclip missing on exotic installs
    pyperclip = None

try:
    import psutil
except Exception:  # pragma: no cover - psutil missing on exotic installs
    psutil = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JARVIS_SYSTEM_PROMPT = (
    "You are Jarvis, a polite, proactive AI assistant in the manner of a "
    "discreet British butler. You are concise, precise, and organised. You "
    "address the user as 'sir' sparingly — at most once or twice per reply — "
    "and never overdo it. You anticipate the next useful step and offer it. "
    "You structure longer answers with short headings or bullet points. You "
    "never invent facts, prices, dates, or file contents you have not been "
    "given. When you do not know something, you say so plainly and suggest "
    "how to find out."
)

JARVIS_GREETING = (
    "Good day, sir. Jarvis at your service. I am ready — ask me anything, "
    "or use the Tools tab for the clipboard summariser, an email draft, or "
    "a system health check."
)

LOCAL_FOOTER = "\n\n*(Local Mode — answered on-device, without a live model.)*"

PAGE_TITLE = "Jarvis AI Assistant"

DARK_CSS = """
<style>
  .stApp { background-color: #0b0f14; color: #e6edf3; }
  section[data-testid="stSidebar"] { background-color: #10161d; border-right: 1px solid #1f2a36; }
  h1, h2, h3 { color: #f0f6fc !important; }
  .jarvis-header { padding: 0.4rem 0 0.2rem 0; }
  .jarvis-subtitle { color: #8b98a5; font-size: 0.95rem; margin-top: -0.4rem; }
  .metric-card {
    background: #121a22;
    border: 1px solid #26313d;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .metric-card .metric-label { color: #8b98a5; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; }
  .metric-card .metric-value { color: #f0f6fc; font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
  .metric-card .metric-sub { color: #8b98a5; font-size: 0.82rem; }
  .badge {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em;
  }
  .badge-local { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }
  .badge-live-openai { background: #23863633; color: #3fb950; border: 1px solid #238636; }
  .badge-live-anthropic { background: #a371f733; color: #d2a8ff; border: 1px solid #a371f7; }
  .badge-off { background: #6e768133; color: #8b98a5; border: 1px solid #6e7681; }
  .tool-card {
    background: #121a22; border: 1px solid #26313d; border-radius: 14px;
    padding: 18px; margin-bottom: 14px;
  }
  .tool-card h3 { margin-top: 0; }
  .stChatMessage { background: #121a22; border: 1px solid #26313d; border-radius: 12px; }
  .stTextInput input, .stTextArea textarea, .stSelectbox select {
    background-color: #0d1217 !important; color: #e6edf3 !important;
    border: 1px solid #26313d !important; border-radius: 8px !important;
  }
  .stButton button { border-radius: 8px; font-weight: 600; }
  div[data-testid="stMetric"] {
    background: #121a22; border: 1px solid #26313d; border-radius: 14px; padding: 12px 14px;
  }
  div[data-testid="stMetric"] label { color: #8b98a5 !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f0f6fc !important; }
</style>
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "messages": [{"role": "assistant", "content": JARVIS_GREETING}],
        "openai_key": "",
        "anthropic_key": "",
        "provider": "OpenAI",
        "local_mode": True,
        "clipboard_summary": "",
        "clipboard_raw": "",
        "email_draft": "",
        "email_subject": "",
        "health": None,
        "health_checked_at": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Helpers — formatting
# ---------------------------------------------------------------------------

def fmt_bytes(num_bytes: float) -> str:
    """Human-readable byte count (e.g. 16.00 GB)."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def mode_badge_html() -> str:
    """Status badge for the current operating mode."""
    if st.session_state["local_mode"]:
        return '<span class="badge badge-local">LOCAL MODE</span>'
    provider = st.session_state["provider"]
    key = st.session_state["openai_key"] if provider == "OpenAI" else st.session_state["anthropic_key"]
    if key:
        cls = "badge-live-openai" if provider == "OpenAI" else "badge-live-anthropic"
        return f'<span class="badge {cls}">LIVE &mdash; {provider.upper()}</span>'
    return '<span class="badge badge-off">LIVE SELECTED &mdash; NO KEY</span>'


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# System tools
# ---------------------------------------------------------------------------

def read_clipboard() -> tuple[str, str]:
    """Return (text, error). One of the two is always empty."""
    if pyperclip is None:
        return "", "pyperclip is not installed, so I cannot read the clipboard on this machine."
    try:
        text = pyperclip.paste()
    except Exception as exc:  # pyperclip raises PyperclipException on headless hosts
        return "", f"I could not read the clipboard: {exc}"
    if text is None or not str(text).strip():
        return "", "The clipboard is empty — copy some text first, then try again, sir."
    return str(text), ""


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    """Extractive summary: leading sentences + bullet-style key points."""
    clean = re.sub(r"\s+", " ", text).strip()
    words = clean.split()
    word_count = len(words)
    reading_minutes = max(1, round(word_count / 200))

    # Split into sentences, keeping the punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    lead = " ".join(sentences[:max_sentences]) if sentences else clean[:500]

    # Key points: lines that already look like bullets/numbered items.
    points = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^([-*•\u2022]|\d+[.)])\s+\S", stripped):
            points.append(re.sub(r"^([-*•\u2022]|\d+[.)])\s+", "", stripped))
    points = points[:6]

    # Also surface any ALL-CAPS headings as topics.
    topics = sorted({w.strip(" :") for w in re.findall(r"\b[A-Z][A-Z &]{3,}\b", text)})[:6]

    parts = [f"**Summary** — {word_count} words, about {reading_minutes} minute(s) to read."]
    parts.append("")
    parts.append(lead)
    if points:
        parts.append("")
        parts.append("**Key points**")
        parts.extend(f"- {p}" for p in points)
    if topics:
        parts.append("")
        parts.append("**Topics mentioned:** " + ", ".join(topics) + ".")
    return "\n".join(parts)


def summarize_clipboard() -> tuple[str, str]:
    """Run the clipboard summariser. Returns (summary, error)."""
    text, error = read_clipboard()
    if error:
        return "", error
    st.session_state["clipboard_raw"] = text
    summary = extractive_summary(text)
    st.session_state["clipboard_summary"] = summary
    return summary, ""


def build_email_draft(recipient: str, context: str, bullets: str) -> tuple[str, str]:
    """Turn bullet points into a polished executive email draft.

    Returns (subject, body). Raises ValueError on missing input.
    """
    recipient = recipient.strip()
    context = context.strip()
    if not recipient:
        raise ValueError("Please enter a recipient name first, sir.")
    points = [line.strip(" -•\t") for line in bullets.splitlines() if line.strip()]
    if not points:
        raise ValueError("Please enter at least one bullet point to build the email from, sir.")

    topic = points[0]
    topic_short = topic if len(topic) <= 70 else topic[:67].rstrip() + "..."
    subject = f"Follow-up: {topic_short}"

    lines = [f"Dear {recipient},", ""]
    if context:
        lines.append(f"Thank you for your time discussing {context}. "
                     "I wanted to follow up in writing so we are aligned on next steps.")
    else:
        lines.append("Thank you for your time. I wanted to follow up in writing "
                     "so we are aligned on next steps.")
    lines.append("")
    lines.append("To summarise:")
    for point in points:
        lines.append(f"• {point}")
    lines.append("")
    lines.append("Please let me know if anything above needs correcting, or if "
                 "there is anything further you need from me.")
    lines.append("")
    lines.append("Kind regards,")
    lines.append("Julian Brown")
    lines.append("Licensed Illinois Real Estate Broker — Realty of America / Team Emelin")
    return subject, "\n".join(lines)


def get_system_health() -> dict:
    """Live CPU / RAM / disk figures via psutil. Raises RuntimeError if unavailable."""
    if psutil is None:
        raise RuntimeError("psutil is not installed, so I cannot read system health on this machine.")
    cpu_pct = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk_path = "C:\\" if os.name == "nt" else "/"
    try:
        disk = psutil.disk_usage(disk_path)
    except Exception:
        disk_path = "/"
        disk = psutil.disk_usage(disk_path)
    return {
        "cpu_pct": cpu_pct,
        "cpu_count": psutil.cpu_count(logical=True),
        "mem_used": mem.used,
        "mem_total": mem.total,
        "mem_pct": mem.percent,
        "disk_free": disk.free,
        "disk_total": disk.total,
        "disk_pct": disk.percent,
        "disk_path": disk_path,
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Chat — local responder
# ---------------------------------------------------------------------------

def local_respond(prompt: str) -> str:
    """Built-in local responder. Understands tool commands by keyword and
    answers general questions with structured guidance. Always states it is
    running locally."""
    text = prompt.strip()
    lowered = text.lower()

    if re.search(r"\b(hello|hi|hey|good (morning|afternoon|evening))\b", lowered):
        return (
            "Good day, sir. All systems are ready. You can ask me to summarise "
            "your clipboard, draft a follow-up email, or run a system health "
            "check — or simply ask a question and I shall do my best with what "
            "I have on hand." + LOCAL_FOOTER
        )

    if "who are you" in lowered or "your name" in lowered:
        return (
            "I am Jarvis, your AI assistant, sir. In Local Mode I run entirely "
            "on this machine with no live model — I can still summarise your "
            "clipboard, draft emails, and check system health. Add an API key "
            "in the sidebar and switch Local Mode off for live answers." + LOCAL_FOOTER
        )

    if any(word in lowered for word in ("clipboard", "summarise", "summarize", "summary")):
        summary, error = summarize_clipboard()
        if error:
            return f"{error}" + LOCAL_FOOTER
        return (
            "Here is your clipboard summary, sir:\n\n" + summary + LOCAL_FOOTER
        )

    if any(word in lowered for word in ("draft", "email", "follow-up", "follow up")):
        return (
            "Of course, sir. Head to the **Tools** tab, fill in the recipient, "
            "context, and bullet points under **Draft Follow-Up Email**, and I "
            "shall turn them into a polished executive draft you can copy or "
            "download. If you paste the bullet points here instead, I shall "
            "draft it in this chat." + LOCAL_FOOTER
        )

    if any(word in lowered for word in ("health", "cpu", "ram", "memory", "disk", "system")):
        try:
            health = get_system_health()
        except Exception as exc:
            return f"I could not read system health: {exc}" + LOCAL_FOOTER
        st.session_state["health"] = health
        return (
            "System health, sir:\n\n"
            f"- **CPU:** {health['cpu_pct']:.1f}% across {health['cpu_count']} logical cores\n"
            f"- **RAM:** {fmt_bytes(health['mem_used'])} used of {fmt_bytes(health['mem_total'])} "
            f"({health['mem_pct']:.1f}%)\n"
            f"- **Disk ({health['disk_path']}):** {fmt_bytes(health['disk_free'])} free of "
            f"{fmt_bytes(health['disk_total'])} ({health['disk_pct']:.1f}% used)\n\n"
            f"Checked at {health['checked_at']}." + LOCAL_FOOTER
        )

    if "time" in lowered or "date" in lowered:
        now = datetime.now()
        return (
            f"It is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}, sir."
            + LOCAL_FOOTER
        )

    if "help" in lowered or "what can you do" in lowered:
        return (
            "At your service, sir. I can:\n\n"
            "- **Summarise your clipboard** — copy any text, then ask me.\n"
            "- **Draft a follow-up email** — give me bullet points and a recipient.\n"
            "- **Check system health** — CPU, RAM, and disk, live from this machine.\n"
            "- **Answer questions** — with structured guidance, though in Local "
            "Mode I have no live model, so for research or writing at length, "
            "add an API key in the sidebar and switch Local Mode off." + LOCAL_FOOTER
        )

    # General fallback: structured guidance, honest about local limits.
    return (
        "A sound question, sir. In Local Mode I have no live model, so I shall "
        "give you a practical framework rather than a researched answer:\n\n"
        "**1. Clarify the goal** — write down the exact outcome you want.\n"
        "**2. Break it into steps** — the smallest next action first.\n"
        "**3. Check what you already have** — notes, emails, and files often "
        "hold half the answer.\n"
        "**4. Ask me again with detail** — the more context you give, the "
        "better I can help.\n\n"
        "For a full researched answer, add an OpenAI or Anthropic key in the "
        "sidebar and switch Local Mode off." + LOCAL_FOOTER
    )


# ---------------------------------------------------------------------------
# Chat — live API
# ---------------------------------------------------------------------------

def live_respond(prompt: str) -> str:
    """Call the selected provider's live API. Raises on any failure so the
    caller can fall back to Local Mode."""
    provider = st.session_state["provider"]
    history = st.session_state["messages"][-11:-1]  # last 5 turns, excluding new prompt
    turns = [{"role": m["role"], "content": m["content"]} for m in history]
    turns.append({"role": "user", "content": prompt})

    if provider == "OpenAI":
        from openai import OpenAI

        client = OpenAI(api_key=st.session_state["openai_key"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": JARVIS_SYSTEM_PROMPT}] + turns,
            temperature=0.7,
            max_tokens=1200,
        )
        return (response.choices[0].message.content or "").strip()

    from anthropic import Anthropic

    client = Anthropic(api_key=st.session_state["anthropic_key"])
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1200,
        temperature=0.7,
        system=JARVIS_SYSTEM_PROMPT,
        messages=turns,
    )
    return "".join(block.text for block in response.content if hasattr(block, "text")).strip()


def respond(prompt: str) -> str:
    """Route a chat turn: live API when configured, otherwise Local Mode."""
    if st.session_state["local_mode"]:
        return local_respond(prompt)
    provider = st.session_state["provider"]
    key = st.session_state["openai_key"] if provider == "OpenAI" else st.session_state["anthropic_key"]
    if not key:
        return (
            f"No {provider} API key is set, sir — I have answered locally instead. "
            "Add a key in the sidebar, or switch Local Mode back on."
            + LOCAL_FOOTER
        )
    try:
        return live_respond(prompt)
    except Exception as exc:
        return (
            f"The live {provider} call failed: {exc}\n\n"
            "I have answered locally for this turn, sir. Please check the key "
            "and your connection, then try again."
            + "\n\n" + local_respond(prompt).replace(LOCAL_FOOTER, "")
        )


# ---------------------------------------------------------------------------
# UI — sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Jarvis Config")
        st.markdown(mode_badge_html(), unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("**API keys**")
        st.markdown(
            "Keys are held only in this session's memory (`st.session_state`) "
            "and are never written to disk.",
            unsafe_allow_html=True,
        )
        st.session_state["openai_key"] = st.text_input(
            "OpenAI API key", type="password", value=st.session_state["openai_key"],
            help="Used only for live OpenAI calls in this session.",
        )
        st.session_state["anthropic_key"] = st.text_input(
            "Anthropic API key", type="password", value=st.session_state["anthropic_key"],
            help="Used only for live Anthropic calls in this session.",
        )
        st.session_state["provider"] = st.selectbox(
            "Live provider", ["OpenAI", "Anthropic"], index=0 if st.session_state["provider"] == "OpenAI" else 1,
        )
        st.session_state["local_mode"] = st.toggle(
            "Local Mode (works with no keys)", value=st.session_state["local_mode"],
            help="On: built-in local responder. Off: live API calls with the key above.",
        )

        st.markdown("---")
        if st.button("Clear chat history", use_container_width=True):
            st.session_state["messages"] = [{"role": "assistant", "content": JARVIS_GREETING}]
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<div style='color:#8b98a5;font-size:0.82rem;'>"
            "Jarvis AI Assistant · runs locally in Pinokio on port 8502.<br>"
            "Local Mode needs no keys and no internet model call."
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# UI — chat tab
# ---------------------------------------------------------------------------

def render_chat_tab() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask Jarvis anything, sir…")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Jarvis is thinking…"):
                answer = respond(prompt)
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# UI — tools tab
# ---------------------------------------------------------------------------

def render_tools_tab() -> None:
    st.markdown("### System Action Tools")
    st.markdown(
        "<p class='jarvis-subtitle'>Three tools, all running on this machine. "
        "No data leaves your computer unless you choose a live provider in the sidebar.</p>",
        unsafe_allow_html=True,
    )

    # -- Tool 1: Summarize Clipboard -------------------------------------
    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
    st.markdown("#### Summarize Clipboard")
    st.markdown("Copies text from your desktop clipboard and summarises it.")
    if st.button("Summarize Clipboard", key="btn_clipboard", use_container_width=True):
        summary, error = summarize_clipboard()
        if error:
            st.warning(error)
        else:
            st.success("Clipboard summarised.")
    if st.session_state["clipboard_summary"]:
        st.markdown(st.session_state["clipboard_summary"])
        with st.expander("Original clipboard text"):
            st.code(st.session_state["clipboard_raw"][:4000])
    st.markdown("</div>", unsafe_allow_html=True)

    # -- Tool 2: Draft Follow-Up Email ------------------------------------
    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
    st.markdown("#### Draft Follow-Up Email")
    st.markdown("Turn bullet points into a polished executive email draft.")
    recipient = st.text_input("Recipient name", key="email_recipient",
                              placeholder="e.g. Catrina Swanigan")
    context = st.text_input("Meeting / context (optional)", key="email_context",
                            placeholder="e.g. our call about 6700 South Shore Drive")
    bullets = st.text_area(
        "Bullet points (one per line)", key="email_bullets", height=140,
        placeholder="• Inspection completed, unit as-is\n• Buyer asked about the ceiling leak\n• Next step: HOA engineer visit",
    )
    if st.button("Draft Email", key="btn_email", use_container_width=True):
        try:
            subject, body = build_email_draft(recipient, context, bullets)
            st.session_state["email_subject"] = subject
            st.session_state["email_draft"] = body
            st.success("Draft ready, sir.")
        except ValueError as exc:
            st.warning(str(exc))
    if st.session_state["email_draft"]:
        st.markdown(f"**Subject:** {st.session_state['email_subject']}")
        st.code(st.session_state["email_draft"], language="text")
        st.download_button(
            "Download draft (.txt)", data=st.session_state["email_draft"],
            file_name="follow-up-email.txt", mime="text/plain",
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # -- Tool 3: System Health Check --------------------------------------
    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
    st.markdown("#### System Health Check")
    st.markdown("Live CPU, RAM, and disk figures from this Windows machine.")
    if st.button("Run Health Check", key="btn_health", use_container_width=True):
        try:
            st.session_state["health"] = get_system_health()
            st.success("Health check complete.")
        except Exception as exc:
            st.error(f"Health check failed: {exc}")
    health = st.session_state["health"]
    if health:
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("CPU Usage", f"{health['cpu_pct']:.1f}%",
                        f"{health['cpu_count']} logical cores")
        with col2:
            metric_card("RAM Used", f"{health['mem_pct']:.1f}%",
                        f"{fmt_bytes(health['mem_used'])} of {fmt_bytes(health['mem_total'])}")
        with col3:
            metric_card("Disk Used", f"{health['disk_pct']:.1f}%",
                        f"{fmt_bytes(health['disk_free'])} free of {fmt_bytes(health['disk_total'])}"
                        f" on {health['disk_path']}")
        st.caption(f"Last checked: {health['checked_at']}")
    else:
        st.info("No health check run yet — press the button above, sir.")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🤖", layout="wide")
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    init_state()
    render_sidebar()

    st.markdown('<div class="jarvis-header">', unsafe_allow_html=True)
    st.markdown(f"# 🤖 {PAGE_TITLE}  {mode_badge_html()}", unsafe_allow_html=True)
    st.markdown(
        "<p class='jarvis-subtitle'>Your polite, proactive AI assistant — "
        "chat, tools, and system health, all in one place.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tab_chat, tab_tools = st.tabs(["💬 Chat", "🛠️ Tools"])
    with tab_chat:
        render_chat_tab()
    with tab_tools:
        render_tools_tab()


if __name__ == "__main__":
    main()
