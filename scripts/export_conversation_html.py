"""
Native Antigravity Chat Interactive Conversation Exporter
Parses Antigravity transcript JSONL and renders a native-fidelity Antigravity IDE chat experience.
"""

import json
import html
import re
import os
import sys
from pathlib import Path
from datetime import datetime

def clean_user_prompt(raw: str) -> str:
    if not raw:
        return ""
    # Extract <USER_REQUEST>...</USER_REQUEST> if wrapped
    match = re.search(r'<USER_REQUEST>\s*([\s\S]*?)\s*</USER_REQUEST>', raw)
    if match:
        text = match.group(1)
    else:
        text = raw
        
    # Strip metadata, system instructions, and slash command prompts
    text = re.sub(r'<ADDITIONAL_METADATA>[\s\S]*?</ADDITIONAL_METADATA>', '', text)
    text = re.sub(r'<USER_SETTINGS_CHANGE>[\s\S]*?</USER_SETTINGS_CHANGE>', '', text)
    text = re.sub(r'<[A-Z_]+>[\s\S]*?</[A-Z_]+>', '', text)
    text = re.sub(r'<[A-Z_]+>', '', text)
    text = re.sub(r'</[A-Z_]+>', '', text)
    text = re.sub(r'/(grill-me|plan|boost|browser|learn)\s+is\s+a\s+\[Slash Command\][\s\S]*', '', text)
    return text.strip()

def format_timestamp(ts_str: str) -> str:
    if not ts_str:
        return ""
    try:
        # e.g. 2026-10-10T14:15:43Z -> Oct 10, 14:15
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %H:%M")
    except Exception:
        return ts_str[:16].replace('T', ' ')

def parse_markdown(text: str) -> str:
    if not text:
        return ""
    
    escaped = html.escape(text)
    
    # Code blocks ```lang ... ```
    def format_code(match):
        lang = match.group(1).strip() or "text"
        code = match.group(2)
        return (
            f'<div class="code-block">'
            f'  <div class="code-header">'
            f'    <span class="code-lang">{lang}</span>'
            f'    <button class="copy-btn" onclick="copyCode(this)">Copy</button>'
            f'  </div>'
            f'  <pre><code>{code}</code></pre>'
            f'</div>'
        )
    escaped = re.sub(r'```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```', format_code, escaped)
    
    # Inline code `code`
    escaped = re.sub(r'`([^`\n]+)`', r'<code class="inline-code">\1</code>', escaped)
    
    # GitHub Alert Callouts: > [!NOTE], > [!TIP], > [!IMPORTANT], > [!WARNING], > [!CAUTION]
    alert_patterns = [
        (r'&gt;\s*\[!NOTE\]\s*([\s\S]*?)(?=(?:\n\n|\Z))', 'note', 'Note'),
        (r'&gt;\s*\[!TIP\]\s*([\s\S]*?)(?=(?:\n\n|\Z))', 'tip', 'Tip'),
        (r'&gt;\s*\[!IMPORTANT\]\s*([\s\S]*?)(?=(?:\n\n|\Z))', 'important', 'Important'),
        (r'&gt;\s*\[!WARNING\]\s*([\s\S]*?)(?=(?:\n\n|\Z))', 'warning', 'Warning'),
        (r'&gt;\s*\[!CAUTION\]\s*([\s\S]*?)(?=(?:\n\n|\Z))', 'caution', 'Caution'),
    ]
    for pattern, alert_type, title in alert_patterns:
        def replace_alert(m, atype=alert_type, atitle=title):
            body = m.group(1).replace('&gt;', '').strip()
            return (
                f'<div class="alert alert-{atype}">'
                f'  <div class="alert-title">{atitle}</div>'
                f'  <div class="alert-body">{body}</div>'
                f'</div>'
            )
        escaped = re.sub(pattern, replace_alert, escaped)
    
    # Blockquotes &gt; text
    escaped = re.sub(r'^&gt;\s+(.*)$', r'<blockquote>\1</blockquote>', escaped, flags=re.MULTILINE)
    
    # Bold **text**
    escaped = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', escaped)
    
    # Italic *text*
    escaped = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', escaped)
    
    # Headers # H1, ## H2, ### H3, #### H4
    escaped = re.sub(r'^#### (.*)$', r'<h4>\1</h4>', escaped, flags=re.MULTILINE)
    escaped = re.sub(r'^### (.*)$', r'<h3>\1</h3>', escaped, flags=re.MULTILINE)
    escaped = re.sub(r'^## (.*)$', r'<h2>\1</h2>', escaped, flags=re.MULTILINE)
    escaped = re.sub(r'^# (.*)$', r'<h1>\1</h1>', escaped, flags=re.MULTILINE)
    
    # Lists
    escaped = re.sub(r'^\* (.*)$', r'<li class="bullet-item">\1</li>', escaped, flags=re.MULTILINE)
    escaped = re.sub(r'^- (.*)$', r'<li class="bullet-item">\1</li>', escaped, flags=re.MULTILINE)
    escaped = re.sub(r'^\d+\.\s+(.*)$', r'<li class="num-item">\1</li>', escaped, flags=re.MULTILINE)
    
    # Convert standard paragraphs / breaks outside code-block
    parts = escaped.split('<div class="code-block">')
    out_parts = []
    for i, p in enumerate(parts):
        if i == 0:
            out_parts.append(p.replace('\n\n', '<p class="chat-p"></p>').replace('\n', '<br>'))
        else:
            sub = p.split('</pre></div>')
            code_block_part = sub[0] + '</pre></div>'
            rest = sub[1] if len(sub) > 1 else ''
            rest_formatted = rest.replace('\n\n', '<p class="chat-p"></p>').replace('\n', '<br>')
            out_parts.append(code_block_part + rest_formatted)
            
    return ''.join(out_parts)

def format_diff(diff_text: str) -> str:
    lines = diff_text.strip().split('\n')
    formatted = []
    for line in lines:
        escaped_line = html.escape(line)
        if line.startswith('+') and not line.startswith('+++'):
            formatted.append(f'<div class="diff-line diff-add">{escaped_line}</div>')
        elif line.startswith('-') and not line.startswith('---'):
            formatted.append(f'<div class="diff-line diff-del">{escaped_line}</div>')
        elif line.startswith('@@'):
            formatted.append(f'<div class="diff-line diff-hunk">{escaped_line}</div>')
        else:
            formatted.append(f'<div class="diff-line diff-ctx">{escaped_line}</div>')
    return '\n'.join(formatted)

def load_and_structure_transcript(transcript_path: str):
    turns = []
    current_turn = None
    total_steps = 0
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            step = json.loads(line)
            total_steps += 1
            stype = step.get('type')
            
            if stype == 'USER_INPUT':
                if current_turn:
                    turns.append(current_turn)
                raw_user = step.get('content', '')
                clean_user = clean_user_prompt(raw_user)
                current_turn = {
                    'turn_id': len(turns) + 1,
                    'created_at': step.get('created_at', ''),
                    'time_formatted': format_timestamp(step.get('created_at', '')),
                    'user_prompt': clean_user,
                    'events': []  # list of {type: 'thought'|'tool'|'response', data: ...}
                }
            elif current_turn is not None:
                # 1. Thought
                thinking = step.get('thinking')
                if thinking and thinking.strip():
                    current_turn['events'].append({
                        'type': 'thought',
                        'step_index': step.get('step_index'),
                        'text': thinking.strip()
                    })
                
                # 2. Tool calls
                tool_calls = step.get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get('name', 'tool')
                        args = tc.get('args', {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                pass
                        current_turn['events'].append({
                            'type': 'tool',
                            'step_index': step.get('step_index'),
                            'name': name,
                            'args': args,
                            'result': ''
                        })
                
                # 3. Tool results (GENERIC)
                if stype == 'GENERIC':
                    content = step.get('content', '')
                    # Match with previous tool event without result
                    matched = False
                    for ev in reversed(current_turn['events']):
                        if ev['type'] == 'tool' and not ev.get('result'):
                            ev['result'] = content
                            matched = True
                            break
                    if not matched and content:
                        current_turn['events'].append({
                            'type': 'response',
                            'text': content
                        })
                        
                # 4. Model assistant text response
                if stype == 'PLANNER_RESPONSE':
                    content = step.get('content')
                    if content and content.strip():
                        current_turn['events'].append({
                            'type': 'response',
                            'text': content.strip()
                        })
                        
    if current_turn:
        turns.append(current_turn)
        
    return turns, total_steps

def build_native_html(turns, total_steps: int, conv_id: str, out_path: str):
    html_lines = []
    
    # Antigravity Blue Delta Icon SVG
    antigravity_logo_svg = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2L2 19.5H8.5L12 13L15.5 19.5H22L12 2Z" fill="url(#ag-grad)" />
  <path d="M12 7.5L8 15.5H16L12 7.5Z" fill="#131418" />
  <defs>
    <linearGradient id="ag-grad" x1="2" y1="2" x2="22" y2="20" gradientUnits="userSpaceOnUse">
      <stop stop-color="#388bfd" />
      <stop offset="1" stop-color="#1f6feb" />
    </linearGradient>
  </defs>
</svg>"""

    # Sparkle / Thinking icon SVG
    sparkle_svg = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
</svg>"""

    # Chevron SVG
    chevron_svg = """<svg class="chevron-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <polyline points="6 9 12 15 18 9"></polyline>
</svg>"""

    # Terminal icon SVG
    terminal_svg = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <polyline points="4 17 10 11 4 5"></polyline>
  <line x1="12" y1="19" x2="20" y2="19"></line>
</svg>"""

    # Code/Edit file icon SVG
    edit_file_svg = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
  <polyline points="14 2 14 8 20 8"></polyline>
  <line x1="16" y1="13" x2="8" y2="13"></line>
  <line x1="16" y1="17" x2="8" y2="17"></line>
</svg>"""

    # Question icon SVG
    question_svg = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="12" cy="12" r="10"></circle>
  <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
  <line x1="12" y1="17" x2="12.01" y2="17"></line>
</svg>"""

    html_lines.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Antigravity Chat &bull; Air Raid Alert Defense</title>
  <style>
    :root {{
      --bg-app: #0f1115;
      --bg-sidebar: #15171d;
      --bg-sidebar-hover: #1c1f27;
      --bg-sidebar-active: #222631;
      --bg-main: #0f1115;
      --bg-user-bubble: #232733;
      --bg-card: #161820;
      --bg-card-inner: #111217;
      --bg-code: #0b0c0f;
      --border-subtle: #252834;
      --border-focus: #388bfd;
      --text-main: #f0f6fc;
      --text-muted: #8b949e;
      --text-dim: #656d76;
      --accent-blue: #388bfd;
      --accent-purple: #a371f7;
      --accent-green: #3fb950;
      --accent-red: #f85149;
      --accent-orange: #d29922;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif;
      --font-mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    }}

    body.light {{
      --bg-app: #f6f8fa;
      --bg-sidebar: #ffffff;
      --bg-sidebar-hover: #f1f3f5;
      --bg-sidebar-active: #e9ecef;
      --bg-main: #f6f8fa;
      --bg-user-bubble: #e8edf4;
      --bg-card: #ffffff;
      --bg-card-inner: #f8f9fa;
      --bg-code: #f3f4f6;
      --border-subtle: #e1e4e8;
      --border-focus: #0969da;
      --text-main: #1f2328;
      --text-muted: #57606a;
      --text-dim: #8c959f;
      --accent-blue: #0969da;
      --accent-purple: #8250df;
      --accent-green: #1a7f37;
      --accent-red: #cf222e;
      --accent-orange: #9a6700;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}

    html {{
      height: 100%;
      overflow-x: hidden;
      overflow-y: auto;
      scroll-behavior: smooth;
    }}

    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-app);
      color: var(--text-main);
      min-height: 100%;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      overflow-y: auto;
      -webkit-font-smoothing: antialiased;
    }}

    /* Top Navigation Shell - Sticky Header */
    .app-header {{
      position: sticky;
      top: 0;
      height: 52px;
      background-color: var(--bg-sidebar);
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 1.25rem;
      z-index: 1000;
      user-select: none;
    }}

    .header-left {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}

    .logo-badge {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-weight: 600;
      font-size: 0.95rem;
      color: var(--text-main);
      text-decoration: none;
    }}

    .model-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      background: rgba(56, 139, 253, 0.12);
      color: var(--accent-blue);
      border: 1px solid rgba(56, 139, 253, 0.3);
      padding: 0.15rem 0.55rem;
      border-radius: 9999px;
      font-size: 0.72rem;
      font-weight: 500;
      font-family: var(--font-mono);
    }}

    .header-right {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }}

    .top-btn {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 6px;
      padding: 0.35rem 0.65rem;
      font-size: 0.78rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.15s ease;
      font-family: var(--font-sans);
    }}

    .top-btn:hover {{
      background: var(--bg-sidebar-hover);
      color: var(--text-main);
      border-color: var(--text-muted);
    }}

    .top-btn.active {{
      background: rgba(56, 139, 253, 0.15);
      color: var(--accent-blue);
      border-color: var(--accent-blue);
    }}

    /* Main Container: 2-column layout */
    .app-body {{
      display: flex;
      width: 100%;
      min-height: calc(100vh - 52px);
      position: relative;
    }}

    /* Left Sidebar: Sticky Navigator */
    .sidebar {{
      width: 320px;
      position: sticky;
      top: 52px;
      height: calc(100vh - 52px);
      background-color: var(--bg-sidebar);
      border-right: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      z-index: 900;
      overflow: hidden;
      transition: width 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .sidebar.collapsed {{
      width: 0;
      border-right: none;
      visibility: hidden;
    }}

    .sidebar-header {{
      padding: 0.85rem 1rem 0.5rem 1rem;
      border-bottom: 1px solid var(--border-subtle);
    }}

    .sidebar-title-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
    }}

    .search-box {{
      position: relative;
      margin-bottom: 0.4rem;
    }}

    .search-box input {{
      width: 100%;
      background: var(--bg-app);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.4rem 0.65rem 0.4rem 1.8rem;
      font-size: 0.8rem;
      color: var(--text-main);
      outline: none;
      font-family: var(--font-sans);
    }}

    .search-box input:focus {{
      border-color: var(--border-focus);
    }}

    .search-icon-pos {{
      position: absolute;
      left: 0.55rem;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      font-size: 0.75rem;
    }}

    .turns-list {{
      flex: 1;
      overflow-y: auto;
      padding: 0.5rem 0.6rem;
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }}

    .turn-nav-item {{
      padding: 0.6rem 0.75rem;
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      background: transparent;
      border: 1px solid transparent;
      transition: all 0.12s ease;
      user-select: none;
    }}

    .turn-nav-item:hover {{
      background: var(--bg-sidebar-hover);
    }}

    .turn-nav-item.active {{
      background: var(--bg-sidebar-active);
      border-color: rgba(56, 139, 253, 0.3);
    }}

    .turn-nav-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-mono);
    }}

    .turn-nav-num {{
      color: var(--accent-blue);
      font-weight: 600;
    }}

    .turn-nav-snippet {{
      font-size: 0.82rem;
      color: var(--text-main);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      line-height: 1.35;
    }}

    .turn-nav-meta {{
      display: flex;
      gap: 0.35rem;
      align-items: center;
      margin-top: 0.15rem;
    }}

    .tiny-badge {{
      font-size: 0.68rem;
      padding: 0.1rem 0.35rem;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}

    /* Main Chat Stream */
    .chat-view {{
      flex: 1;
      min-width: 0;
      background: var(--bg-main);
      position: relative;
    }}

    .chat-container {{
      max-width: 880px;
      margin: 0 auto;
      padding: 2rem 1.5rem 8rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 2.5rem;
    }}

    /* Turn Group */
    .turn-group {{
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      scroll-margin-top: 72px;
    }}

    .turn-divider {{
      display: flex;
      align-items: center;
      gap: 1rem;
      margin: 0.5rem 0;
      color: var(--text-dim);
      font-size: 0.72rem;
      font-family: var(--font-mono);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    .turn-divider::before, .turn-divider::after {{
      content: "";
      flex: 1;
      height: 1px;
      background: var(--border-subtle);
    }}

    /* User Message Bubble */
    .user-row {{
      display: flex;
      justify-content: flex-end;
      gap: 0.75rem;
    }}

    .user-bubble {{
      max-width: 85%;
      background: var(--bg-user-bubble);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 0.9rem 1.2rem;
      font-size: 0.95rem;
      line-height: 1.6;
      color: var(--text-main);
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      white-space: pre-wrap;
      word-break: break-word;
    }}

    .user-avatar {{
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: #374151;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
      font-weight: 600;
      color: #e5e7eb;
      flex-shrink: 0;
    }}

    /* Assistant Message Column */
    .assistant-row {{
      display: flex;
      gap: 1rem;
      align-items: flex-start;
    }}

    .assistant-avatar {{
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: #191c24;
      border: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      margin-top: 0.2rem;
    }}

    .assistant-content {{
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1.1rem;
      overflow: hidden;
    }}

    /* Thinking Process Accordion */
    .thinking-accordion {{
      background: rgba(163, 113, 247, 0.05);
      border: 1px solid rgba(163, 113, 247, 0.2);
      border-radius: 8px;
      overflow: hidden;
      transition: border-color 0.15s ease;
    }}

    .thinking-accordion summary {{
      padding: 0.5rem 0.85rem;
      font-size: 0.8rem;
      font-weight: 500;
      color: var(--accent-purple);
      cursor: pointer;
      user-select: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      list-style: none;
    }}

    .thinking-accordion summary::-webkit-details-marker {{ display: none; }}

    .thinking-summary-left {{
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }}

    .thinking-body {{
      padding: 0.85rem 1rem;
      font-size: 0.83rem;
      color: var(--text-muted);
      border-top: 1px solid rgba(163, 113, 247, 0.12);
      line-height: 1.6;
      white-space: pre-wrap;
      font-family: var(--font-sans);
      max-height: 480px;
      overflow-y: auto;
    }}

    /* Tool Call Card Widget */
    .tool-widget {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      overflow: hidden;
      margin: 0.35rem 0;
    }}

    .tool-widget summary {{
      padding: 0.55rem 0.85rem;
      font-size: 0.82rem;
      cursor: pointer;
      user-select: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      list-style: none;
      background: var(--bg-card-inner);
    }}

    .tool-widget summary::-webkit-details-marker {{ display: none; }}

    .tool-summary-left {{
      display: flex;
      align-items: center;
      gap: 0.55rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .tool-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      background: rgba(56, 139, 253, 0.15);
      color: var(--accent-blue);
      font-weight: 600;
    }}

    .tool-desc {{
      font-size: 0.8rem;
      color: var(--text-main);
      font-weight: 500;
    }}

    .tool-body {{
      border-top: 1px solid var(--border-subtle);
      padding: 0.75rem 1rem;
      background: var(--bg-code);
      font-family: var(--font-mono);
      font-size: 0.78rem;
      overflow-x: auto;
    }}

    .tool-section-label {{
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      margin-bottom: 0.4rem;
      font-weight: 600;
    }}

    /* Terminal Window inside run_command */
    .terminal-window {{
      background: #090a0d;
      border: 1px solid #1f232c;
      border-radius: 6px;
      overflow: hidden;
      margin-top: 0.4rem;
    }}

    .terminal-top {{
      background: #141720;
      padding: 0.35rem 0.65rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      border-bottom: 1px solid #1f232c;
    }}

    .terminal-dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
    }}

    .terminal-cmd {{
      color: var(--text-main);
      font-family: var(--font-mono);
      font-size: 0.78rem;
      margin-left: 0.5rem;
      word-break: break-all;
    }}

    .terminal-output {{
      padding: 0.75rem;
      color: #c9d1d9;
      font-size: 0.78rem;
      white-space: pre-wrap;
      max-height: 380px;
      overflow-y: auto;
      line-height: 1.5;
    }}

    /* Diff Lines */
    .diff-container {{
      background: #090a0d;
      border: 1px solid #1f232c;
      border-radius: 6px;
      padding: 0.5rem 0;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      line-height: 1.5;
      max-height: 400px;
      overflow-y: auto;
    }}

    .diff-line {{
      padding: 0.1rem 0.75rem;
      white-space: pre-wrap;
    }}

    .diff-add {{ background: rgba(46, 160, 67, 0.15); color: #3fb950; }}
    .diff-del {{ background: rgba(248, 81, 73, 0.15); color: #f85149; }}
    .diff-hunk {{ background: rgba(163, 113, 247, 0.1); color: #a371f7; }}
    .diff-ctx {{ color: #8b949e; }}

    /* Question Pill Card */
    .question-card {{
      background: var(--bg-card);
      border: 1px solid rgba(56, 139, 253, 0.25);
      border-radius: 8px;
      padding: 1rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}

    .question-title {{
      font-weight: 600;
      color: var(--text-main);
      font-size: 0.9rem;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }}

    .question-options {{
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }}

    .question-option {{
      padding: 0.55rem 0.85rem;
      border-radius: 6px;
      background: var(--bg-card-inner);
      border: 1px solid var(--border-subtle);
      font-size: 0.82rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .question-option.selected {{
      background: rgba(63, 185, 80, 0.12);
      border-color: rgba(63, 185, 80, 0.4);
      color: var(--text-main);
      font-weight: 500;
    }}

    .selected-badge {{
      margin-left: auto;
      font-size: 0.7rem;
      padding: 0.1rem 0.4rem;
      background: var(--accent-green);
      color: #fff;
      border-radius: 4px;
      font-family: var(--font-mono);
    }}

    /* Markdown prose in Assistant Text */
    .prose {{
      font-size: 0.95rem;
      line-height: 1.68;
      color: var(--text-main);
    }}

    .prose p.chat-p {{ margin: 0.8rem 0; }}
    .prose h1, .prose h2, .prose h3, .prose h4 {{
      margin: 1.25rem 0 0.5rem 0;
      color: var(--text-main);
      font-weight: 600;
    }}
    .prose h1 {{ font-size: 1.35rem; }}
    .prose h2 {{ font-size: 1.2rem; }}
    .prose h3 {{ font-size: 1.05rem; }}
    .prose h4 {{ font-size: 0.95rem; }}

    .prose ul, .prose ol {{
      margin: 0.6rem 0 0.6rem 1.4rem;
    }}

    .prose li {{ margin-bottom: 0.35rem; }}

    .inline-code {{
      background: rgba(110, 118, 129, 0.2);
      color: var(--text-main);
      font-family: var(--font-mono);
      font-size: 0.85em;
      padding: 0.15rem 0.35rem;
      border-radius: 4px;
    }}

    .code-block {{
      background: var(--bg-code);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      overflow: hidden;
      margin: 1rem 0;
    }}

    .code-header {{
      background: var(--bg-card-inner);
      padding: 0.35rem 0.85rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-subtle);
      font-size: 0.75rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}

    .copy-btn {{
      background: transparent;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 4px;
      padding: 0.15rem 0.45rem;
      font-size: 0.7rem;
      cursor: pointer;
      font-family: var(--font-sans);
    }}

    .copy-btn:hover {{
      color: var(--text-main);
      border-color: var(--text-muted);
    }}

    .code-block pre {{
      padding: 0.85rem 1rem;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 0.83rem;
      line-height: 1.5;
    }}

    /* GitHub Alert Styles */
    .alert {{
      border-left: 3px solid;
      border-radius: 0 6px 6px 0;
      padding: 0.75rem 1rem;
      margin: 0.85rem 0;
      font-size: 0.9rem;
    }}

    .alert-title {{
      font-weight: 600;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.3rem;
    }}

    .alert-note {{ background: rgba(56, 139, 253, 0.08); border-color: var(--accent-blue); }}
    .alert-note .alert-title {{ color: var(--accent-blue); }}

    .alert-tip {{ background: rgba(63, 185, 80, 0.08); border-color: var(--accent-green); }}
    .alert-tip .alert-title {{ color: var(--accent-green); }}

    .alert-important {{ background: rgba(163, 113, 247, 0.08); border-color: var(--accent-purple); }}
    .alert-important .alert-title {{ color: var(--accent-purple); }}

    .alert-warning {{ background: rgba(210, 153, 34, 0.08); border-color: var(--accent-orange); }}
    .alert-warning .alert-title {{ color: var(--accent-orange); }}

    .alert-caution {{ background: rgba(248, 81, 73, 0.08); border-color: var(--accent-red); }}
    .alert-caution .alert-title {{ color: var(--accent-red); }}

    blockquote {{
      border-left: 3px solid var(--border-subtle);
      padding-left: 0.85rem;
      color: var(--text-muted);
      margin: 0.75rem 0;
    }}

    /* Media Embeds */
    .media-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      overflow: hidden;
      margin: 0.85rem 0;
    }}

    .media-img {{
      width: 100%;
      height: auto;
      max-height: 520px;
      object-fit: contain;
      background: #000;
      display: block;
    }}

    .media-caption {{
      padding: 0.5rem 0.85rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      background: var(--bg-card-inner);
      font-family: var(--font-mono);
      display: flex;
      justify-content: space-between;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border-subtle); border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--text-dim); }}
  </style>
</head>
<body>
  <!-- Header Shell -->
  <header class="app-header">
    <div class="header-left">
      <button id="sidebarToggleBtn" class="top-btn" onclick="toggleSidebar()" title="Toggle Sidebar">☰</button>
      <div class="logo-badge">
        {antigravity_logo_svg}
        <span>Antigravity</span>
      </div>
      <span class="model-badge">Gemini 3.7 Flash</span>
      <span style="font-size: 0.78rem; color: var(--text-muted); font-family: var(--font-mono); margin-left: 0.5rem;">Air Raid Alert Defense</span>
    </div>
    <div class="header-right">
      <button id="toggleThoughtsBtn" class="top-btn" onclick="toggleAllThoughts()">{sparkle_svg} Thoughts</button>
      <button id="toggleToolsBtn" class="top-btn" onclick="toggleAllTools()">{terminal_svg} Tools</button>
      <button id="themeToggleBtn" class="top-btn" onclick="toggleTheme()">🌓 Theme</button>
    </div>
  </header>

  <div class="app-body">
    <!-- Left Navigation Drawer -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-title-row">
          <span>Conversation History</span>
          <span style="font-family: var(--font-mono);">{len(turns)} Turns</span>
        </div>
        <div class="search-box">
          <span class="search-icon-pos">🔍</span>
          <input type="text" id="turnSearchInput" placeholder="Filter conversation..." />
        </div>
      </div>
      <div class="turns-list" id="turnsList">
""")
    
    # Sidebar Turn Items
    for t in turns:
        tid = t['turn_id']
        tool_count = sum(1 for ev in t['events'] if ev['type'] == 'tool')
        thought_count = sum(1 for ev in t['events'] if ev['type'] == 'thought')
        
        # User snippet preview
        prompt_snippet = html.escape(t['user_prompt'][:100].replace('\n', ' '))
        
        meta_badges = []
        if tool_count > 0:
            meta_badges.append(f'<span class="tiny-badge">⚡ {tool_count}</span>')
        if thought_count > 0:
            meta_badges.append(f'<span class="tiny-badge">🧠 {thought_count}</span>')
            
        badges_html = "".join(meta_badges)
        
        html_lines.append(f"""        <div class="turn-nav-item" id="nav-turn-{tid}" onclick="scrollToTurn({tid})">
          <div class="turn-nav-header">
            <span class="turn-nav-num">#{tid:02d}</span>
            <span>{t['time_formatted']}</span>
          </div>
          <div class="turn-nav-snippet">{prompt_snippet}</div>
          <div class="turn-nav-meta">{badges_html}</div>
        </div>""")

    html_lines.append("""      </div>
    </aside>

    <!-- Main Chat Feed Pane -->
    <main class="chat-view" id="chatView">
      <div class="chat-container" id="chatContainer">
""")

    # Render each Turn
    for t in turns:
        tid = t['turn_id']
        user_clean = html.escape(t['user_prompt'])
        
        html_lines.append(f"""        <!-- Turn {tid} -->
        <section class="turn-group" id="turn-{tid}" data-turn-id="{tid}">
          <div class="turn-divider">Turn #{tid} &bull; {t['time_formatted']}</div>

          <!-- User Bubble -->
          <div class="user-row">
            <div class="user-bubble">{user_clean}</div>
            <div class="user-avatar">U</div>
          </div>

          <!-- Assistant Content -->
          <div class="assistant-row">
            <div class="assistant-avatar">{antigravity_logo_svg}</div>
            <div class="assistant-content">
""")

        # Process Turn Events chronologically
        for ev in t['events']:
            ev_type = ev['type']
            
            # 1. Thought Process
            if ev_type == 'thought':
                text = ev['text']
                char_count = len(text)
                escaped_thought = html.escape(text)
                html_lines.append(f"""              <details class="thinking-accordion">
                <summary>
                  <div class="thinking-summary-left">{sparkle_svg} <span>Thinking Process ({char_count:,} chars)</span></div>
                  {chevron_svg}
                </summary>
                <div class="thinking-body">{escaped_thought}</div>
              </details>""")

            # 2. Tool Invocation
            elif ev_type == 'tool':
                tname = ev['name']
                targs = ev['args'] if isinstance(ev['args'], dict) else {}
                result_str = ev.get('result', '')
                
                # Check tool specific rendering
                if tname == 'ask_question':
                    q_list = targs.get('questions', [])
                    if isinstance(q_list, str):
                        try: q_list = json.loads(q_list)
                        except: q_list = []
                        
                    for q in q_list:
                        q_text = html.escape(q.get('question', 'Question'))
                        opts = q.get('options', [])
                        
                        # Find selected answer from result_str
                        selected_idx = -1
                        opt_htmls = []
                        for idx, opt in enumerate(opts):
                            is_sel = opt in result_str or f"A{idx+1}" in result_str or (idx == 0 and "Recommended" in result_str and "Recommended" in opt)
                            sel_class = "selected" if is_sel else ""
                            sel_badge = '<span class="selected-badge">✓ Selected</span>' if is_sel else ''
                            opt_htmls.append(f'<div class="question-option {sel_class}"><span>{html.escape(opt)}</span>{sel_badge}</div>')
                            
                        html_lines.append(f"""              <div class="question-card">
                <div class="question-title">{question_svg} {q_text}</div>
                <div class="question-options">{"".join(opt_htmls)}</div>
              </div>""")

                elif tname == 'run_command':
                    cmd_line = targs.get('CommandLine', '')
                    summary = targs.get('toolSummary', 'Terminal Command')
                    exit_code_pill = '<span style="color: #3fb950;">✓ Exit 0</span>' if 'exited with code 0' in result_str else ''
                    
                    html_lines.append(f"""              <details class="tool-widget">
                <summary>
                  <div class="tool-summary-left">
                    <span class="tool-pill">{terminal_svg} run_command</span>
                    <span class="tool-desc">{html.escape(summary)}</span>
                  </div>
                  <div>{exit_code_pill} {chevron_svg}</div>
                </summary>
                <div class="tool-body">
                  <div class="terminal-window">
                    <div class="terminal-top">
                      <span class="terminal-dot" style="background:#ff5f56;"></span>
                      <span class="terminal-dot" style="background:#ffbd2e;"></span>
                      <span class="terminal-dot" style="background:#27c93f;"></span>
                      <span class="terminal-cmd">$ {html.escape(cmd_line)}</span>
                    </div>
                    <div class="terminal-output">{html.escape(result_str.strip())}</div>
                  </div>
                </div>
              </details>""")

                elif tname == 'replace_file_content':
                    target_file = targs.get('TargetFile', '')
                    file_base = os.path.basename(target_file)
                    desc = targs.get('Description', 'Edit file')
                    
                    # Extract diff block
                    diff_match = re.search(r'\[diff_block_start\]([\s\S]*?)\[diff_block_end\]', result_str)
                    if diff_match:
                        diff_content = format_diff(diff_match.group(1))
                    else:
                        diff_content = html.escape(result_str.strip())
                        
                    html_lines.append(f"""              <details class="tool-widget">
                <summary>
                  <div class="tool-summary-left">
                    <span class="tool-pill">{edit_file_svg} edit_file</span>
                    <span class="tool-desc">{html.escape(file_base)} &mdash; <span style="color:var(--text-muted);font-weight:400;">{html.escape(desc)}</span></span>
                  </div>
                  {chevron_svg}
                </summary>
                <div class="tool-body">
                  <div class="diff-container">{diff_content}</div>
                </div>
              </details>""")

                else:
                    summary = targs.get('toolSummary', tname)
                    arg_dump = html.escape(json.dumps(targs, indent=2))
                    res_dump = html.escape(result_str[:2500] + ('\n... [truncated]' if len(result_str) > 2500 else ''))
                    
                    html_lines.append(f"""              <details class="tool-widget">
                <summary>
                  <div class="tool-summary-left">
                    <span class="tool-pill">{tname}</span>
                    <span class="tool-desc">{html.escape(summary)}</span>
                  </div>
                  {chevron_svg}
                </summary>
                <div class="tool-body">
                  <div class="tool-section-label">Parameters</div>
                  <pre style="margin-bottom:0.5rem;color:var(--text-muted);">{arg_dump}</pre>
                  <div class="tool-section-label">Output</div>
                  <pre>{res_dump}</pre>
                </div>
              </details>""")

            # 3. Model Prose Response
            elif ev_type == 'response':
                resp_text = ev['text']
                
                # Check for images referenced in this response
                img_embeds = []
                for img_name in ['verified_raions_map.png', 'new_dashboard_clean.png', 'regional_performance_heatmap.png', 'regional_onset_performance_heatmap.png', 'benchmark_horizons_f2.png', 'hromada_active_render.png']:
                    if img_name in resp_text:
                        img_embeds.append(f"""<div class="media-card">
  <img src="media/{img_name}" class="media-img" alt="{img_name}" onerror="this.onerror=null; this.src='../brain/media/{img_name}';">
  <div class="media-caption"><span>{img_name}</span><span>Diagnostic Visualization</span></div>
</div>""")
                        
                formatted_markdown = parse_markdown(resp_text)
                media_html = "".join(img_embeds)
                
                html_lines.append(f"""              <div class="prose">
                {formatted_markdown}
                {media_html}
              </div>""")

        html_lines.append("""            </div>
          </div>
        </section>""")

    html_lines.append("""      </div>
    </main>
  </div>

  <script>
    let thoughtsOpen = false;
    let toolsOpen = false;

    function toggleAllThoughts() {
      thoughtsOpen = !thoughtsOpen;
      document.querySelectorAll('.thinking-accordion').forEach(el => { el.open = thoughtsOpen; });
      document.getElementById('toggleThoughtsBtn').classList.toggle('active', thoughtsOpen);
    }

    function toggleAllTools() {
      toolsOpen = !toolsOpen;
      document.querySelectorAll('.tool-widget').forEach(el => { el.open = toolsOpen; });
      document.getElementById('toggleToolsBtn').classList.toggle('active', toolsOpen);
    }

    function toggleTheme() {
      document.body.classList.toggle('light');
      localStorage.setItem('ag_theme', document.body.classList.contains('light') ? 'light' : 'dark');
    }

    if (localStorage.getItem('ag_theme') === 'light') {
      document.body.classList.add('light');
    }

    function toggleSidebar() {
      const sb = document.getElementById('sidebar');
      sb.classList.toggle('collapsed');
      document.getElementById('sidebarToggleBtn').classList.toggle('active', !sb.classList.contains('collapsed'));
    }

    function scrollToTurn(turnId) {
      const el = document.getElementById('turn-' + turnId);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        document.querySelectorAll('.turn-nav-item').forEach(i => i.classList.remove('active'));
        const navItem = document.getElementById('nav-turn-' + turnId);
        if (navItem) navItem.classList.add('active');
      }
    }

    function copyCode(btn) {
      const pre = btn.closest('.code-block').querySelector('pre code');
      if (pre) {
        navigator.clipboard.writeText(pre.innerText).then(() => {
          const original = btn.innerText;
          btn.innerText = 'Copied!';
          setTimeout(() => { btn.innerText = original; }, 2000);
        });
      }
    }

    // Live Turn Filter in Sidebar
    const searchInput = document.getElementById('turnSearchInput');
    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      const items = document.querySelectorAll('.turn-nav-item');
      items.forEach(item => {
        if (!q) {
          item.style.display = 'flex';
        } else {
          const txt = item.textContent.toLowerCase();
          item.style.display = txt.includes(q) ? 'flex' : 'none';
        }
      });
    });

    // IntersectionObserver to highlight active turn in sidebar as user scrolls
    let userHasScrolled = false;
    window.addEventListener('scroll', () => { userHasScrolled = true; }, { once: true });

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const tid = entry.target.getAttribute('data-turn-id');
          document.querySelectorAll('.turn-nav-item').forEach(i => i.classList.remove('active'));
          const navItem = document.getElementById('nav-turn-' + tid);
          if (navItem) {
            navItem.classList.add('active');
            if (userHasScrolled) {
              navItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
          }
        }
      });
    }, { root: null, rootMargin: '-70px 0px -70% 0px', threshold: 0 });

    document.querySelectorAll('.turn-group').forEach(el => observer.observe(el));
  </script>
</body>
</html>
""")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_lines))
    print(f"Generated Native Antigravity Chat UI: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    transcript_path = repo_root / "brain" / "logs" / "transcript_full.jsonl"
    if not transcript_path.exists():
        transcript_path = repo_root / "brain" / "logs" / "transcript.jsonl"
        
    print(f"Parsing transcript: {transcript_path}...")
    turns, total_steps = load_and_structure_transcript(str(transcript_path))
    print(f"Loaded {len(turns)} clean conversation turns across {total_steps} steps.")
    
    conv_id = "6e0c7109-5304-44a3-8c3d-a42d1e47513c"
    
    # Generate docs/index.html
    docs_index = repo_root / "docs" / "index.html"
    build_native_html(turns, total_steps, conv_id, str(docs_index))
    
    # Mirror in brain/conversation_viewer.html
    brain_viewer = repo_root / "brain" / "conversation_viewer.html"
    build_native_html(turns, total_steps, conv_id, str(brain_viewer))

if __name__ == "__main__":
    main()
