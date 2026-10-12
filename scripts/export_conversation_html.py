"""
Native Antigravity Chat Interactive Conversation Exporter (v3)
Renders an authentic Antigravity IDE chat experience with:
1. Antigravity process timeline (Thought for Xs, Edited +X -Y, Analyzed #LX-Y, Command summaries with hexagon icons)
2. Robust placeholder-based markdown & table parser (prevents unclosed tags or table squishing)
3. Universal natural window scrolling with sticky header and sticky sidebar
4. Zero system metadata / prompt slop
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
    # Extract <USER_REQUEST>...</USER_REQUEST> if present
    match = re.search(r'<USER_REQUEST>\s*([\s\S]*?)\s*</USER_REQUEST>', raw)
    if match:
        text = match.group(1)
    else:
        text = raw
        
    # Strip metadata, system instructions, and slash command templates
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
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %H:%M")
    except Exception:
        return ts_str[:16].replace('T', ' ')

def get_file_icon(filename: str) -> str:
    if filename.endswith('.py'):
        return '🐍'
    elif filename.endswith(('.html', '.htm')):
        return '🌐'
    elif filename.endswith(('.json', '.geojson')):
        return '📦'
    elif filename.endswith('.md'):
        return '📝'
    elif filename.endswith('.ipynb'):
        return '📓'
    elif filename.endswith(('.png', '.jpg', '.svg')):
        return '🖼️'
    return '📄'

def render_markdown(text: str) -> str:
    if not text:
        return ""
        
    code_blocks = []
    def save_code(m):
        lang = m.group(1).strip() or "text"
        code = m.group(2)
        escaped_code = html.escape(code)
        idx = len(code_blocks)
        html_code = (
            f'<div class="code-block">'
            f'  <div class="code-header">'
            f'    <span class="code-lang">{lang}</span>'
            f'    <button class="copy-btn" onclick="copyCode(this)">Copy</button>'
            f'  </div>'
            f'  <pre><code>{escaped_code}</code></pre>'
            f'</div>'
        )
        code_blocks.append(html_code)
        return f"___CODE_BLOCK_{idx}___"
        
    # 1. Protect code blocks
    text = re.sub(r'```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```', save_code, text)
    
    # 2. Protect and render tables
    tables = []
    def save_table(m):
        raw_table = m.group(0).strip()
        lines = [l.strip() for l in raw_table.split('\n') if l.strip()]
        if len(lines) < 2:
            return raw_table
        headers = [c.strip() for c in lines[0].strip('|').split('|')]
        body_rows = []
        for l in lines[1:]:
            if re.match(r'^[\|\s\:\-]+$', l):
                continue
            cells = [c.strip() for c in l.strip('|').split('|')]
            body_rows.append(cells)
            
        h_html = "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"
        b_html = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in body_rows)
        idx = len(tables)
        t_html = f'<div class="table-wrapper"><table class="chat-table"><thead>{h_html}</thead><tbody>{b_html}</tbody></table></div>'
        tables.append(t_html)
        return f"___TABLE_{idx}___"
        
    table_pattern = re.compile(r'((?:^\|.+?\|$\n?){2,})', re.MULTILINE)
    text = table_pattern.sub(save_table, text)
    
    # 3. HTML escape remaining text
    text = html.escape(text)
    
    # 4. Links: [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank" class="chat-link">\1</a>', text)
    
    # 5. Inline code `code`
    text = re.sub(r'`([^`\n]+)`', r'<code class="inline-code">\1</code>', text)
    
    # 6. GitHub Alert Callouts
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
        text = re.sub(pattern, replace_alert, text)
    
    # 7. Blockquotes
    text = re.sub(r'^&gt;\s+(.*)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    
    # 8. Bold & Italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    
    # 9. Headers
    text = re.sub(r'^#### (.*)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # 10. Lists
    text = re.sub(r'^\* (.*)$', r'<li class="bullet-item">\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^- (.*)$', r'<li class="bullet-item">\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+(.*)$', r'<li class="num-item">\1</li>', text, flags=re.MULTILINE)
    
    # 11. Paragraphs & Linebreaks
    text = text.replace('\n\n', '<p class="chat-p"></p>').replace('\n', '<br>')
    
    # 12. Restore protected blocks
    for idx, c_html in enumerate(code_blocks):
        text = text.replace(f"___CODE_BLOCK_{idx}___", c_html)
    for idx, t_html in enumerate(tables):
        text = text.replace(f"___TABLE_{idx}___", t_html)
        
    return text

def format_diff(diff_text: str) -> tuple[str, int, int]:
    lines = diff_text.strip().split('\n')
    formatted = []
    adds = 0
    dels = 0
    for line in lines:
        escaped_line = html.escape(line)
        if line.startswith('+') and not line.startswith('+++'):
            formatted.append(f'<div class="diff-line diff-add">{escaped_line}</div>')
            adds += 1
        elif line.startswith('-') and not line.startswith('---'):
            formatted.append(f'<div class="diff-line diff-del">{escaped_line}</div>')
            dels += 1
        elif line.startswith('@@'):
            formatted.append(f'<div class="diff-line diff-hunk">{escaped_line}</div>')
        else:
            formatted.append(f'<div class="diff-line diff-ctx">{escaped_line}</div>')
    return '\n'.join(formatted), adds, dels

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
                    'processes': [],      # ordered list of process items (thoughts, tools)
                    'responses': []       # list of model prose responses
                }
            elif current_turn is not None:
                # 1. Thought Process Item
                thinking = step.get('thinking')
                if thinking and thinking.strip():
                    th_text = thinking.strip()
                    est_seconds = max(2, min(45, len(th_text) // 120))
                    current_turn['processes'].append({
                        'kind': 'thought',
                        'step_index': step.get('step_index'),
                        'label': f'Thought for {est_seconds}s',
                        'text': th_text
                    })
                
                # 2. Tool Calls
                tool_calls = step.get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        name = tc.get('name', 'tool')
                        args = tc.get('args', {})
                        if isinstance(args, str):
                            try: args = json.loads(args)
                            except: pass
                        current_turn['processes'].append({
                            'kind': 'tool',
                            'step_index': step.get('step_index'),
                            'name': name,
                            'args': args,
                            'result': ''
                        })
                
                # 3. Tool Results (GENERIC)
                if stype == 'GENERIC':
                    content = step.get('content', '')
                    matched = False
                    for p in reversed(current_turn['processes']):
                        if p['kind'] == 'tool' and not p.get('result'):
                            p['result'] = content
                            matched = True
                            break
                    if not matched and content:
                        current_turn['responses'].append(content)
                        
                # 4. Model Text Responses
                if stype == 'PLANNER_RESPONSE':
                    content = step.get('content')
                    if content and content.strip():
                        current_turn['responses'].append(content.strip())
                        
    if current_turn:
        turns.append(current_turn)
        
    return turns, total_steps

def build_native_html(turns, total_steps: int, conv_id: str, out_path: str):
    html_lines = []
    
    # Antigravity Blue Delta Icon SVG
    antigravity_logo_svg = """<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 2L2 19.5H8.5L12 13L15.5 19.5H22L12 2Z" fill="url(#ag-grad)" />
  <path d="M12 7.5L8 15.5H16L12 7.5Z" fill="#131418" />
  <defs>
    <linearGradient id="ag-grad" x1="2" y1="2" x2="22" y2="20" gradientUnits="userSpaceOnUse">
      <stop stop-color="#388bfd" />
      <stop offset="1" stop-color="#1f6feb" />
    </linearGradient>
  </defs>
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
      --bg-process-hover: rgba(255, 255, 255, 0.04);
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
      --bg-process-hover: rgba(0, 0, 0, 0.03);
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

    html, body {{
      height: 100%;
      width: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden; /* Entire site is locked to viewport, never scrolls */
    }}

    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-app);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      -webkit-font-smoothing: antialiased;
    }}

    /* Fixed Application Header */
    .app-header {{
      height: 50px;
      width: 100%;
      background-color: var(--bg-sidebar);
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 1.25rem;
      flex-shrink: 0;
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

    /* Body 2-Column Shell */
    .app-body {{
      display: flex;
      width: 100%;
      height: calc(100vh - 50px);
      overflow: hidden;
      position: relative;
    }}

    /* Left Sidebar: Fixed Navigator */
    .sidebar {{
      width: 320px;
      height: 100%;
      background-color: var(--bg-sidebar);
      border-right: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      z-index: 900;
      overflow: hidden;
      transition: width 0.2s cubic-bezier(0.16, 1, 0.3, 1), margin-left 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}

    .sidebar.collapsed {{
      width: 0;
      margin-left: -1px;
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

    .search-box input {{
      width: 100%;
      background: var(--bg-app);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.4rem 0.65rem;
      font-size: 0.8rem;
      color: var(--text-main);
      outline: none;
      font-family: var(--font-sans);
    }}

    .search-box input:focus {{ border-color: var(--border-focus); }}

    .turns-list {{
      flex: 1;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 0.5rem 0.6rem;
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }}

    .turn-nav-item {{
      padding: 0.55rem 0.75rem;
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

    .turn-nav-item:hover {{ background: var(--bg-sidebar-hover); }}
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

    .turn-nav-num {{ color: var(--accent-blue); font-weight: 600; }}

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
      height: 100%;
      overflow-y: auto;
      overflow-x: hidden;
      -webkit-overflow-scrolling: touch;
      background: var(--bg-main);
      position: relative;
      scroll-behavior: smooth;
    }}

    .chat-container {{
      max-width: 880px;
      margin: 0 auto;
      padding: 2rem 1.5rem 8rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 2.5rem;
    }}

    .turn-group {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
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

    /* Assistant Stream */
    .assistant-row {{
      display: flex;
      gap: 1rem;
      align-items: flex-start;
    }}

    .assistant-avatar {{
      width: 30px;
      height: 30px;
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
      gap: 0.85rem;
      overflow: hidden;
    }}

    /* ANTIGRAVITY PROCESS TIMELINE (Matches Screenshot 2) */
    .process-timeline {{
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
      background: rgba(255, 255, 255, 0.015);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 0.35rem 0.45rem;
      margin-bottom: 0.5rem;
    }}

    .process-item {{
      border-radius: 6px;
      overflow: hidden;
      transition: background 0.12s ease;
    }}

    .process-item[open] {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      margin: 0.25rem 0;
    }}

    .process-summary {{
      padding: 0.32rem 0.6rem;
      font-size: 0.82rem;
      font-family: var(--font-sans);
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      list-style: none;
      border-radius: 5px;
      transition: background 0.1s ease;
    }}

    .process-summary::-webkit-details-marker {{ display: none; }}
    .process-summary:hover {{ background: var(--bg-process-hover); color: var(--text-main); }}

    .process-left {{
      display: flex;
      align-items: center;
      gap: 0.45rem;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }}

    .process-action {{
      color: var(--text-dim);
      font-weight: 400;
    }}

    .process-file {{
      color: var(--text-main);
      font-weight: 500;
      font-family: var(--font-mono);
      font-size: 0.8rem;
    }}

    .diff-add-pill {{
      color: var(--accent-green);
      font-family: var(--font-mono);
      font-size: 0.75rem;
      font-weight: 600;
      margin-left: 0.2rem;
    }}

    .diff-del-pill {{
      color: var(--accent-red);
      font-family: var(--font-mono);
      font-size: 0.75rem;
      font-weight: 600;
      margin-left: 0.15rem;
    }}

    .line-range-pill {{
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-muted);
      font-family: var(--font-mono);
      font-size: 0.72rem;
      padding: 0.1rem 0.4rem;
      border-radius: 4px;
      margin-left: 0.2rem;
    }}

    .process-right {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-shrink: 0;
    }}

    .process-status-icon {{
      font-size: 0.75rem;
      color: var(--text-dim);
    }}

    .process-chevron {{
      font-size: 0.85rem;
      color: var(--text-dim);
      transition: transform 0.15s ease;
    }}

    .process-item[open] .process-chevron {{
      transform: rotate(90deg);
    }}

    .process-drawer {{
      padding: 0.75rem 0.85rem;
      background: var(--bg-card-inner);
      border-top: 1px solid var(--border-subtle);
      font-size: 0.8rem;
      font-family: var(--font-mono);
    }}

    /* Terminal Widget */
    .terminal-box {{
      background: #090a0d;
      border: 1px solid #1f232c;
      border-radius: 6px;
      overflow: hidden;
    }}

    .terminal-head {{
      background: #141720;
      padding: 0.35rem 0.65rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      border-bottom: 1px solid #1f232c;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      color: var(--text-main);
    }}

    .terminal-body {{
      padding: 0.75rem;
      color: #c9d1d9;
      font-size: 0.78rem;
      white-space: pre-wrap;
      max-height: 380px;
      overflow-y: auto;
      line-height: 1.5;
    }}

    /* Diff View */
    .diff-box {{
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

    .diff-line {{ padding: 0.1rem 0.75rem; white-space: pre-wrap; }}
    .diff-add {{ background: rgba(46, 160, 67, 0.15); color: #3fb950; }}
    .diff-del {{ background: rgba(248, 81, 73, 0.15); color: #f85149; }}
    .diff-hunk {{ background: rgba(163, 113, 247, 0.1); color: #a371f7; }}
    .diff-ctx {{ color: #8b949e; }}

    /* Question Widget */
    .question-box {{
      background: var(--bg-card);
      border: 1px solid rgba(56, 139, 253, 0.25);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }}

    .q-opt {{
      padding: 0.45rem 0.75rem;
      border-radius: 6px;
      background: var(--bg-card-inner);
      border: 1px solid var(--border-subtle);
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .q-opt.selected {{
      background: rgba(63, 185, 80, 0.12);
      border-color: rgba(63, 185, 80, 0.4);
      color: var(--text-main);
      font-weight: 500;
    }}

    .q-badge {{
      font-size: 0.68rem;
      padding: 0.1rem 0.35rem;
      background: var(--accent-green);
      color: #fff;
      border-radius: 4px;
    }}

    /* Markdown Prose */
    .prose {{
      font-size: 0.95rem;
      line-height: 1.68;
      color: var(--text-main);
    }}

    .prose p.chat-p {{ margin: 0.75rem 0; }}
    .prose h1, .prose h2, .prose h3, .prose h4 {{
      margin: 1.25rem 0 0.5rem 0;
      color: var(--text-main);
      font-weight: 600;
    }}
    .prose h1 {{ font-size: 1.3rem; }}
    .prose h2 {{ font-size: 1.15rem; }}
    .prose h3 {{ font-size: 1.05rem; }}
    .prose h4 {{ font-size: 0.95rem; }}

    .prose ul, .prose ol {{ margin: 0.5rem 0 0.5rem 1.4rem; }}
    .prose li {{ margin-bottom: 0.35rem; }}

    .inline-code {{
      background: rgba(110, 118, 129, 0.2);
      color: var(--text-main);
      font-family: var(--font-mono);
      font-size: 0.85em;
      padding: 0.15rem 0.35rem;
      border-radius: 4px;
    }}

    .chat-link {{
      color: var(--accent-blue);
      text-decoration: underline;
      text-underline-offset: 3px;
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
    }}

    .copy-btn:hover {{ color: var(--text-main); border-color: var(--text-muted); }}

    .code-block pre {{
      padding: 0.85rem 1rem;
      overflow-x: auto;
      font-family: var(--font-mono);
      font-size: 0.83rem;
      line-height: 1.5;
    }}

    /* Markdown Tables */
    .table-wrapper {{
      width: 100%;
      overflow-x: auto;
      margin: 1rem 0;
    }}

    .chat-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      overflow: hidden;
    }}

    .chat-table th {{
      background: var(--bg-card-inner);
      padding: 0.65rem 0.85rem;
      text-align: left;
      border-bottom: 1px solid var(--border-subtle);
      font-weight: 600;
      color: var(--text-main);
    }}

    .chat-table td {{
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-muted);
    }}

    .chat-table tr:last-child td {{ border-bottom: none; }}

    /* GitHub Alerts */
    .alert {{
      border-left: 3px solid;
      border-radius: 0 6px 6px 0;
      padding: 0.75rem 1rem;
      margin: 0.85rem 0;
      font-size: 0.9rem;
    }}
    .alert-title {{ font-weight: 600; font-size: 0.82rem; text-transform: uppercase; margin-bottom: 0.3rem; }}
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

    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border-subtle); border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--text-dim); }}
  </style>
</head>
<body>
  <!-- Header -->
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
      <button id="toggleThoughtsBtn" class="top-btn" onclick="toggleAllProcesses()">&gt;_ All Processes</button>
      <button id="themeToggleBtn" class="top-btn" onclick="toggleTheme()">🌓 Theme</button>
    </div>
  </header>

  <div class="app-body">
    <!-- Left Navigator Sidebar -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-title-row">
          <span>Conversation History</span>
          <span style="font-family: var(--font-mono);">{len(turns)} Turns</span>
        </div>
        <div class="search-box">
          <input type="text" id="turnSearchInput" placeholder="Filter turns..." />
        </div>
      </div>
      <div class="turns-list" id="turnsList">
""")

    # Render Sidebar Items
    for t in turns:
        tid = t['turn_id']
        tool_count = sum(1 for p in t['processes'] if p['kind'] == 'tool')
        thought_count = sum(1 for p in t['processes'] if p['kind'] == 'thought')
        prompt_snippet = html.escape(t['user_prompt'][:90].replace('\n', ' '))
        
        badges = []
        if tool_count > 0:
            badges.append(f'<span class="tiny-badge">⚡ {tool_count}</span>')
        if thought_count > 0:
            badges.append(f'<span class="tiny-badge">🧠 {thought_count}</span>')
            
        html_lines.append(f"""        <div class="turn-nav-item" id="nav-turn-{tid}" onclick="scrollToTurn({tid})">
          <div class="turn-nav-header">
            <span class="turn-nav-num">#{tid:02d}</span>
            <span>{t['time_formatted']}</span>
          </div>
          <div class="turn-nav-snippet">{prompt_snippet}</div>
          <div class="turn-nav-meta">{"".join(badges)}</div>
        </div>""")

    html_lines.append("""      </div>
    </aside>

    <!-- Main Chat Feed -->
    <main class="chat-view" id="chatView">
      <div class="chat-container" id="chatContainer">
""")

    # Render each Conversation Turn
    for t in turns:
        tid = t['turn_id']
        user_clean = html.escape(t['user_prompt'])
        
        html_lines.append(f"""        <!-- Turn {tid} -->
        <section class="turn-group" id="turn-{tid}" data-turn-id="{tid}">
          <div class="turn-divider">Turn #{tid} &bull; {t['time_formatted']}</div>

          <!-- User Prompt -->
          <div class="user-row">
            <div class="user-bubble">{user_clean}</div>
            <div class="user-avatar">U</div>
          </div>

          <!-- Assistant Stream -->
          <div class="assistant-row">
            <div class="assistant-avatar">{antigravity_logo_svg}</div>
            <div class="assistant-content">
""")

        # Process Timeline (Matches Screenshot 2)
        if t['processes']:
            html_lines.append("""              <div class="process-timeline">""")
            for p in t['processes']:
                p_kind = p['kind']
                
                # 1. Thought item
                if p_kind == 'thought':
                    label = p['label']
                    th_text = html.escape(p['text'])
                    html_lines.append(f"""                <details class="process-item">
                  <summary class="process-summary">
                    <span class="process-left">
                      <span class="process-action">{label}</span>
                    </span>
                    <span class="process-chevron">&rsaquo;</span>
                  </summary>
                  <div class="process-drawer" style="white-space: pre-wrap; color: var(--text-muted);">{th_text}</div>
                </details>""")

                # 2. Tool item
                elif p_kind == 'tool':
                    tname = p['name']
                    targs = p['args'] if isinstance(p['args'], dict) else {}
                    tresult = p.get('result', '')
                    
                    if tname == 'replace_file_content':
                        tf = targs.get('TargetFile', '')
                        fn = os.path.basename(tf)
                        icon = get_file_icon(fn)
                        diff_match = re.search(r'\[diff_block_start\]([\s\S]*?)\[diff_block_end\]', tresult)
                        if diff_match:
                            diff_html, adds, dels = format_diff(diff_match.group(1))
                        else:
                            diff_html = html.escape(tresult.strip())
                            adds, dels = 1, 1
                            
                        html_lines.append(f"""                <details class="process-item">
                  <summary class="process-summary">
                    <span class="process-left">
                      <span class="process-action">Edited</span>
                      <span class="process-file">{icon} {html.escape(fn)}</span>
                      <span class="diff-add-pill">+{adds}</span>
                      <span class="diff-del-pill">-{dels}</span>
                    </span>
                    <span class="process-chevron">&rsaquo;</span>
                  </summary>
                  <div class="process-drawer">
                    <div class="diff-box">{diff_html}</div>
                  </div>
                </details>""")

                    elif tname == 'write_to_file':
                        tf = targs.get('TargetFile', '')
                        fn = os.path.basename(tf)
                        icon = get_file_icon(fn)
                        code_cnt = targs.get('CodeContent', '')
                        line_cnt = len(code_cnt.split('\n'))
                        
                        html_lines.append(f"""                <details class="process-item">
                  <summary class="process-summary">
                    <span class="process-left">
                      <span class="process-action">Created</span>
                      <span class="process-file">{icon} {html.escape(fn)}</span>
                      <span class="diff-add-pill">+{line_cnt}</span>
                    </span>
                    <span class="process-chevron">&rsaquo;</span>
                  </summary>
                  <div class="process-drawer">
                    <pre style="white-space: pre-wrap; max-height: 350px; overflow-y: auto;">{html.escape(code_cnt[:2500])}</pre>
                  </div>
                </details>""")

                    elif tname == 'view_file':
                        tf = targs.get('AbsolutePath', '')
                        fn = os.path.basename(tf)
                        icon = get_file_icon(fn)
                        s_line = targs.get('StartLine', 1)
                        e_line = targs.get('EndLine', '')
                        range_str = f"#L{s_line}-{e_line}" if e_line else f"#L{s_line}"
                        
                        html_lines.append(f"""                <details class="process-item">
                  <summary class="process-summary">
                    <span class="process-left">
                      <span class="process-action">Analyzed</span>
                      <span class="process-file">{icon} {html.escape(fn)}</span>
                      <span class="line-range-pill">{range_str}</span>
                    </span>
                    <span class="process-chevron">&rsaquo;</span>
                  </summary>
                  <div class="process-drawer">
                    <pre style="white-space: pre-wrap; max-height: 350px; overflow-y: auto;">{html.escape(tresult[:2500])}</pre>
                  </div>
                </details>""")

                    elif tname == 'run_command':
                        cmd = targs.get('CommandLine', '')
                        summary = targs.get('toolSummary') or targs.get('toolAction') or cmd[:40]
                        
                        html_lines.append(f"""                <details class="process-item">
                  <summary class="process-summary">
                    <span class="process-left">
                      <span class="process-title">{html.escape(summary)}</span>
                    </span>
                    <span class="process-right">
                      <span class="process-status-icon">&x2B21;</span>
                      <span class="process-chevron">&rsaquo;</span>
                    </span>
                  </summary>
                  <div class="process-drawer">
                    <div class="terminal-box">
                      <div class="terminal-head">$ {html.escape(cmd)}</div>
                      <div class="terminal-body">{html.escape(tresult.strip())}</div>
                    </div>
                  </div>
                </details>""")

                    elif tname == 'ask_question':
                        q_list = targs.get('questions', [])
                        if isinstance(q_list, str):
                            try: q_list = json.loads(q_list)
                            except: q_list = []
                            
                        for q in q_list:
                            q_text = html.escape(q.get('question', 'Question'))
                            opts = q.get('options', [])
                            opt_htmls = []
                            for idx, opt in enumerate(opts):
                                is_sel = opt in tresult or f"A{idx+1}" in tresult or (idx == 0 and "Recommended" in tresult and "Recommended" in opt)
                                sel_cls = "selected" if is_sel else ""
                                sel_bdg = '<span class="q-badge">&check; Selected</span>' if is_sel else ''
                                opt_htmls.append(f'<div class="q-opt {sel_cls}"><span>{html.escape(opt)}</span>{sel_bdg}</div>')
                                
                            html_lines.append(f"""                <details class="process-item">
                  <summary class="process-summary">
                    <span class="process-left">
                      <span class="process-action">Interview</span>
                      <span class="process-title">{q_text}</span>
                    </span>
                    <span class="process-chevron">&rsaquo;</span>
                  </summary>
                  <div class="process-drawer">
                    <div class="question-box">
                      <div style="font-weight: 600; margin-bottom: 0.35rem;">{q_text}</div>
                      {"".join(opt_htmls)}
                    </div>
                  </div>
                </details>""")

                    else:
                        summary = targs.get('toolSummary') or tname
                        html_lines.append(f"""                <details class="process-item">
                  <summary class="process-summary">
                    <span class="process-left">
                      <span class="process-action">{html.escape(tname)}</span>
                      <span class="process-title">{html.escape(summary)}</span>
                    </span>
                    <span class="process-chevron">&rsaquo;</span>
                  </summary>
                  <div class="process-drawer">
                    <pre style="white-space: pre-wrap; max-height: 300px; overflow-y: auto;">{html.escape(tresult[:2000])}</pre>
                  </div>
                </details>""")

            html_lines.append("""              </div>""")

        # Assistant Responses (Prose & Images)
        if t['responses']:
            combined_resp = "\n\n".join(t['responses'])
            
            # Check for media attachments
            media_items = []
            for img_name in ['verified_raions_map.png', 'new_dashboard_clean.png', 'regional_performance_heatmap.png', 'regional_onset_performance_heatmap.png', 'benchmark_horizons_f2.png', 'hromada_active_render.png']:
                if img_name in combined_resp:
                    media_items.append(f"""<div class="media-card">
  <img src="media/{img_name}" class="media-img" alt="{img_name}" onerror="this.onerror=null; this.src='../brain/media/{img_name}';">
  <div class="media-caption"><span>{img_name}</span><span>Diagnostic Visualization</span></div>
</div>""")
                    
            rendered_prose = render_markdown(combined_resp)
            media_html = "".join(media_items)
            
            html_lines.append(f"""              <div class="prose">
                {rendered_prose}
                {media_html}
              </div>""")

        html_lines.append("""            </div>
          </div>
        </section>""")

    html_lines.append("""      </div>
    </main>
  </div>

  <script>
    let processesExpanded = false;

    function toggleAllProcesses() {
      processesExpanded = !processesExpanded;
      document.querySelectorAll('.process-item').forEach(el => {
        el.open = processesExpanded;
      });
      document.getElementById('toggleThoughtsBtn').classList.toggle('active', processesExpanded);
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
      const chatView = document.getElementById('chatView');
      if (el && chatView) {
        chatView.scrollTo({
          top: el.offsetTop - 20,
          behavior: 'smooth'
        });
        document.querySelectorAll('.turn-nav-item').forEach(i => i.classList.remove('active'));
        const navItem = document.getElementById('nav-turn-' + turnId);
        if (navItem) navItem.classList.add('active');
      }
    }

    function copyCode(btn) {
      const pre = btn.closest('.code-block').querySelector('pre code');
      if (pre) {
        navigator.clipboard.writeText(pre.innerText).then(() => {
          const orig = btn.innerText;
          btn.innerText = 'Copied!';
          setTimeout(() => { btn.innerText = orig; }, 2000);
        });
      }
    }

    // Live Turn Filter
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

    // Auto highlight active turn in sidebar on scroll without window jumping
    const chatView = document.getElementById('chatView');
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const tid = entry.target.getAttribute('data-turn-id');
          document.querySelectorAll('.turn-nav-item').forEach(i => i.classList.remove('active'));
          const navItem = document.getElementById('nav-turn-' + tid);
          if (navItem) {
            navItem.classList.add('active');
            const list = document.getElementById('turnsList');
            if (list) {
              const itTop = navItem.offsetTop - list.offsetTop;
              if (itTop < list.scrollTop || itTop > list.scrollTop + list.clientHeight - 60) {
                list.scrollTop = itTop - list.clientHeight / 2;
              }
            }
          }
        }
      });
    }, { root: chatView, rootMargin: '-20px 0px -70% 0px', threshold: 0.05 });

    document.querySelectorAll('.turn-group').forEach(el => observer.observe(el));
  </script>
</body>
</html>
""")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_lines))
    print(f"Generated Native Antigravity Chat UI v3: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    transcript_path = repo_root / "brain" / "logs" / "transcript_full.jsonl"
    if not transcript_path.exists():
        transcript_path = repo_root / "brain" / "logs" / "transcript.jsonl"
        
    print(f"Parsing transcript: {transcript_path}...")
    turns, total_steps = load_and_structure_transcript(str(transcript_path))
    print(f"Loaded {len(turns)} clean conversation turns across {total_steps} steps.")
    
    conv_id = "6e0c7109-5304-44a3-8c3d-a42d1e47513c"
    
    # 1. Output to docs/index.html
    docs_index = repo_root / "docs" / "index.html"
    build_native_html(turns, total_steps, conv_id, str(docs_index))
    
    # 2. Mirror in brain/conversation_viewer.html
    brain_viewer = repo_root / "brain" / "conversation_viewer.html"
    build_native_html(turns, total_steps, conv_id, str(brain_viewer))

if __name__ == "__main__":
    main()
