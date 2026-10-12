"""
Impeccable Antigravity Chat Interactive Brain Exporter
Transforms the conversation transcript and brain artifacts into a pixel-perfect,
native Google DeepMind Antigravity IDE Chat interface.
"""

import json
import html
import re
import os
import sys
from pathlib import Path
from datetime import datetime

def render_markdown(text: str) -> str:
    """Renders GitHub Flavored Markdown to HTML with Antigravity styling."""
    if not text:
        return ""
    
    # 1. Extract and protect code blocks
    code_blocks = []
    def save_code_block(match):
        lang = match.group(1).strip() if match.group(1) else "text"
        code = match.group(2)
        idx = len(code_blocks)
        code_blocks.append((lang, code))
        return f"__CODE_BLOCK_{idx}__"
    
    # Match ```lang ... ```
    text = re.sub(r'```([a-zA-Z0-9_\-\+]*)\r?\n([\s\S]*?)```', save_code_block, text)
    
    # 2. Escape HTML
    text = html.escape(text)
    
    # 3. Callout / Alerts (> [!NOTE], > [!IMPORTANT], etc.)
    def replace_alerts(match):
        alert_type = match.group(1).upper()
        raw_content = match.group(2)
        lines = [re.sub(r'^&gt;\s?', '', l) for l in raw_content.split('\n')]
        content = '<br>'.join(lines).strip()
        icon_map = {
            "NOTE": ("ℹ️", "note"),
            "TIP": ("💡", "tip"),
            "IMPORTANT": ("📌", "important"),
            "WARNING": ("⚠️", "warning"),
            "CAUTION": ("🛑", "caution"),
        }
        icon, css_class = icon_map.get(alert_type, ("ℹ️", "note"))
        return f'<div class="callout callout-{css_class}"><div class="callout-header"><span class="callout-icon">{icon}</span><span class="callout-title">{alert_type}</span></div><div class="callout-body">{content}</div></div>'

    text = re.sub(r'^&gt; \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\r?\n((?:&gt;.*(?:\r?\n|$))+)', 
                  replace_alerts, text, flags=re.MULTILINE)

    # 4. Standard Blockquotes
    text = re.sub(r'^&gt; (.*)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    
    # 5. Headers
    text = re.sub(r'^#### (.*)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # 6. Inline elements: Bold, Italic, Inline Code
    text = re.sub(r'`([^`\n]+)`', r'<code class="inline-code">\1</code>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    
    # 7. Unordered and Ordered Lists
    text = re.sub(r'^\* (.*)$', r'<li class="list-item">\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^- (.*)$', r'<li class="list-item">\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\. (.*)$', r'<li class="list-item-ordered">\1</li>', text, flags=re.MULTILINE)
    
    # 8. Tables
    def parse_tables(block):
        lines = block.strip().split('\n')
        if len(lines) >= 2 and '|' in lines[0] and re.match(r'^[|\s\-:]+$', lines[1]):
            headers = [h.strip() for h in lines[0].split('|')[1:-1]]
            rows = []
            for line in lines[2:]:
                if '|' in line:
                    cols = [c.strip() for c in line.split('|')[1:-1]]
                    rows.append(cols)
            
            table_html = '<div class="table-container"><table class="data-table"><thead><tr>'
            for h in headers:
                table_html += f'<th>{h}</th>'
            table_html += '</tr></thead><tbody>'
            for row in rows:
                table_html += '<tr>'
                for c in row:
                    table_html += f'<td>{c}</td>'
                table_html += '</tr>'
            table_html += '</tbody></table></div>'
            return table_html
        return block

    paragraphs = text.split('\n\n')
    parsed_paragraphs = []
    for p in paragraphs:
        if '|' in p and '\n' in p:
            parsed_paragraphs.append(parse_tables(p))
        else:
            # Newlines inside paragraph to <br>
            parsed_paragraphs.append(p.replace('\n', '<br>'))
            
    text = '<p>' + '</p><p>'.join(parsed_paragraphs) + '</p>'
    text = text.replace('<p></p>', '')
    
    # 9. Restore Code Blocks
    for idx, (lang, code) in enumerate(code_blocks):
        escaped_code = html.escape(code)
        
        # Check if it's a diff
        if lang == 'diff' or ('@@' in code and ('+' in code or '-' in code)):
            diff_lines = []
            for dl in escaped_code.split('\n'):
                if dl.startswith('+'):
                    diff_lines.append(f'<span class="diff-add">{dl}</span>')
                elif dl.startswith('-'):
                    diff_lines.append(f'<span class="diff-sub">{dl}</span>')
                elif dl.startswith('@'):
                    diff_lines.append(f'<span class="diff-info">{dl}</span>')
                else:
                    diff_lines.append(f'<span class="diff-ctx">{dl}</span>')
            formatted_code = '\n'.join(diff_lines)
            card = f'''<div class="code-block-wrapper">
              <div class="code-block-header">
                <span class="code-lang">diff</span>
                <button class="copy-btn" onclick="copySnippet(this)">Copy</button>
              </div>
              <pre class="code-content language-diff"><code>{formatted_code}</code></pre>
            </div>'''
        else:
            lang_label = lang if lang else "code"
            card = f'''<div class="code-block-wrapper">
              <div class="code-block-header">
                <span class="code-lang">{lang_label}</span>
                <button class="copy-btn" onclick="copySnippet(this)">Copy</button>
              </div>
              <pre class="code-content language-{lang_label}"><code>{escaped_code}</code></pre>
            </div>'''
        text = text.replace(f"__CODE_BLOCK_{idx}__", card)
        
    return text

def parse_turn_data(transcript_path: str):
    turns = []
    current_turn = None
    total_steps = 0
    total_tool_calls = 0
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            step = json.loads(line)
            total_steps += 1
            stype = step.get('type')
            
            if stype == 'USER_INPUT':
                if current_turn is not None:
                    turns.append(current_turn)
                current_turn = {
                    'turn_id': len(turns) + 1,
                    'created_at': step.get('created_at', ''),
                    'user_prompt': step.get('content', ''),
                    'model_version': 'Gemini 3.7 Flash',
                    'thoughts': [],
                    'tool_calls': [],
                    'assistant_responses': [],
                }
            elif current_turn is not None:
                # Capture model thinking / chain of thought
                thinking = step.get('thinking')
                if thinking and thinking.strip():
                    current_turn['thoughts'].append({
                        'step_index': step.get('step_index'),
                        'text': thinking.strip()
                    })
                
                # Capture tool executions
                tool_calls = step.get('tool_calls', [])
                if tool_calls:
                    for tc in tool_calls:
                        total_tool_calls += 1
                        name = tc.get('name', 'tool')
                        args = tc.get('args', {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                pass
                        current_turn['tool_calls'].append({
                            'step_index': step.get('step_index'),
                            'name': name,
                            'args': args,
                            'result': ''
                        })
                
                # Capture tool results
                if stype == 'GENERIC':
                    content = step.get('content', '')
                    if current_turn['tool_calls']:
                        for tc in reversed(current_turn['tool_calls']):
                            if not tc['result']:
                                tc['result'] = content
                                break
                    elif content:
                        current_turn['assistant_responses'].append(content)
                        
                # Capture assistant content
                if stype == 'PLANNER_RESPONSE':
                    content = step.get('content')
                    if content and content.strip():
                        current_turn['assistant_responses'].append(content.strip())
                        
    if current_turn is not None:
        turns.append(current_turn)
        
    return turns, total_steps, total_tool_calls

def load_artifacts_and_media(repo_root: Path):
    artifacts = []
    art_dir = repo_root / "brain" / "artifacts"
    if art_dir.exists():
        for fname in sorted(os.listdir(art_dir)):
            if fname.endswith('.md'):
                fpath = art_dir / fname
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        raw = f.read()
                    # Clean title
                    title = fname.replace('.md', '').replace('_', ' ').title()
                    first_line = raw.strip().split('\n')[0]
                    if first_line.startswith('# '):
                        title = first_line.replace('# ', '').strip()
                    artifacts.append({
                        'filename': fname,
                        'title': title,
                        'size': len(raw),
                        'content': raw
                    })
                except Exception as e:
                    print(f"Error loading artifact {fname}: {e}")

    media = []
    med_dir = repo_root / "brain" / "media"
    if med_dir.exists():
        for fname in sorted(os.listdir(med_dir)):
            if fname.endswith(('.png', '.jpg', '.svg')):
                title = fname.replace('.png', '').replace('.jpg', '').replace('_', ' ').title()
                media.append({
                    'filename': fname,
                    'title': title,
                    'path': f"media/{fname}"
                })
                
    return artifacts, media

def build_native_antigravity_html(turns, total_steps, total_tools, artifacts, media, conv_id):
    """Builds a complete, modern single-page Antigravity IDE application."""

    # Pre-render turn snippets for sidebar
    sidebar_items_html = []
    feed_cards_html = []

    for t in turns:
        tid = t['turn_id']
        raw_prompt = t['user_prompt']
        
        # Clean user prompt: strip <USER_REQUEST>, <ADDITIONAL_METADATA> tags
        clean_prompt = re.sub(r'<USER_REQUEST>\s*', '', raw_prompt)
        clean_prompt = re.sub(r'\s*</USER_REQUEST>[\s\S]*', '', clean_prompt).strip()
        
        # Check for slash commands in prompt
        slash_cmds = re.findall(r'(/[a-zA-Z0-9_\-]+)', clean_prompt)
        
        created_time = t['created_at'].split('T')[1][:5] if 'T' in t['created_at'] else ''
        date_str = t['created_at'].split('T')[0] if 'T' in t['created_at'] else ''
        
        prompt_snippet = clean_prompt.replace('\n', ' ')[:55]
        if len(clean_prompt) > 55:
            prompt_snippet += '...'
            
        tool_count = len(t['tool_calls'])
        thought_count = len(t['thoughts'])
        
        # Sidebar item
        sidebar_items_html.append(f'''
        <div class="turn-nav-item" id="nav-turn-{tid}" onclick="scrollToTurn({tid})">
          <div class="turn-nav-top">
            <span class="turn-nav-badge">#{tid}</span>
            <span class="turn-nav-time">{created_time}</span>
          </div>
          <div class="turn-nav-text">{html.escape(prompt_snippet)}</div>
          <div class="turn-nav-meta">
            {f'<span class="meta-tag tools-tag">⚡ {tool_count} tools</span>' if tool_count else ''}
            {f'<span class="meta-tag cot-tag">🧠 CoT</span>' if thought_count else ''}
            {' '.join([f'<span class="meta-tag cmd-tag">{cmd}</span>' for cmd in slash_cmds[:2]])}
          </div>
        </div>''')

        # Main feed card
        user_body_html = render_markdown(clean_prompt)
        
        # Build thoughts block
        thought_block_html = ""
        if t['thoughts']:
            all_thoughts = "\n\n---\n\n".join([f"[{th.get('step_index')}]: {th['text']}" for th in t['thoughts']])
            thought_chars = len(all_thoughts)
            thought_block_html = f'''
            <div class="cot-container">
              <details class="cot-details">
                <summary class="cot-summary">
                  <div class="cot-summary-left">
                    <span class="cot-icon">🧠</span>
                    <span class="cot-title">Thought process</span>
                    <span class="cot-meta">• {thought_count} iterations • {thought_chars:,} chars</span>
                  </div>
                  <span class="cot-chevron">▾</span>
                </summary>
                <div class="cot-body">
                  <pre class="cot-text">{html.escape(all_thoughts)}</pre>
                </div>
              </details>
            </div>'''
            
        # Build tool executions block
        tools_block_html = ""
        if t['tool_calls']:
            tools_list_html = []
            for tc in t['tool_calls']:
                t_name = tc.get('name', 'tool')
                t_args = tc.get('args', {})
                t_result = tc.get('result', '')
                
                # Determine icon & summary
                t_summary = ""
                t_action = ""
                if isinstance(t_args, dict):
                    t_summary = t_args.get('toolSummary', '')
                    t_action = t_args.get('toolAction', '')
                    cmd_line = t_args.get('CommandLine', '')
                    target_file = t_args.get('TargetFile', '')
                else:
                    cmd_line = ""
                    target_file = ""
                    
                icon = "⚡"
                if t_name == "run_command":
                    icon = "🖥️"
                    display_title = f"Terminal: {cmd_line[:60]}" if cmd_line else t_summary or "Execute Command"
                elif t_name in ("replace_file_content", "write_to_file"):
                    icon = "📝"
                    display_title = f"Edit: {Path(target_file).name if target_file else t_summary}"
                elif t_name == "view_file":
                    icon = "📄"
                    display_title = f"View: {Path(t_args.get('AbsolutePath', '')).name if isinstance(t_args, dict) else 'file'}"
                elif t_name == "ask_question":
                    icon = "💬"
                    display_title = "User Dialogue Question"
                elif t_name == "search_web":
                    icon = "🌐"
                    display_title = f"Web Search: {t_args.get('query', '')[:40]}"
                else:
                    display_title = t_summary or t_name
                    
                # Format args and result
                args_json = json.dumps(t_args, indent=2) if isinstance(t_args, (dict, list)) else str(t_args)
                
                # Truncate very long terminal output
                truncated_note = ""
                if len(t_result) > 5000:
                    truncated_note = f"\n... [Truncated {len(t_result)-5000} additional characters for view performance] ..."
                    display_result = t_result[:5000] + truncated_note
                else:
                    display_result = t_result
                    
                tools_list_html.append(f'''
                <details class="tool-card">
                  <summary class="tool-header">
                    <div class="tool-header-left">
                      <span class="tool-icon">{icon}</span>
                      <span class="tool-badge">{html.escape(t_name)}</span>
                      <span class="tool-action-title">{html.escape(display_title)}</span>
                    </div>
                    <div class="tool-header-right">
                      <span class="tool-status-pill">✓ Success</span>
                      <span class="tool-chevron">▾</span>
                    </div>
                  </summary>
                  <div class="tool-body">
                    <div class="tool-pane-section">
                      <div class="tool-pane-label">PARAMETERS / INPUT</div>
                      <pre class="tool-code"><code>{html.escape(args_json)}</code></pre>
                    </div>
                    {f"""<div class="tool-pane-section">
                      <div class="tool-pane-label">EXECUTION OUTPUT</div>
                      <div class="terminal-window">
                        <div class="terminal-bar">
                          <span class="term-dot red"></span>
                          <span class="term-dot yellow"></span>
                          <span class="term-dot green"></span>
                          <span class="term-title">Console Output</span>
                        </div>
                        <pre class="terminal-code"><code>{html.escape(display_result)}</code></pre>
                      </div>
                    </div>""" if display_result else ""}
                  </div>
                </details>''')
                
            tools_block_html = f'''
            <div class="tools-deck">
              <div class="tools-deck-header">
                <span class="tools-deck-label">🔧 Tool Executions ({len(t['tool_calls'])})</span>
              </div>
              {''.join(tools_list_html)}
            </div>'''
            
        # Assistant response block
        assistant_html = ""
        if t['assistant_responses']:
            combined_response = "\n\n".join(t['assistant_responses'])
            assistant_body_html = render_markdown(combined_response)
            assistant_html = f'''
            <div class="agent-response-container">
              <div class="agent-header">
                <div class="agent-avatar">✦</div>
                <div class="agent-meta">
                  <span class="agent-name">Antigravity</span>
                  <span class="model-badge">Gemini 3.7 Flash</span>
                  <span class="agent-time">{created_time}</span>
                </div>
              </div>
              <div class="agent-message-body">
                {assistant_body_html}
              </div>
            </div>'''
            
        feed_cards_html.append(f'''
        <div class="turn-anchor" id="turn-{tid}" data-turn-id="{tid}">
          <!-- Turn Divider -->
          <div class="turn-divider">
            <span class="turn-divider-line"></span>
            <span class="turn-divider-pill">Turn #{tid} &bull; {date_str} {created_time}</span>
            <span class="turn-divider-line"></span>
          </div>

          <!-- User Prompt -->
          <div class="user-message-card">
            <div class="user-header">
              <div class="user-avatar">You</div>
              <div class="user-meta">
                <span class="user-name">Andrew</span>
                <span class="user-time">{created_time}</span>
                {' '.join([f'<span class="user-cmd-pill">{c}</span>' for c in slash_cmds])}
              </div>
            </div>
            <div class="user-content markdown-body">
              {user_body_html}
            </div>
          </div>

          <!-- Agent Activities -->
          <div class="agent-turn-wrapper">
            {thought_block_html}
            {tools_block_html}
            {assistant_html}
          </div>
        </div>''')

    # Build Artifact Drawer items
    artifacts_json = json.dumps([{
        'filename': a['filename'],
        'title': a['title'],
        'content': render_markdown(a['content']),
        'size': f"{a['size'] / 1024:.1f} KB"
    } for a in artifacts])
    
    media_json = json.dumps(media)

    # Full HTML Template
    template = f'''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Antigravity Chat &mdash; Air Raid Alert Defense</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-canvas: #090b10;
      --bg-sidebar: #0f121a;
      --bg-card: #131722;
      --bg-card-hover: #171d2b;
      --bg-user-card: #141926;
      --bg-tool-card: #10141f;
      --bg-cot: #121422;
      --border-subtle: #1e2433;
      --border-accent: #2e3852;
      --border-glow: rgba(99, 102, 241, 0.35);
      
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      
      --accent-primary: #6366f1;
      --accent-primary-hover: #4f46e5;
      --accent-glow: rgba(99, 102, 241, 0.2);
      
      --accent-purple: #a855f7;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      
      --font-ui: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      --font-code: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      
      --header-height: 56px;
      --sidebar-width: 320px;
    }}

    [data-theme="light"] {{
      --bg-canvas: #f8fafc;
      --bg-sidebar: #ffffff;
      --bg-card: #ffffff;
      --bg-card-hover: #f1f5f9;
      --bg-user-card: #f8fafc;
      --bg-tool-card: #f1f5f9;
      --bg-cot: #f5f3ff;
      --border-subtle: #e2e8f0;
      --border-accent: #cbd5e1;
      --border-glow: rgba(99, 102, 241, 0.2);
      
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #94a3b8;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    
    body {{
      font-family: var(--font-ui);
      background-color: var(--bg-canvas);
      color: var(--text-main);
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    /* TOP HEADER */
    .app-header {{
      height: var(--header-height);
      background: var(--bg-sidebar);
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 1.25rem;
      z-index: 100;
      flex-shrink: 0;
    }}

    .header-left {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}

    .logo-container {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-weight: 700;
      font-size: 0.95rem;
      color: var(--text-main);
      letter-spacing: -0.01em;
    }}

    .logo-badge {{
      width: 26px;
      height: 26px;
      border-radius: 7px;
      background: linear-gradient(135deg, #6366f1, #a855f7);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-size: 0.85rem;
      box-shadow: 0 0 12px var(--accent-glow);
    }}

    .workspace-tag {{
      background: rgba(99, 102, 241, 0.12);
      color: var(--accent-primary);
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: 6px;
      padding: 0.2rem 0.55rem;
      font-size: 0.75rem;
      font-family: var(--font-code);
      font-weight: 500;
    }}

    .session-title {{
      font-size: 0.85rem;
      color: var(--text-muted);
      border-left: 1px solid var(--border-subtle);
      padding-left: 0.75rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 320px;
    }}

    .header-center {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .stat-chip {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 9999px;
      padding: 0.2rem 0.65rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }}

    .stat-chip.highlight {{
      color: var(--accent-emerald);
      border-color: rgba(16, 185, 129, 0.3);
      background: rgba(16, 185, 129, 0.08);
    }}

    .header-right {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }}

    .btn {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 6px;
      padding: 0.4rem 0.7rem;
      font-size: 0.78rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.15s ease;
      font-family: var(--font-ui);
    }}

    .btn:hover {{
      background: var(--bg-card-hover);
      color: var(--text-main);
      border-color: var(--border-accent);
    }}

    .btn.active {{
      background: var(--accent-primary);
      color: #fff;
      border-color: var(--accent-primary);
    }}

    .btn-github {{
      background: rgba(255, 255, 255, 0.06);
      color: var(--text-main);
      text-decoration: none;
      border: 1px solid var(--border-accent);
    }}

    .btn-github:hover {{
      background: rgba(255, 255, 255, 0.12);
    }}

    /* MAIN APP LAYOUT (SPLIT PANE) */
    .app-workspace {{
      display: flex;
      flex: 1;
      height: calc(100vh - var(--header-height));
      overflow: hidden;
    }}

    /* SIDEBAR */
    .app-sidebar {{
      width: var(--sidebar-width);
      background: var(--bg-sidebar);
      border-right: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }}

    .sidebar-tabs {{
      display: flex;
      border-bottom: 1px solid var(--border-subtle);
      background: rgba(0, 0, 0, 0.1);
    }}

    .tab-btn {{
      flex: 1;
      padding: 0.65rem 0.5rem;
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.15s;
    }}

    .tab-btn:hover {{
      color: var(--text-main);
    }}

    .tab-btn.active {{
      color: var(--accent-primary);
      border-bottom-color: var(--accent-primary);
      background: rgba(99, 102, 241, 0.05);
    }}

    .sidebar-search {{
      padding: 0.75rem;
      border-bottom: 1px solid var(--border-subtle);
    }}

    .search-field {{
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 0.45rem 0.65rem;
      color: var(--text-main);
      font-size: 0.8rem;
      font-family: var(--font-ui);
      outline: none;
      transition: border-color 0.2s;
    }}

    .search-field:focus {{
      border-color: var(--accent-primary);
    }}

    .sidebar-pane {{
      flex: 1;
      overflow-y: auto;
      display: none;
    }}

    .sidebar-pane.active {{
      display: block;
    }}

    .turn-nav-item {{
      padding: 0.75rem 0.85rem;
      border-bottom: 1px solid var(--border-subtle);
      cursor: pointer;
      transition: background 0.15s;
      border-left: 3px solid transparent;
    }}

    .turn-nav-item:hover {{
      background: var(--bg-card-hover);
    }}

    .turn-nav-item.active {{
      background: var(--bg-card);
      border-left-color: var(--accent-primary);
    }}

    .turn-nav-top {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.25rem;
      font-size: 0.75rem;
    }}

    .turn-nav-badge {{
      font-weight: 700;
      color: var(--accent-primary);
      font-family: var(--font-code);
    }}

    .turn-nav-time {{
      color: var(--text-dim);
      font-size: 0.7rem;
    }}

    .turn-nav-text {{
      font-size: 0.8rem;
      color: var(--text-muted);
      line-height: 1.35;
      margin-bottom: 0.4rem;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}

    .turn-nav-meta {{
      display: flex;
      gap: 0.35rem;
      flex-wrap: wrap;
    }}

    .meta-tag {{
      font-size: 0.68rem;
      border-radius: 4px;
      padding: 0.1rem 0.35rem;
      font-family: var(--font-code);
      font-weight: 500;
    }}

    .meta-tag.tools-tag {{ background: rgba(6, 182, 212, 0.12); color: var(--accent-cyan); }}
    .meta-tag.cot-tag {{ background: rgba(168, 85, 247, 0.12); color: var(--accent-purple); }}
    .meta-tag.cmd-tag {{ background: rgba(245, 158, 11, 0.12); color: var(--accent-amber); }}

    /* ARTIFACTS LIST */
    .artifact-item {{
      padding: 0.85rem;
      border-bottom: 1px solid var(--border-subtle);
      cursor: pointer;
      transition: background 0.15s;
    }}

    .artifact-item:hover {{
      background: var(--bg-card-hover);
    }}

    .artifact-title {{
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 0.4rem;
      margin-bottom: 0.2rem;
    }}

    .artifact-meta {{
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-code);
    }}

    /* GALLERY LIST */
    .gallery-grid {{
      padding: 0.75rem;
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.75rem;
    }}

    .gallery-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      overflow: hidden;
      cursor: pointer;
      transition: transform 0.15s, border-color 0.2s;
    }}

    .gallery-card:hover {{
      transform: translateY(-2px);
      border-color: var(--accent-primary);
    }}

    .gallery-thumb {{
      width: 100%;
      height: 120px;
      object-fit: cover;
      display: block;
      background: #000;
    }}

    .gallery-label {{
      padding: 0.4rem 0.6rem;
      font-size: 0.75rem;
      color: var(--text-muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    /* CHAT STREAM (CENTER) */
    .chat-container {{
      flex: 1;
      overflow-y: auto;
      padding: 1.5rem 2rem 5rem 2rem;
      scroll-behavior: smooth;
    }}

    .chat-feed {{
      max-width: 900px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 2.5rem;
    }}

    .turn-anchor {{
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      scroll-margin-top: 1.5rem;
    }}

    .turn-divider {{
      display: flex;
      align-items: center;
      gap: 1rem;
      margin: 0.5rem 0;
    }}

    .turn-divider-line {{
      flex: 1;
      height: 1px;
      background: var(--border-subtle);
    }}

    .turn-divider-pill {{
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-code);
      background: var(--bg-sidebar);
      border: 1px solid var(--border-subtle);
      padding: 0.15rem 0.6rem;
      border-radius: 9999px;
    }}

    /* USER MESSAGE CARD */
    .user-message-card {{
      background: var(--bg-user-card);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 1.15rem 1.35rem;
      box-shadow: 0 4px 14px rgba(0,0,0,0.15);
      border-left: 4px solid #3b82f6;
    }}

    .user-header {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
      margin-bottom: 0.75rem;
    }}

    .user-avatar {{
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: linear-gradient(135deg, #2563eb, #3b82f6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.7rem;
      font-weight: 700;
      color: #fff;
    }}

    .user-name {{
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-main);
    }}

    .user-time {{
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-code);
    }}

    .user-cmd-pill {{
      background: rgba(245, 158, 11, 0.12);
      color: var(--accent-amber);
      border: 1px solid rgba(245, 158, 11, 0.25);
      border-radius: 4px;
      padding: 0.1rem 0.4rem;
      font-size: 0.7rem;
      font-family: var(--font-code);
      font-weight: 600;
    }}

    .user-content {{
      font-size: 0.92rem;
      line-height: 1.6;
      color: var(--text-main);
    }}

    /* AGENT TURN WRAPPER */
    .agent-turn-wrapper {{
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}

    /* COT (CHAIN OF THOUGHT) ACCORDION */
    .cot-container {{
      border-radius: 8px;
      overflow: hidden;
    }}

    .cot-details {{
      background: var(--bg-cot);
      border: 1px solid rgba(168, 85, 247, 0.2);
      border-radius: 8px;
    }}

    .cot-summary {{
      padding: 0.6rem 0.9rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      user-select: none;
      transition: background 0.15s;
    }}

    .cot-summary:hover {{
      background: rgba(168, 85, 247, 0.08);
    }}

    .cot-summary-left {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .cot-icon {{ font-size: 0.95rem; }}

    .cot-title {{
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--accent-purple);
    }}

    .cot-meta {{
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-code);
    }}

    .cot-chevron {{
      font-size: 0.8rem;
      color: var(--accent-purple);
      transition: transform 0.2s;
    }}

    .cot-details[open] .cot-chevron {{
      transform: rotate(180deg);
    }}

    .cot-body {{
      padding: 0.85rem 1rem;
      border-top: 1px solid rgba(168, 85, 247, 0.15);
      background: rgba(0, 0, 0, 0.15);
      max-height: 380px;
      overflow-y: auto;
    }}

    .cot-text {{
      font-family: var(--font-code);
      font-size: 0.78rem;
      color: var(--text-muted);
      white-space: pre-wrap;
      line-height: 1.55;
    }}

    /* TOOLS DECK */
    .tools-deck {{
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }}

    .tools-deck-header {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      font-weight: 700;
      padding-left: 0.2rem;
    }}

    .tool-card {{
      background: var(--bg-tool-card);
      border: 1px solid var(--border-subtle);
      border-radius: 7px;
      overflow: hidden;
      transition: border-color 0.15s;
    }}

    .tool-card:hover {{
      border-color: var(--border-accent);
    }}

    .tool-header {{
      padding: 0.5rem 0.8rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      user-select: none;
      background: rgba(0, 0, 0, 0.1);
    }}

    .tool-header-left {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .tool-badge {{
      background: rgba(6, 182, 212, 0.15);
      color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.25);
      padding: 0.12rem 0.4rem;
      border-radius: 4px;
      font-family: var(--font-code);
      font-size: 0.72rem;
      font-weight: 600;
    }}

    .tool-action-title {{
      font-size: 0.8rem;
      color: var(--text-main);
      font-weight: 500;
    }}

    .tool-header-right {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .tool-status-pill {{
      font-size: 0.68rem;
      color: var(--accent-emerald);
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.25);
      padding: 0.1rem 0.4rem;
      border-radius: 9999px;
      font-weight: 500;
    }}

    .tool-chevron {{
      font-size: 0.75rem;
      color: var(--text-dim);
      transition: transform 0.2s;
    }}

    .tool-card[open] .tool-chevron {{
      transform: rotate(180deg);
    }}

    .tool-body {{
      padding: 0.85rem;
      border-top: 1px solid var(--border-subtle);
      background: rgba(0, 0, 0, 0.2);
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }}

    .tool-pane-section {{
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }}

    .tool-pane-label {{
      font-size: 0.68rem;
      font-family: var(--font-code);
      font-weight: 700;
      color: var(--text-dim);
      letter-spacing: 0.05em;
    }}

    .tool-code {{
      background: var(--bg-canvas);
      border: 1px solid var(--border-subtle);
      border-radius: 5px;
      padding: 0.5rem 0.75rem;
      font-family: var(--font-code);
      font-size: 0.75rem;
      color: var(--text-muted);
      overflow-x: auto;
      max-height: 250px;
    }}

    /* TERMINAL WINDOW SIMULATION */
    .terminal-window {{
      background: #080a0f;
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      overflow: hidden;
    }}

    .terminal-bar {{
      background: #11141d;
      padding: 0.35rem 0.65rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      border-bottom: 1px solid var(--border-subtle);
    }}

    .term-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }}

    .term-dot.red {{ background: #ef4444; }}
    .term-dot.yellow {{ background: #f59e0b; }}
    .term-dot.green {{ background: #10b981; }}

    .term-title {{
      font-size: 0.68rem;
      font-family: var(--font-code);
      color: var(--text-dim);
      margin-left: 0.25rem;
    }}

    .terminal-code {{
      padding: 0.65rem 0.85rem;
      font-family: var(--font-code);
      font-size: 0.75rem;
      color: #94a3b8;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 350px;
      overflow-y: auto;
      line-height: 1.45;
    }}

    /* AGENT RESPONSE CONTAINER */
    .agent-response-container {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 1.35rem 1.6rem;
      box-shadow: 0 4px 16px rgba(0,0,0,0.15);
      border-left: 4px solid var(--accent-primary);
    }}

    .agent-header {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
      margin-bottom: 1rem;
    }}

    .agent-avatar {{
      width: 26px;
      height: 26px;
      border-radius: 50%;
      background: linear-gradient(135deg, #6366f1, #a855f7);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.8rem;
      color: #fff;
      box-shadow: 0 0 10px var(--accent-glow);
    }}

    .agent-meta {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .agent-name {{
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--text-main);
    }}

    .model-badge {{
      background: rgba(99, 102, 241, 0.12);
      color: var(--accent-primary);
      border: 1px solid rgba(99, 102, 241, 0.25);
      border-radius: 4px;
      padding: 0.1rem 0.45rem;
      font-size: 0.7rem;
      font-family: var(--font-code);
      font-weight: 500;
    }}

    .agent-time {{
      font-size: 0.72rem;
      color: var(--text-dim);
      font-family: var(--font-code);
    }}

    /* MARKDOWN STYLING */
    .markdown-body {{
      font-size: 0.92rem;
      line-height: 1.68;
      color: var(--text-main);
    }}

    .markdown-body p {{
      margin-bottom: 0.85rem;
    }}

    .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4 {{
      color: var(--text-main);
      font-weight: 700;
      margin: 1.25rem 0 0.6rem 0;
      letter-spacing: -0.01em;
    }}

    .markdown-body h1 {{ font-size: 1.35rem; }}
    .markdown-body h2 {{ font-size: 1.18rem; border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.35rem; }}
    .markdown-body h3 {{ font-size: 1.05rem; }}
    .markdown-body h4 {{ font-size: 0.95rem; }}

    .inline-code {{
      font-family: var(--font-code);
      font-size: 0.82em;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 0.12rem 0.35rem;
      border-radius: 4px;
      color: #38bdf8;
    }}

    /* CODE BLOCKS */
    .code-block-wrapper {{
      background: #080a0f;
      border: 1px solid var(--border-subtle);
      border-radius: 7px;
      margin: 0.9rem 0;
      overflow: hidden;
    }}

    .code-block-header {{
      background: #11141e;
      padding: 0.35rem 0.75rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-subtle);
    }}

    .code-lang {{
      font-family: var(--font-code);
      font-size: 0.7rem;
      color: var(--text-dim);
      text-transform: uppercase;
      font-weight: 600;
    }}

    .copy-btn {{
      background: transparent;
      border: 1px solid var(--border-subtle);
      color: var(--text-dim);
      border-radius: 4px;
      padding: 0.15rem 0.45rem;
      font-size: 0.68rem;
      cursor: pointer;
      transition: all 0.15s;
    }}

    .copy-btn:hover {{
      color: var(--text-main);
      background: var(--bg-card);
    }}

    .code-content {{
      padding: 0.85rem 1rem;
      font-family: var(--font-code);
      font-size: 0.8rem;
      line-height: 1.5;
      color: #cbd5e1;
      overflow-x: auto;
    }}

    /* DIFF STYLING */
    .diff-add {{ display: block; background: rgba(34, 197, 94, 0.15); color: #4ade80; }}
    .diff-sub {{ display: block; background: rgba(239, 68, 68, 0.15); color: #f87171; }}
    .diff-info {{ display: block; color: #38bdf8; font-weight: 600; }}
    .diff-ctx {{ display: block; color: var(--text-dim); }}

    /* CALLOUT ALERTS */
    .callout {{
      border-radius: 7px;
      margin: 1rem 0;
      overflow: hidden;
      border: 1px solid;
    }}

    .callout-header {{
      padding: 0.45rem 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
    }}

    .callout-body {{
      padding: 0.65rem 0.85rem;
      font-size: 0.86rem;
      line-height: 1.55;
    }}

    .callout-note {{ border-color: rgba(56, 189, 248, 0.3); background: rgba(56, 189, 248, 0.05); }}
    .callout-note .callout-header {{ background: rgba(56, 189, 248, 0.12); color: #38bdf8; }}

    .callout-tip {{ border-color: rgba(34, 197, 94, 0.3); background: rgba(34, 197, 94, 0.05); }}
    .callout-tip .callout-header {{ background: rgba(34, 197, 94, 0.12); color: #4ade80; }}

    .callout-important {{ border-color: rgba(168, 85, 247, 0.3); background: rgba(168, 85, 247, 0.05); }}
    .callout-important .callout-header {{ background: rgba(168, 85, 247, 0.12); color: #c084fc; }}

    .callout-warning {{ border-color: rgba(245, 158, 11, 0.3); background: rgba(245, 158, 11, 0.05); }}
    .callout-warning .callout-header {{ background: rgba(245, 158, 11, 0.12); color: #fbbf24; }}

    /* TABLES */
    .table-container {{
      overflow-x: auto;
      margin: 1rem 0;
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      text-align: left;
    }}

    .data-table th {{
      background: var(--bg-card);
      padding: 0.55rem 0.85rem;
      font-weight: 600;
      color: var(--text-main);
      border-bottom: 1px solid var(--border-subtle);
    }}

    .data-table td {{
      padding: 0.5rem 0.85rem;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-muted);
    }}

    .data-table tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}

    /* ARTIFACT DRAWER MODAL */
    .modal-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(4px);
      z-index: 200;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }}

    .modal-overlay.active {{
      display: flex;
    }}

    .modal-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-accent);
      border-radius: 10px;
      width: 100%;
      max-width: 900px;
      max-height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
      overflow: hidden;
    }}

    .modal-header {{
      padding: 1rem 1.25rem;
      background: var(--bg-sidebar);
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}

    .modal-title {{
      font-size: 1rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .modal-close {{
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-size: 1.25rem;
      cursor: pointer;
    }}

    .modal-close:hover {{
      color: var(--text-main);
    }}

    .modal-body {{
      padding: 1.5rem;
      overflow-y: auto;
      font-size: 0.92rem;
    }}

    /* RESPONSIVE */
    @media (max-width: 900px) {{
      .app-sidebar {{ display: none; }}
      .chat-container {{ padding: 1rem; }}
    }}
  </style>
</head>
<body>

  <!-- TOP APP HEADER -->
  <header class="app-header">
    <div class="header-left">
      <div class="logo-container">
        <div class="logo-badge">✦</div>
        <span>Antigravity</span>
      </div>
      <span class="workspace-tag">AgenticAlert</span>
      <div class="session-title">Time Series Alert Defense STGNN</div>
    </div>

    <div class="header-center">
      <div class="stat-chip highlight">● Session Archived</div>
      <div class="stat-chip">Turns: <strong>{len(turns)}</strong></div>
      <div class="stat-chip">Steps: <strong>{total_steps:,}</strong></div>
      <div class="stat-chip">Tools: <strong>{total_tools:,}</strong></div>
    </div>

    <div class="header-right">
      <button class="btn" id="btnToggleThoughts" onclick="toggleAllThoughts()">🧠 Thoughts</button>
      <button class="btn" id="btnToggleTools" onclick="toggleAllTools()">🔧 Tools</button>
      <button class="btn" id="btnTheme" onclick="toggleTheme()">🌓 Theme</button>
      <a class="btn btn-github" href="https://github.com/TiDatm/AgenticAlert" target="_blank">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
        GitHub
      </a>
    </div>
  </header>

  <!-- WORKSPACE SPLIT-PANE -->
  <div class="app-workspace">

    <!-- LEFT SIDEBAR -->
    <aside class="app-sidebar">
      <div class="sidebar-tabs">
        <button class="tab-btn active" id="tabTurns" onclick="switchSidebarTab('turns')">Turns ({len(turns)})</button>
        <button class="tab-btn" id="tabArtifacts" onclick="switchSidebarTab('artifacts')">Artifacts ({len(artifacts)})</button>
        <button class="tab-btn" id="tabGallery" onclick="switchSidebarTab('gallery')">Gallery ({len(media)})</button>
      </div>

      <div class="sidebar-search">
        <input type="text" class="search-field" id="searchInput" placeholder="Filter prompts, code, tools..." />
      </div>

      <!-- TURNS PANE -->
      <div class="sidebar-pane active" id="paneTurns">
        {''.join(sidebar_items_html)}
      </div>

      <!-- ARTIFACTS PANE -->
      <div class="sidebar-pane" id="paneArtifacts">
        {''.join([f"""<div class="artifact-item" onclick="openArtifactModal('{a['filename']}')">
          <div class="artifact-title">📑 {html.escape(a['title'])}</div>
          <div class="artifact-meta">{a['filename']} &bull; {a['size'] / 1024:.1f} KB</div>
        </div>""" for a in artifacts])}
      </div>

      <!-- GALLERY PANE -->
      <div class="sidebar-pane" id="paneGallery">
        <div class="gallery-grid">
          {''.join([f"""<div class="gallery-card" onclick="openLightbox('{m['path']}', '{html.escape(m['title'])}')">
            <img class="gallery-thumb" src="{m['path']}" alt="{m['title']}" loading="lazy" />
            <div class="gallery-label">{html.escape(m['title'])}</div>
          </div>""" for m in media])}
        </div>
      </div>
    </aside>

    <!-- CENTER CHAT STREAM -->
    <main class="chat-container" id="chatContainer">
      <div class="chat-feed" id="chatFeed">
        {''.join(feed_cards_html)}
      </div>
    </main>

  </div>

  <!-- ARTIFACT MODAL -->
  <div class="modal-overlay" id="artifactModal" onclick="closeModalOnBg(event)">
    <div class="modal-card">
      <div class="modal-header">
        <div class="modal-title" id="artifactModalTitle">📑 Artifact</div>
        <button class="modal-close" onclick="closeArtifactModal()">&times;</button>
      </div>
      <div class="modal-body markdown-body" id="artifactModalBody"></div>
    </div>
  </div>

  <!-- LIGHTBOX MODAL -->
  <div class="modal-overlay" id="lightboxModal" onclick="closeLightboxOnBg(event)">
    <div class="modal-card" style="max-width: 1000px;">
      <div class="modal-header">
        <div class="modal-title" id="lightboxTitle">Image</div>
        <button class="modal-close" onclick="closeLightbox()">&times;</button>
      </div>
      <div class="modal-body" style="text-align: center; padding: 1rem;">
        <img id="lightboxImg" src="" style="max-width: 100%; max-height: 70vh; border-radius: 6px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);" />
      </div>
    </div>
  </div>

  <!-- EMBEDDED DATA -->
  <script>
    const ARTIFACTS = {artifacts_json};
    const MEDIA = {media_json};

    // Sidebar tab switching
    function switchSidebarTab(tabName) {{
      document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
      document.querySelectorAll('.sidebar-pane').forEach(p => p.classList.remove('active'));
      
      if (tabName === 'turns') {{
        document.getElementById('tabTurns').classList.add('active');
        document.getElementById('paneTurns').classList.add('active');
      }} else if (tabName === 'artifacts') {{
        document.getElementById('tabArtifacts').classList.add('active');
        document.getElementById('paneArtifacts').classList.add('active');
      }} else if (tabName === 'gallery') {{
        document.getElementById('tabGallery').classList.add('active');
        document.getElementById('paneGallery').classList.add('active');
      }}
    }}

    // Scroll to turn with highlight
    function scrollToTurn(tid) {{
      const el = document.getElementById('turn-' + tid);
      if (el) {{
        el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        
        document.querySelectorAll('.turn-nav-item').forEach(i => i.classList.remove('active'));
        const navItem = document.getElementById('nav-turn-' + tid);
        if (navItem) navItem.classList.add('active');
      }}
    }}

    // Toggle All Thoughts
    let thoughtsOpen = false;
    function toggleAllThoughts() {{
      thoughtsOpen = !thoughtsOpen;
      document.querySelectorAll('.cot-details').forEach(el => el.open = thoughtsOpen);
      document.getElementById('btnToggleThoughts').classList.toggle('active', thoughtsOpen);
    }}

    // Toggle All Tools
    let toolsOpen = false;
    function toggleAllTools() {{
      toolsOpen = !toolsOpen;
      document.querySelectorAll('.tool-card').forEach(el => el.open = toolsOpen);
      document.getElementById('btnToggleTools').classList.toggle('active', toolsOpen);
    }}

    // Theme Toggle
    function toggleTheme() {{
      const html = document.documentElement;
      const current = html.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      localStorage.setItem('antigravity_theme', next);
    }}

    const savedTheme = localStorage.getItem('antigravity_theme');
    if (savedTheme) {{
      document.documentElement.setAttribute('data-theme', savedTheme);
    }}

    // Real-time Search Filter
    const searchInput = document.getElementById('searchInput');
    searchInput.addEventListener('input', function(e) {{
      const query = e.target.value.toLowerCase().trim();
      const turns = document.querySelectorAll('.turn-anchor');
      const navItems = document.querySelectorAll('.turn-nav-item');
      
      turns.forEach((turn, idx) => {{
        const text = turn.textContent.toLowerCase();
        const nav = navItems[idx];
        if (!query || text.includes(query)) {{
          turn.style.display = 'flex';
          if (nav) nav.style.display = 'block';
        }} else {{
          turn.style.display = 'none';
          if (nav) nav.style.display = 'none';
        }}
      }});
    }});

    // Artifact Modal
    function openArtifactModal(fname) {{
      const art = ARTIFACTS.find(a => a.filename === fname);
      if (art) {{
        document.getElementById('artifactModalTitle').innerHTML = '📑 ' + art.title;
        document.getElementById('artifactModalBody').innerHTML = art.content;
        document.getElementById('artifactModal').classList.add('active');
      }}
    }}

    function closeArtifactModal() {{
      document.getElementById('artifactModal').classList.remove('active');
    }}

    function closeModalOnBg(e) {{
      if (e.target.id === 'artifactModal') closeArtifactModal();
    }}

    // Lightbox Modal
    function openLightbox(src, title) {{
      document.getElementById('lightboxTitle').textContent = title;
      document.getElementById('lightboxImg').src = src;
      document.getElementById('lightboxModal').classList.add('active');
    }}

    function closeLightbox() {{
      document.getElementById('lightboxModal').classList.remove('active');
    }}

    function closeLightboxOnBg(e) {{
      if (e.target.id === 'lightboxModal') closeLightbox();
    }}

    // Copy Snippet
    function copySnippet(btn) {{
      const code = btn.closest('.code-block-wrapper').querySelector('code').textContent;
      navigator.clipboard.writeText(code).then(() => {{
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = orig, 1500);
      }});
    }}
  </script>
</body>
</html>
'''
    return template

def main():
    repo_root = Path(__file__).resolve().parent.parent
    transcript_path = repo_root / "brain" / "logs" / "transcript_full.jsonl"
    if not transcript_path.exists():
        transcript_path = repo_root / "brain" / "logs" / "transcript.jsonl"
        
    print(f"Loading transcript from {transcript_path}...")
    turns, total_steps, total_tools = parse_turn_data(str(transcript_path))
    print(f"Parsed {len(turns)} conversation turns, {total_steps} total steps, {total_tools} tool calls.")
    
    print("Loading artifacts and visual media...")
    artifacts, media = load_artifacts_and_media(repo_root)
    print(f"Loaded {len(artifacts)} markdown artifacts and {len(media)} media files.")
    
    conv_id = "6e0c7109-5304-44a3-8c3d-a42d1e47513c"
    
    html_content = build_native_antigravity_html(turns, total_steps, total_tools, artifacts, media, conv_id)
    
    # 1. Output to docs/index.html (GitHub Pages)
    docs_index = repo_root / "docs" / "index.html"
    docs_index.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_index, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Written: {docs_index} ({os.path.getsize(docs_index)/(1024*1024):.2f} MB)")
    
    # 2. Output copy to brain/conversation_viewer.html
    brain_viewer = repo_root / "brain" / "conversation_viewer.html"
    with open(brain_viewer, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Written: {brain_viewer} ({os.path.getsize(brain_viewer)/(1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
