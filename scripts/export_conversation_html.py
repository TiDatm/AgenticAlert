"""
Interactive HTML Conversation Brain Exporter
Converts Antigravity transcript JSONL into a standalone, interactive, searchable HTML viewer.
"""

import json
import html
import re
import os
import sys
from pathlib import Path
from datetime import datetime

def parse_markdown_simple(text: str) -> str:
    if not text:
        return ""
    
    # Escape HTML special chars first
    escaped = html.escape(text)
    
    # Code blocks ```lang ... ```
    def replace_code_block(match):
        code = match.group(2)
        lang = match.group(1) or ""
        return f'<pre><code class="language-{lang}">{code}</code></pre>'
    
    escaped = re.sub(r'```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```', replace_code_block, escaped)
    
    # Inline code `code`
    escaped = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', escaped)
    
    # Bold **text**
    escaped = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', escaped)
    
    # Italic *text*
    escaped = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', escaped)
    
    # Headers # H1, ## H2, ### H3
    escaped = re.sub(r'^### (.*)$', r'<h3>\1</h3>', escaped, flags=re.MULTILINE)
    escaped = re.sub(r'^## (.*)$', r'<h2>\1</h2>', escaped, flags=re.MULTILINE)
    escaped = re.sub(r'^# (.*)$', r'<h1>\1</h1>', escaped, flags=re.MULTILINE)
    
    # Blockquotes > text
    escaped = re.sub(r'^> (.*)$', r'<blockquote>\1</blockquote>', escaped, flags=re.MULTILINE)
    
    # Lists
    escaped = re.sub(r'^\* (.*)$', r'<li>\1</li>', escaped, flags=re.MULTILINE)
    escaped = re.sub(r'^- (.*)$', r'<li>\1</li>', escaped, flags=re.MULTILINE)
    
    # Convert newlines to <br> unless inside <pre>
    parts = escaped.split('<pre>')
    new_parts = [parts[0].replace('\n\n', '<p></p>').replace('\n', '<br>')]
    for p in parts[1:]:
        sub = p.split('</pre>')
        new_parts.append(sub[0] + '</pre>' + sub[1].replace('\n\n', '<p></p>').replace('\n', '<br>'))
    return '<pre>'.join(new_parts)

def load_transcript(transcript_path: str):
    turns = []
    current_turn = None
    step_count = 0
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            step = json.loads(line)
            step_count += 1
            stype = step.get('type')
            
            if stype == 'USER_INPUT':
                if current_turn is not None:
                    turns.append(current_turn)
                current_turn = {
                    'turn_id': len(turns) + 1,
                    'created_at': step.get('created_at', ''),
                    'user_prompt': step.get('content', ''),
                    'thoughts': [],
                    'tool_calls': [],
                    'assistant_responses': [],
                }
            elif current_turn is not None:
                # Capture thoughts
                thinking = step.get('thinking')
                if thinking and thinking.strip():
                    current_turn['thoughts'].append({
                        'step_index': step.get('step_index'),
                        'text': thinking.strip()
                    })
                
                # Capture tool calls
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
                        current_turn['tool_calls'].append({
                            'step_index': step.get('step_index'),
                            'name': name,
                            'args': args,
                            'result': ''
                        })
                
                # Capture tool outputs (GENERIC type)
                if stype == 'GENERIC':
                    content = step.get('content', '')
                    if current_turn['tool_calls']:
                        # Attach result to most recent tool call without result
                        for tc in reversed(current_turn['tool_calls']):
                            if not tc['result']:
                                tc['result'] = content
                                break
                    elif content:
                        current_turn['assistant_responses'].append(content)
                        
                # Capture model assistant responses
                if stype == 'PLANNER_RESPONSE':
                    content = step.get('content')
                    if content and content.strip():
                        current_turn['assistant_responses'].append(content.strip())
                        
                # Capture system messages if noteworthy
                if stype == 'SYSTEM_MESSAGE':
                    content = step.get('content', '')
                    if 'error' in content.lower() or 'failed' in content.lower():
                        current_turn['assistant_responses'].append(f"> [SYSTEM ALERT] {content}")
                        
    if current_turn is not None:
        turns.append(current_turn)
        
    return turns, step_count

def generate_html(turns, total_steps: int, conv_id: str, out_path: str):
    html_lines = []
    
    html_lines.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Conversation Brain Viewer &mdash; {conv_id[:8]}</title>
  <style>
    :root {{
      --bg-primary: #0d1117;
      --bg-secondary: #161b22;
      --bg-tertiary: #21262d;
      --border-color: #30363d;
      --text-primary: #c9d1d9;
      --text-muted: #8b949e;
      --accent-blue: #58a6ff;
      --accent-purple: #bc8cff;
      --accent-green: #3fb950;
      --accent-orange: #d29922;
      --accent-red: #f85149;
      --card-user: #1f2937;
      --card-thought: #1a2234;
      --card-tool: #1c2128;
      --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    
    body.light {{
      --bg-primary: #f6f8fa;
      --bg-secondary: #ffffff;
      --bg-tertiary: #eaeef2;
      --border-color: #d0d7de;
      --text-primary: #24292f;
      --text-muted: #57606a;
      --accent-blue: #0969da;
      --accent-purple: #8250df;
      --accent-green: #1a7f37;
      --accent-orange: #9a6700;
      --accent-red: #cf222e;
      --card-user: #eef2ff;
      --card-thought: #f3f0ff;
      --card-tool: #f6f8fa;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: var(--font-sans);
      background-color: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.6;
      padding: 0;
    }}

    header {{
      background-color: var(--bg-secondary);
      border-bottom: 1px solid var(--border-color);
      padding: 1.25rem 2rem;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }}

    .header-content {{
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
    }}

    .title-area h1 {{
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--accent-blue);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}

    .badges {{
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-top: 0.4rem;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      background-color: var(--bg-tertiary);
      border: 1px solid var(--border-color);
      border-radius: 9999px;
      padding: 0.2rem 0.6rem;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      color: var(--text-muted);
    }}

    .badge.highlight {{
      color: var(--accent-green);
      border-color: rgba(63, 185, 80, 0.4);
    }}

    .controls {{
      display: flex;
      gap: 0.75rem;
      align-items: center;
      flex-wrap: wrap;
    }}

    .search-input {{
      background: var(--bg-primary);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 0.45rem 0.75rem;
      color: var(--text-primary);
      font-size: 0.85rem;
      width: 240px;
      outline: none;
      transition: border-color 0.2s, width 0.2s;
    }}

    .search-input:focus {{
      border-color: var(--accent-blue);
      width: 300px;
    }}

    .btn {{
      background: var(--bg-tertiary);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      border-radius: 6px;
      padding: 0.45rem 0.75rem;
      font-size: 0.8rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.15s ease;
    }}

    .btn:hover {{
      background: var(--border-color);
    }}

    .btn.active {{
      background: var(--accent-blue);
      color: #fff;
      border-color: var(--accent-blue);
    }}

    .container {{
      max-width: 1200px;
      margin: 1.5rem auto;
      padding: 0 1.5rem;
    }}

    .stats-bar {{
      display: flex;
      gap: 1rem;
      margin-bottom: 1.5rem;
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 0.75rem 1.25rem;
      font-size: 0.85rem;
      justify-content: space-between;
      align-items: center;
    }}

    .feed {{
      display: flex;
      flex-direction: column;
      gap: 2rem;
    }}

    .turn-card {{
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      transition: transform 0.15s, border-color 0.2s;
    }}

    .turn-card.highlight {{
      border-color: var(--accent-blue);
    }}

    .turn-header {{
      background: var(--bg-tertiary);
      padding: 0.6rem 1.25rem;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.8rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }}

    .turn-number {{
      font-weight: 700;
      color: var(--accent-blue);
      font-size: 0.9rem;
    }}

    .turn-body {{
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}

    .user-box {{
      background: var(--card-user);
      border-left: 4px solid var(--accent-blue);
      border-radius: 6px;
      padding: 1rem 1.25rem;
      font-size: 0.95rem;
    }}

    .user-label {{
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--accent-blue);
      margin-bottom: 0.4rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}

    .thought-details {{
      background: var(--card-thought);
      border: 1px solid rgba(188, 140, 255, 0.25);
      border-radius: 6px;
      margin-top: 0.5rem;
      overflow: hidden;
    }}

    .thought-details summary {{
      padding: 0.5rem 0.85rem;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--accent-purple);
      cursor: pointer;
      user-select: none;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}

    .thought-details summary:hover {{
      background: rgba(188, 140, 255, 0.08);
    }}

    .thought-content {{
      padding: 0.75rem 1rem;
      font-size: 0.82rem;
      border-top: 1px solid rgba(188, 140, 255, 0.15);
      color: var(--text-muted);
      white-space: pre-wrap;
      font-family: var(--font-mono);
      max-height: 400px;
      overflow-y: auto;
    }}

    .tools-container {{
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      margin-top: 0.5rem;
    }}

    .tool-details {{
      background: var(--card-tool);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      overflow: hidden;
    }}

    .tool-details summary {{
      padding: 0.45rem 0.75rem;
      font-size: 0.8rem;
      font-family: var(--font-mono);
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      user-select: none;
    }}

    .tool-badge {{
      background: rgba(88, 166, 255, 0.15);
      color: var(--accent-blue);
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-weight: 600;
    }}

    .tool-body {{
      padding: 0.75rem 1rem;
      border-top: 1px solid var(--border-color);
      font-size: 0.8rem;
      font-family: var(--font-mono);
      background: var(--bg-primary);
    }}

    .tool-body pre {{
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 350px;
      overflow-y: auto;
      padding: 0.5rem;
      background: var(--bg-secondary);
      border-radius: 4px;
      border: 1px solid var(--border-color);
    }}

    .assistant-box {{
      background: var(--bg-secondary);
      padding: 0.75rem 0;
      font-size: 0.95rem;
      line-height: 1.65;
    }}

    .assistant-label {{
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--accent-green);
      margin-bottom: 0.6rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }}

    pre {{
      background: var(--bg-primary);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 0.75rem;
      overflow-x: auto;
      margin: 0.75rem 0;
      font-family: var(--font-mono);
      font-size: 0.85rem;
    }}

    code {{
      font-family: var(--font-mono);
      background: rgba(110, 118, 129, 0.2);
      padding: 0.15rem 0.35rem;
      border-radius: 4px;
      font-size: 0.85em;
    }}

    pre code {{
      background: transparent;
      padding: 0;
    }}

    blockquote {{
      border-left: 3px solid var(--border-color);
      padding-left: 1rem;
      color: var(--text-muted);
      margin: 0.75rem 0;
    }}

    h1, h2, h3, h4 {{
      margin: 1rem 0 0.5rem 0;
      color: var(--accent-blue);
    }}

    ul, ol {{
      margin-left: 1.5rem;
      margin-bottom: 0.75rem;
    }}

    li {{
      margin-bottom: 0.25rem;
    }}

    .jump-selector {{
      background: var(--bg-primary);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      border-radius: 6px;
      padding: 0.45rem;
      font-size: 0.8rem;
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-content">
      <div class="title-area">
        <h1><span>🛡️</span> Air Raid Defense Forecasting: Conversation Brain</h1>
        <div class="badges">
          <span class="badge highlight">● Interactive Archive</span>
          <span class="badge">Session: {conv_id}</span>
          <span class="badge">Turns: {len(turns)}</span>
          <span class="badge">Total Steps: {total_steps}</span>
        </div>
      </div>
      <div class="controls">
        <input type="text" id="searchInput" class="search-input" placeholder="Search prompts, tools, output..." />
        <select id="jumpSelector" class="jump-selector">
          <option value="">Jump to Turn...</option>
""")
    
    for t in turns:
        t_id = t['turn_id']
        snippet = html.escape(t['user_prompt'][:40].replace('\n', ' '))
        html_lines.append(f'          <option value="turn-{t_id}">Turn {t_id}: {snippet}...</option>')
        
    html_lines.append(f"""        </select>
        <button id="toggleThoughtsBtn" class="btn" onclick="toggleAllThoughts()">🧠 Thoughts</button>
        <button id="toggleToolsBtn" class="btn" onclick="toggleAllTools()">🔧 Tools</button>
        <button id="themeBtn" class="btn" onclick="toggleTheme()">🌓 Theme</button>
      </div>
    </div>
  </header>

  <main class="container">
    <div class="stats-bar">
      <div><strong>{len(turns)}</strong> conversation exchanges documented from start to complete hierarchical defense model verification.</div>
      <div id="matchCount" style="color: var(--text-muted); font-size: 0.8rem;"></div>
    </div>

    <div class="feed" id="conversationFeed">
""")

    for t in turns:
        t_id = t['turn_id']
        created_at = t['created_at'].replace('T', ' ').replace('Z', '')
        user_prompt_html = parse_markdown_simple(t['user_prompt'])
        
        # Tools count and thoughts count
        num_tools = len(t['tool_calls'])
        num_thoughts = len(t['thoughts'])
        
        html_lines.append(f"""      <div class="turn-card" id="turn-{t_id}" data-turn-id="{t_id}">
        <div class="turn-header">
          <span class="turn-number">Turn #{t_id}</span>
          <span>{created_at} &bull; {num_tools} tool calls &bull; {num_thoughts} reasoning steps</span>
        </div>
        <div class="turn-body">
          <div class="user-box">
            <div class="user-label">👤 User Request</div>
            <div class="user-prompt">{user_prompt_html}</div>
          </div>
""")

        # Thoughts accordion if present
        if t['thoughts']:
            all_thoughts_text = "\n\n---\n\n".join([f"[{th.get('step_index', '')}]: {th['text']}" for th in t['thoughts']])
            escaped_thought = html.escape(all_thoughts_text)
            html_lines.append(f"""          <details class="thought-details">
            <summary>🧠 Assistant Reasoning & Strategy ({num_thoughts} steps, {len(all_thoughts_text)} chars)</summary>
            <div class="thought-content">{escaped_thought}</div>
          </details>
""")

        # Tools executed
        if t['tool_calls']:
            html_lines.append("""          <div class="tools-container">""")
            for tc in t['tool_calls']:
                name = tc.get('name', 'tool')
                args_str = json.dumps(tc.get('args', {}), indent=2)
                result_str = tc.get('result', '')
                
                # Truncate very large outputs to 4000 chars for snappy browser rendering
                if len(result_str) > 4000:
                    result_disp = result_str[:4000] + "\n... [truncated for browser performance] ..."
                else:
                    result_disp = result_str
                    
                summary = tc.get('args', {}).get('toolSummary', '') if isinstance(tc.get('args'), dict) else ''
                summary_label = f" &mdash; <em>{html.escape(summary)}</em>" if summary else ""
                
                html_lines.append(f"""            <details class="tool-details">
              <summary><span class="tool-badge">{html.escape(name)}</span>{summary_label}</summary>
              <div class="tool-body">
                <strong>Arguments:</strong>
                <pre>{html.escape(args_str)}</pre>
                <strong>Execution Output:</strong>
                <pre>{html.escape(result_disp)}</pre>
              </div>
            </details>""")
            html_lines.append("""          </div>""")

        # Assistant responses
        if t['assistant_responses']:
            combined_response = "\n\n".join(t['assistant_responses'])
            asst_html = parse_markdown_simple(combined_response)
            html_lines.append(f"""          <div class="assistant-box">
            <div class="assistant-label">🤖 Antigravity Assistant Response</div>
            <div class="assistant-text">{asst_html}</div>
          </div>""")

        html_lines.append("""        </div>
      </div>""")

    html_lines.append("""    </div>
  </main>

  <script>
    let thoughtsExpanded = false;
    let toolsExpanded = false;

    function toggleAllThoughts() {
      thoughtsExpanded = !thoughtsExpanded;
      document.querySelectorAll('.thought-details').forEach(el => {
        el.open = thoughtsExpanded;
      });
      document.getElementById('toggleThoughtsBtn').classList.toggle('active', thoughtsExpanded);
    }

    function toggleAllTools() {
      toolsExpanded = !toolsExpanded;
      document.querySelectorAll('.tool-details').forEach(el => {
        el.open = toolsExpanded;
      });
      document.getElementById('toggleToolsBtn').classList.toggle('active', toolsExpanded);
    }

    function toggleTheme() {
      document.body.classList.toggle('light');
      localStorage.setItem('theme', document.body.classList.contains('light') ? 'light' : 'dark');
    }

    if (localStorage.getItem('theme') === 'light') {
      document.body.classList.add('light');
    }

    // Jump to turn
    document.getElementById('jumpSelector').addEventListener('change', function(e) {
      const targetId = e.target.value;
      if (targetId) {
        const el = document.getElementById(targetId);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          el.classList.add('highlight');
          setTimeout(() => el.classList.remove('highlight'), 2000);
        }
      }
    });

    // Real-time search filter
    const searchInput = document.getElementById('searchInput');
    const matchCount = document.getElementById('matchCount');
    searchInput.addEventListener('input', function(e) {
      const query = e.target.value.toLowerCase().trim();
      const cards = document.querySelectorAll('.turn-card');
      let visible = 0;
      
      cards.forEach(card => {
        if (!query) {
          card.style.display = 'block';
          visible++;
        } else {
          const text = card.textContent.toLowerCase();
          if (text.includes(query)) {
            card.style.display = 'block';
            visible++;
          } else {
            card.style.display = 'none';
          }
        }
      });
      
      if (query) {
        matchCount.textContent = `Showing ${visible} of ${cards.length} turns matching "${query}"`;
      } else {
        matchCount.textContent = '';
      }
    });
  </script>
</body>
</html>
""")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_lines))
    print(f"Generated interactive HTML: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    transcript_path = repo_root / "brain" / "logs" / "transcript_full.jsonl"
    if not transcript_path.exists():
        transcript_path = repo_root / "brain" / "logs" / "transcript.jsonl"
        
    print(f"Loading transcript from {transcript_path}...")
    turns, total_steps = load_transcript(str(transcript_path))
    print(f"Parsed {len(turns)} conversation turns across {total_steps} steps.")
    
    conv_id = "6e0c7109-5304-44a3-8c3d-a42d1e47513c"
    
    # 1. Output to docs/index.html (standard for GitHub Pages)
    docs_index = repo_root / "docs" / "index.html"
    generate_html(turns, total_steps, conv_id, str(docs_index))
    
    # 2. Output copy to brain/conversation_viewer.html
    brain_viewer = repo_root / "brain" / "conversation_viewer.html"
    generate_html(turns, total_steps, conv_id, str(brain_viewer))

if __name__ == "__main__":
    main()
