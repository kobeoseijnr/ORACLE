"""
Convert markdown report to HTML with embedded images and styling
"""
import re
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
MD_FILE = BASE_DIR / "AUTOCKT_20_SAMPLE_ANALYSIS_REPORT.md"
HTML_FILE = BASE_DIR / "AUTOCKT_20_SAMPLE_ANALYSIS_REPORT.html"

def markdown_to_html(md_content):
    """Convert markdown to HTML with proper formatting."""
    html = md_content
    
    # Headers
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # Horizontal rules
    html = re.sub(r'^---$', r'<hr>', html, flags=re.MULTILINE)
    
    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Code blocks
    html = re.sub(r'```\n(.+?)\n```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```(.+?)\n(.+?)\n```', r'<pre><code class="language-\1">\2</code></pre>', html, flags=re.DOTALL)
    
    # Inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Images
    html = re.sub(r'!\[([^\]]+)\]\(([^)]+)\)', r'<img src="\2" alt="\1" class="figure">', html)
    
    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
    
    # Lists
    lines = html.split('\n')
    in_list = False
    result_lines = []
    
    for line in lines:
        # Unordered list items
        if re.match(r'^- (.+)$', line):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            content = re.match(r'^- (.+)$', line).group(1)
            result_lines.append(f'<li>{content}</li>')
        # Ordered list items
        elif re.match(r'^\d+\. (.+)$', line):
            if not in_list:
                result_lines.append('<ol>')
                in_list = True
            content = re.match(r'^\d+\. (.+)$', line).group(1)
            result_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                result_lines.append('</ul>' if '<ul>' in '\n'.join(result_lines[-10:]) else '</ol>')
                in_list = False
            if line.strip():
                result_lines.append(line)
            else:
                result_lines.append('')
    
    if in_list:
        result_lines.append('</ul>')
    
    html = '\n'.join(result_lines)
    
    # Tables
    html = convert_markdown_tables(html)
    
    # Paragraphs (lines that aren't already HTML tags)
    lines = html.split('\n')
    result_lines = []
    current_para = []
    
    for line in lines:
        if line.strip() and not line.strip().startswith('<') and not line.strip().endswith('>'):
            current_para.append(line.strip())
        else:
            if current_para:
                result_lines.append('<p>' + ' '.join(current_para) + '</p>')
                current_para = []
            result_lines.append(line)
    
    if current_para:
        result_lines.append('<p>' + ' '.join(current_para) + '</p>')
    
    html = '\n'.join(result_lines)
    
    # Clean up multiple empty lines
    html = re.sub(r'\n{3,}', '\n\n', html)
    
    return html

def convert_markdown_tables(html):
    """Convert markdown tables to HTML tables."""
    lines = html.split('\n')
    result = []
    in_table = False
    table_rows = []
    
    for line in lines:
        # Check if line is a table row
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            
            # Skip separator rows
            if re.match(r'^\|[\s\-:]+\|', line):
                continue
            
            # Parse table row
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            table_rows.append(cells)
        else:
            if in_table:
                # Convert table rows to HTML
                result.append('<table class="data-table">')
                for i, row in enumerate(table_rows):
                    tag = 'th' if i == 0 else 'td'
                    result.append('<tr>')
                    for cell in row:
                        result.append(f'<{tag}>{cell}</{tag}>')
                    result.append('</tr>')
                result.append('</table>')
                table_rows = []
                in_table = False
            
            result.append(line)
    
    # Handle table at end of file
    if in_table:
        result.append('<table class="data-table">')
        for i, row in enumerate(table_rows):
            tag = 'th' if i == 0 else 'td'
            result.append('<tr>')
            for cell in row:
                result.append(f'<{tag}>{cell}</{tag}>')
            result.append('</tr>')
        result.append('</table>')
    
    return '\n'.join(result)

def create_html_document(content):
    """Create complete HTML document with styling."""
    # Use double braces to escape format placeholders
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoCkt 20-Sample Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            font-size: 2em;
        }}
        
        h3 {{
            color: #555;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        
        h4 {{
            color: #666;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}
        
        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}
        
        ul, ol {{
            margin-left: 30px;
            margin-bottom: 20px;
        }}
        
        li {{
            margin-bottom: 8px;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #e74c3c;
        }}
        
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        pre code {{
            background-color: transparent;
            color: #ecf0f1;
            padding: 0;
        }}
        
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        table.data-table th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        
        table.data-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        
        table.data-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        table.data-table tr:hover {{
            background-color: #f1f1f1;
        }}
        
        img.figure {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            border-radius: 5px;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .header-info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        
        @media print {{
            body {{
                background-color: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>"""
    
    return html_template.format(content=content)

# Read markdown file
print("Reading markdown file...")
with open(MD_FILE, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert to HTML
print("Converting markdown to HTML...")
html_content = markdown_to_html(md_content)

# Create complete HTML document
print("Creating HTML document...")
html_document = create_html_document(html_content)

# Save HTML file
print(f"Writing HTML file to {HTML_FILE}...")
with open(HTML_FILE, 'w', encoding='utf-8') as f:
    f.write(html_document)

print(f"HTML report created successfully: {HTML_FILE}")
print(f"  File size: {HTML_FILE.stat().st_size / 1024:.1f} KB")
