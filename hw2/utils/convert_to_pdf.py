#!/usr/bin/env python3
import re

with open('report.md', 'r', encoding='utf-8') as f:
    content = f.read()

html_content = content

html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
html_content = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html_content, flags=re.MULTILINE)

html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
html_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_content)

html_content = re.sub(r'```(.+?)```', r'<pre><code>\1</code></pre>', html_content, flags=re.DOTALL)
html_content = re.sub(r'`(.+?)`', r'<code>\1</code>', html_content)

html_content = re.sub(
    r'<img src="(.+?)"(.+?)/>',
    r'<img src="\1"\2>',
    html_content
)

html_content = html_content.replace('\n\n', '</p><p>')
html_content = '<p>' + html_content + '</p>'

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Image Processing - Homework 2 Report</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        
        body {{
            font-family: "Times New Roman", "Songti SC", serif;
            line-height: 1.6;
            color: #000;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20px;
            background: white;
        }}
        
        a {{
            color: #000 !important;
            text-decoration: none !important;
            cursor: text !important;
        }}
        
        h1 {{
            font-size: 24pt;
            margin-top: 20px;
            margin-bottom: 15px;
            color: #000;
            border-bottom: 2px solid #000;
            padding-bottom: 8px;
        }}
        
        h2 {{
            font-size: 18pt;
            margin-top: 18px;
            margin-bottom: 12px;
            color: #000;
            border-bottom: 1px solid #666;
            padding-bottom: 6px;
        }}
        
        h3 {{
            font-size: 14pt;
            margin-top: 14px;
            margin-bottom: 10px;
            color: #000;
        }}
        
        h4 {{
            font-size: 12pt;
            margin-top: 12px;
            margin-bottom: 8px;
            color: #000;
        }}
        
        p {{
            margin: 10px 0;
            text-align: justify;
        }}
        
        code {{
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 10pt;
        }}
        
        pre {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border: 1px solid #ddd;
        }}
        
        pre code {{
            background: none;
            padding: 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 10pt;
        }}
        
        th, td {{
            border: 1px solid #000;
            padding: 8px;
            text-align: left;
        }}
        
        th {{
            background-color: #e0e0e0;
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px auto;
        }}
        
        em {{
            font-style: italic;
            color: #333;
        }}
        
        strong {{
            font-weight: bold;
        }}
        
        hr {{
            border: none;
            border-top: 1px solid #ccc;
            margin: 20px 0;
        }}
        
        @media print {{
            body {{
                margin: 0;
                padding: 0;
            }}
            
            a {{
                color: #000 !important;
                text-decoration: none !important;
            }}
            
            h1, h2, h3, h4 {{
                page-break-after: avoid;
            }}
            
            img {{
                page-break-inside: avoid;
            }}
            
            table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

with open('report.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("Successfully generated report.html")
print("To export PDF, please open report.html in browser and print it as PDF")

