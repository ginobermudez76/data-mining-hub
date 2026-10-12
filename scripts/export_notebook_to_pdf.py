"""Convertidor autónomo de Jupyter Notebook (.ipynb) a HTML y PDF imprimible.

Autor: Gino Maximiliano Bermúdez Santos
Repositorio: data-mining-hub
"""

import base64
import html
import json
import os
import re
import subprocess
import sys


def markdown_to_html(md_text: str) -> str:
    """Convierte markdown básico (encabezados, tablas, listas, alertas y bloques) a HTML."""
    lines = md_text.split("\n")
    html_lines = []
    in_table = False
    table_header_done = False
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Alertas estilo GitHub
        if stripped.startswith("> [!IMPORTANT]"):
            html_lines.append('<div class="alert alert-important"><strong>IMPORTANTE:</strong>')
            continue
        elif stripped.startswith("> [!WARNING]"):
            html_lines.append('<div class="alert alert-warning"><strong>ADVERTENCIA:</strong>')
            continue
        elif stripped.startswith("> [!NOTE]"):
            html_lines.append('<div class="alert alert-note"><strong>NOTA:</strong>')
            continue
        elif stripped.startswith(">"):
            content = line.lstrip("> ").strip()
            if html_lines and html_lines[-1].startswith('<div class="alert'):
                html_lines.append(f"<p>{html.escape(content)}</p></div>")
            else:
                html_lines.append(f"<blockquote>{html.escape(content)}</blockquote>")
            continue

        # Cerrar listas si la línea no es ítem
        if in_list and not (stripped.startswith("- ") or stripped.startswith("* ") or re.match(r"^\d+\.\s", stripped)):
            html_lines.append("</ul>")
            in_list = False

        # Tablas markdown
        if "|" in stripped and stripped.startswith("|") and stripped.endswith("|"):
            parts = [p.strip() for p in stripped.split("|")[1:-1]]
            if not in_table:
                html_lines.append('<table class="rendered-table"><thead><tr>')
                for p in parts:
                    html_lines.append(f"<th>{html.escape(p)}</th>")
                html_lines.append("</tr></thead><tbody>")
                in_table = True
                table_header_done = False
                continue
            elif not table_header_done and all(set(p).issubset({"-", ":", " "}) for p in parts):
                table_header_done = True
                continue
            else:
                html_lines.append("<tr>")
                for p in parts:
                    # Permitir negrita dentro de tabla
                    p_formatted = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(p))
                    html_lines.append(f"<td>{p_formatted}</td>")
                html_lines.append("</tr>")
                continue
        elif in_table:
            html_lines.append("</tbody></table>")
            in_table = False

        if not stripped:
            html_lines.append("<br/>")
            continue

        # Encabezados
        if stripped.startswith("# "):
            html_lines.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("#### "):
            html_lines.append(f"<h4>{html.escape(stripped[5:])}</h4>")
        elif stripped == "---":
            html_lines.append("<hr/>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = html.escape(stripped[2:])
            content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.*?)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
        elif re.match(r"^\d+\.\s", stripped):
            if not in_list:
                html_lines.append("<ol>")
                in_list = True
            m = re.match(r"^\d+\.\s(.*)", stripped)
            content = html.escape(m.group(1)) if m else ""
            content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.*?)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
        else:
            text = html.escape(stripped)
            # Formato en línea
            text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
            text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
            html_lines.append(f"<p>{text}</p>")

    if in_table:
        html_lines.append("</tbody></table>")
    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def convert_notebook_to_html(ipynb_path: str, output_html_path: str) -> str:
    """Convierte el archivo .ipynb a un documento HTML con diseño profesional."""
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Pipeline de Ingesta, Exploración y Calidad de Datos - Jupyter Notebook</title>
<style>
    @page {
        size: A4;
        margin: 1.5cm;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #24292e;
        line-height: 1.5;
        background-color: #ffffff;
        margin: 0;
        padding: 20px;
    }
    .notebook-container {
        max-width: 900px;
        margin: 0 auto;
    }
    h1, h2, h3, h4 {
        color: #0366d6;
        margin-top: 24px;
        margin-bottom: 12px;
        font-weight: 600;
    }
    h1 { font-size: 24px; border-bottom: 2px solid #eaecef; padding-bottom: 8px; color: #1f4e79; }
    h2 { font-size: 20px; border-bottom: 1px solid #eaecef; padding-bottom: 6px; }
    h3 { font-size: 16px; }
    p { margin-top: 0; margin-bottom: 10px; }
    code {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 12px;
        background-color: #f6f8fa;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .cell {
        margin-bottom: 16px;
        page-break-inside: avoid;
    }
    .input-cell {
        background-color: #f8f9fa;
        border: 1px solid #e1e4e8;
        border-left: 4px solid #0366d6;
        border-radius: 4px;
        margin-bottom: 6px;
        overflow-x: auto;
    }
    .cell-prompt {
        font-family: monospace;
        font-size: 11px;
        color: #6a737d;
        padding: 4px 8px;
        background: #f1f3f5;
        border-bottom: 1px solid #e1e4e8;
        font-weight: bold;
    }
    .code-content {
        margin: 0;
        padding: 10px 12px;
        font-family: Consolas, "Liberation Mono", Courier, monospace;
        font-size: 11.5px;
        line-height: 1.45;
        white-space: pre-wrap;
        word-wrap: break-word;
        color: #24292e;
    }
    .output-cell {
        margin-left: 10px;
        margin-bottom: 8px;
    }
    .output-prompt {
        font-family: monospace;
        font-size: 10px;
        color: #d73a49;
        font-weight: bold;
        margin-bottom: 2px;
    }
    .output-stream {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 3px;
        padding: 8px 10px;
        font-family: Consolas, monospace;
        font-size: 11px;
        line-height: 1.4;
        white-space: pre-wrap;
        color: #333333;
        margin-top: 4px;
    }
    .output-result {
        background-color: #ffffff;
        border: 1px solid #d1d5da;
        border-radius: 3px;
        padding: 8px 10px;
        font-family: Consolas, monospace;
        font-size: 11px;
        line-height: 1.4;
        white-space: pre-wrap;
        color: #111111;
        margin-top: 4px;
    }
    .output-image {
        text-align: center;
        margin: 10px 0;
    }
    .output-image img {
        max-width: 100%;
        height: auto;
        border: 1px solid #e1e4e8;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .rendered-table {
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
        font-size: 11.5px;
    }
    .rendered-table th, .rendered-table td {
        border: 1px solid #d1d5da;
        padding: 6px 10px;
        text-align: left;
    }
    .rendered-table th {
        background-color: #f1f3f5;
        font-weight: 600;
    }
    .rendered-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    .alert {
        padding: 10px 14px;
        margin: 12px 0;
        border-radius: 4px;
        border-left: 4px solid;
        font-size: 12px;
    }
    .alert-important {
        background-color: #e8f4fd;
        border-left-color: #0366d6;
        color: #0366d6;
    }
    .alert-warning {
        background-color: #fff8e6;
        border-left-color: #f39c12;
        color: #8a6d3b;
    }
    .alert-note {
        background-color: #edf7ed;
        border-left-color: #2ecc71;
        color: #27ae60;
    }
    hr {
        border: 0;
        height: 1px;
        background: #e1e4e8;
        margin: 20px 0;
    }
    @media print {
        body { padding: 0; }
        .cell { page-break-inside: avoid; }
    }
</style>
</head>
<body>
<div class="notebook-container">
""")

    code_counter = 1
    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type")
        source = "".join(cell.get("source", []))

        if ctype == "markdown":
            html_parts.append('<div class="cell markdown-cell">')
            html_parts.append(markdown_to_html(source))
            html_parts.append('</div>')

        elif ctype == "code":
            exec_count = cell.get("execution_count") or code_counter
            html_parts.append('<div class="cell">')
            html_parts.append('<div class="input-cell">')
            html_parts.append(f'<div class="cell-prompt">In [{exec_count}]:</div>')
            html_parts.append(f'<pre class="code-content">{html.escape(source)}</pre>')
            html_parts.append('</div>')

            # Renderizar outputs
            for out in cell.get("outputs", []):
                out_type = out.get("output_type")
                if out_type == "stream":
                    text = "".join(out.get("text", []))
                    html_parts.append('<div class="output-cell">')
                    html_parts.append(f'<pre class="output-stream">{html.escape(text)}</pre>')
                    html_parts.append('</div>')
                elif out_type in ("execute_result", "display_data"):
                    data = out.get("data", {})
                    if "image/png" in data:
                        b64_img = data["image/png"]
                        html_parts.append('<div class="output-cell output-image">')
                        html_parts.append(f'<img src="data:image/png;base64,{b64_img}" alt="Gráfico EDA" />')
                        html_parts.append('</div>')
                    elif "text/plain" in data:
                        text = "".join(data["text/plain"])
                        html_parts.append('<div class="output-cell">')
                        html_parts.append(f'<div class="output-prompt">Out [{exec_count}]:</div>')
                        html_parts.append(f'<pre class="output-result">{html.escape(text)}</pre>')
                        html_parts.append('</div>')

            html_parts.append('</div>')
            code_counter += 1

    html_parts.append("""
</div>
</body>
</html>
""")

    full_html = "\n".join(html_parts)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    return output_html_path


def export_html_to_pdf_via_browser(html_path: str, pdf_path: str) -> bool:
    """Convierte el archivo HTML a PDF utilizando Microsoft Edge headless."""
    abs_html = os.path.abspath(html_path).replace("\\", "/")
    abs_pdf = os.path.abspath(pdf_path)

    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_exe = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_exe = p
            break

    if not edge_exe:
        print("[!] No se encontró ejecutable de Microsoft Edge.", file=sys.stderr)
        return False

    temp_profile = os.path.join(os.environ.get("TEMP", "C:/Temp"), "edge_nb_export")
    os.makedirs(temp_profile, exist_ok=True)

    cmd = [
        "powershell", "-NoProfile", "-Command",
        f'Start-Process -FilePath "{edge_exe}" -ArgumentList '
        f'"--headless=new", "--disable-gpu", "--no-pdf-header-footer", '
        f'"--user-data-dir=`"{temp_profile}`"", '
        f'"--print-to-pdf=`"{abs_pdf}`"", '
        f'"file:///{abs_html}" -Wait'
    ]

    try:
        subprocess.run(cmd, check=True, timeout=30)
        return os.path.exists(abs_pdf) and os.path.getsize(abs_pdf) > 0
    except Exception as e:
        print(f"[!] Error ejecutando exportación PDF: {e}", file=sys.stderr)
        return False


def main():
    ipynb_path = os.path.join("notebooks", "pipeline_ingesta_exploracion_calidad.ipynb")
    html_path = os.path.join("notebooks", "pipeline_ingesta_exploracion_calidad.html")
    pdf_path = os.path.join("notebooks", "pipeline_ingesta_exploracion_calidad.pdf")

    if not os.path.exists(ipynb_path):
        print(f"[!] Error: No se encontró {ipynb_path}")
        sys.exit(1)

    print(f"[1/2] Generando HTML estilizado desde: {ipynb_path}")
    convert_notebook_to_html(ipynb_path, html_path)
    print(f"      -> HTML generado: {html_path} ({os.path.getsize(html_path):,} bytes)")

    print(f"[2/2] Exportando a documento PDF: {pdf_path}")
    success = export_html_to_pdf_via_browser(html_path, pdf_path)
    if success:
        print(f"      -> [OK] PDF generado exitosamente: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
    else:
        print(f"      -> [!] No se pudo generar automáticamente el PDF.")
        print(f"         Abra el archivo '{html_path}' en su navegador y presione Ctrl + P -> Guardar como PDF.")


if __name__ == "__main__":
    main()
