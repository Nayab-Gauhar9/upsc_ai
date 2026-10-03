import os
import io
import markdown
from xhtml2pdf import pisa
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor

# -------------------------------------------------------------------
# Configuration & Asset Paths
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOGO_PATH = os.path.join(CURRENT_DIR, "logo.jpg")

# Set default logo path (or provide an HTTPS URL)
POLITY_MENTOR_LOGO_URL = DEFAULT_LOGO_PATH

# Warm Amber-Gold Palette aligned with FlutterFlow UI
GOLD_PRIMARY = "#E0997B"      # Main warm accent
GOLD_LIGHT = "#F6C197"        # Bright highlight for titles
GOLD_DARK = "#92400E"         # Subtle borders & divider lines
DARK_BG = "#0A0E17"           # Dark canvas background
DARK_SURFACE = "#111827"      # Callout container cards
TEXT_MAIN = "#F1F5F9"         # Crisp readable body text
TEXT_MUTED = "#94A3B8"        # Metadata / secondary subtitles

GOLD_THEME_CSS = f"""
<style>
    @page {{
        size: a4 portrait;
        margin-top: 2.2cm;
        margin-bottom: 2.2cm;
        margin-left: 1.5cm;
        margin-right: 1.5cm;
        background-color: {DARK_BG};
    }}

    body {{
        font-family: Helvetica, Arial, sans-serif;
        color: {TEXT_MAIN};
        background-color: {DARK_BG};
        line-height: 1.5;
        font-size: 10pt;
    }}

    /* Top Branded Header Bar */
    .header-table {{
        width: 100%;
        border-bottom: 1.5px solid {GOLD_PRIMARY};
        padding-bottom: 8px;
        margin-bottom: 16px;
    }}
    .brand-title {{
        font-size: 14pt;
        font-weight: bold;
        color: {GOLD_PRIMARY};
        letter-spacing: 0.8px;
    }}
    .brand-subtitle {{
        font-size: 8pt;
        color: {TEXT_MUTED};
    }}
    .badge {{
        font-size: 8pt;
        color: {GOLD_LIGHT};
        font-weight: bold;
        text-align: right;
    }}

    /* Content Headings */
    h1 {{
        font-size: 17pt;
        color: {GOLD_LIGHT};
        margin-top: 4px;
        margin-bottom: 8px;
        line-height: 1.3;
    }}
    h2 {{
        font-size: 11pt;
        color: {GOLD_PRIMARY};
        margin-top: 14px;
        margin-bottom: 6px;
        border-bottom: 1px solid {GOLD_DARK};
        padding-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }}
    h3 {{
        font-size: 10pt;
        color: {GOLD_LIGHT};
        margin-top: 10px;
        margin-bottom: 4px;
    }}

    /* Callout & Summary Cards */
    blockquote {{
        background-color: {DARK_SURFACE};
        border-left: 3.5px solid {GOLD_PRIMARY};
        margin: 8px 0;
        padding: 8px 12px;
        color: #E2E8F0;
        font-size: 9.5pt;
    }}

    /* Bullet Lists */
    ul {{
        margin: 6px 0;
        padding-left: 18px;
    }}
    li {{
        margin-bottom: 5px;
        color: {TEXT_MAIN};
    }}

    hr {{
        border: 0;
        height: 1px;
        background: {GOLD_DARK};
        margin: 12px 0;
    }}

    code {{
        font-family: Courier, monospace;
        color: {GOLD_LIGHT};
        background-color: {DARK_SURFACE};
        padding: 1px 4px;
        font-size: 8.5pt;
    }}
</style>
"""


def _create_watermark_and_footer_layer(
    width: float,
    height: float,
    page_num: int,
    total_pages: int
) -> io.BytesIO:
    """Generates an overlay with the diagonal watermark, clickable website link, and page counter."""
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(width, height))

    # 1. Diagonal Watermark
    can.saveState()
    can.translate(width / 2.0, height / 2.0)
    can.rotate(45)
    can.setFillColor(HexColor(GOLD_PRIMARY), alpha=0.07)
    can.setFont("Helvetica-Bold", 54)
    can.drawCentredString(0, 0, "POLITY MENTOR")
    can.setFont("Helvetica-Bold", 16)
    can.drawCentredString(0, -32, "UPSC INTELLIGENCE ARCHIVE")
    can.restoreState()

# 2. Running Footer Divider
    can.saveState()
    can.setStrokeColor(HexColor(GOLD_DARK))
    can.setLineWidth(0.6)
    can.line(42, 38, width - 42, 38)

    # 3. Footer Text: Clickable Web Link (Left) & Page Counter (Right)
    can.setFont("Helvetica-Bold", 8)
    can.setFillColor(HexColor(GOLD_PRIMARY))
    can.drawString(42, 24, "politymentor.com")

    # Explicit clickable URL hotspot over the footer text
    can.linkURL("https://politymentor.com", (40, 20, 115, 34), relative=0)

    can.setFont("Helvetica", 8)
    can.setFillColor(HexColor(TEXT_MUTED))
    can.drawString(120, 24, "• UPSC Civil Services GS-II Intelligence")
    can.drawRightString(width - 42, 24, f"Page {page_num} of {total_pages}")
    can.restoreState()

    can.save()
    packet.seek(0)
    return packet


def generate_pdf_from_markdown(title: str, markdown_content: str, logo_url: str = "") -> io.BytesIO:
    """
    Renders markdown to a gold-themed dark PDF, embedding the enlarged logo next to the
    header title, a diagonal watermark, and running footers with clickable links.
    """
    html_body = markdown.markdown(markdown_content, extensions=['extra'])

    # Determine image tag for logo
    active_logo = logo_url or POLITY_MENTOR_LOGO_URL
    if active_logo and os.path.exists(active_logo):
        clean_logo_path = os.path.abspath(active_logo).replace("\\", "/")
        logo_tag = f'<img src="{clean_logo_path}" width="42" height="42" style="vertical-align: middle;" />'
    elif active_logo and (active_logo.startswith("http://") or active_logo.startswith("https://")):
        logo_tag = f'<img src="{active_logo}" width="42" height="42" style="vertical-align: middle;" />'
    else:
        logo_tag = '<span style="font-size: 16pt; margin-right: 6px;">🏛️</span>'

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        {GOLD_THEME_CSS}
    </head>
    <body>
        <table class="header-table">
            <tr>
                <td style="width: 65%; vertical-align: middle;">
                    <table style="border: none;">
                        <tr>
                            <td style="width: 48px; vertical-align: middle;">
                                {logo_tag}
                            </td>
                            <td style="vertical-align: middle; padding-left: 6px;">
                                <span class="brand-title">POLITY MENTOR</span><br/>
                                <span class="brand-subtitle">AI-Synthesized UPSC Prelims & Mains Intelligence</span>
                            </td>
                        </tr>
                    </table>
                </td>
                <td style="width: 35%; vertical-align: middle;" class="badge">
                    CONFIDENTIAL & EXCLUSIVE<br/>
                    <a href="https://politymentor.com" style="color: {GOLD_PRIMARY}; text-decoration: none; font-weight: normal;">politymentor.com</a>
                </td>
            </tr>
        </table>

        {html_body}
    </body>
    </html>
    """

    # 1. Render base HTML layout to PDF
    base_pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(src=full_html, dest=base_pdf_buffer)
    if pisa_status.err:
        raise RuntimeError("Failed to generate PDF from markdown.")

    base_pdf_buffer.seek(0)

    # 2. Merge dynamic watermark & footer page numbers across every page
    reader = PdfReader(base_pdf_buffer)
    writer = PdfWriter()
    total_pages = len(reader.pages)

    for idx, page in enumerate(reader.pages, start=1):
        overlay_stream = _create_watermark_and_footer_layer(
            width=float(A4[0]),
            height=float(A4[1]),
            page_num=idx,
            total_pages=total_pages
        )
        overlay_page = PdfReader(overlay_stream).pages[0]
        page.merge_page(overlay_page, over=True)
        writer.add_page(page)

    final_buffer = io.BytesIO()
    writer.write(final_buffer)
    final_buffer.seek(0)
    return final_buffer
