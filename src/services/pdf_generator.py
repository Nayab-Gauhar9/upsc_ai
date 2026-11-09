import os
import io
import re
import unicodedata
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

POLITY_MENTOR_LOGO_URL = DEFAULT_LOGO_PATH
WEBSITE_URL = "https://politymentor.com"

# -------------------------------------------------------------------
# Executive Black & Slate Dual-Tone Palette
# -------------------------------------------------------------------
OUTER_MARGIN_BG = "#0F172A"  # Perimeter margin background outside the border
INNER_CANVAS_BG = "#000000"  # Pure Black interior reading background
BORDER_COLOR = "#A8997B"     # Warm Muted Gold for border frame and dividers
HEADLINE_COLOR = "#A8997B"   # Warm Muted Gold for titles and headings
SURFACE_CARD = "#111827"     # Subtle elevated container for blockquotes/cards
SURFACE_INNER = "#0B0F19"    # Table header inset background
TEXT_MAIN = "#F8FAFC"        # Crisp readable white text
TEXT_MUTED = "#94A3B8"       # Secondary slate for meta and subtitles

# -------------------------------------------------------------------
# Comprehensive Unicode Glyph Sanitizer (Eliminates '■' Tofu Blocks)
# -------------------------------------------------------------------
SYMBOL_REPLACEMENTS = {
    "\u20b9": "Rs. ",   # Indian Rupee symbol (₹) -> Rs.
    "\u00a3": "GBP ",
    "\u20ac": "EUR ",
    "\u00a5": "JPY ",
    "\u2018": "'",      # Left single quote
    "\u2019": "'",      # Right single quote
    "\u201c": '"',      # Left double quote
    "\u201d": '"',      # Right double quote
    "\u2022": "* ",     # Bullet point
    "\u2026": "...",    # Ellipsis
    "\u00a0": " ",      # Non-breaking space
    "\u2713": "[OK]",   # Checkmark
    "\u2717": "[X]",    # Cross
}

# Matches all Unicode dashes, hyphens, and horizontal bars:
# \u2010 (Hyphen), \u2011 (Non-breaking hyphen), \u2012 (Figure dash), \u2013 (En-dash),
# \u2014 (Em-dash), \u2015 (Horizontal bar), \u2212 (Minus), \u00ad (Soft hyphen),
# \ufe58, \ufe63, \uff0d (Small/fullwidth hyphens)
DASH_REGEX = re.compile(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00ad\ufe58\ufe63\uff0d\u2043\u2500]')

# Invisible zero-width format characters that cause phantom square blocks
ZERO_WIDTH_REGEX = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff]')

def sanitize_for_pdf(text: str) -> str:
    """
    Cleans all problematic Unicode characters, dashes, and currency marks
    so ReportLab and xhtml2pdf render clean, crisp text with zero tofu blocks.
    """
    if not text:
        return ""

    for char, replacement in SYMBOL_REPLACEMENTS.items():
        text = text.replace(char, replacement)

    # Normalize all Unicode hyphens/dashes to standard ASCII '-'
    text = DASH_REGEX.sub('-', text)

    # Strip zero-width & invisible format characters
    text = ZERO_WIDTH_REGEX.sub('', text)

    # Decompose remaining complex glyphs into closest base Latin characters
    text = unicodedata.normalize('NFKD', text)

    return text

# -------------------------------------------------------------------
# Document CSS Stylesheet
# -------------------------------------------------------------------
MODERN_SLATE_CSS = f"""
<style>
    @page {{
        size: a4 portrait;
        margin-top: 2.3cm;
        margin-bottom: 2.3cm;
        margin-left: 1.6cm;
        margin-right: 1.6cm;
    }}

    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: {TEXT_MAIN};
        line-height: 1.6;
        font-size: 9.8pt;
    }}

    /* Clickable Link Styling */
    a {{
        color: {HEADLINE_COLOR};
        text-decoration: none;
    }}

    /* Top Branded Header Bar */
    .header-table {{
        width: 100%;
        border-bottom: 1.5px solid {BORDER_COLOR};
        padding-bottom: 10px;
        margin-bottom: 16px;
    }}
    .brand-title {{
        font-size: 13.5pt;
        font-weight: bold;
        color: {HEADLINE_COLOR};
        letter-spacing: 0.9px;
        text-decoration: none;
    }}
    .brand-subtitle {{
        font-size: 7.8pt;
        color: {TEXT_MUTED};
        letter-spacing: 0.3px;
    }}
    .header-badge {{
        font-size: 8pt;
        color: {HEADLINE_COLOR};
        font-weight: bold;
        text-align: right;
        line-height: 1.4;
    }}

    /* Typography & Hierarchy */
    h1 {{
        font-size: 15.5pt;
        color: {HEADLINE_COLOR};
        margin-top: 6px;
        margin-bottom: 8px;
        line-height: 1.35;
        font-weight: bold;
        letter-spacing: -0.2px;
    }}
    h2 {{
        font-size: 10.8pt;
        color: {HEADLINE_COLOR};
        margin-top: 14px;
        margin-bottom: 6px;
        border-bottom: 1px solid {BORDER_COLOR};
        padding-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: bold;
    }}
    h3 {{
        font-size: 9.8pt;
        color: {HEADLINE_COLOR};
        margin-top: 10px;
        margin-bottom: 4px;
        font-weight: bold;
    }}

    /* Modern Inset Dossier Cards */
    blockquote {{
        background-color: {SURFACE_CARD};
        border-left: 3.5px solid {BORDER_COLOR};
        margin: 10px 0;
        padding: 9px 13px;
        color: {TEXT_MAIN};
        font-size: 9.3pt;
        border-radius: 4px;
    }}

    /* Bullet Lists */
    ul {{
        margin: 6px 0;
        padding-left: 18px;
    }}
    li {{
        margin-bottom: 5px;
        color: {TEXT_MAIN};
        line-height: 1.55;
    }}

    hr {{
        border: 0;
        height: 1px;
        background: {BORDER_COLOR};
        margin: 14px 0;
    }}

    /* Tables */
    table.content-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        background-color: {SURFACE_CARD};
        border: 1px solid {BORDER_COLOR};
    }}
    table.content-table th {{
        background-color: {SURFACE_INNER};
        color: {HEADLINE_COLOR};
        font-weight: bold;
        padding: 6px 8px;
        font-size: 8.5pt;
        border-bottom: 1px solid {BORDER_COLOR};
        text-align: left;
    }}
    table.content-table td {{
        padding: 6px 8px;
        font-size: 8.5pt;
        border-bottom: 1px solid #1E293B;
        color: {TEXT_MAIN};
    }}

    code {{
        font-family: Courier, monospace;
        color: {HEADLINE_COLOR};
        background-color: {SURFACE_CARD};
        padding: 1px 4px;
        font-size: 8.5pt;
        border-radius: 3px;
    }}
</style>
"""


def _create_underlay_layer(width: float, height: float) -> io.BytesIO:
    """
    Renders the background underlay:
    1. Outer perimeter (outside the border): #0F172A
    2. Framed inner canvas (inside the border): #000000 (Pure Black)
    3. Border outline: #A8997B (Width 1.2pt)
    """
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(width, height))

    # Fill full page with outer slate color (#0F172A)
    can.setFillColor(HexColor(OUTER_MARGIN_BG))
    can.rect(0, 0, width, height, fill=1, stroke=0)

    # Fill inner framed canvas with pure black (#000000) and stroke border (#A8997B)
    can.setFillColor(HexColor(INNER_CANVAS_BG))
    can.setStrokeColor(HexColor(BORDER_COLOR))
    can.setLineWidth(1.2)
    can.rect(20, 20, width - 40, height - 40, fill=1, stroke=1)

    can.save()
    packet.seek(0)
    return packet


def _create_overlay_layer(
    width: float,
    height: float,
    page_num: int,
    total_pages: int
) -> io.BytesIO:
    """
    Renders top overlays:
    1. Subtle diagonal watermark
    2. Clickable header hotspot
    3. Clickable footer URL and running page numbers
    """
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(width, height))

    # 1. Subtle Diagonal Watermark
    can.saveState()
    can.translate(width / 2.0, height / 2.0)
    can.rotate(45)
    can.setFillColor(HexColor(HEADLINE_COLOR), alpha=0.035)
    can.setFont("Helvetica-Bold", 50)
    can.drawCentredString(0, 0, "POLITY MENTOR")
    can.setFont("Helvetica-Bold", 14)
    can.drawCentredString(0, -28, "UPSC INTELLIGENCE ARCHIVE")
    can.restoreState()

    # 2. Clickable Header Hotspot (Logo + Brand Title on every page)
    can.linkURL(WEBSITE_URL, (42, height - 75, 230, height - 25), relative=0)

    # 3. Running Footer Divider
    can.saveState()
    can.setStrokeColor(HexColor(BORDER_COLOR))
    can.setLineWidth(0.6)
    can.line(42, 38, width - 42, 38)

    # 4. Clickable Footer Link (politymentor.com)
    can.setFont("Helvetica-Bold", 8.5)
    can.setFillColor(HexColor(HEADLINE_COLOR))
    can.drawString(42, 24, "politymentor.com")

    # Link hotspot spanning the footer URL
    can.linkURL(WEBSITE_URL, (40, 20, 125, 34), relative=0)

    # Secondary Footer Text & Page Counter
    can.setFont("Helvetica", 8)
    can.setFillColor(HexColor(TEXT_MUTED))
    can.drawString(128, 24, "• UPSC Civil Services GS-II Intelligence")
    can.drawRightString(width - 42, 24, f"Page {page_num} of {total_pages}")
    can.restoreState()

    can.save()
    packet.seek(0)
    return packet


def generate_pdf_from_markdown(title: str, markdown_content: str, logo_url: str = "") -> io.BytesIO:
    """
    Renders markdown to a print-calibrated, slate-and-black dual-tone PDF:
    - Outer margin: #0F172A
    - Inner reading canvas: Pure Black (#000000)
    - Full-page outer border & headings: #A8997B
    - Sanitized text with zero tofu blocks (■)
    - Clickable logo, header, and footer links
    - Subtle watermark and page counters
    """
    clean_title = sanitize_for_pdf(title)
    clean_markdown = sanitize_for_pdf(markdown_content)

    html_body = markdown.markdown(clean_markdown, extensions=['extra', 'tables'])

    # Determine image tag for logo
    active_logo = logo_url or POLITY_MENTOR_LOGO_URL
    if active_logo and os.path.exists(active_logo):
        clean_logo_path = os.path.abspath(active_logo).replace("\\", "/")
        logo_tag = f'<a href="{WEBSITE_URL}"><img src="{clean_logo_path}" width="40" height="40" style="vertical-align: middle; border: 0;" /></a>'
    elif active_logo and (active_logo.startswith("http://") or active_logo.startswith("https://")):
        logo_tag = f'<a href="{WEBSITE_URL}"><img src="{active_logo}" width="40" height="40" style="vertical-align: middle; border: 0;" /></a>'
    else:
        logo_tag = f'<a href="{WEBSITE_URL}" style="text-decoration: none;"><span style="font-size: 16pt; margin-right: 6px;">🏛️</span></a>'

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{clean_title}</title>
        {MODERN_SLATE_CSS}
    </head>
    <body>
        <table class="header-table">
            <tr>
                <td style="width: 65%; vertical-align: middle;">
                    <table style="border: none;">
                        <tr>
                            <td style="width: 46px; vertical-align: middle;">
                                {logo_tag}
                            </td>
                            <td style="vertical-align: middle; padding-left: 8px;">
                                <a href="{WEBSITE_URL}" style="text-decoration: none;">
                                    <span class="brand-title">POLITY MENTOR</span>
                                </a><br/>
                                <span class="brand-subtitle">AI-Synthesized UPSC Prelims & Mains Intelligence</span>
                            </td>
                        </tr>
                    </table>
                </td>
                <td style="width: 35%; vertical-align: middle;" class="header-badge">
                    CONFIDENTIAL & EXCLUSIVE<br/>
                    <a href="{WEBSITE_URL}">politymentor.com &rarr;</a>
                </td>
            </tr>
        </table>

        {html_body}
    </body>
    </html>
    """

    # 1. Render base HTML layout
    base_pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(src=full_html, dest=base_pdf_buffer)
    if pisa_status.err:
        raise RuntimeError("Failed to generate PDF from markdown.")

    base_pdf_buffer.seek(0)

    # 2. Merge dual-tone underlay (underneath) and dynamic overlay (on top) across every page
    reader = PdfReader(base_pdf_buffer)
    writer = PdfWriter()
    total_pages = len(reader.pages)

    width = float(A4[0])
    height = float(A4[1])

    underlay_stream = _create_underlay_layer(width=width, height=height)
    underlay_page = PdfReader(underlay_stream).pages[0]

    for idx, page in enumerate(reader.pages, start=1):
        # Step A: Place the #0F172A perimeter and #000000 inner canvas under the content
        page.merge_page(underlay_page, over=False)

        # Step B: Stamp watermark, clickable links, and footer counters on top
        overlay_stream = _create_overlay_layer(
            width=width,
            height=height,
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