"""
Markdown -> PDF (real, selectable text) using ReportLab.

The browser sends the report markdown to /download/pdf; this module renders it as a
clean research-paper-style PDF with headings, paragraphs, bullet/numbered lists,
Markdown pipe tables, bold/italic, and clickable links. Pure Python — no system
libraries — so it builds on Render without extra setup.
"""

import html
import io
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem,
)


def _styles():
    ss = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold",
                                fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold",
                             fontSize=13, leading=16, spaceBefore=12, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Times-Bold",
                             fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman",
                              fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=7),
        "cell": ParagraphStyle("c", parent=ss["BodyText"], fontName="Times-Roman",
                              fontSize=9, leading=12),
    }
    return styles


def _inline(text: str) -> str:
    """Convert Markdown inline syntax to ReportLab mini-markup (escaped, safe)."""
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  r'<a href="\2" color="#0645ad">\1</a>', text)          # links
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)                  # bold
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)         # italic
    return text


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.count("|") >= 2


def _table(rows, styles):
    data = []
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):   # separator row
            continue
        data.append([Paragraph(_inline(c), styles["cell"]) for c in cells])
    if not data:
        return None
    t = Table(data, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def markdown_to_pdf(md: str, title: str = "Research Report") -> bytes:
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, title=title,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
    )
    flow = []
    lines = (md or "").replace("\r\n", "\n").split("\n")
    i, first_heading = 0, True

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # table block
        if _is_table_row(line):
            block = []
            while i < len(lines) and _is_table_row(lines[i]):
                block.append(lines[i]); i += 1
            tbl = _table(block, styles)
            if tbl:
                flow += [Spacer(1, 4), tbl, Spacer(1, 6)]
            continue

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level, txt = len(m.group(1)), m.group(2)
            style = styles["title"] if (first_heading and level == 1) else (
                styles["h2"] if level <= 2 else styles["h3"])
            flow.append(Paragraph(_inline(txt), style))
            first_heading = False
            i += 1
            continue

        # a lone **Bold Heading** line (writer sometimes bolds section titles)
        mb = re.match(r"^\*\*(.+?)\*\*:?$", stripped)
        if mb:
            flow.append(Paragraph(_inline(mb.group(1)),
                                  styles["title"] if first_heading else styles["h2"]))
            first_heading = False
            i += 1
            continue

        # bullet list
        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                items.append(ListItem(Paragraph(
                    _inline(re.sub(r"^\s*[-*]\s+", "", lines[i])), styles["body"])))
                i += 1
            flow.append(ListFlowable(items, bulletType="bullet", leftIndent=14))
            continue

        # numbered list
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(ListItem(Paragraph(
                    _inline(re.sub(r"^\s*\d+\.\s+", "", lines[i])), styles["body"])))
                i += 1
            flow.append(ListFlowable(items, bulletType="1", leftIndent=14))
            continue

        # plain paragraph
        flow.append(Paragraph(_inline(stripped), styles["body"]))
        i += 1

    if not flow:
        flow.append(Paragraph("(empty report)", styles["body"]))

    doc.build(flow)
    return buf.getvalue()