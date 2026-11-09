#!/usr/bin/env python3
"""Build the CMDC integration meeting report as a verified DOCX."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("/Users/yellowich/Desktop/CMDC数据集接入汇报与后续优化建议.docx")
REPORT_CSV = PROJECT_ROOT / "output" / "cmdc_benchmark_report.csv"
SUMMARY_JSON = PROJECT_ROOT / "output" / "cmdc_benchmark_summary.json"
AUDIT_JSON = PROJECT_ROOT / "output" / "cmdc_data_audit.json"
CHART_DIR = Path("/tmp/cmdc_meeting_charts")

FONT_CN = "Arial Unicode MS"
FONT_MONO = "Menlo"
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
INK = RGBColor(24, 31, 38)
MUTED = RGBColor(91, 103, 112)
WHITE = RGBColor(255, 255, 255)
NAVY = "17324D"
TEAL = "0F766E"
PALE_BLUE = "F3F7FA"
LIGHT_GRAY = "F6F7F8"
BORDER = "D9E0E5"


def set_font(
    run,
    name: str = FONT_CN,
    size: float | None = None,
    bold=None,
    east_asia_name: str | None = None,
):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:ascii"), name)
    rpr.rFonts.set(qn("w:hAnsi"), name)
    rpr.rFonts.set(qn("w:eastAsia"), east_asia_name or name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold


def fmt_paragraph(p, before=0, after=5, line=1.28, keep=False):
    f = p.paragraph_format
    f.space_before = Pt(before)
    f.space_after = Pt(after)
    f.line_spacing = line
    f.keep_with_next = keep


def add_text(doc, text: str, *, bold_lead: str = "", after=6, size=10.5):
    p = doc.add_paragraph()
    fmt_paragraph(p, after=after)
    if bold_lead and text.startswith(bold_lead):
        lead = p.add_run(bold_lead)
        set_font(lead, size=size, bold=True)
        rest = p.add_run(text[len(bold_lead) :])
        set_font(rest, size=size)
    else:
        run = p.add_run(text)
        set_font(run, size=size)
    return p


def add_bullet(doc, text: str):
    p = doc.add_paragraph(style="List Bullet")
    fmt_paragraph(p, after=3, line=1.2)
    run = p.add_run(text)
    set_font(run, size=10.2)
    return p


def add_heading(doc, text: str, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    run = p.add_run(text)
    set_font(run, size=17 if level == 1 else 12.5, bold=True)
    run.font.color.rgb = RGBColor(0, 0, 0)
    fmt_paragraph(p, before=13 if level == 1 else 9, after=7, line=1.08, keep=True)
    return p


def shade_cell(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc_pr = cell._tc.get_or_add_tcPr()
    mar = tc_pr.first_child_found_in("w:tcMar")
    if mar is None:
        mar = OxmlElement("w:tcMar")
        tc_pr.append(mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def table_borders(table):
    pr = table._tbl.tblPr
    borders = pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "5")
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), BORDER)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader")
    hdr.set(qn("w:val"), "true")
    tr_pr.append(hdr)


def no_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    flag = OxmlElement("w:cantSplit")
    flag.set(qn("w:val"), "true")
    tr_pr.append(flag)


def add_table(doc, headers, rows, widths=None, font_size=8.9):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table_borders(table)
    header = table.rows[0]
    repeat_header(header)
    no_split(header)
    for i, value in enumerate(headers):
        cell = header.cells[i]
        shade_cell(cell, NAVY)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cell_margins(cell)
        p = cell.paragraphs[0]
        fmt_paragraph(p, after=0, line=1.08)
        run = p.add_run(str(value))
        set_font(run, size=font_size + 0.2, bold=True)
        run.font.color.rgb = WHITE
        if widths:
            cell.width = Cm(widths[i])
    for row_index, values in enumerate(rows):
        row = table.add_row()
        no_split(row)
        for i, value in enumerate(values):
            cell = row.cells[i]
            shadow = PALE_BLUE if row_index % 2 else "FFFFFF"
            shade_cell(cell, shadow)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cell_margins(cell)
            p = cell.paragraphs[0]
            fmt_paragraph(p, after=0, line=1.14)
            run = p.add_run(str(value))
            set_font(run, size=font_size)
            if widths:
                cell.width = Cm(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run("内部汇报  ")
    set_font(run, size=8.5)
    run.font.color.rgb = MUTED
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld1, instr, fld2])


def page_break(doc):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)
    fmt_paragraph(p, after=0)


def load_data():
    report = pd.read_csv(REPORT_CSV)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    return report, summary, audit


def build_charts(report: pd.DataFrame, audit: dict):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    cn_font = FontProperties(fname=FONT_PATH)
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.7

    labels = {
        "text": "文本",
        "acoustic": "声学",
        "audio": "语音融合",
        "visual": "视觉",
        "text_audio": "文本＋语音",
        "complete_multimodal": "全模态",
    }
    df = report.copy()
    df["label"] = df["variant_name"].map(labels).fillna(df["variant_name"])
    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=220)
    colors = ["#2563EB", "#0F766E", "#D97706", "#7C3AED", "#0B6E99", "#B91C1C"]
    bars = ax.bar(df["label"], df["auc_mean"], color=colors[: len(df)], width=0.64)
    for bar, auc, subjects in zip(bars, df["auc_mean"], df["subjects"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{auc:.3f}\nn={subjects}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontproperties=cn_font,
            color="#24323D",
        )
    ax.set_ylim(0.78, 1.05)
    ax.set_ylabel("AUC", fontproperties=cn_font, fontsize=9.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], fontproperties=cn_font, fontsize=9)
    ax.grid(axis="y", color="#E7EDF2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    auc_path = CHART_DIR / "cmdc_auc.png"
    fig.savefig(auc_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    mods = audit["modalities"]
    names = ["文本", "视觉", "语音嵌入", "openSMILE"]
    keys = ["text", "visual", "audio_deep", "acoustic"]
    counts = [mods[key]["subjects_covered"] for key in keys]
    fig, ax = plt.subplots(figsize=(8.4, 3.2), dpi=220)
    bars = ax.barh(names, counts, color=["#2563EB", "#7C3AED", "#0F766E", "#D97706"])
    for bar, count in zip(bars, counts):
        ax.text(
            count + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{count}/78",
            va="center",
            fontsize=8.8,
            fontproperties=cn_font,
            color="#24323D",
        )
    ax.set_xlim(0, 86)
    ax.set_xlabel("覆盖受试者人数", fontproperties=cn_font, fontsize=9.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontproperties=cn_font, fontsize=9)
    ax.grid(axis="x", color="#E7EDF2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    coverage_path = CHART_DIR / "cmdc_coverage.png"
    fig.savefig(coverage_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return auc_path, coverage_path


def build() -> Document:
    report, summary, audit = load_data()
    auc_chart, coverage_chart = build_charts(report, audit)
    primary = summary["primary_result"]

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = FONT_CN
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT_CN)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_CN)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK
    normal.paragraph_format.line_spacing = 1.28
    normal.paragraph_format.space_after = Pt(5)
    for name, size in (("Title", 27), ("Heading 1", 17), ("Heading 2", 12.5)):
        style = doc.styles[name]
        style.font.name = FONT_CN
        style._element.rPr.rFonts.set(qn("w:ascii"), FONT_CN)
        style._element.rPr.rFonts.set(qn("w:hAnsi"), FONT_CN)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(0 if name == "Title" else 12)
        style.paragraph_format.space_after = Pt(7)
    title_style = doc.styles["Title"]
    title_p_pr = title_style._element.get_or_add_pPr()
    for border in title_p_pr.findall(qn("w:pBdr")):
        title_p_pr.remove(border)

    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.25)
    section.right_margin = Cm(2.25)
    add_footer(section)

    p = doc.add_paragraph()
    fmt_paragraph(p, before=55, after=8)
    run = p.add_run("项目组会汇报")
    set_font(run, size=10, bold=True)
    run.font.color.rgb = MUTED

    p = doc.add_paragraph(style="Title")
    fmt_paragraph(p, before=0, after=10, line=1.08)
    run = p.add_run("CMDC 数据集接入汇报")
    set_font(run, size=27, bold=True)
    run.add_break()
    run = p.add_run("与后续优化建议")
    set_font(run, size=27, bold=True)

    p = doc.add_paragraph()
    fmt_paragraph(p, after=20)
    run = p.add_run("2026-11-09 项目组会")
    set_font(run, size=12)
    run.font.color.rgb = RGBColor(50, 62, 72)

    add_text(
        doc,
        "本次汇报说明 CMDC 数据接入后系统发生了什么变化、目前取得了哪些结果、哪些结果可以写入论文，以及下一阶段还可以从数据、模型、评估和 Agent 四个方向怎样提升。",
        after=12,
        size=11,
    )
    add_table(
        doc,
        ["汇报结论", "当前判断"],
        [
            ["主要收益", "新增独立的中文多模态外部验证域，提高数据多样性和科研可信度"],
            ["主要结果", f"文本＋语音 77 人，受试者级 AUC {primary['auc_mean']:.3f}"],
            ["是否改变主评分", "否。CMDC 不进入当前老年主风险融合模型"],
            ["下一步重点", "置信区间与校准、缺失模态、模型对比、人口学与老年外部数据"],
        ],
        widths=[3.8, 11.5],
        font_size=9.2,
    )

    page_break(doc)

    add_heading(doc, "1 结论先行", 1)
    add_text(
        doc,
        "新数据集带来的核心提升不是“把准确率数字冲高”，而是让系统第一次拥有了另一套独立的中文访谈多模态验证数据。",
        bold_lead="核心结论：",
    )
    add_bullet(doc, "系统由 CHARLS 表格基线和 MPDD 老年多模态两个来源，扩展为三个独立数据域。")
    add_bullet(doc, "新增 78 名受试者、每人 12 道半结构化访谈题和约 10.83 小时音频。")
    add_bullet(doc, "新增文本、语音嵌入、openSMILE 声学和访谈视觉四类受试者级特征。")
    add_bullet(doc, "形成严格按 subject_id 切分的重复交叉验证，避免同一人的 12 个回答跨训练和验证。")
    add_bullet(doc, "新增工作台页面、管理员接口、数据健康限制和独立模型产物。")
    add_text(
        doc,
        "CMDC 是混合成人样本，缺少结构化年龄、性别、共病和量表分数，因此当前只作为外部验证和预训练来源，不并入老年主风险评分。",
        bold_lead="必须明确的边界：",
    )

    add_heading(doc, "2 接入前后的变化", 1)
    add_table(
        doc,
        ["层面", "接入前", "接入后"],
        [
            ["数据来源", "CHARLS、MPDD", "增加 CMDC 中文访谈数据域"],
            ["数据规模", "原有老年表格与多模态数据", "新增 78 人、936 个受试者×问题位置"],
            ["模态", "表格、语音、面部、步态、人格", "增加中文文本、128 维语音嵌入、openSMILE 和访谈视觉"],
            ["评估", "主要依赖单一来源内部验证", "增加独立中文多模态外部验证"],
            ["前端", "一个多模态页", "拆分 MPDD 与 CMDC 两个独立来源标签"],
            ["接口", "MPDD 数据与训练接口", "增加 CMDC 构建、训练、摘要和报告接口"],
            ["治理", "MPDD/CHARLS 使用规则", "增加 CMDC EULA、单次密码、派生文件 0600 和健康提示"],
        ],
        widths=[3.0, 5.7, 6.6],
        font_size=8.8,
    )

    add_heading(doc, "3 数据规模与完整性", 1)
    data_cols = doc.add_table(rows=1, cols=4)
    data_cols.alignment = WD_TABLE_ALIGNMENT.CENTER
    data_cols.autofit = False
    table_borders(data_cols)
    repeat_header(data_cols.rows[0])
    metrics = [
        ("受试者", "78 人"),
        ("HC / MDD", "52 / 26"),
        ("访谈音频", "10.83 小时"),
        ("归档成员", "5680 个"),
    ]
    for index, (label, value) in enumerate(metrics):
        cell = data_cols.rows[0].cells[index]
        shade_cell(cell, LIGHT_GRAY)
        cell_margins(cell, top=150, bottom=150)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(value + "\n")
        set_font(run, size=13, bold=True)
        run2 = p.add_run(label)
        set_font(run2, size=8.5)
        run2.font.color.rgb = MUTED
        cell.width = Cm(3.8)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    add_text(
        doc,
        "8 个加密归档全部流式读取成功，共约 12.69 GiB，无读取错误。派生表包含 78 行、3504 个数值特征，原始音频和转写不进入项目目录。",
    )
    doc.add_picture(str(coverage_chart), width=Inches(6.35))
    doc.inline_shapes[-1]._inline.docPr.set(
        "descr",
        "CMDC 数据集中文本、视觉、语音嵌入和 openSMILE 声学特征覆盖受试者人数的柱状图",
    )
    doc.inline_shapes[-1]._inline.docPr.set("title", "CMDC 模态覆盖")
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(doc.paragraphs[-1], after=3)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(p, after=8)
    run = p.add_run("图 1 各模态受试者覆盖情况")
    set_font(run, size=8.5)
    run.font.color.rgb = MUTED
    add_text(
        doc,
        "视觉嵌入只覆盖 45 人，四模态完整病例为 44 人。因此全模态 AUC 不能与 78 人的语音结果直接等同比较。",
        bold_lead="阅读边界：",
    )

    add_heading(doc, "4 建模流程的变化", 1)
    steps = [
        ("01", "读取", "从加密归档读取文本、视觉、语音和声学数值特征，不复制原始访谈。"),
        ("02", "聚合", "同一受试者的 12 个问题分别计算均值和标准差，形成一条受试者级样本。"),
        ("03", "切分", "按 subject_id 做 5 折、重复 10 次的分层交叉验证。"),
        ("04", "评估", "分别训练文本、声学、语音、视觉和全模态模型。"),
        ("05", "报告", "输出 AUC、PR-AUC、灵敏度、特异度、F1 和 Brier，并保存独立模型。"),
    ]
    add_table(
        doc,
        ["步骤", "操作", "关键控制"],
        steps,
        widths=[1.4, 2.8, 11.1],
        font_size=9.0,
    )
    add_text(
        doc,
        "模型使用折内中位数填补、标准化和固定正则化 Logistic Regression。当前结果用于验证数据管线和研究信号，不是最终深度模型或临床模型。",
        bold_lead="模型口径：",
    )

    add_heading(doc, "5 实验结果", 1)
    add_table(
        doc,
        ["方案", "人数", "AUC", "PR-AUC", "灵敏度", "特异度", "F1", "Brier"],
        [
            ["文本", "77", "0.921", "0.883", "0.852", "0.810", "0.763", "0.131"],
            ["openSMILE", "78", "0.955", "0.958", "0.963", "0.909", "0.905", "0.089"],
            ["语音融合", "78", "0.961", "0.973", "0.924", "0.979", "0.937", "0.050"],
            ["视觉", "45", "0.987", "0.990", "0.943", "0.969", "0.950", "0.055"],
            ["文本＋语音", "77", "0.963", "0.967", "0.920", "0.960", "0.918", "0.054"],
            ["全模态完整病例", "44", "0.983", "0.985", "0.945", "0.958", "0.944", "0.047"],
        ],
        widths=[3.2, 1.4, 1.6, 1.8, 1.8, 1.8, 1.5, 1.6],
        font_size=8.2,
    )
    doc.add_picture(str(auc_chart), width=Inches(6.35))
    doc.inline_shapes[-1]._inline.docPr.set(
        "descr",
        "CMDC 文本、声学、语音、视觉、文本加语音和全模态方案的受试者级 AUC 柱状图",
    )
    doc.inline_shapes[-1]._inline.docPr.set("title", "CMDC 外部验证结果")
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(doc.paragraphs[-1], after=3)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(p, after=8)
    run = p.add_run("图 2 各方案受试者级 AUC 与样本量")
    set_font(run, size=8.5)
    run.font.color.rgb = MUTED
    add_text(
        doc,
        "主外部验证方案预先选择“文本＋语音”，因为覆盖 77 人且不把样本缩减到视觉完整病例。视觉和全模态结果作为敏感性分析，不能据此宣称模型已接近临床可用。",
        bold_lead="主结果解释：",
    )

    page_break(doc)

    add_heading(doc, "6 为什么不能直接并入老年主模型", 1)
    add_table(
        doc,
        ["限制", "对模型的影响", "当前处理"],
        [
            ["混合成人样本", "不能代表 60 岁以上老年人群", "只做外部验证和领域适配"],
            ["无结构化年龄、性别、共病", "无法进行人口学和共病校正", "不进入 CHARLS 风险融合"],
            ["无 PHQ-9/HAMD 分数", "无法做严重程度回归", "只使用 HC/MDD 二分类标签"],
            ["视觉仅 45 人", "全模态分析样本进一步缩小", "同时报告完整病例和缺失情况"],
            ["样本仅 78 人", "AUC 波动大、外推有限", "重复交叉验证并注明探索性"],
            ["与 MPDD 特征提取不同", "不能把特征列直接拼接", "保持独立数据域，不做行级混合"],
        ],
        widths=[4.0, 7.0, 4.3],
        font_size=8.8,
    )
    add_text(
        doc,
        "如果把 CMDC 直接混入主训练集，数字可能变好看，但研究对象、采集协议和标签定义会混杂，论文中的老年风险结论反而更弱。",
        bold_lead="研究判断：",
    )

    add_heading(doc, "7 后续提升路线", 1)
    add_table(
        doc,
        ["优先级", "提升方向", "具体工作", "预期作用"],
        [
            ["P0", "统计可信度", "Bootstrap 95% CI、校准曲线、决策曲线、DeLong 检验", "判断高 AUC 是否稳定，补足论文证据"],
            ["P0", "缺失模态", "完整病例、缺失指示、多重插补和模态缺失模型对比", "解释 45 人与 78 人结果差异"],
            ["P0", "模型对比", "嵌套调参的 Logistic、Elastic Net、Random Forest、XGBoost", "避免固定单一模型造成误判"],
            ["P1", "受试者级融合", "早期融合、后期融合、门控融合、模态 dropout", "充分利用文本、语音和视觉互补信号"],
            ["P1", "数据补齐", "申请年龄、性别、共病、教育、用药和量表分数", "支持亚组、共病和严重程度分析"],
            ["P1", "老年外部验证", "寻找或建立独立老年中文多模态样本", "验证从成人混合样本到老年人的迁移性"],
            ["P2", "领域适配", "用 CMDC 预训练文本/语音编码器，再在老年数据微调", "提高小样本条件下的表征能力"],
            ["P2", "前瞻性验证", "固定采集协议、盲法评估、人工复核和随访", "从回顾性实验走向真实筛查验证"],
        ],
        widths=[1.5, 3.4, 7.2, 4.6],
        font_size=8.4,
    )

    add_heading(doc, "8 未来 30 天建议计划", 1)
    add_table(
        doc,
        ["时间", "工作", "可交付结果"],
        [
            ["第 1 周", "补齐 CMDC 的 Bootstrap 置信区间、校准曲线和决策曲线", "更完整的 CMDC 结果表与图"],
            ["第 2 周", "完成缺失模态三方案和五类模型对比", "选择稳定的主分析和敏感性分析"],
            ["第 3 周", "尝试受试者级早期/后期/门控融合", "比较不同融合策略，不以最高 AUC 单独选型"],
            ["第 4 周", "申请或确认人口学、共病和量表数据，形成老年外部验证申请清单", "下一步数据申请方案与导师决策材料"],
        ],
        widths=[2.2, 8.4, 6.1],
        font_size=8.9,
    )
    add_text(
        doc,
        "在获得新的老年真实数据之前，不建议继续通过合成 CMDC 数据提高分数。合成数据可以补充训练，但不能替代独立真实外部验证。",
        bold_lead="数据扩充原则：",
    )

    add_heading(doc, "9 对话智能体还能怎样提升", 1)
    add_text(
        doc,
        "CMDC 的文本和语音信号不能直接替代对话 Agent，但可以用于改进问询设计、语言特征词典和文本编码器。",
    )
    add_bullet(doc, "把 CMDC 常见表达归纳为主题词、否定词、模糊表达和功能影响，不复制真实原句。")
    add_bullet(doc, "使用 Prompt v2 的“单一关键追问或行动选择”结构，由组员提交候选 Prompt 和多轮案例。")
    add_bullet(doc, "对睡眠、兴趣、情绪、身体不适、认知、社交和无望自责建立同一套回答类型矩阵。")
    add_bullet(doc, "对自伤、急症、自行停药和诊断请求继续使用本地硬规则，不能只依赖云端 Prompt。")
    add_bullet(doc, "建立人工盲评集，比较访谈、陪伴、安全和医学边界四个维度的不同模型版本。")
    add_bullet(doc, "把对话缺失信息转化为结构化字段，为后续风险模型提供可解释变量。")

    add_heading(doc, "10 组会汇报口径", 1)
    add_text(
        doc,
        "“这次新接入的是 CMDC 中文半结构化访谈数据，共 78 人，每人 12 题，约 10.83 小时音频。系统新增了文本、语音和视觉三类数值特征，并把同一受试者的 12 个回答聚合后按受试者切分验证。”",
    )
    add_text(
        doc,
        "“主结果采用文本＋语音，77 人，受试者级 AUC 0.963。视觉完整病例只有 45 人，全模态完整病例 44 人，因此这些结果作为敏感性分析。”",
    )
    add_text(
        doc,
        "“当前最重要的变化是增加了独立中文多模态外部验证和完整工程链路，而不是直接提高老年主风险评分。下一阶段要做置信区间、校准、缺失模态、模型比较和老年外部数据补齐。”",
    )

    add_heading(doc, "11 建议向导师确认的事项", 1)
    add_table(
        doc,
        ["确认事项", "建议决策"],
        [
            ["CMDC 的论文定位", "作为中文多模态外部验证，不并入老年主队列"],
            ["是否继续申请元数据", "优先确认年龄、性别、共病和量表分数是否可获得"],
            ["是否寻找老年多模态外部数据", "若论文目标包含老年外部验证，应列为下一阶段重点"],
            ["下一步实验优先级", "先置信区间、校准和缺失模态，再做复杂融合模型"],
            ["合成数据使用边界", "只用于训练侧增强，验证和测试始终使用真实数据"],
        ],
        widths=[5.0, 10.2],
        font_size=9.0,
    )
    add_text(
        doc,
        "建议组会形成一个明确决定：CMDC 当前作为独立外部验证章节保留，同时把获得老年真实数据和元数据作为后续最高优先级，而不是继续在 78 人样本上追求更高的单一 AUC。",
        bold_lead="会议落点：",
    )

    return doc


def main():
    doc = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
