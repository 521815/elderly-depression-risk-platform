#!/usr/bin/env python3
"""Build the CMDC integration meeting report as a verified DOCX."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
ADVANCED_SUMMARY_JSON = PROJECT_ROOT / "output" / "cmdc_advanced_summary.json"
HOLDOUT_SUMMARY_JSON = PROJECT_ROOT / "output" / "cmdc_locked_holdout_summary.json"
PAIRED_CSV = PROJECT_ROOT / "output" / "cmdc_paired_comparison.csv"
MISSING_MODALITY_CSV = PROJECT_ROOT / "output" / "cmdc_missing_modality_report.csv"
MODEL_COMPARISON_CSV = PROJECT_ROOT / "output" / "cmdc_model_comparison.csv"
CALIBRATION_CSV = PROJECT_ROOT / "output" / "cmdc_calibration_curve.csv"
DECISION_CURVE_CSV = PROJECT_ROOT / "output" / "cmdc_decision_curve.csv"
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
    advanced = json.loads(ADVANCED_SUMMARY_JSON.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_SUMMARY_JSON.read_text(encoding="utf-8"))
    paired = pd.read_csv(PAIRED_CSV)
    missing = pd.read_csv(MISSING_MODALITY_CSV)
    model_comparison = pd.read_csv(MODEL_COMPARISON_CSV)
    calibration = pd.read_csv(CALIBRATION_CSV)
    decision_curve = pd.read_csv(DECISION_CURVE_CSV)
    return (
        report,
        summary,
        audit,
        advanced,
        holdout,
        paired,
        missing,
        model_comparison,
        calibration,
        decision_curve,
    )


def build_charts(
    report: pd.DataFrame,
    audit: dict,
    paired: pd.DataFrame,
    model_comparison: pd.DataFrame,
    calibration: pd.DataFrame,
    decision_curve: pd.DataFrame,
    holdout: dict,
):
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

    paired_labels = [
        "文本＋语音\n对 文本",
        "文本＋语音\n对 声学",
        "文本＋语音\n对 语音",
        "全模态\n对 视觉",
        "全模态\n对 文本＋语音",
    ]
    paired_values = paired["auc_difference"].to_numpy()
    paired_low = paired["ci_low"].to_numpy()
    paired_high = paired["ci_high"].to_numpy()
    errors = np.vstack(
        [paired_values - paired_low, paired_high - paired_values]
    )
    fig, ax = plt.subplots(figsize=(8.6, 3.9), dpi=220)
    colors = ["#2563EB" if value >= 0 else "#D97706" for value in paired_values]
    y = np.arange(len(paired_labels))
    ax.errorbar(
        paired_values,
        y,
        xerr=errors,
        fmt="none",
        ecolor="#64748B",
        elinewidth=1.2,
        capsize=4,
    )
    ax.scatter(paired_values, y, color=colors, s=56, zorder=3)
    ax.axvline(0, color="#B91C1C", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(paired_labels, fontproperties=cn_font, fontsize=8.4)
    ax.invert_yaxis()
    ax.set_xlabel("配对 AUC 差值及 95% Bootstrap 区间", fontproperties=cn_font, fontsize=9.3)
    ax.grid(axis="x", color="#E7EDF2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    paired_path = CHART_DIR / "cmdc_paired_difference.png"
    fig.savefig(paired_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    model_labels = {
        "logistic_l2": "Logistic L2",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
    }
    model_df = model_comparison.copy()
    model_df["label"] = model_df["model"].map(model_labels).fillna(model_df["model"])
    values = model_df["auc"].to_numpy()
    lower = values - model_df["auc_ci_low"].to_numpy()
    upper = model_df["auc_ci_high"].to_numpy() - values
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=220)
    bars = ax.bar(
        model_df["label"],
        values,
        color=["#2563EB", "#0F766E", "#D97706"],
        width=0.58,
    )
    ax.errorbar(
        model_df["label"],
        values,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#475569",
        capsize=5,
        linewidth=1.2,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.02, bar.get_height() + 0.012),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontproperties=cn_font,
        )
    ax.set_ylim(0.82, 1.03)
    ax.set_ylabel("AUC", fontproperties=cn_font, fontsize=9.5)
    ax.set_xticks(range(len(model_df)))
    ax.set_xticklabels(model_df["label"], fontproperties=cn_font, fontsize=9)
    ax.grid(axis="y", color="#E7EDF2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    model_path = CHART_DIR / "cmdc_model_comparison.png"
    fig.savefig(model_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    locked = (holdout.get("holdout_metrics") or {}).get(
        "locked_threshold",
        {},
    )
    sensitivity_ci = locked.get("sensitivity_exact_ci") or {}
    specificity_ci = locked.get("specificity_exact_ci") or {}
    estimate_values = np.asarray(
        [
            float(sensitivity_ci.get("estimate", 0)),
            float(specificity_ci.get("estimate", 0)),
        ]
    )
    lower = estimate_values - np.asarray(
        [
            float(sensitivity_ci.get("ci_low", 0)),
            float(specificity_ci.get("ci_low", 0)),
        ]
    )
    upper = np.asarray(
        [
            float(sensitivity_ci.get("ci_high", 0)),
            float(specificity_ci.get("ci_high", 0)),
        ]
    ) - estimate_values
    fig, ax = plt.subplots(figsize=(7.5, 2.85), dpi=220)
    labels_holdout = ["灵敏度\n5/5", "特异度\n10/11"]
    ax.errorbar(
        labels_holdout,
        estimate_values,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        markersize=7,
        color="#0F766E",
        ecolor="#64748B",
        capsize=7,
        linewidth=1.4,
    )
    for index, value in enumerate(estimate_values):
        ax.text(
            index + 0.08,
            value,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            fontproperties=cn_font,
            color="#24323D",
        )
    ax.set_ylim(0.38, 1.04)
    ax.set_ylabel("比例及 95% 精确区间", fontproperties=cn_font, fontsize=9)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels_holdout, fontproperties=cn_font, fontsize=9)
    ax.grid(axis="y", color="#E7EDF2", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    holdout_path = CHART_DIR / "cmdc_locked_holdout_ci.png"
    fig.savefig(holdout_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.45), dpi=220)
    ax = axes[0]
    ax.plot(
        calibration["predicted_frequency"],
        calibration["observed_frequency"],
        marker="o",
        color="#2563EB",
        linewidth=1.8,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#B91C1C", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("预测概率", fontproperties=cn_font, fontsize=8.8)
    ax.set_ylabel("实际阳性率", fontproperties=cn_font, fontsize=8.8)
    ax.set_title("校准曲线", fontproperties=cn_font, fontsize=10)
    ax.grid(color="#E7EDF2", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(
        decision_curve["threshold"],
        decision_curve["net_benefit"],
        color="#0F766E",
        linewidth=1.8,
        label="模型",
    )
    ax.plot(
        decision_curve["threshold"],
        decision_curve["treat_all_net_benefit"],
        color="#D97706",
        linewidth=1.2,
        label="全部干预",
    )
    ax.plot(
        decision_curve["threshold"],
        decision_curve["treat_none_net_benefit"],
        color="#64748B",
        linewidth=1.0,
        label="不干预",
    )
    ax.set_xlabel("阈值", fontproperties=cn_font, fontsize=8.8)
    ax.set_ylabel("净获益", fontproperties=cn_font, fontsize=8.8)
    ax.set_title("决策曲线", fontproperties=cn_font, fontsize=10)
    ax.grid(color="#E7EDF2", linewidth=0.6)
    ax.legend(prop=cn_font, fontsize=7.8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    calibration_path = CHART_DIR / "cmdc_calibration_decision.png"
    fig.savefig(calibration_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return (
        auc_path,
        coverage_path,
        paired_path,
        model_path,
        holdout_path,
        calibration_path,
    )


def build() -> Document:
    (
        report,
        summary,
        audit,
        advanced,
        holdout,
        paired,
        missing,
        model_comparison,
        calibration,
        decision_curve,
    ) = load_data()
    (
        auc_chart,
        coverage_chart,
        paired_chart,
        model_chart,
        holdout_chart,
        calibration_chart,
    ) = build_charts(
        report,
        audit,
        paired,
        model_comparison,
        calibration,
        decision_curve,
        holdout,
    )
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
            ["锁定留出", "61 人训练、16 人测试；AUC 1.000，但精确区间宽，不能视为临床性能证明"],
            ["融合是否已证明更好", "点估计有提升倾向，但配对 Bootstrap 区间跨 0，暂不能宣称稳定提升"],
            ["是否改变主评分", "否。CMDC 不进入当前老年主风险融合模型"],
            ["下一步重点", "多中心或老年独立留出、人口学与共病元数据、跨中心校准"],
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
    add_bullet(doc, "落地配对 Bootstrap、缺失模态动态融合、嵌套模型比较、校准和决策曲线。")
    add_bullet(doc, "完成固定种子的 61/16 受试者级留出，阈值只在训练集选择并评估一次锁定测试集。")
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

    add_heading(doc, "5.1 配对比较与置信区间", 2)
    bootstrap = advanced.get("primary_bootstrap_ci", {})
    auc_ci = bootstrap.get("auc", {})
    add_text(
        doc,
        f"文本＋语音的 Bootstrap AUC 为 {auc_ci.get('mean', 0):.3f}，"
        f"95% 区间 {auc_ci.get('ci_low', 0):.3f} 至 "
        f"{auc_ci.get('ci_high', 0):.3f}。区间较宽，说明 78 人样本下仍需谨慎解释。",
        bold_lead="整体不确定性：",
    )
    doc.add_picture(str(paired_chart), width=Inches(6.35))
    doc.inline_shapes[-1]._inline.docPr.set(
        "descr",
        "文本加语音、全模态与其他模态之间的配对 AUC 差值及 Bootstrap 置信区间",
    )
    doc.inline_shapes[-1]._inline.docPr.set("title", "CMDC 配对 AUC 差异")
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(doc.paragraphs[-1], after=3)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(p, after=8)
    run = p.add_run("图 3 同一受试者上的配对 AUC 差异与 95% Bootstrap 区间")
    set_font(run, size=8.5)
    run.font.color.rgb = MUTED

    paired_rows = []
    paired_name_labels = {
        "text_audio": "文本＋语音",
        "text": "文本",
        "acoustic": "声学",
        "audio": "语音融合",
        "visual": "视觉",
        "complete_multimodal": "全模态",
    }
    for _, row in paired.iterrows():
        paired_rows.append(
            [
                f"{paired_name_labels.get(row['model_a'], row['model_a'])} vs "
                f"{paired_name_labels.get(row['model_b'], row['model_b'])}",
                str(int(row["subjects"])),
                f"{row['auc_a']:.3f}",
                f"{row['auc_b']:.3f}",
                f"{row['auc_difference']:+.3f}",
                f"{row['ci_low']:+.3f} 至 {row['ci_high']:+.3f}",
                f"{row['paired_t_p_value']:.4f}",
            ]
        )
    add_table(
        doc,
        ["比较", "人数", "AUC A", "AUC B", "差值", "Bootstrap 95% CI", "折级 p"],
        paired_rows,
        widths=[5.3, 1.2, 1.5, 1.5, 1.4, 3.4, 1.4],
        font_size=7.9,
    )
    add_text(
        doc,
        "文本＋语音相对文本的点估计提高约 0.046，但受试者级 Bootstrap 区间仍跨 0；相对声学和语音融合的差异更小。"
        "全模态相对视觉和文本＋语音也没有表现出稳定优势。因此当前结果支持“融合有潜力”，尚不支持“融合已稳定提高准确性”。",
        bold_lead="配对结论：",
    )

    page_break(doc)

    add_heading(doc, "5.2 缺失模态与动态融合", 2)
    missing_labels = {
        "available_modality_dynamic_fusion": "可用模态动态融合",
        "dynamic_fusion_complete_subjects": "动态融合限定完整病例",
        "fixed_complete_multimodal": "固定四模态完整病例",
    }
    missing_rows = []
    for _, row in missing.iterrows():
        missing_rows.append(
            [
                missing_labels.get(row["strategy"], row["strategy"]),
                str(int(row["subjects"])),
                f"{row['auc']:.3f}",
                f"{row['auc_ci_low']:.3f} 至 {row['auc_ci_high'] if 'auc_ci_high' in row else float('nan'):.3f}",
                f"{row['brier']:.3f}",
            ]
        )
    add_table(
        doc,
        ["策略", "人数", "AUC", "Bootstrap 95% CI", "Brier"],
        missing_rows,
        widths=[5.5, 1.5, 1.5, 4.4, 1.8],
        font_size=8.6,
    )
    missing_comparison = advanced.get("missing_modality_comparison", {})
    add_text(
        doc,
        "动态融合允许受试者只要具备文本、视觉或语音中的可用模态就参与，不强制删除视觉缺失者。"
        f"它的覆盖率从 44 人扩展到 78 人，AUC 为 {missing.loc[missing['strategy'] == 'available_modality_dynamic_fusion', 'auc'].iloc[0]:.3f}。"
        f"在共同完整病例上，动态融合与固定四模态的 AUC 差为 "
        f"{missing_comparison.get('auc_difference', 0):+.3f}，"
        f"95% 区间 {missing_comparison.get('ci_low', 0):+.3f} 至 "
        f"{missing_comparison.get('ci_high', 0):+.3f}，没有显著差异。",
        bold_lead="覆盖与性能：",
    )

    add_heading(doc, "5.3 嵌套调参的多模型比较", 2)
    model_labels = {
        "logistic_l2": "Logistic L2",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
    }
    model_rows = []
    for _, row in model_comparison.iterrows():
        model_rows.append(
            [
                model_labels.get(row["model"], row["model"]),
                f"{row['auc']:.3f}",
                f"{row['auc_ci_low']:.3f}",
                f"{row['sensitivity']:.3f}",
                f"{row['specificity']:.3f}",
                f"{row['brier']:.3f}",
            ]
        )
    add_table(
        doc,
        ["模型", "AUC", "Bootstrap CI 下限", "灵敏度", "特异度", "Brier"],
        model_rows,
        widths=[4.1, 1.7, 3.1, 2.1, 2.1, 2.1],
        font_size=8.8,
    )
    doc.add_picture(str(model_chart), width=Inches(6.35))
    doc.inline_shapes[-1]._inline.docPr.set(
        "descr",
        "Logistic L2、Random Forest 和 Extra Trees 的受试者级 AUC 及 Bootstrap 置信区间",
    )
    doc.inline_shapes[-1]._inline.docPr.set("title", "CMDC 嵌套模型比较")
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(doc.paragraphs[-1], after=3)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(p, after=8)
    run = p.add_run("图 4 嵌套调参模型的 AUC 与不确定性")
    set_font(run, size=8.5)
    run.font.color.rgb = MUTED
    add_text(
        doc,
        f"Extra Trees 的点估计 AUC 最高，但三种模型的置信区间明显重叠，且 Extra Trees 的灵敏度较低。"
        f"当前自动选择结果为 {advanced.get('best_nested_model', '-')}，但它只是按 AUC 选择，"
        "最终上线模型还需要综合校准、灵敏度、复杂度和部署成本。",
        bold_lead="模型选择：",
    )

    add_heading(doc, "5.4 校准与决策曲线", 2)
    add_text(
        doc,
        f"已生成 {len(calibration)} 个校准分箱和 {len(decision_curve)} 个决策阈值。"
        "低概率分箱的预期风险接近 0，高概率分箱接近 1，但中间分箱样本较少，校准仍不稳定。"
        "决策曲线在多数组阈值下优于“全部干预”和“全部不干预”，但仍需独立样本验证。",
        bold_lead="新增可信度分析：",
    )
    doc.add_picture(str(calibration_chart), width=Inches(6.35))
    doc.inline_shapes[-1]._inline.docPr.set(
        "descr",
        "CMDC 外部验证模型的校准曲线和决策曲线",
    )
    doc.inline_shapes[-1]._inline.docPr.set("title", "CMDC 校准与决策曲线")
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(doc.paragraphs[-1], after=3)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(p, after=8)
    run = p.add_run("图 5 校准曲线与决策曲线")
    set_font(run, size=8.5)
    run.font.color.rgb = MUTED

    add_heading(doc, "5.5 固定受试者留出与阈值锁定", 2)
    holdout_metrics = holdout.get("holdout_metrics") or {}
    locked = holdout_metrics.get("locked_threshold") or {}
    sensitivity_ci = locked.get("sensitivity_exact_ci") or {}
    specificity_ci = locked.get("specificity_exact_ci") or {}
    train_metrics = holdout.get("train_cv_metrics") or {}
    add_text(
        doc,
        "为减少反复使用同一批交叉验证结果造成的乐观偏差，新增一个预先固定随机种子 "
        f"{holdout.get('holdout_seed', '-')} 的受试者级 80/20 留出分析。"
        "先用训练集内部折外概率选择筛查阈值，再固定模型和阈值评估测试集，测试标签不参与调参。",
        bold_lead="协议：",
    )
    add_table(
        doc,
        ["评估", "人数", "AUC", "PR-AUC", "灵敏度", "特异度", "Brier"],
        [
            [
                "训练集交叉验证",
                str(holdout.get("train_subjects", 0)),
                f"{train_metrics.get('auc', 0):.3f}",
                f"{train_metrics.get('pr_auc', 0):.3f}",
                f"{train_metrics.get('sensitivity', 0):.3f}",
                f"{train_metrics.get('specificity', 0):.3f}",
                f"{train_metrics.get('brier', 0):.3f}",
            ],
            [
                "锁定测试集",
                str(holdout.get("test_subjects", 0)),
                f"{holdout_metrics.get('auc', 0):.3f}",
                f"{holdout_metrics.get('pr_auc', 0):.3f}",
                f"{locked.get('sensitivity', 0):.3f}",
                f"{locked.get('specificity', 0):.3f}",
                f"{holdout_metrics.get('brier', 0):.3f}",
            ],
        ],
        widths=[4.0, 1.5, 1.7, 1.8, 1.9, 1.9, 1.7],
        font_size=8.6,
    )
    add_text(
        doc,
        f"训练集 {holdout.get('train_hc', 0)} 名 HC、{holdout.get('train_mdd', 0)} 名 MDD；"
        f"测试集 {holdout.get('test_hc', 0)} 名 HC、{holdout.get('test_mdd', 0)} 名 MDD。"
        f"训练内选出的锁定阈值为 {holdout.get('locked_threshold', 0):.4f}。"
        f"测试集混淆矩阵为 TP={locked.get('tp', 0)}、TN={locked.get('tn', 0)}、"
        f"FP={locked.get('fp', 0)}、FN={locked.get('fn', 0)}。",
        bold_lead="样本与阈值：",
    )
    doc.add_picture(str(holdout_chart), width=Inches(5.9))
    doc.inline_shapes[-1]._inline.docPr.set(
        "descr",
        "CMDC 锁定测试集灵敏度和特异度点估计及 95% 精确二项区间",
    )
    doc.inline_shapes[-1]._inline.docPr.set("title", "CMDC 锁定留出精确区间")
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(doc.paragraphs[-1], after=3)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fmt_paragraph(p, after=8)
    run = p.add_run("图 6 锁定测试集的精确二项区间")
    set_font(run, size=8.5)
    run.font.color.rgb = MUTED
    add_text(
        doc,
        f"灵敏度 {sensitivity_ci.get('successes', 0)}/{sensitivity_ci.get('trials', 0)}，"
        f"点估计 {sensitivity_ci.get('estimate', 0):.3f}，95% 精确区间 "
        f"{sensitivity_ci.get('ci_low', 0):.3f}-{sensitivity_ci.get('ci_high', 0):.3f}；"
        f"特异度 {specificity_ci.get('successes', 0)}/{specificity_ci.get('trials', 0)}，"
        f"点估计 {specificity_ci.get('estimate', 0):.3f}，95% 精确区间 "
        f"{specificity_ci.get('ci_low', 0):.3f}-{specificity_ci.get('ci_high', 0):.3f}。"
        "测试集只有 16 人，完美 AUC 会使普通 Bootstrap 区间退化；精确区间更能反映信息不足。",
        bold_lead="不要误读：",
    )
    add_text(
        doc,
        "这次留出验证证明固定划分、训练内选阈值和锁定评估的流程已经跑通，且点估计没有与交叉验证方向冲突；"
        "但它仍是同一 CMDC 数据源的小样本留出，不是多中心或老年独立外部测试，不能据此宣称模型已具备稳定临床性能。",
        bold_lead="证据边界：",
    )

    add_heading(doc, "5.6 多模态提升结论", 2)
    add_bullet(doc, "文本＋语音相对纯文本的点估计更高，但受试者级 Bootstrap 区间跨 0，不能宣称稳定显著提升。")
    add_bullet(doc, "四模态完整病例没有稳定优于视觉或文本＋语音；当前数据不足以证明“模态越多一定越好”。")
    add_bullet(doc, "可用模态动态融合把覆盖从 44 人扩展到 78 人，在完整病例上没有显著性能损失，更适合真实缺失场景。")
    add_bullet(doc, "Extra Trees 的 AUC 点估计最高，但置信区间与 Logistic、Random Forest 重叠，且灵敏度较低。")
    add_text(
        doc,
        "当前证据支持“中文多模态有研究价值、动态融合有工程价值”，尚不支持“复杂多模态融合已经稳定提高准确性”。",
        bold_lead="一句话结论：",
    )

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
    add_text(
        doc,
        "本轮已经把可直接实施的稳健性措施落地：同一受试者配对 Bootstrap、95% 置信区间、缺失模态动态融合、三种模型的嵌套调参比较、校准曲线和决策曲线；"
        "并完成固定种子的受试者级 61/16 留出、训练内阈值选择和锁定测试。"
        "下一阶段不再重复这些基础实验，而是争取多中心或老年独立留出、补齐元数据并验证跨中心迁移。",
        bold_lead="已完成：",
    )
    add_table(
        doc,
        ["优先级", "提升方向", "具体工作", "预期作用"],
        [
            ["P0", "多中心或老年独立留出", "在外部机构或独立老年队列预先锁定特征、模型和阈值", "检验本地 61/16 留出结果能否跨人群复现"],
            ["P0", "阈值与校准决策", "在训练内部选择阈值，再评估外部灵敏度、特异度和决策曲线", "避免用测试集反向选择阈值"],
            ["P1", "复杂门控融合", "在已完成的动态融合基础上比较注意力门控和模态 dropout", "只有独立验证优于简单融合时才采用"],
            ["P1", "数据补齐", "申请年龄、性别、共病、教育、用药和量表分数", "支持亚组、共病和严重程度分析"],
            ["P1", "老年外部验证", "寻找或建立独立老年中文多模态样本", "验证从成人混合样本到老年人的迁移性"],
            ["P2", "领域适配", "取得可复现的原始文本/语音后，用 CMDC 预训练编码器再在老年数据微调", "提高小样本条件下的表征能力"],
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
            ["第 1 周", "复核本地 61/16 锁定留出，完成模型卡与失败病例分析", "可复核的留出报告和错误案例清单"],
            ["第 2 周", "复核动态融合、门控融合和校准，比较覆盖与临床错误成本", "确定主分析与敏感性分析"],
            ["第 3 周", "申请或确认年龄、性别、共病、用药和量表数据", "更完整的外部验证和亚组分析"],
            ["第 4 周", "形成老年多模态外部验证申请清单与采集方案", "导师决策材料和下一步数据计划"],
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
        "“我们还固定了随机种子，把 77 人按受试者分成 61 人训练和 16 人测试，阈值只在训练集选择。锁定测试集 AUC 为 1.000，"
        "但测试集只有 11 名 HC 和 5 名 MDD；灵敏度 5/5 的 95% 精确区间是 0.478 到 1.000，所以这是一项流程和点估计验证，不是临床性能证明。”",
    )
    add_text(
        doc,
        "“我们进一步完成了配对 Bootstrap、95% 置信区间、缺失模态动态融合、嵌套模型比较、校准和决策曲线。"
        "配对结果显示文本＋语音的点估计更高，但置信区间仍跨 0；动态融合把覆盖从 44 人扩展到 78 人，没有观察到显著性能损失。”",
    )
    add_text(
        doc,
        "“所以当前可以确认的是中文多模态信号有研究价值、系统具备更完整的验证能力，但还不能宣称融合已经稳定提高准确性。"
        "下一步会争取多中心或老年独立留出、补齐元数据并验证跨中心迁移，而不是继续在 78 人样本上追求更高单点 AUC。”",
    )

    add_heading(doc, "11 建议向导师确认的事项", 1)
    add_table(
        doc,
        ["确认事项", "建议决策"],
        [
            ["CMDC 的论文定位", "作为中文多模态外部验证，不并入老年主队列"],
            ["是否继续申请元数据", "优先确认年龄、性别、共病和量表分数是否可获得"],
            ["是否寻找老年多模态外部数据", "若论文目标包含老年外部验证，应列为下一阶段重点"],
            ["下一步实验优先级", "先取得多中心或老年独立留出，再决定是否采用复杂融合"],
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
