"""Update the advisor progress report with the corrected MPDD evaluation."""

from __future__ import annotations

import copy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
DOCX_PATH = ROOT / "docs" / "老年抑郁风险预测大模型_导师汇报材料.docx"
FIGURE_PATH = ROOT / "paper" / "figures" / "chapters" / "fig3_multimodal.png"
DATA_HEALTH_FIGURE_PATH = (
    ROOT / "paper" / "figures" / "chapters" / "fig_data_health.png"
)


def find_paragraph(document, startswith: str):
    for paragraph in document.paragraphs:
        if paragraph.text.strip().startswith(startswith):
            return paragraph
    raise ValueError(f"Paragraph not found: {startswith}")


def replace_text(paragraph, text: str) -> None:
    if not paragraph.runs:
        paragraph.add_run(text)
        return
    paragraph.runs[0].text = text
    for run in paragraph.runs[1:]:
        run.text = ""


def remove_paragraph(paragraph) -> None:
    parent = paragraph._p.getparent()
    if parent is not None:
        parent.remove(paragraph._p)


def format_body_paragraph(paragraph) -> None:
    paragraph.style = paragraph.style
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.1
    for run in paragraph.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(11)
        run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "PingFang SC")


def append_styled_row(table, values: list[str]):
    new_tr = copy.deepcopy(table.rows[-1]._tr)
    table._tbl.append(new_tr)
    row = table.rows[-1]
    for idx, value in enumerate(values):
        cell = row.cells[idx]
        cell.text = value
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.name = "Calibri"
                run.font.size = Pt(9)
                run._element.get_or_add_rPr().rFonts.set(
                    qn("w:eastAsia"), "PingFang SC"
                )
    return row


def replace_figure(document, caption_prefix: str, image_path: Path, width=None) -> None:
    image_paragraph = None
    image_index = None
    for index, paragraph in enumerate(document.paragraphs):
        if not paragraph._p.xpath(".//a:blip"):
            continue
        following = [
            item.text.strip()
            for item in document.paragraphs[index + 1 : index + 3]
            if item.text.strip()
        ]
        if following and following[0].startswith(caption_prefix):
            image_paragraph = paragraph
            image_index = index
            break
    if image_paragraph is None or image_index is None:
        raise ValueError(f"Figure not found: {caption_prefix}")

    shape_index = sum(
        1
        for paragraph in document.paragraphs[:image_index]
        if paragraph._p.xpath(".//a:blip")
    )
    image_width = width or document.inline_shapes[shape_index].width
    for child in list(image_paragraph._p):
        if child.tag == qn("w:r"):
            image_paragraph._p.remove(child)
    image_paragraph.add_run().add_picture(str(image_path), width=image_width)
    image_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def resize_figure(document, caption_prefix: str, width) -> None:
    for index, paragraph in enumerate(document.paragraphs):
        if not paragraph._p.xpath(".//a:blip"):
            continue
        following = [
            item.text.strip()
            for item in document.paragraphs[index + 1 : index + 3]
            if item.text.strip()
        ]
        if not following or not following[0].startswith(caption_prefix):
            continue
        shape_index = sum(
            1
            for item in document.paragraphs[:index]
            if item._p.xpath(".//a:blip")
        )
        shape = document.inline_shapes[shape_index]
        ratio = shape.height / shape.width
        shape.width = width
        shape.height = int(width * ratio)
        return
    raise ValueError(f"Figure not found: {caption_prefix}")


def add_page_break_before(paragraph) -> None:
    previous = paragraph._p.getprevious()
    if previous is not None and previous.xpath(".//w:br[@w:type='page']"):
        return
    page_break = paragraph.insert_paragraph_before()
    page_break.add_run().add_break(WD_BREAK.PAGE)


def remove_page_break_before(paragraph) -> None:
    previous = paragraph._p.getprevious()
    if previous is not None and previous.xpath(".//w:br[@w:type='page']"):
        previous.getparent().remove(previous)


def add_page_break_before_table(document, table) -> None:
    previous = table._tbl.getprevious()
    if previous is not None and previous.xpath(".//w:br[@w:type='page']"):
        return
    page_break = document.add_paragraph()
    page_break.add_run().add_break(WD_BREAK.PAGE)
    table._tbl.addprevious(page_break._p)


def main() -> None:
    document = Document(DOCX_PATH)

    replace_text(find_paragraph(document, "日期："), "日期：2026 年 11 月 10 日")
    replace_text(
        find_paragraph(document, "材料版本："),
        "材料版本：V2.0（数据准确性与模型口径复核）",
    )
    replace_text(
        find_paragraph(document, "当前数据基础："),
        "当前数据基础：CHARLS 标准表 32,936 条；MPDD 受试者 110 名"
        "（有官方标签 87 人、测试 23 人）、15,506 维多模态特征"
        "（语音、面部、步态、人格）；对话模拟组 500 条、表格模拟组 2,000 条；"
        "用户记录页支持多用户档案与评估历史管理。",
    )
    replace_text(
        find_paragraph(document, "数据健康页自动检查"),
        "数据健康页自动检查 CHARLS、MPDD、模拟组和用户记录质量，并新增"
        "官方 label2 覆盖、阳性/阴性人数及 PHQ-9 阈值口径分歧审计，"
        "接新数据前可先发现标签错配、类别失衡和模态缺失。",
    )

    mpdd_paragraph = next(
        (
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.strip().startswith(
                ("MPDD 采用受试者级", "MPDD 二分类主分析")
            )
        ),
        None,
    )
    if mpdd_paragraph is None:
        raise ValueError("MPDD evaluation paragraph not found")
    replace_text(
        mpdd_paragraph,
        "MPDD 二分类主分析优先使用官方 label2：87 名有标签受试者中阳性 28 人、"
        "阴性 59 人；旧 PHQ-9 ≥8 仅识别 17 名阳性，漏掉 11 名官方阳性。"
        "数据表中的 hamd_score 只是 PHQ-9 兼容别名，不是 HAMD-24。",
    )

    image_paragraph = next(
        paragraph
        for index, paragraph in enumerate(document.paragraphs)
        if paragraph._p.xpath(".//a:blip")
        and any(
            item.text.strip().startswith("图 9")
            for item in document.paragraphs[index + 1 : index + 3]
        )
    )
    model_paragraph = next(
        (
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.strip().startswith(
                ("正式预测流程改为", "在同一协议下")
            )
        ),
        None,
    )
    if model_paragraph is None:
        model_paragraph = image_paragraph.insert_paragraph_before()
        format_body_paragraph(model_paragraph)
    replace_text(
        model_paragraph,
        "本次模型改为受试者片段均值聚合、训练折内 64 维 PCA 和强正则化 L2 "
        "逻辑回归，可用模态等权融合并固定阈值 0.5。5×5 受试者分组验证得到 "
        "AUC 0.627、PR-AUC 0.531、Brier 0.212、准确率 0.656；同协议旧 PHQ-9 "
        "阈值标签的 AUC 0.747 只作口径对照，不作为正式结果。",
    )
    for paragraph in list(document.paragraphs):
        if paragraph is model_paragraph:
            continue
        if paragraph.text.strip().startswith("在同一协议下"):
            remove_paragraph(paragraph)

    replace_figure(document, "图 4", DATA_HEALTH_FIGURE_PATH)
    replace_figure(document, "图 9", FIGURE_PATH, width=Inches(5.2))
    resize_figure(document, "图 11", Inches(5.2))

    fusion_paragraph = next(
        (
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.strip().startswith(
                ("初始使用文献先验权重", "MPDD 组内采用等权概率融合")
            )
        ),
        None,
    )
    if fusion_paragraph is None:
        raise ValueError("Fusion paragraph not found")
    replace_text(
        fusion_paragraph,
        "MPDD 组内采用等权概率融合，最终多源风险仍保留文献先验或学习型融合。"
        "本次 17 人留出 AUC 0.571、Brier 0.245，仅用于流程检查，"
        "稳定性能仍需外部真实数据验证。",
    )

    conclusion_paragraph = next(
        (
            paragraph
            for paragraph in document.paragraphs
            if paragraph.text.strip().startswith(
                ("本轮把“数据准确性”", "项目已从单一模型脚本")
            )
        ),
        None,
    )
    if conclusion_paragraph is None:
        raise ValueError("Conclusion paragraph not found")
    replace_text(
        conclusion_paragraph,
        "项目已从单一模型脚本发展为可演示、可复现、可接真实数据的完整科研工作台。"
        "本轮把“数据准确性”落实到标签、训练和评估三层：统一官方 label2，"
        "修正 PHQ-9/HAMD 字段语义，按受试者聚合片段特征，采用强正则化模型、"
        "固定阈值和受试者分组验证。正式融合结果虽低于旧阈值口径的表面数字，"
        "但任务定义更真实、流程可追溯，后续可在更大队列上继续优化。",
    )

    test_table = document.tables[2]
    replacements = {
        "单元测试": "MPDD、多模态、API、模型卡等核心 32 项通过",
        "数据健康": "当前 healthy=True，0 个阻断问题，7 个风险提示",
    }
    for row in test_table.rows[1:]:
        key = row.cells[0].text.strip()
        if key in replacements:
            row.cells[1].text = replacements[key]
    if not any(row.cells[0].text.strip() == "Prompt 审计" for row in test_table.rows):
        append_styled_row(test_table, ["Prompt 审计", "访谈与陪伴 2 个版本全部通过"])

    products_table = document.tables[3]
    if not any(
        row.cells[0].text.strip() == "MPDD 标签口径对照"
        for row in products_table.rows
    ):
        append_styled_row(
            products_table,
            [
                "MPDD 标签口径对照",
                "output/mpdd_label_protocol_comparison.csv",
                "官方 label2 与旧 PHQ-9 阈值的同协议对照",
            ],
        )

    add_page_break_before_table(document, document.tables[1])
    remove_page_break_before(find_paragraph(document, "六、验证与测试"))

    document.core_properties.title = "老年抑郁风险预测大模型导师汇报材料"
    document.core_properties.subject = "数据准确性、预测流程与主要突破"
    document.core_properties.description = "2026 年 11 月 10 日汇报版"
    document.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
