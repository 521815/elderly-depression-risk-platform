#!/usr/bin/env python3
"""Build the mentor-facing dataset application report DOCX."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


HEADING_BLUE = RGBColor(0x2E, 0x74, 0xB5)
HEADING_DARK = RGBColor(0x1F, 0x4D, 0x78)
INK = RGBColor(0x0B, 0x25, 0x45)
MUTED = RGBColor(0x55, 0x55, 0x55)
BLACK = RGBColor(0x00, 0x00, 0x00)
HEADER_FILL = "E8EEF5"
ZEBRA_FILL = "F7F9FC"

EA_BODY = "宋体"
EA_HEAD = "黑体"
LATIN = "Calibri"

OUT = Path.home() / "Desktop" / "数据集申请与导师汇报材料.docx"


def set_font_ea(target, latin: str, east: str) -> None:
    if hasattr(target, "font"):
        target.font.name = latin
        rpr = target.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        rfonts.set(qn("w:ascii"), latin)
        rfonts.set(qn("w:hAnsi"), latin)
        rfonts.set(qn("w:eastAsia"), east)
    else:
        run = target
        run.font.name = latin
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        rfonts.set(qn("w:ascii"), latin)
        rfonts.set(qn("w:hAnsi"), latin)
        rfonts.set(qn("w:eastAsia"), east)


def setup_document(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.left_margin = Inches(1)
    sec.right_margin = Inches(1)
    sec.top_margin = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.header_distance = Inches(0.492)
    sec.footer_distance = Inches(0.492)

    normal = doc.styles["Normal"]
    normal.font.size = Pt(11)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    set_font_ea(normal, LATIN, EA_BODY)

    for name, size, color, before, after, ea in [
        ("Heading 1", 16, HEADING_BLUE, 18, 10, EA_HEAD),
        ("Heading 2", 13, HEADING_BLUE, 14, 7, EA_HEAD),
        ("Heading 3", 12, HEADING_DARK, 10, 5, EA_HEAD),
    ]:
        style = doc.styles[name]
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        set_font_ea(style, LATIN, ea)

    header_para = sec.header.paragraphs[0]
    header_para.text = ""
    header_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = header_para.add_run("老年抑郁风险预测大模型  |  数据集申请与导师汇报材料")
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED
    set_font_ea(run, LATIN, EA_BODY)

    footer = sec.footer.paragraphs[0]
    footer.text = ""
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = footer.add_run("第 ")
    r.font.size = Pt(9)
    r.font.color.rgb = MUTED
    set_font_ea(r, LATIN, EA_BODY)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    footer._p.append(fld)
    r2 = footer.add_run(" 页")
    r2.font.size = Pt(9)
    r2.font.color.rgb = MUTED
    set_font_ea(r2, LATIN, EA_BODY)


def add_title_block(doc: Document) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run("老年抑郁风险预测大模型")
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = INK
    set_font_ea(run, LATIN, EA_HEAD)

    p2 = doc.add_paragraph()
    p2.paragraph_format.space_after = Pt(4)
    run2 = p2.add_run("数据集申请与导师汇报材料：真实样本扩展路径与低/中/高样本量下的模型优化路线")
    run2.font.size = Pt(13)
    run2.font.color.rgb = MUTED
    set_font_ea(run2, LATIN, EA_BODY)

    metadata = [
        ("汇报日期", "2026-09-03"),
        ("提交主体", "大创课题组；对外申请建议以导师名义提交"),
        ("当前家底", "CHARLS 表格基线 AUC 约 0.83；MPDD-Elderly 有标签 87 人，受试者级融合 AUC 约 0.70"),
        ("建议下一步", "优先确认 MPDD-Young / MPDD-AVG-2026 许可范围，再启动其他公开数据申请"),
    ]
    for label, value in metadata:
        mp = doc.add_paragraph()
        mp.paragraph_format.space_after = Pt(3)
        mp.paragraph_format.line_spacing = 1.15
        lr = mp.add_run(f"{label}：")
        lr.font.size = Pt(10)
        lr.font.bold = True
        lr.font.color.rgb = INK
        set_font_ea(lr, LATIN, EA_BODY)
        vr = mp.add_run(value)
        vr.font.size = Pt(10)
        vr.font.color.rgb = BLACK
        set_font_ea(vr, LATIN, EA_BODY)

    rule = doc.add_paragraph()
    rule.paragraph_format.space_before = Pt(8)
    rule.paragraph_format.space_after = Pt(2)
    pPr = rule._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "12")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "2E74B5")
    pBdr.append(bottom)
    pPr.append(pBdr)


def set_cell_margins(table) -> None:
    tbl = table._tbl
    tblPr = tbl.tblPr
    margins = OxmlElement("w:tblCellMar")
    for name, val in [("top", 80), ("bottom", 80), ("start", 120), ("end", 120)]:
        el = OxmlElement(f"w:{name}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        margins.append(el)
    tblPr.append(margins)


def set_table_geometry(table, widths_inches: list[float]) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    total_dxa = int(sum(widths_inches) * 1440)
    widths_dxa = [int(w * 1440) for w in widths_inches]
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:w"), str(total_dxa))
    tblW.set(qn("w:type"), "dxa")
    tblLayout = OxmlElement("w:tblLayout")
    tblLayout.set(qn("w:type"), "fixed")
    tblPr.append(tblLayout)
    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths_dxa):
            gc.set(qn("w:w"), str(w))
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            if idx < len(widths_dxa):
                cell.width = Inches(widths_inches[idx])
    set_cell_margins(table)


def shade_cell(cell, fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def set_header_row_repeat(table) -> None:
    if not table.rows:
        return
    tr = table.rows[0]._tr
    trPr = tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def format_cell(cell, text: str, bold: bool = False, size: float = 9.5, fill: str | None = None) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.12
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if idx > 0:
            run = p.add_run()
            run.add_break()
        run = p.add_run(line)
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = BLACK
        set_font_ea(run, LATIN, EA_BODY)
    if fill:
        shade_cell(cell, fill)


def add_table(doc: Document, rows: list[list[str]], widths: list[float], zebra: bool = True) -> None:
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return
    padded = [r + [""] * (n_cols - len(r)) for r in rows]
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.style = "Table Grid"
    set_table_geometry(table, widths)
    for ridx, row in enumerate(padded):
        for cidx, text in enumerate(row):
            cell = table.cell(ridx, cidx)
            if ridx == 0:
                format_cell(cell, text, bold=True, size=9.5, fill=HEADER_FILL)
            else:
                fill = ZEBRA_FILL if (zebra and ridx % 2 == 0) else None
                format_cell(cell, text, size=9, fill=fill)
    set_header_row_repeat(table)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(2)
    spacer.paragraph_format.space_after = Pt(4)
    spacer.add_run("")


def add_body(doc: Document, text: str, bold_lead: str | None = None, size: float = 11, after: float = 6) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.25
    if bold_lead:
        lead = p.add_run(bold_lead)
        lead.font.bold = True
        lead.font.size = Pt(size)
        lead.font.color.rgb = INK
        set_font_ea(lead, LATIN, EA_BODY)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.color.rgb = BLACK
    set_font_ea(run, LATIN, EA_BODY)


def main() -> None:
    doc = Document()
    setup_document(doc)
    add_title_block(doc)

    doc.add_heading("一、汇报摘要", level=1)
    add_body(
        doc,
        "本项目现有两条真实数据主线：CHARLS 中国中老年纵向表格数据，以及经官方许可申请获得的 MPDD-Elderly 多模态数据。"
        "当前不是模型结构不够新，而是带标签的中文老年多模态真实样本太少，导致复杂模型容易过拟合、外部验证难以成立。"
        "模拟数据已在训练侧做了多轮科学验证，但不能替代独立真实样本。",
    )
    add_body(
        doc,
        "本材料记录已采用的公开数据，以及需申请但免费的数据申请入口，并按“未来真实样本量”给出低、中、高三条模型路线，"
        "供导师决定下一步申请哪些数据、以及论文口径如何表述。所有链接均于 2026-09-03 复核。",
    )

    doc.add_heading("二、当前数据家底", level=1)
    add_body(doc, "先固定现状，再谈扩展，避免把不同协议、不同标签口径的数据直接混用。")
    add_table(
        doc,
        [
            ["数据", "可用内容", "现状与限制"],
            [
                "CHARLS",
                "中国中老年纵向调查；人口学、共病、功能、生活方式、CESD-10",
                "本机已有 2011-2020 与 Harmonized 数据；标准表测试受试者 2617 人，随机森林/XGBoost/逻辑回归 AUC 约 0.83。表格数据，无对话音视频。",
            ],
            [
                "MPDD-Elderly",
                "中国有基础病老年人；中文半结构化访谈音视频、步态 IMU、BigFive 人格、PHQ-9",
                "110 人；训练 87 人有公开标签，官方测试 23 人标签不公开。受试者级融合 AUC 约 0.70；原始数据与聚合特征不可再分发。",
            ],
        ],
        widths=[1.10, 2.35, 3.05],
    )

    doc.add_heading("三、采用的公开免费数据集（老年队列 + 跨文化扩展）", level=1)
    add_body(
        doc,
        "本项目面向上海等多元城市，公开可直接收录项分为两类：中国老年队列 CLHLS 用于同人群跨队列验证；HRS、ELSA、SHARE 用于跨文化敏感性分析，检验表格模型在多元人口背景下的稳定性。上述队列均注册或接受数据条款后免费下载。"
        "公开免费数据中没有面向中国老年人群的音视频多模态真实样本，因此本阶段只能扩大表格模型的真实样本外与跨文化验证；"
        "多模态真实样本仍须走 MPDD 同源申请或长期队列。",
    )
    add_table(
        doc,
        [
            ["数据集", "人群与模态", "规模", "官方入口", "建议用途"],
            [
                "CLHLS",
                "中国 65+ 高龄老人为主；问卷",
                "多波次，样本量大",
                "opendata.pku.edu.cn/dataverse/CHADS",
                "CHARLS 表格模型真实样本外扩展与跨队列验证",
            ],
            [
                "HRS",
                "美国 50+；问卷",
                "多波次，样本量大",
                "hrs.isr.umich.edu",
                "多元文化背景下的跨文化敏感性分析",
            ],
            [
                "ELSA",
                "英国 50+；问卷",
                "多波次，样本量大",
                "elsa-project.ac.uk/accessing-elsa-data",
                "跨文化敏感性分析",
            ],
            [
                "SHARE",
                "欧洲多国 50+；问卷",
                "多波次，样本量大",
                "share-eric.eu/data/data-access",
                "跨文化敏感性分析",
            ],
        ],
        widths=[1.10, 1.60, 1.10, 1.60, 1.10],
    )

    doc.add_heading("四、免费但需申请或签协议的数据集", level=1)
    add_body(
        doc,
        "以下数据均免费用于学术研究，但需要教师签名、机构注册或数据方人工审核。建议统一由导师作为申请主体。",
    )
    add_table(
        doc,
        [
            ["数据集", "人群与模态", "规模", "申请方式", "建议用途"],
            [
                "MPDD-Elderly 2025",
                "中国有基础病老年人；访谈音视频 + 步态 + 人格",
                "110 人",
                "已由导师申请获得；扩量前先确认协议覆盖范围",
                "当前核心数据集",
            ],
            [
                "MPDD-Young 2025",
                "中国年轻人；同一采集协议",
                "110 人",
                "hacilab.github.io/MPDDChallenge.github.io；发 sstcneu@163.com",
                "两年龄组对比、跨组泛化",
            ],
            [
                "MPDD-AVG 2026",
                "更新版 Young 110 + Elderly 110；音视频 + 步态 + 原始人格/人口学/健康",
                "220 人",
                "HF gated：huggingface.co/datasets/chasonfff/MPDD-AVG-2026 页面 Request access，人工审核",
                "同源更新样本；先确认许可关系",
            ],
            [
                "D-Vlog",
                "YouTube vlog；语音与视觉特征",
                "961 条、约 160 小时",
                "作者页 Google Form 后发 yoonjeewoo@gmail.com",
                "野外特征预训练，非老年外部验证",
            ],
            [
                "MODMA",
                "中文；EEG + 语音",
                "128 导 EEG 53 人、3 导 EEG 55 人、语音 52 人",
                "modma.lzu.edu.cn/data/index 注册申请",
                "中文抑郁语音/生理信号补充",
            ],
            [
                "CMDC",
                "中文；半结构化访谈音视频 + 文本",
                "以官方发布为准",
                "IEEE DataPort 下载后获密码；签 EULA 发 zoubochao@ustb.edu.cn",
                "中文多模态迁移",
            ],
            [
                "DAIC-WOZ / E-DAIC",
                "英语；临床访谈",
                "189 会话 / 275 人",
                "dcapswoz.ict.usc.edu Apply Now，填表签署，需学术邮箱",
                "基准对比与预训练",
            ],
            [
                "MMDA",
                "中文；门诊与模拟门诊访谈音视频文本",
                "以官方论文为准",
                "无统一公开表单；先确认论文与实验室可用性，再联系通讯作者",
                "中文多模态候选，待核验",
            ],
        ],
        widths=[1.18, 1.70, 1.02, 1.55, 1.05],
    )

    doc.add_heading("五、申请动作与合规要点", level=1)
    add_body(doc, "对外申请建议由导师作为全职教师/研究员提交，学生不直接以个人名义申请。", bold_lead="申请主体：")
    add_body(
        doc,
        "研究题目与目的、所属单位、数据使用范围、参与成员、申请的子集名称、拟发表成果类型、数据保管与删除承诺。",
        bold_lead="材料内容：",
    )
    add_body(
        doc,
        "对 MPDD 系列每次申请新子集，都先问清现有许可是否覆盖，还是需要另签协议，并保留数据方书面确认。",
        bold_lead="许可确认：",
    )
    add_body(
        doc,
        "仅学术研究；不商用；不向课题组外再分发；不披露个人可识别信息；项目结束或数据方要求时删除。MPDD 原始数据与派生数据不进入公开仓库、网盘或微信传输。",
        bold_lead="合规承诺：",
    )
    add_body(
        doc,
        "先建变量与标签口径映射表；同一采集协议才考虑合并训练；跨协议数据先做分布诊断，再决定迁移或独立验证。",
        bold_lead="接入检查：",
    )

    doc.add_heading("六、按未来真实样本量的模型路线", level=1)
    add_body(
        doc,
        "样本量不同，允许用的模型复杂度就不同。路线按“能拿到的真实标签量”设计，不把模拟数据当作真实样本计入。",
    )
    add_table(
        doc,
        [
            ["情形", "预计真实标签量", "模型策略", "验证口径"],
            [
                "低样本：现状",
                "MPDD-Elderly 87 + CHARLS 表格",
                "低方差模型、特征筛选、受试者级重复交叉验证、概率校准与阈值选择；不引入大参数深度模型",
                "内部验证 + 时间外验证；模拟组只用于训练侧研究",
            ],
            [
                "中样本：同源补充",
                "补充 MPDD-Young 或 AVG-2026 后约 200 上下",
                "同协议合并训练；两年龄组对比、跨年龄组泛化、人格/共病亚组分析；轻量融合头",
                "真实受试者级重复验证；新增样本重跑全部基线",
            ],
            [
                "高样本：中文多模态齐全",
                "MPDD 同源补充 + CMDC/MODMA 等申请到位后",
                "预训练特征 + 轻量分类头 + 域适应；CMDC 补中文访谈迁移，MODMA 作生理/语音补充",
                "跨协议先做变量映射与分布诊断；只对协议可比的真实样本称外部验证",
            ],
            [
                "外部队列",
                "CHARLS + CLHLS + HRS/ELSA/SHARE",
                "CLHLS 做中国老年跨队列真实样本外验证；HRS/ELSA/SHARE 做跨文化敏感性分析",
                "表格模型验证用；不进入多模态训练；不冒充上海本地前瞻验证",
            ],
        ],
        widths=[1.05, 1.30, 2.20, 1.95],
    )
    add_body(
        doc,
        "需申请的英语或非老年语料只作为预训练特征来源或领域泛化参考，不写入“已完成外部验证”的结论。",
        bold_lead="边界提醒：",
    )

    doc.add_heading("七、需要导师决策的事项", level=1)
    add_body(
        doc,
        "确认现有 MPDD 许可是否覆盖 MPDD-Young 与 MPDD-AVG-2026；若覆盖即可申请同源补充，若未覆盖则按官方流程另签协议。",
        bold_lead="事项一：",
    )
    add_body(
        doc,
        "是否以导师名义启动 CMDC、MODMA 等中文数据申请。",
        bold_lead="事项二：",
    )
    add_body(
        doc,
        "是否在伦理与知情同意程序可行后，把桌面工作台作为长期科研筛查工具积累真实随访队列。",
        bold_lead="事项三：",
    )

    doc.add_heading("八、给导师的汇报要点", level=1)
    add_body(
        doc,
        "若本周目标是扩大表格模型的验证规模，建议先注册下载 CLHLS，按 CESD-10 与人口学/共病变量映射接入，与 CHARLS 做中国老年跨队列真实样本外验证；HRS、ELSA、SHARE 同步纳入跨文化敏感性分析，支撑上海多元人群场景。",
        bold_lead="表格真实验证：",
    )
    add_body(
        doc,
        "多模态真实样本仍以同源 MPDD-Young / MPDD-AVG-2026 为最直接补充；CMDC、MODMA 等申请数据到位后只做迁移与补充，不改变“老年中文多模态真实验证”的论文口径。",
        bold_lead="多模态样本：",
    )
    add_body(
        doc,
        "模拟组只用于训练侧研究；外部验证结论只能来自真实且协议可比的数据。公开直接收录部分只记录本材料采用项，CLHLS 写中国老年跨队列验证，HRS/ELSA/SHARE 写跨文化敏感性分析，不冒充上海本地前瞻验证。",
        bold_lead="论文口径：",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
