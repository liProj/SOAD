"""Generate report.docx: the Chinese working report of the round. Numbers come from results/paper_numbers.json,
results/summary.json and results/tables/main.csv (the same sources as the paper); every available figure is embedded.
A run that has not finished is reported as unfinished, never filled in."""
import json, os, sys
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
B = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, f"{B}/code")
from write_paper import FIGS
ZH = {"fig:mod": "数据模态：RGB、深度、曲率图 B（凸度）与 N（起伏度）；上为训练地块，下为留出地块（数据集中图像本身旋转了 90°）",
      "fig:size": "果实尺寸、密度与类别分布；左图上轴为整帧缩到 640 px 后的尺寸", "fig:shift": "训练地块与留出地块的差异：公制宽度（像素宽×深度）不变，果实亮度/饱和度明显偏移",
      "fig:pipe": "本文流程：(1)(2) 改变训练数据，(3)–(5) 只在推理时起作用", "fig:align": "左：几何图是否随图像一起增强（探针实验）；右：公开代码训练的 concat 融合模型同一权重喂/不喂几何的验证 mAP50",
      "fig:curves": "验证曲线：左为公开代码整帧训练，右为切片训练", "fig:ladder": "留出地块上全部已完成模型的对比（虚线为原文印出值）", "fig:froc": "留出地块 FROC 与 PR 曲线",
      "fig:count": "按果实尺寸的召回率；每帧检测数 vs. 标注数（计数误差）", "fig:per": "证据重打分的消融与特征重要性", "fig:rob": "几何输入被遮挡/加噪时的鲁棒性",
      "fig:printed": "原文印出值、公开代码按 README 设置的实测值、本文流程三者对比", "fig:cost": "精度与推理代价（每帧 640 px 前向次数）", "fig:qual": "留出地块定性对比（绿=正确，红=误检，白虚线=标注）"}


def add_df(doc, df, title, max_rows=70, nd=3):
    doc.add_heading(title, level=3)
    df = df.head(max_rows)
    t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for j, c in enumerate(df.columns):
        t.rows[0].cells[j].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = "" if pd.isna(v) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(8)


def figs(doc, labs):
    avail = [f for f in FIGS if os.path.exists(f"{B}/figures/{f[0]}.png")]
    for n, _, lab in avail:
        if lab in labs:
            doc.add_picture(f"{B}/figures/{n}.png", width=Inches(6.2))
            doc.add_paragraph(f"图 {[f[2] for f in avail].index(lab) + 1}　{ZH[lab]}").runs[0].font.size = Pt(9)


def main():
    N = {k: str(v).replace("\\%", "%").replace("--", "–").replace("$-$", "−").replace("$<$", "<") for k, v in json.load(open(f"{B}/results/paper_numbers.json")).items()}
    S = json.load(open(f"{B}/results/summary.json")); R = S["runs"]; has = lambda t: t in R
    scr = pd.read_csv(f"{B}/screening.csv", keep_default_na=False)
    doc = Document()
    doc.add_heading("Artificial Intelligence in Agriculture 第 1 轮工作报告：GeoSphere-DETR（RGB-D 柑橘检测）的复现、公开代码审计与“原生分辨率切片”新流程（含对齐几何的对照检验）", 0)
    doc.add_paragraph("期刊：Artificial Intelligence in Agriculture（KeAi/Elsevier，金色开放获取，ISSN 2589-7217；主页 https://www.sciencedirect.com/journal/artificial-intelligence-in-agriculture ，"
                      "KeAi 页 https://www.keaipublishing.com/en/journals/artificial-intelligence-in-agriculture/ ，DOI 前缀 10.1016/j.aiia）。任务里给的 Springer collections 链接只是“地址示例”，不是该刊。"
                      "目标论文：GeoSphere-DETR: RGB-D citrus fruit detection in cluttered orchards via geometry-gated fusion and geometry-aware confidence recalibration，doi:10.1016/j.aiia.2026.08.010（正文 9 张图）。"
                      "本报告与 paper/main.pdf 配套，数字同源（results/paper_numbers.json）。")
    doc.add_heading("0 排除列表与下载（第 0、1 步）", 1)
    doc.add_paragraph("本机此前没有该期刊的任何记录（“刚下载的 100 篇”不存在），排除列表从本轮新建，放在公共位置：~/Documents/mdpi_papers/exclude/aiia_2589-7217.csv（230 条 DOI = 最新 230 条 Crossref 记录，含 1 条勘误）。"
                      "按发表日期倒序取最近 200 篇研究论文（2024-10 至 2026-10）→ latest200.csv（OpenAlex 摘要 191/200）。"
                      "下一轮：python tools/fetch_round.py --issn 2589-7217 --slug aiia --journal \"Artificial Intelligence in Agriculture\" --out AIIA2 --n 200 --round 2。")
    doc.add_paragraph("重要限制（如实说明）：该刊全文无法用程序获取——ScienceDirect 对脚本返回 403，KeAi 页面声明 tdm-reservation，Elsevier API 无 key 只给题录，Europe PMC 不收录该刊；"
                      "用 Chrome 打开文章页出现 “Are you a robot?” 验证码，按规则没有绕过。因此 200 篇的评估是“摘要级”的（screening.csv 的 evidence 列逐行注明），"
                      "只有通过 GitHub / Hugging Face 实际核实过的条目写了“checked”。若需要全文级筛选：请在已打开的 Chrome 标签页里手动过一次验证码，或提供 Elsevier API key。")
    doc.add_heading("1 200 篇逐篇评估（第 2–8 步）", 1)
    c = scr.groupby("category").size().sort_values(ascending=False).reset_index(); c.columns = ["类别", "篇数"]
    doc.add_paragraph(f"逐篇结果见 screening.csv（{len(scr)} 行：数据是否公开、位置与大小、是否实测可下载、复现难度、能否设计新方法、能否超过、综合评分）。"
                      f"该刊以田间机器人/采摘/导航、光谱、遥感估产为主，绝大多数论文使用自采田间数据且摘要中没有数据声明。"
                      f"经实际核实：数据与代码都能下载的只有 {int((scr.downloadable_checked.str.startswith('checked: dataset')).sum())} 篇（即目标论文）；CropFusionNet 的仓库只有代码、数据不入库；ADeepWeeD 的仓库只有 README。"
                      "按第 12 条“公开不充分的数据集不要”的要求，其余候选全部放弃。")
    add_df(doc, c, "表 1　200 篇的类别分布")
    add_df(doc, scr.sort_values("score", ascending=False)[["idx", "title", "category", "data_public", "downloadable_checked", "score"]].head(12).assign(title=lambda d: d.title.str.slice(0, 70)), "表 2　评分最高的 12 篇")
    doc.add_heading("2 数据对账（第 3、4、12 步）", 1)
    doc.add_paragraph(f"数据集：Hugging Face ariel0818/citrus-fruit-rgbd（CC BY 4.0，13,213 个文件，14.5 GB），已全部下载到 data/citrus/。对账：图像 {N['n_img']} 张、实例 {N['n_inst']} 个，与原文完全一致；"
                      f"划分 {N['train_img']}/{N['val_img']}/{N['test_img']}；无重复图像（MD5，含跨划分）；每张图都有标签、深度、几何图。"
                      f"有 9 个标签文件含负坐标（训练 8、测试 1），公开代码的加载器会整帧丢弃 → 实际训练 {N['train_tframes']} 帧，测试 {N['test_frames']} 帧 / {N['test_gt']} 个果。"
                      f"图像 1920×1200；果实边长中位数 {N['side_med']} px，{N['small_share']}% 小于 32 px；整帧缩到 640 px 后只有约 {N['side_640']} px。结论：这是公开充分、与原文对得上的数据集。详见 data_check.md。")
    figs(doc, ["fig:mod", "fig:size", "fig:shift"])
    doc.add_heading("3 复现原文方法与公开代码审计（第 5 步）", 1)
    doc.add_paragraph("公开代码是魔改的 ultralytics 8.4.9（commit be9cb88）。README 只给了一组训练设置：640 输入、100 轮、batch 8、从 rtdetr-l.pt 初始化——本轮严格按此运行。实测发现 4 个问题：")
    for s in ["(1) 原仓库无法 import：模型注册表引用了已删除的子包。处理：做了一个可运行副本，只删去悬空的类（原仓库原样保留，补丁在 results/fork_patch.diff）。",
              "(2) FPPI 代码读取一个从未被赋值的属性后提前返回，永远算不出 FPPI。处理：按同一定义（conf≥0.25、IoU 0.5 的误检数 / 图像数）离线计算。",
              f"(3) 最关键：几何图没有随图像一起做增强。探针实验（把每张图的灰度图当作它的“几何图”）：验证流程相关系数 {N['c_val']}；默认训练增强（mosaic 开，前 90 轮）下 40/40 个样本里根本没有几何图；mosaic 关掉后几何图存在但相关系数只有 {N['c_mosoff']}（错位）。"
              f"后果：用公开代码训练的 concat 融合模型训练损失正常下降，但验证 mAP50 从第 1 轮起≈0；同一权重喂几何 {N['probe_with']}，不喂几何 {N['probe_without']}。即前 90 轮融合支路根本没被训练。",
              "(4) 验证器永远给模型传 geom 参数，非 RT-DETR 模型（YOLO11m）在第一次验证时就报错，原仓库无法训练 YOLO 基线。处理：有几何才传。"]:
        doc.add_paragraph(s, style="List Bullet")
    doc.add_paragraph(f"基线复现：RT-DETR-L（RGB）在留出地块上 mAP50 {N['E1_rtdetr_m50']} / mAP50:95 {N['E1_rtdetr_m95']} / FPPI {N['E1_rtdetr_fppi']}（验证集 {N['E1_rtdetr_v50']} / {N['E1_rtdetr_v95']}）；原文印出 0.786 / 0.562 / 7.24。"
                      "离线评估器与公开代码的验证器数字一致到小数点后四位，所以不是评估口径问题。53.2 个误检/图中有 52.6 个是与任何标注 IoU<0.5 的框，重复框 0.09、错类 0.53。"
                      "原文的训练分辨率无法确认（全文拿不到，仓库和图里都没写）；640 px 下果实只有约 7 px，原文很可能用了更大的输入。因此下面所有对比都给两种：对“我跑的公开代码”（同条件）和对“原文印出值”（非同条件）。")
    if has("E4_rescore_s0"):
        doc.add_paragraph(f"原文完整模型（GeoSphere-DETR，公开代码原样）：mAP50 {N['E4_rescore_m50']} / mAP50:95 {N['E4_rescore_m95']} / FPPI {N['E4_rescore_fppi']}（印出 0.863 / 0.642 / 3.29）。")
    else:
        doc.add_paragraph("原文完整模型（GeoSphere-DETR，公开代码原样）的 100 轮训练在生成本报告时尚未跑完（第一次运行在第 4 轮验证时卡住约 2 小时，为让新方法试点先出结果而中止后重跑）。跑完后重新生成本报告即可补上。")
    if has("Y11m_s0"):
        doc.add_paragraph(f"YOLO11m（修补验证器后才能训练）：mAP50 {N['Y11m_m50']} / FPPI {N['Y11m_fppi']}（印出 0.715 / 8.735）。")
    figs(doc, ["fig:align", "fig:curves", "fig:printed"])
    doc.add_heading("4 新方法（第 6、9 步）", 1)
    doc.add_paragraph("诊断：在 README 设置下，主要误差来源不是“没有几何信息”，而是“分辨率丢了”——22 px 的果缩成 7 px；同时公开的训练流程没有把几何图送到正确的位置。据此设计的流程不改网络结构（仍是原文的 RT-DETR-L + 几何门控颈部 + 球面头），只改数据和推理：")
    for s in [f"① 原生分辨率切片：每帧切成 4×3 个 640×640 切片（重叠 213/280 px），训练 {N['train_tiles']} 个切片 / {N['train_tboxes']} 个框；10 轮 = 18,490 次迭代，与整帧 100 轮（18,700 次）预算相同。推理时丢弃贴着内部切片边界的框，平移回整帧后做类内 NMS，最多保留 300 个。",
              f"② 几何对齐（只用于参考结构这一变体，用来公平检验几何支路）：把 B、N 两张曲率图作为第 4、5 通道与图像叠在一起再做 mosaic/仿射/翻转，之后再拆开。探针相关系数：无增强 {N['c_al']}，完整训练增强 {N['c_almos']}。",
              "③ 翻转融合：原图与水平镜像各推理一次，用加权框融合（WBF，IoU 0.55）合并；推理代价翻倍。",
              f"④ 证据重打分：对每个检测框计算 39 个特征（检测分数与帧内上下文、框几何、深度、公制尺寸、曲率、颜色），用 LightGBM 在验证集检测结果上拟合“是否真果”分类器和 IoU 回归器，新分数 = p^(1−α)·IoU^α；α={N['alpha']} 只在验证集 5 折袋外预测上选，留出地块只用于最终评估。"]:
        doc.add_paragraph(s, style="List Bullet")
    figs(doc, ["fig:pipe"])
    doc.add_heading("5 主要结果（第 7 步）", 1)
    doc.add_paragraph(f"留出地块（{N['test_frames']} 帧 / {N['test_gt']} 个果），所有模型都在整帧上用同一个评估器打分（mAP50 / mAP50:95 / conf 0.25 的 FPPI）：基线 RT-DETR-L {N['E1_rtdetr_m50']} / {N['E1_rtdetr_m95']} / {N['E1_rtdetr_fppi']}；"
                      + (f"原文完整模型（公开代码按 README 设置）{N['E4_rescore_m50']} / {N['E4_rescore_m95']} / {N['E4_rescore_fppi']}，比同条件的 RGB 基线还低；" if "E4_rescore_m50" in N else "原文完整模型（公开代码）尚未跑完；")
                      + f"切片 RGB（不用几何）{N['T_rgb_m50']} / {N['T_rgb_m95']} / {N['T_rgb_fppi']}；切片 + 对齐几何（参考结构）{N['T_E4_aligned_m50']} / {N['T_E4_aligned_m95']} / {N['T_E4_aligned_fppi']}。"
                      f"主方法按验证集选择（翻转融合后验证集 mAP50:95：切片 RGB {N['hv_rgb']}，切片+几何 {N['hv_geo']}），选中的是切片 RGB；加翻转融合和重打分后为 {N['H_wbf_per_m50']} / {N['H_wbf_per_m95']} / {N['H_wbf_per_fppi']}，"
                      f"几何变体做同样两步为 {N['T_E4_aligned_wbf_per_m50']} / {N['T_E4_aligned_wbf_per_m95']} / {N['T_E4_aligned_wbf_per_fppi']}。"
                      f"召回 0.70 时每图误检：基线 {N['E1_rtdetr_f70']} → 主方法 {N['H_wbf_per_f70']}。每帧计数误差（conf 0.25）：基线 {N['E1_rtdetr_cmae']} → 主方法 {N['H_wbf_per_cmae']}。"
                      f"按测试帧配对 bootstrap（1000 次）：基线→主方法 mAP50 {N['b_all_50']}（{N['b_all_50_ci']}），mAP50:95 {N['b_all_95']}（{N['b_all_95_ci']}），p {N['b_all_95_p']}。")
    doc.add_paragraph("如实结论：① 新流程大幅超过“公开代码按 README 设置跑出来”的基线和完整模型，也超过原文印出的 RT-DETR 基线（0.786 / 0.562），但没有达到原文印出的 GeoSphere-DETR（0.863 / 0.642 / FPPI 3.29）。"
                      "由于原文训练分辨率无法确认，与印出值的比较不是同条件比较；论文里没有声称超过原文印出结果。"
                      "② 提升几乎全部来自原生分辨率切片；在分辨率恢复、几何对齐之后，参考结构的几何支路没有带来可测的精度增益，它改变的只是置信度标定（固定阈值下更保守）。"
                      "③ 切片、翻转融合、二阶段重打分都是已有技术，方法学创新有限；本文的价值主要在测量和复现审计。")
    main = pd.read_csv(f"{B}/results/tables/main.csv").drop(columns=["run", "family"])
    add_df(doc, main, "表 3　留出地块主结果（全部已完成的模型）")
    add_df(doc, pd.DataFrame(S["printed"], columns=["原文印出", "mAP50", "mAP50:95", "FPPI"]), "表 4　原文印出值")
    figs(doc, ["fig:ladder", "fig:froc", "fig:count", "fig:qual"])
    doc.add_heading("6 消融实验（第 10 步）", 1)
    doc.add_paragraph(f"增益分解（配对 bootstrap，95% 区间）：整帧→切片 RGB mAP50 {N['b_tile_50']}（{N['b_tile_50_ci']}）、mAP50:95 {N['b_tile_95']}（{N['b_tile_95_ci']}），是主要来源；"
                      f"翻转融合 {N['b_flip_50']} / {N['b_flip_95']}；证据重打分 {N['b_per_50']} / {N['b_per_95']}。")
    if "b_geo_50" in N:
        doc.add_paragraph(f"几何支路的对照：切片 RGB → 切片 + 对齐几何（参考结构）mAP50 {N['b_geo_50']}（{N['b_geo_50_ci']}），mAP50:95 {N['b_geo_95']}（{N['b_geo_95_ci']}，p = {N['b_geo_95_p']}）——没有可测的精度增益。"
                          f"同召回率下误检相同（召回 0.70 时 {N['T_E4_aligned_f70']} 对 {N['T_rgb_f70']}）；差别只在固定置信度 0.25 处：几何模型 FPPI {N['T_E4_aligned_fppi']}（召回 {N['T_E4_aligned_rec']}），RGB 模型 {N['T_rgb_fppi']}（召回 {N['T_rgb_rec']}），是分数标定不同。"
                          "只有 1 个种子，这个小差值只能读作“没有可测增益”，不能读作“几何有害”。")
    if "b_e1e4_50" in N:
        doc.add_paragraph(f"公开代码内部对比：RGB 基线 → 原文完整模型 mAP50 {N['b_e1e4_50']}、mAP50:95 {N['b_e1e4_95']}（{N['b_e1e4_95_ci']}）——公开流程复现不出原文“加几何后提升”的方向，与几何未对齐的发现一致。")
    if has("T_E4_unaligned_s0_full"):
        doc.add_paragraph(f"切片 + 作者原加载器（几何不对齐）：{N['T_E4_unaligned_m50']} / {N['T_E4_unaligned_m95']}" + (f"；对齐减不对齐 = {N['b_align_50']} / {N['b_align_95']}（mAP50:95 的 p = {N['b_align_95_p']}）。也就是说在切片条件下，几何对不对齐也测不出差别——这与门控探针的结论一致：门控值几乎是常数，网络基本没有用这条支路，所以输入错位也伤不到它。对齐缺陷是真实存在的，对拼接融合是致命的，但修好它并不能让门控支路变得有用。" if "b_align_50" in N else "。"))
    else:
        doc.add_paragraph("“切片 + 作者原加载器”的消融（T_E4_unaligned_s0）在生成本报告时尚未跑完。")
    if has("T_rgb20_s0_full"):
        doc.add_paragraph(f"切片 RGB 第二个 10 轮周期（共 20 轮）：单视图 {N['T_rgb20_m50']} / {N['T_rgb20_m95']}" + (f"（相对第一个周期 {N['b_long_50']} / {N['b_long_95']}）" if "b_long_50" in N else "") + (f"；加翻转融合和重打分 {N['T_rgb20_wbf_per_m50']} / {N['T_rgb20_wbf_per_m95']}。" if "T_rgb20_wbf_per_m50" in N else "。"))
    else:
        doc.add_paragraph("切片 RGB 的第二个 10 轮训练周期（T_rgb20_s0）在生成本报告时尚未跑完。")
    doc.add_paragraph(f"证据重打分的分组消融（主方法的翻转融合结果上）：mAP50 {N['ab_base']} → {N['ab_all']}（全部证据）；只做分数再校准 {N['ab_score']}；上下文各加一组：框几何 {N['ab_box']}、深度 {N['ab_depth']}、公制尺寸 {N['ab_size']}、曲率 {N['ab_curv']}、颜色 {N['ab_colour']}；去掉颜色 {N['ab_nocolour']}。深度和曲率没有贡献。"
                      f"在检测器自身召回率下，留出地块每图误检为 {N['per_fr_before']}（重打分前）对 {N['per_fr_after']}（重打分后，{N['per_seeds']} 个种子）；它稳定改变的是分数尺度（conf 0.25 处 FPPI {N['H_wbf_fppi']} → {N['H_wbf_per_fppi']}，召回 {N['H_wbf_rec']} → {N['H_wbf_per_rec']}）。"
                      "所以它是一个标定步骤，排序收益很小，只作为可选步骤，不作为主要贡献。")
    rs = json.load(open(f"{B}/results/rescore_{S['headline']}_wbf.json"))["ablation"]
    add_df(doc, pd.DataFrame([dict(证据组=k, 验证集袋外_mAP5095=v["val_oof"]["map5095"], 测试_mAP50=v["test"]["map50"], 测试_mAP5095=v["test"]["map5095"], 同召回_FPPI=v["test"]["fppi_at_base_recall"]) for k, v in rs.items()]), "表 5　证据重打分分组消融")
    import glob
    rob = sorted(glob.glob(f"{B}/results/det/T_E4_aligned_s0_miss*.json")) + sorted(glob.glob(f"{B}/results/det/T_E4_aligned_s0_noise*.json"))
    if rob:
        doc.add_paragraph(f"几何鲁棒性（切片几何模型，留出地块，干净输入 mAP50:95 {N['T_E4_aligned_m95']} / FPPI {N['T_E4_aligned_fppi']}）：" + "；".join(
            f"{'遮挡' if 'miss' in f else '噪声σ'} {os.path.basename(f)[:-5].split('miss')[-1].split('noise')[-1]} → {json.load(open(f))['test']['map5095']:.3f} / {json.load(open(f))['test']['fppi']:.2f}" for f in rob)
            + "。即使把曲率图全部遮掉，精度也基本不变。")
    else:
        doc.add_paragraph("几何遮挡/噪声鲁棒性评估在生成本报告时尚未跑完。")
    if os.path.exists(f"{B}/results/gate_probe.json"):
        gp = json.load(open(f"{B}/results/gate_probe.json")); lv = [k for k in gp["clean"] if k.startswith("level")]
        doc.add_paragraph(f"原因（实测）：参考结构的融合是“特征 ×(1 + α·门控)”，α = {gp['alpha']:g}，几何只能把特征放大 1～{1 + gp['alpha']:g} 倍。训练后的门控在三个尺度上的均值为 "
                          + "、".join(f"{gp['clean'][k]['mean']:.2f}" for k in lv) + "，单张图内的标准差只有 " + "、".join(f"{gp['clean'][k]['std_within_image']:.3f}" for k in lv)
                          + "；曲率图全部遮掉后均值为 " + "、".join(f"{gp['masked'][k]['mean']:.2f}" for k in lv)
                          + "。也就是说门控近似常数，融合近似整体缩放，检测器实际上接近一个 RGB 检测器——这同时解释了“几何没有增益”和“去掉几何也不掉点”。所以“对几何缺失鲁棒”在这里并不能说明几何被用好了。")
    figs(doc, ["fig:per", "fig:rob", "fig:cost"])
    doc.add_paragraph("被否定的想法（如实记录）：① 不重新训练、仅在推理时把切片放大到 960 px（几何变体）：验证集 mAP50 "
                      + (f"{N['T_E4_aligned_v50']} → {N['im960_v50']}，mAP50:95 {N['T_E4_aligned_v95']} → {N['im960_v95']}" if "im960_v50" in N else "下降")
                      + f"，在验证集上就被否决，没有在留出地块上试。② 全局亮度（gamma）归一化：帧级亮度中位数训练 {N['v_train']}、留出 {N['v_test']}，几乎相同——偏移只发生在果实本身（在阴影里），全局校正无效。"
                      "③ 深度/曲率证据对重打分没有贡献。④ 参考结构的几何支路在原生分辨率、几何对齐的条件下没有精度增益（见上）。")
    if "T960_E4_aligned_m50" in N:
        doc.add_paragraph(f"960 px 微调（4 轮，单视图）：{N['T960_E4_aligned_m50']} / {N['T960_E4_aligned_m95']}（640 px 切片模型 {N['T_E4_aligned_m50']} / {N['T_E4_aligned_m95']}），每切片代价 2.25 倍。")
    doc.add_heading("7 本文的意义与如实说明", 1)
    doc.add_paragraph("意义：① 对实践者——不需要新网络结构、也不需要深度通道的做法：按原生分辨率切片、镜像视图融合、在验证集上确定工作点；并按同召回率比较误检。在该数据集上把每帧多报近 50 个果的检测器变成误差约 4–6 个的检测器。若要用深度派生通道，必须让它和图像“粘在一起”过增强。"
                      "② 对研究记录——给出了核对过的数据集、可运行的参考代码补丁，以及一组可复用的探针实验（验证多模态输入是否在增强后仍然对齐）。"
                      "③ 对原文的结论——在公开流程里，“拼接 vs. 门控”的比较实际上比较的是“各自如何忽略一个没被训练过的输入”；几何对齐并恢复分辨率后，我们测不到几何支路的精度增益，融合模块的贡献应当相对于“看得清果实的 RGB 检测器”来衡量。")
    doc.add_paragraph("如实说明：① 没有达到原文印出的完整模型数字；与印出值的比较不是同条件。② 主模型只训练了 1 个种子，bootstrap 区间反映的是测试帧抽样的不确定性，不是训练随机性。"
                      "③ 切片模型每帧需要 12–24 次前向，适合离线估产，不适合 30 Hz 控制。④ 重打分在 200 张验证帧上拟合，对留出地块只部分迁移。⑤ 200 篇的筛选是摘要级的。"
                      "⑥ 本机同时有其他会话的任务在占用 GPU，推理耗时没有可靠测量，所以只报告前向次数。")
    doc.add_heading("8 输出位置", 1)
    for s in ["工作目录：~/Documents/mdpi_papers/AIIA/", "论文：paper/main.pdf（LaTeX 源 paper/main.tex）", "本报告：report.docx", "图：figures/（PDF+PNG）", "结果：results/summary.json、results/tables/main.csv、results/bootstrap.json、results/rescore_*.json、results/det/、results/dets/",
              "筛选表：screening.csv（200 行）、results/shortlist.csv", "数据对账：data_check.md、results/data_check.json", "数据：data/citrus/（原始）、data/tiles/、data/tiles_flip/", "代码：code/（一键重建图表/论文/报告：bash code/finalize.sh）",
              "参考代码：third_party/Geosphere-DETR/（原样）、third_party/Geosphere-DETR-run/（可运行副本）、results/fork_patch.diff", "排除列表（公共位置）：~/Documents/mdpi_papers/exclude/aiia_2589-7217.csv", "清单：OUTPUTS.md"]:
        doc.add_paragraph(s, style="List Bullet")
    doc.save(f"{B}/report.docx")
    print("wrote report.docx")


if __name__ == "__main__":
    main()
