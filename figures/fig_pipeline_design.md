# SOAD 主图：用户指定版本的矢量重绘

- 以用户指定的 exec-5c6bb5a2-6a7c-420f-b465-bb1dbaed5410.png 为布局和图像来源。参考图副本为 fig_pipeline_selected_reference.png。
- 保留该版本的宽幅五分区、照片位置、深度与曲率图排列、特征组括号及验证集监督示意。
- 文字为嵌入字体；网络方块、特征堆叠、树模型、连线、候选框和边框为矢量路径。
- 照片与深度图等嵌图裁自用户指定的生成式图稿，仅用于流程示意，不是实验检测结果或实测几何图。部分照片内部原有叠加标记保留在照片像素内。
- 输出 fig_pipeline.pdf、fig_pipeline.svg 与预览 fig_pipeline.png。论文引用 PDF。
- 重建命令：python code/fig_pipeline_selected.py。旧入口 fig_pipeline_vector.py、install_pipeline_figure.py 和 figs_extra.py 的 pipeline 均调用此版本。
- 上一轮用户不采用的第三版布局已备份为 fig_pipeline_third_layout.pdf/png/svg，绘图源为 code/fig_pipeline_vector_previous.py。
