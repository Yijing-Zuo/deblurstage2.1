# deblurstage2.1

读取 deblur 后的 **Out**，输出逐行英文文字 PDF。重新下载固定版本的官方 Paddle 模型、重新识别；不加载旧 OCR 缓存、v3 微调权重、Qwen 或 DeepSeek。

本版重点是修复重复框、漏行和异常短读数，并在字形支持下选择有限的字母／单词候选。它不会把模糊证据强行改写成通顺文章，也不保证恢复原文。Clear、Blur 只出现在最后的比较图中，不进入检测、识别、词典或候选评分。

## 服务器安装

在 ICRN 的 JupyterLab **Terminal** 执行。代码与 `qaoa` 同级；输入、权重和结果在独立的 `deblur-assets` 中。本项目不会删除旧文件或环境。

```text
~/
├── qaoa/
├── deblurstage2.1/                # 只放代码
└── deblur-assets/
    ├── data/docsity/samples.jsonl
    ├── data/docsity/images/       # Out、Blur
    ├── data/docsity_references/   # 可选的 Clear 对照
    ├── model-cache/               # 本版新下载的模型
    └── runs/stage2_1/formal/       # 本版实验结果和证据
```

`samples.jsonl` 和图片必须先保留／上传到上述位置。允许复用原始输入图片；不需要保留旧模型和旧运行结果。每行至少有 `id`、`out`；`out`、`blur` 路径相对样本清单所在目录：

```json
{"id":"4_004","document_id":"4","out":"images/Out_4_004.png","blur":"images/Blur_4_004.png"}
```

可选的 `data/docsity_references/clear.jsonl` 每行是 `{"id":"4_004","clear":"images/Clear_4_004.png"}`。清单或图片缺失时比较 PDF 标明 Clear unavailable；恢复流程不需要它。

先取得新仓库代码。下面同时适用于已克隆的空仓库和还未克隆的情况；命令成功后再安装环境。

```bash
cd ~ &&
if [ -d deblurstage2.1/.git ]; then
  git -C deblurstage2.1 -c safe.directory="$HOME/deblurstage2.1" pull --ff-only
else
  git clone https://github.com/Yijing-Zuo/deblurstage2.1.git
fi &&
cd ~/deblurstage2.1
```

新建一个环境。已存在 `deblur21` 时从激活开始，不必重复创建。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n deblur21 -c conda-forge python=3.11 libgl=1.7 -y &&
conda activate deblur21 &&
python -m pip install paddlepaddle-gpu==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ &&
python -m pip install -r requirements.txt &&
python -m pip check
```

沿用此前服务器已使用的 Paddle 3.2.0/cu126 和 PaddleX 3.7.0 接口版本，但使用全新的环境。无需 Torch、Transformers、FlashAttention 或大语言模型。`libgl` 为 OpenCV 提供原生库；运行命令仅对当前进程设置它的搜索路径，不修改系统库。

## 正式运行全部数据

默认 `config.yaml` 选取 **157 个完整区域**（4号74个、14号83个）。0号缺片区域不包含在本次清单中；其他训练文档不需要 Out。157个区域曾用于查看 Clear 和设计方案，所以应称开发／回归实验。

下载三套固定 SHA 的官方模型，不运行 GPU 推理：

```bash
cd ~/deblurstage2.1 &&
conda activate deblur21 &&
python pipeline.py download
```

首次下载可能需要访问 Hugging Face。模型缓存路径由配置明确指定，不读取旧 `HF_HOME`。若该路径存在非本版缓存，程序会拒绝混用；请在配置里选一个新的空目录，不必删除其他项目的缓存。模型下载中断后，重复下载命令。

接着直接运行完整实验，包含 OCR、候选选择和 PDF，不包含 GPU smoke test：

```bash
cd ~/deblurstage2.1 &&
conda activate deblur21 &&
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
python -u pipeline.py all --offline
```

如果要脱离终端运行，**用下面命令替代上面的正式运行命令，不要同时启动两个**：

```bash
cd ~/deblurstage2.1 &&
conda activate deblur21 &&
mkdir -p ../deblur-assets/runs/stage2_1 &&
nohup env LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  python -u pipeline.py all --offline \
  > ../deblur-assets/runs/stage2_1/formal.log 2>&1 < /dev/null &
```

日志在 `~/deblur-assets/runs/stage2_1/formal.log`。`nohup` 可以应对终端断开，但不能阻止服务器／容器被关闭；重新启动后可按下文恢复。

## 查看结果

JupyterLab 左侧进入 `deblur-assets/runs/stage2_1/formal/`：

| 文件 | 用途 |
|---|---|
| `ocr.pdf` | 补读和视觉选择后、词典纠错前的逐行文字 |
| `recovered.pdf` | 保守局部修改后的最终白底黑字页面，可复制文字 |
| `comparison.pdf` | **Clear｜Out｜Blur｜Final** 四列，Final 超长时有续页 |
| `review.html` | 行裁图、两模型原读数、候选、修改和复核原因 |
| `ocr.jsonl` | 本次新识别的页面、行框和每次读数 |
| `pages.jsonl` / `lines.jsonl` | 最终页面／逐行结果及修改依据 |
| `run.json` | 当前运行状态、配置、模型文件哈希、耗时和失败数 |
| `cache/` / `crops/` | 本次生成的字符概率、逐项缓存和裁图，续跑时需保留 |

HTML 的裁图为相对路径。下载 HTML 查看时应一起下载 `crops/`；只下载 HTML 不会带走图片。PDF 独立可用。

`done` 只表示流程完成；`needs_review` 表示字词、接缝或覆盖仍有疑问；`error` 表示实际处理失败。失败区域仍会进入 PDF 并有提示，不会悄悄消失。存在失败时程序最终返回退出码2，并保留已完成结果。

PDF 默认512×768逻辑尺寸、10.5pt字、15pt行距，保留阅读顺序，每条物理行单独开始。长行自动换行、内容超出则续页；它是可读的重新排版，不是逐像素复刻原页面。

## 中断、重跑和更新

中断后，先确认原进程已停止，再**重复同一个正式运行命令和同一个 run_dir**。完整页面直接跳过；失败页面重试，但其中已保存、校验通过的识别观察继续复用。每次识别立即保存裁图、字符概率和哈希，JSON 使用原子替换。断电发生在一次识别尚未落盘时，该次识别需要重做。

只改词典或候选选择设置时，无需重新跑 GPU OCR：

```bash
cd ~/deblurstage2.1 &&
conda activate deblur21 &&
python pipeline.py decode &&
python pipeline.py render
```

只改字号、行距等排版设置，运行 `python pipeline.py render`。要保留调参前的 PDF，可先从 JupyterLab 下载。解码变化自动改变解码缓存键；同一运行目录下的最终 PDF／JSON 是当前版本，候选缓存仍保留。

`--mode visual` 仅供 `decode/all`，关闭词典修改。默认 `ocr.pdf` 已经提供同一视觉基线，因此通常不用额外运行它。选子集时必须给独立运行目录，例如 `--documents 14 --run-dir ../deblur-assets/runs/stage2_1/doc14`；同一运行目录不能混用不同样本集合。

更新代码（不要在正在运行的进程中更新）：

```bash
cd ~/deblurstage2.1 &&
git -c safe.directory="$PWD" pull --ff-only
```

这里的 Git 信任设置只作用于这次命令，避免此前共享服务器 ownership 报错。拉取失败时先处理报错，不要继续假定已更新。

## 代码与选择规则

```text
pipeline.py   命令入口、有限补读、缓存和断点恢复
ocr.py        固定模型下载、PaddleX接口、原始CTC概率
layout.py     拆行、同栏合并／去重、阅读顺序、残余区域
ctc.py        CTC束搜索和精确候选评分
decode.py     两模型证据选择、局部词／词组候选
render.py     PDF和HTML
storage.py    数据路径、字符范围、原子文件和哈希
config.yaml   唯一配置
```

检测器为 `PP-OCRv6_medium_det`，识别器为 `PP-OCRv6_medium_rec` 与 `en_PP-OCRv5_mobile_rec`。模型和版本在配置中固定，不用 `latest`。PaddleX内部接口也锁到3.7.0，防止升级后悄悄改变概率或裁图含义。

Out检测后先拆高框、合并同栏同基线片段并去重。残余笔画只触发一次局部再检测，最多8个ROI；不会凭暗像素直接制造文字行。读数异常时先收紧上下空白，再最多3个重叠短段；接缝不可靠时保留原读数并标记复核。

允许 A–Z/a–z、0–9、ASCII标点和空格。原始非英语读数保留供审查，但不把未知文字音译成英文。曲引号、破折号及常见排版连字有明确的文字正规化表；正规化读数可成为候选，仍须通过视觉评分，**不合并／重归一CTC概率来制造支持**。被排除字符的概率质量另外保存。

每词保留原读数、另一模型读数、小束搜索和少量英文词典候选；默认编辑距离2，短于5字符上限1，首版不自动扩大到3。数字串不做拼写修正。每3词一个小窗口，保留实际OCR完整短语，再加入有界组合。固定窗口边界会遗漏部分跨窗口合词，不能当作完整语言理解。

对每个模型比较修改前后的相对CTC支持，每个模型仅使用一组覆盖合适的视图。两个模型都要有支持，明显反对时否决；通用单词／相邻词频只作弱排序。无法判定时允许保留近似字母串。阈值在配置中可见，尚未在2.1真实结果上校准。

## 验证范围与参考

仓库包含CPU回归测试：CTC重复字符、概率质量、错误词频覆盖、分段拼接、几何去重／分栏、坏缓存恢复、中断续跑、PDF长词／续页／缺失结果。PDF还做了合成页面的渲染检查。**这些验证的是代码行为，不是本数据的OCR正确率；本地没有运行H200真实模型实验。**

开发者可安装额外测试依赖后运行（服务器正式实验不要求这一步）：

```bash
python -m pip install pypdf==6.10.0
python -m unittest discover -s tests -v
```

官方实现参考：[PaddleX 3.7.0](https://github.com/PaddlePaddle/PaddleX/tree/v3.7.0)、[CTC识别接口](https://github.com/PaddlePaddle/PaddleX/tree/v3.7.0/paddlex/inference/models/text_recognition)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[SymSpell](https://github.com/mammothb/symspellpy)。模型来源和不可变SHA见 `config.yaml`。
