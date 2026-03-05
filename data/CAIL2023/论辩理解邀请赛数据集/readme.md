# 论辩理解邀请赛 输入输出格式说明

本赛道邀请赛阶段包含两个任务：争议观点对抽取（下称单模态）和多模态争议识别（下称多模态）。邀请赛测试集的案由与入门赛不重叠，且涵盖刑事与民事案件。所有数据均以 JSON Line 格式存储，每个样本一个 JSON 字符串；文件中出现的 Unicode 字符（中文等）以转义方式存储，可用 `pandas` 或 `datasets` 包直接读取。

两个任务下发的数据集包括训练集 `*train_*.jsonl` 和测试集 `*test_*.jsonl`，不同任务的数据（以及同一任务下不同种类的数据）以文件名前缀区分。

单模态数据集中包含的文件有：

- `train_text.jsonl`：包含了庭审笔录中辩诉双方辩护全文的分句数据，共 $8094$ 条。每条数据包含的字段内容如下：
  - `sentence_id`：句子 ID
  - `text_id`：庭审笔录 ID
  - `category`：刑事、民事案件分类
  - `chapter`：刑事罪名或民事案由所在章节
  - `crime`：具体的刑事罪名或民事案由
  - `position`：诉方（sc）与辩方（bc）标志
  - `sentence`：句子文本

- `train_pair.jsonl`：包含了 $3425$ 对庭审笔录中的互动论点对，每条数据包含的字段内容如下：
  - `text_id`：庭审笔录 ID
  - `sc_id`：诉方论点，以 `multi_train_text.jsonl` 中的 `sentence_id` 表示
  - `bc_id`：辩方论点，以 `multi_train_text.jsonl` 中的 `sentence_id` 表示

- `dirty_train_text.jsonl` ：从入门赛训练数据抽取得到的补充训练数据，共 $24578$ 条，格式与 `train_text.jsonl` 一致。
- `dirty_train_pair.jsonl`：从入门赛训练数据抽取得到的补充训练数据，共 $3112$ 条，格式与 `train_pair.jsonl` 一致，但未包含所有可能的观点对。
- `test_text.jsonl`：格式与 `multi_train_text.jsonl` 完全一致，共 $12634$ 条数据。

多模态数据集中包含的文件有：

- `multi_train_text.jsonl`：`train_text.jsonl` 的多模态任务版本，共 $645$ 条数据。
- `multi_train_pair.jsonl`：`train_pair.jsonl` 的多模态任务版本，共 $196$ 条数据。
- `multi_test_text.jsonl`：`test_text.jsonl` 的多模态任务版本，共 $940$ 条数据。

除庭审笔录数据外，本赛道还为多模态任务提供庭审音频和转录文本。每一份庭审笔录的多模态数据都存储在以庭审笔录 ID 命名的文件夹内，其中有一份 `.wav` 格式的庭审音频文件和一份 `.txt` 转录文本文件。转录文本包括人工标注的发言者和自动转录的发言内容，按发言顺序记录。

具体可供下载的文件有：

- `train.7z`：多模态训练集的庭审音频和转录文本，共 $73$ 份。
  SHA256：D4935F4F656914736E26A181A0C49FA707FA69964233616202A7CEFE646FC704
  百度网盘链接: <https://pan.baidu.com/s/1EvjFIDAaD-Whygs4ybb8qA>
  提取码: LbLj

- `test.7z`：多模态测试集的庭审音频和转录文本，共 $61$ 份。特别地，`test_text.jsonl` 中有多模态数据的属于多模态任务，没有多模态数据的属于单模态任务。
  SHA256：9F5C504A3DF7DAF42478D483FD636C1A8F2FC54CDD41A6157941E90CDCB030F6
  百度网盘链接: <https://pan.baidu.com/s/1SfddGGWJIxowho5epU-IXA>
  提取码: LbLj

参加邀请赛的选手需要提交与 `test_text.jsonl` 和 `multi_test_text.jsonl` 相对应的 `test_pair.jsonl` 文件，其中包含**两个任务**的预测结果（数据保证两个任务测试集的 `text_id`、`sc_id` 和 `bc_id` 不重叠，训练集不保证）。提交文件的格式请参照 `train_pair.jsonl`，并须保证双方论点的句子 ID（`sc_id` 和 `bc_id`）和庭审笔录 ID（`text_id`）能够配对——评测脚本会自动忽略不配对的预测条目。
