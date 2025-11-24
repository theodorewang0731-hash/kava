# 模块与方法清单（供人工核对）

说明：下列为仓库中主要 Python 模块及其类/函数/方法签名与简短用途说明。若需要我可以进一步列出参数、返回值或调用站点。

---

## 顶层脚本

- `train.py`
  - `parse_args()`：解析 CLI 参数。
  - `main()`：训练入口，加载配置并调用 `KAVATrainer`。

- `evaluate.py`
  - `parse_args()`：解析 CLI 参数。
  - `main()`：评估入口，使用 `KAVAEvaluator`。
  - `KAVAEvaluator` 类：
    - `__init__(...)`：初始化评估器（模型路径、tokenizer 等）。
    - `load_model(checkpoint_path: str)`：加载模型与检查点。
    - `generate_answer(...)`：基于输入生成答案/解答。
    - `evaluate_dataset(...)`：对某个数据集批量评估并返回结果。

- `inference.py`
  - `parse_args()`、`main()`：推理入口。
  - `KAVAInference` 类：
    - `__init__(...)`：初始化（模型、tokenizer、设备）。
    - `generate(...)`：逐条生成答案。
    - `batch_generate(...)`：批量生成接口。
    - `interactive_mode()`：交互式模式（REPL）。

- `run_multi_seed.py`
  - `MultiSeedRunner` 类：
    - `__init__(...)`：配置多次实验（种子、输出路径）。
    - `run_single_seed(seed: int) -> Dict`：运行单个种子实验并返回结果。
    - `_find_latest_checkpoint(output_dir: Path) -> Path`：查找最新 checkpoint。
    - `run_all_seeds() -> List[Dict]`：依序运行所有种子并收集结果。
    - `_save_intermediate_results(results: List[Dict])`：保存中间结果。
    - `aggregate_results(all_results: List[Dict]) -> Dict`：聚合多次运行结果。
    - `print_summary_table(aggregated: Dict)`：打印汇总表。
    - `save_summary(aggregated: Dict)`：保存汇总到文件。
    - `run_full_experiment()`：完整实验流程。
  - `parse_args()`、`main()`：脚本入口。

- `quick_model_test.py`
  - `test_model_loading()`：快速检查模型能否加载。

- `smoke_test.py` / `smoke_test_lite.py`
  - `test_directory_structure()`：检查目录结构是否齐全。
  - `test_src_modules()`（lite）：检查 `src` 模块导入/存在性。
  - `test_configs()`：检查配置文件是否存在且符合预期字段。
  - `test_data_loading()`：数据加载基本检查。
  - `test_rkv_compression()`：RKV 压缩模块的快速自检。
  - `test_losses()`：损失计算单元测试样例。
  - `main()`：运行一组 smoke tests。

- `pre_training_check.py`
  - `check_file_exists(path: str, description: str) -> bool`：检查文件是否存在。
  - `check_import(module_name: str) -> bool`：尝试导入模块。
  - `check_config_files() -> bool`：检查配置文件完整性。
  - `check_core_files() -> bool`：检查核心源码文件。
  - `check_src_modules() -> bool`：检查 `src` 包模块。
  - `check_dependencies() -> bool`：检查依赖包安装情况。
  - `check_gpu() -> bool`：GPU 可用性检查。
  - `check_hpc_models() -> bool`：HPC/集群相关模型检查。
  - `check_slurm_config() -> bool`：SLURM 配置检查。
  - `main()`：组合运行所有检查。


## `src` 子模块（按文件）

- `src/rkv_compression.py`
  - `class RKVCompressor`：实现 R-KV 压缩与选择逻辑。
    - `__init__(...)`：配置初始化（k, latent_tokens, lambda, layerwise_std 等）。
    - `compute_importance_score(...)`：计算 token importance 分数（用于筛选）。
    - `compute_redundancy_score(...)`：计算冗余评分。
    - `select_top_tokens(...)`：基于得分选择 top-k token indices。
    - `compress(...)`：对 KV 张量执行压缩流程，返回压缩后 KV 或索引。
    - `apply_projection(...)`：可选投影矩阵应用（低秩投影/线性变换）。
    - `normalize_layerwise(...)`：按层标准化 KV（若启用 layerwise_std）。
  - `extract_kv_from_outputs(outputs)`（模块级函数）：从模型输出中抽取 KV 张量的助手函数。

- `src/losses.py`
  - `class KVDistillationLoss(nn.Module)`：
    - `__init__(...)`：初始化（loss type、权重、layerwise 标准化选项等）。
    - `normalize_layerwise(...)`：按层标准化张量（若需要）。
    - `compute_loss(pred_kv, target_kv, ...)`：计算 KV 蒸馏损失（mse 或 smooth_l1）。
    - `forward(...)`：返回 KV 蒸馏损失值。
  - `class CODILoss(nn.Module)`：
    - `__init__(...)`：CODI 相关超参设置。
    - `forward(...)`：CODI 损失计算（用于潜变量一致性/对齐）。
  - `class KAVALoss(nn.Module)`：
    - `__init__(...)`：KAVA 总损失组合（CE + KV distill + CODI 等权重）。
    - `compute_ce_loss(...)`：计算交叉熵部分。
    - `forward(...)`：组合各项损失并返回字典/标量。

- `src/latent_reasoning.py`
  - `class LatentReasoningModule(nn.Module)`：实现潜变量推理与 Jacobi 迭代流程。
    - `__init__(...)`：设置 latent tokens 数量、Jacobi 迭代次数、投影层、dropout 等。
    - `initialize_latent_tokens(...)`：初始化潜变量（可学习或基于输入初始化）。
    - `jacobi_iteration(...)`：一次 Jacobi 更新步骤，迭代以收敛 latent 表示。
    - `forward_student(...)`：学生模型侧的前向流程，返回学生侧 latent KV 与 logits 等。
    - `extract_latent_kv(...)`：从 latent token 表示构建 KV 张量以供 loss/蒸馏使用。
    - `forward_teacher(...)`：教师模型前向（用于生成 target KV/labels）；有时作为独立路径。
  - 模块级函数：
    - `prepare_labels_for_student(...)`：根据教师输出准备学生侧标签。
    - `prepare_labels_for_teacher(...)`：为教师准备标签（如果需要）。

- `src/data_utils.py`
  - `class GSM8KDataset`：处理 GSM8K 数据集与分割。
    - `__init__(...)`：加载原始数据路径、tokenizer、split 等。
    - `verify_dataset_sizes(self)`：检查 train/val/test 划分数量。
    - `add_special_tokens(self)`：向 tokenizer/模型添加特殊 token（若需要）。
    - `format_teacher_prompt(self, sample)`：格式化教师输入 prompt。
    - `format_student_prompt(self, sample)`：格式化学生 prompt（可能省略步骤/chain）。
    - `tokenize_teacher_sample(self, sample)`：将教师样本转为 token ids。
    - `tokenize_student_sample(self, sample)`：将学生样本转为 token ids。
    - `get_train_dataset(self) -> Dataset`：返回训练用 Dataset 对象。
    - `get_val_dataset(self) -> Dataset`：返回验证集 Dataset 对象。
    - `get_test_dataset(self) -> Dataset`：返回测试集 Dataset 对象。
  - `collate_fn_teacher(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict`：teacher 批处理函数。
  - `collate_fn_student(batch: List[Dict], tokenizer: PreTrainedTokenizer) -> Dict`：student 批处理函数。
  - `extract_answer_number(answer_text: str) -> Optional[float]`：从字符串提取数值答案（辅助评测）。

- `src/trainer.py`
  - `class KAVATrainer`：主训练器。
    - `__init__(self, config: Dict)`：加载训练配置。
    - `setup_model(self)`：模型加载、LoRA/adapter/参数冻结等设置。
    - `setup_data(self)`：dataset/dataloader 初始化。
    - `setup_training(self)`：优化器、scheduler、loss、混合精度设置等。
    - `train_step(self, batch_data: Dict) -> Dict`：单步训练逻辑（前向、损失、反向、梯度裁剪、优化器 step）。
    - `train(self, use_wandb: bool = False)`：完整训练循环（epoch、验证、checkpoint 保存）。
    - `save_checkpoint(self, epoch: int)`：保存模型检查点与训练状态。
  - `load_config(config_path: str) -> Dict`：读取 YAML/JSON 配置文件。

- `src/evaluation_datasets.py`
  - `class EvaluationDatasets`：封装多个评测集加载函数。
    - `load_gsm8k(split: str = "test") -> Dataset`：加载 GSM8K 指定 split。
    - `load_gsm8k_hard() -> Dataset`：加载更难样本集（如 paper 的 hard subset）。
    - `load_svamp() -> Dataset`：加载 SVAMP 数据集。
    - `normalize_sample(sample: Dict, dataset_name: str) -> Dict`：标准化样本格式以统一评测流程。
    - `extract_numerical_answer(answer_str: str) -> Optional[float]`：提取数值答案。
    - `load_all_evaluation_datasets() -> Dict[str, Dataset]`：返回所有支持的数据集字典。
  - `test_dataset_loading()`：自测函数，验证数据集加载正确。

- `src/utils.py`
  - `extract_answer_number(text: str) -> Optional[str]`：从文本抽出数值/答案字符串。
  - `normalize_number(number_str: str) -> str`：标准化数字格式（浮点/整数、小数点处理）。
  - `exact_match_numeric(pred: str, gold: str, tolerance: float = 1e-3) -> bool`：数值精确匹配判定（带容差）。
  - `normalize_answer(text: str) -> str`：将生成文本归一化到可比较形式。
  - `calculate_accuracy(predictions: list, golds: list) -> float`：计算准确率。
  - `format_metrics(metrics: dict, decimals: int = 4) -> dict`：格式化输出的 metric 字典。
  - 若包含单元测试函数：`test_extract_answer_number()`、`test_exact_match_numeric()`、`test_normalize_number()`。

- `src/model_configs.py`
  - `class ModelConfig`：数据类，封装模型/训练配置。
    - `__post_init__(self)`：校验/补全字段。
    - `to_dict(self) -> Dict`：导出 dict 表示。
    - `to_yaml(self, output_path: str)`：导出 yaml。
    - `to_json(self, output_path: str)`：导出 json。
  - 实用函数：
    - `get_config(config_name: str) -> ModelConfig`：按名返回配置对象。
    - `list_configs(verbose: bool = False)`：列出可用配置。
    - `compare_configs(config1_name: str, config2_name: str)`：比较两个配置差异。
    - `export_all_configs(output_dir: str = "configs", format: str = "yaml")`：导出所有配置文件。
    - `get_model_by_size(size: str) -> List[str]`：按模型大小查询名称。
    - `get_configs_by_dataset(dataset: str) -> List[str]`：按数据集筛选配置。


## 其他脚本/工具

- `validate_and_visualize.py`：
  - `class KAVAResultValidator`：
    - `__init__(results_dir: str = "outputs")`：初始化路径等。
    - `load_results() -> pd.DataFrame`：加载实验结果到 DataFrame。
    - `validate_results(df: pd.DataFrame) -> Dict`：进行一致性/完整性检查。
    - `compare_with_paper(df: pd.DataFrame)`：与论文表格比较。
    - `plot_results(df: pd.DataFrame)`、私有绘图函数 `_plot_all_metrics`、`_plot_seed_variance`、`_plot_paper_comparison`、`_plot_training_curves`。
    - `generate_report(df, validation, comparison)`：生成最终报告（文件）。

- `aggregate_results.py` / `aggregate_multi_seed.py` / `analyze_results.py`：结果加载、聚合、统计、LaTeX/CSV 导出函数。
  - 常见函数：`load_experiment_summary`、`load_seed_results`、`aggregate_multi_seed`、`compute_statistics`、`generate_latex_table`、`print_summary_table`、`create_paper_table` 等。

- `download_from_hf.py`：模型/数据集下载助手。
  - `download_models()`、`download_datasets()`、`check_hf_cache()`。

- `benchmark_incremental_decoding.py`：推理速度/增量解码基准。
  - `class IncrementalDecodingBenchmark`：`generate_naive`、`generate_cached`、`generate_transformers_optimized`、`run_comparison`。

- `format_results.py`：表格/CSV/LaTeX 输出助手。
  - `load_aggregated_results`、`format_value`、`generate_latex_table_accuracy`、`generate_latex_table_efficiency`、`generate_csv`、`collect_results`、`main()`。


---

如果你需要：
- 我可以继续为每个方法列出参数签名与返回类型（若源码中有注释/类型提示）。
- 我可以搜索并列出每个方法的调用站点（who calls whom），生成调用图或调用链。
- 我可以将上表拆成单文件的 CSV 便于核对。

我已把此报告写入 `REPORT_MODULE_METHODS.md`。下一步你想要我：
- 列出每个方法的完整签名（参数 + 返回）？
- 列出每个方法在仓库中的调用位置（文件:line）？
- 生成 CSV/Excel 版本便于打钩核对？

请选择一个或多个，我会继续执行。