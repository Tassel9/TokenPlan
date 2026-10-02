# 智慧路灯 RAG 检索评测说明

## 当前结论

当前知识集已经切换为智慧路灯市政运维域，覆盖路灯巡检、告警、遥测、终端接入、证书安全、能耗与维修工单。旧业务域上产生的召回率和延迟结果不能直接作为 UrbanOps 指标，因此本仓库不沿用那些数字。

当前冻结输入为：

- `evaluation/fixtures/urbanops_agentic_rag_ragas_blueprint_v1.json`：20 个主题的知识蓝图；
- `evaluation/fixtures/urbanops_agentic_rag_ragas_cases_v1.json`：60 篇文档、200 个问题，其中 calibration 50 例、holdout 150 例；
- `evaluation/fixtures/urbanops_agentic_rag_ragas_latest_manifest.json`：数据集哈希与生成参数。

这些数据是策展合成评测集，不是生产流量。报告只能说明当前实现对这组智慧路灯场景的离线表现，不能外推到真实城市规模、百万级语料或生产 SLA。

## 评测口径

- 召回：参考文档是否进入检索结果，使用 Recall@k 等指标；
- 排序：比较稠密召回、词法召回、RRF 融合和 BGE 重排后的结果；
- 端到端质量：同时检查回答是否由证据支持、是否覆盖任务、是否安全；
- 延迟：检索链路延迟与模型生成延迟分开报告，硬件、并发、缓存和预热状态必须随报告记录；
- 证据边界：任何历史报告若不是在当前 manifest 对应的数据集上运行，都不得标注为 UrbanOps 当前成绩。

## 复现

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
$py = ".\.venv-win\Scripts\python.exe"

# 校验并重新生成冻结数据集
& $py evaluation\build_agentic_rag_ragas_dataset.py

# 召回、延迟与候选池实验
& $py evaluation\benchmarks\diagnose_recall_gap.py
& $py evaluation\benchmarks\bench_retrieval_latency.py
& $py evaluation\benchmarks\sweep_rerank_candidates.py
& $py evaluation\benchmarks\bench_concurrency.py

# Agentic RAG 两臂评测
& $py evaluation\evaluate_agentic_rag_ragas_pipeline.py
& $py evaluation\evaluate_agentic_rag_ragas_judge.py
```

新报告应写入 `evaluation/reports/`，并保留数据集 SHA、代码版本、运行环境、重复次数、原始样本结果与 `production_evidence=false` 标记。
