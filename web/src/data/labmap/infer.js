// 阶段 5 · llm_infer 实验台挂载表 (每章 ≤ 3 个; 已有的 AttnMaskLab / SoftmaxTempLab 由 models.js 的 widgets 挂载)
export default {
  'infer-kv-memory': ['InferKvCacheLab', 'InferPagedAttentionLab'],
  'infer-scheduler': ['InferContinuousBatchingLab', 'InferChunkedPrefillLab', 'InferPdDisaggLab'],
  'infer-decode-control': ['InferSpecDecodeLab', 'InferSamplingLab'],
  'infer-compute': ['InferFlashAttnLab'],
  'infer-prefix-radix': ['InferRadixCacheLab'],
  'infer-structured-output': ['InferGrammarLab'],
  'infer-quant-awq': ['InferQuantLab', 'InferKvQuantLab'],
  'infer-kv-footprint': ['InferKvFootprintLab'],
  'infer-attention-sinks': ['InferAttnSinkLab'],
  'infer-tree-speculation': ['InferTreeSpecLab'],
  'infer-kv-offload': ['InferKvOffloadLab'],
  'infer-moe-serving': ['InferEplbLab'],
  'infer-sparse-decode': ['InferSparseDecodeLab'],
}
