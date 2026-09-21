// 阶段 1 · llm_basic: 章节 -> 实验台 (BpeLab 由 topics/basic.js 的 widgets 挂在 basic-data)
export default {
  'basic-forward': ['BasicShapeFlowLab'],
  'basic-backward': ['BasicBackpropLab', 'BasicGradcheckLab'],
  'basic-optim-sample': ['BasicOptimLab', 'SoftmaxTempLab'],
}
