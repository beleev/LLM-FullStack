// 阶段 2 · llm_models: 章节 -> 实验台 (AttnMaskLab / MtpLab 已由 models.js 的 widgets 挂在 models-mtp)
export default {
  'models-mtp': ['ModelLinearAttnLab'],
  'models-generation': ['ModelKvGenLab'],
  'models-qknorm-yarn': ['ModelQkNormLab', 'ModelRopeScaleLab'],
  'models-mamba': ['ModelMambaLab'],
  'models-moe-balance': ['ModelAuxFreeLab'],
  'models-dsa': ['ModelDsaLab'],
  'models-gptoss': ['ModelSwaLayersLab', 'ModelAttnSinkLab'],
  'models-llada': ['ModelLladaLab'],
  'models-var': ['ModelVarLab'],
  blocks: ['ModelInitLab'], // 绑权重 + 初始化尺度: 本仓库真实修过的 bug
}
