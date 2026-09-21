// 阶段 4 · llm_finetune: 章节路由 -> 实验台。
// finetune-rlhf 另有 topics/finetune.js 里的 widgets (GrpoLab, SoftmaxTempLab)。
export default {
  'finetune-sft': ['FtSftMaskLab'],
  'finetune-lora': ['FtLoraLab'],
  'finetune-qlora': ['FtNf4Lab'],
  'finetune-dora': ['FtDoraLab'],
  'finetune-dpo': ['FtDpoLab'],
  'finetune-simpo-orpo': ['FtPrefLab'],
  'finetune-rlhf': ['FtBtLab'],
  'finetune-grpo-variants': ['FtClipLab', 'FtGrpoAdvLab', 'FtGspoLab'],
  'finetune-onpolicy-distill': ['FtKlLab', 'FtDistillT2Lab'],
  'finetune-rlvr': ['FtRlvrLab'],
  'finetune-runs': ['FtChooseLab'],
}
