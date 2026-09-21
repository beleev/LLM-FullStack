// 阶段 3 · llm_train: 章节路由 -> 实验台 (每章最多 3 个; train-moe-seq 另有 widgets 里的 MoeRouteLab / RingAttnLab)
export default {
  'train-model-parallel': ['PipelineLab', 'TrainTpSplitLab'],
  'train-memory': ['TrainZeroMemoryLab', 'TrainActCkptLab'],
  'train-precision-stability': ['TrainFloatLineLab'],
  'train-moe-seq': ['TrainZigzagLab'],
  'train-collectives-loop': ['TrainRingAllReduceLab'],
  'train-pipeline-schedules': ['TrainInterleavedLab'],
  'train-lr-schedule': ['TrainLrScheduleLab'],
  'train-low-precision': ['TrainBlockScaleLab'],
  'train-muon': ['TrainMuonLab', 'TrainMuonVsAdamLab'],
  'train-ulysses': ['TrainUlyssesLab'],
}
