import { createRouter, createWebHashHistory } from 'vue-router'

// meta.stage 用于侧栏分组与面包屑；meta.title 是章节名。
const routes = [
  {
    path: '/',
    name: 'home',
    component: () => import('@/views/Home.vue'),
    meta: { title: '主线总览', stage: 'intro', chapter: '序章' },
  },

  // ── 阶段 1 · llm_basic ────────────────────────────────────────────
  {
    path: '/basic',
    name: 'basic',
    component: () => import('@/views/Basic.vue'),
    meta: { title: '最小可跑闭环', stage: 'basic', chapter: '阶段 1 · llm_basic' },
  },
  {
    path: '/basic/data',
    name: 'basic-data',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '数据与 tokenizer', stage: 'basic', chapter: '阶段 1 · llm_basic' },
  },
  {
    path: '/basic/forward',
    name: 'basic-forward',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'forward 与形状流', stage: 'basic', chapter: '阶段 1 · llm_basic' },
  },
  {
    path: '/basic/backward',
    name: 'basic-backward',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '手写 backward', stage: 'basic', chapter: '阶段 1 · llm_basic' },
  },
  {
    path: '/basic/optim-sample',
    name: 'basic-optim-sample',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'Adam 与采样', stage: 'basic', chapter: '阶段 1 · llm_basic' },
  },

  // ── 阶段 2 · llm_models (5 章) ────────────────────────────────────
  {
    path: '/attention',
    name: 'attention',
    component: () => import('@/views/Attention.vue'),
    meta: { title: '注意力演进', stage: 'models', chapter: '阶段 2 · llm_models' },
  },
  {
    path: '/position',
    name: 'position',
    component: () => import('@/views/Position.vue'),
    meta: { title: '位置编码与 RoPE', stage: 'models', chapter: '阶段 2 · llm_models' },
  },
  {
    path: '/blocks',
    name: 'blocks',
    component: () => import('@/views/Blocks.vue'),
    meta: { title: 'Block 组装器', stage: 'models', chapter: '阶段 2 · llm_models' },
  },
  {
    path: '/moe',
    name: 'moe',
    component: () => import('@/views/MoE.vue'),
    meta: { title: 'MoE 路由', stage: 'models', chapter: '阶段 2 · llm_models' },
  },
  {
    path: '/diffusion',
    name: 'diffusion',
    component: () => import('@/views/Diffusion.vue'),
    meta: { title: '扩散生成', stage: 'models', chapter: '阶段 2 · llm_models' },
  },

  // ── 阶段 3 · llm_train ────────────────────────────────────────────
  {
    path: '/train',
    name: 'train',
    component: () => import('@/views/Train.vue'),
    meta: { title: '规模化训练', stage: 'train', chapter: '阶段 3 · llm_train' },
  },
  {
    path: '/train/batch-ddp',
    name: 'train-batch-ddp',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'batch 与 DDP', stage: 'train', chapter: '阶段 3 · llm_train' },
  },
  {
    path: '/train/model-parallel',
    name: 'train-model-parallel',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'TP / PP 切模型', stage: 'train', chapter: '阶段 3 · llm_train' },
  },
  {
    path: '/train/memory',
    name: 'train-memory',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '状态与显存', stage: 'train', chapter: '阶段 3 · llm_train' },
  },
  {
    path: '/train/precision-stability',
    name: 'train-precision-stability',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '精度与稳定性', stage: 'train', chapter: '阶段 3 · llm_train' },
  },
  {
    path: '/train/collectives-loop',
    name: 'train-collectives-loop',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '通信与 full_loop', stage: 'train', chapter: '阶段 3 · llm_train' },
  },

  // ── 阶段 4 · llm_finetune ─────────────────────────────────────────
  {
    path: '/finetune',
    name: 'finetune',
    component: () => import('@/views/Finetune.vue'),
    meta: { title: 'SFT / LoRA / DPO', stage: 'finetune', chapter: '阶段 4 · llm_finetune' },
  },
  {
    path: '/finetune/sft',
    name: 'finetune-sft',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'SFT 数据与 loss', stage: 'finetune', chapter: '阶段 4 · llm_finetune' },
  },
  {
    path: '/finetune/lora',
    name: 'finetune-lora',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'LoRA 参数高效微调', stage: 'finetune', chapter: '阶段 4 · llm_finetune' },
  },
  {
    path: '/finetune/dpo',
    name: 'finetune-dpo',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'DPO 偏好对齐', stage: 'finetune', chapter: '阶段 4 · llm_finetune' },
  },
  {
    path: '/finetune/runs',
    name: 'finetune-runs',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '训练脚本与落盘', stage: 'finetune', chapter: '阶段 4 · llm_finetune' },
  },

  // ── 阶段 5 · llm_infer ────────────────────────────────────────────
  {
    path: '/infer',
    name: 'infer',
    component: () => import('@/views/Infer.vue'),
    meta: { title: '推理与部署优化', stage: 'infer', chapter: '阶段 5 · llm_infer' },
  },
  {
    path: '/infer/kv-memory',
    name: 'infer-kv-memory',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'KV 与缓存内存', stage: 'infer', chapter: '阶段 5 · llm_infer' },
  },
  {
    path: '/infer/scheduler',
    name: 'infer-scheduler',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '调度与 prefill', stage: 'infer', chapter: '阶段 5 · llm_infer' },
  },
  {
    path: '/infer/decode-control',
    name: 'infer-decode-control',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '解码加速与约束', stage: 'infer', chapter: '阶段 5 · llm_infer' },
  },
  {
    path: '/infer/compute',
    name: 'infer-compute',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '算子与压缩', stage: 'infer', chapter: '阶段 5 · llm_infer' },
  },
  {
    path: '/infer/engine',
    name: 'infer-engine',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'mini-vLLM 引擎', stage: 'infer', chapter: '阶段 5 · llm_infer' },
  },

  // ── 阶段 6 · llm_agent ────────────────────────────────────────────
  {
    path: '/agent',
    name: 'agent',
    component: () => import('@/views/Agent.vue'),
    meta: { title: 'Agent 应用层', stage: 'agent', chapter: '阶段 6 · llm_agent' },
  },
  {
    path: '/agent/loop',
    name: 'agent-loop',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'Agent loop', stage: 'agent', chapter: '阶段 6 · llm_agent' },
  },
  {
    path: '/agent/tools-permissions',
    name: 'agent-tools-permissions',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '工具与权限', stage: 'agent', chapter: '阶段 6 · llm_agent' },
  },
  {
    path: '/agent/context-memory',
    name: 'agent-context-memory',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '上下文与记忆', stage: 'agent', chapter: '阶段 6 · llm_agent' },
  },
  {
    path: '/agent/extensibility',
    name: 'agent-extensibility',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'Hooks / Skills / MCP', stage: 'agent', chapter: '阶段 6 · llm_agent' },
  },
  {
    path: '/agent/state-subagents',
    name: 'agent-state-subagents',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: '持久化与子智能体', stage: 'agent', chapter: '阶段 6 · llm_agent' },
  },
  {
    path: '/agent/full-loop',
    name: 'agent-full-loop',
    component: () => import('@/views/StageTopic.vue'),
    meta: { title: 'mini Agent harness', stage: 'agent', chapter: '阶段 6 · llm_agent' },
  },

  // ── 终章 · 全书索引 ────────────────────────────────────────────────
  {
    path: '/compare',
    name: 'compare',
    component: () => import('@/views/Compare.vue'),
    meta: { title: '总览对照表', stage: 'outro', chapter: '终章' },
  },
]

export default createRouter({
  history: createWebHashHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})
