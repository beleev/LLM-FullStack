// 划重点: 把 75 个章节分成三层, 让人知道哪些必须读、哪些可以跳。
//
//   冲刺  12 章  只有一天, 就读这些 —— 每个阶段最核心的那一两页。
//   主干  37 章  完整的课程主线。读完能说清"一个大模型从训练到上线到能行动"整条链路。
//   扩展  36 章  深水区与分支: 2026 年的前沿技术、同一问题的其他解法、生成模型分支。
//                跳过不影响理解主线, 但每一章都对应真实系统里正在用的东西。
//
// 分层只是阅读建议, 不改变任何章节的内容或可达性。
// 新增章节默认归为扩展 —— 主线是要守住的, 加内容不该让它变长。

// 冲刺: 每个阶段挑出"不读这页就接不上下一阶段"的那几页
const SPRINT = [
  'basic-forward', 'basic-backward',                      // 形状怎么流、梯度怎么回来
  'attention', 'blocks', 'models-generation',             // 注意力、Block 怎么拼、KV cache
  'train-batch-ddp', 'train-memory',                      // 为什么多卡等价于大 batch、显存花在哪
  'finetune-sft', 'finetune-lora',                        // 怎么教模型做事、怎么只训 1% 参数
  'infer-kv-memory', 'infer-scheduler',                   // 推理的瓶颈、怎么把请求喂进去
  'agent-loop',                                           // 模型怎么变成会行动的系统
]

// 主干: 冲刺 + 课程主线的其余部分
const CORE = [
  ...SPRINT,
  'home',
  'basic', 'basic-data', 'basic-optim-sample',
  'models', 'position', 'moe',
  'train', 'train-model-parallel', 'train-precision-stability', 'train-collectives-loop',
  'finetune', 'finetune-dpo', 'finetune-rlhf', 'finetune-runs',
  'infer', 'infer-decode-control', 'infer-compute', 'infer-engine',
  'agent', 'agent-tools-permissions', 'agent-context-memory',
  'agent-extensibility', 'agent-state-subagents', 'agent-full-loop',
]

// 工具页: 不属于阅读主线, 需要时来查
const REF = ['compare', 'glossary', 'fast-track']

const sprintSet = new Set(SPRINT)
const coreSet = new Set(CORE)
const refSet = new Set(REF)

export const isSprint = (route) => sprintSet.has(route)
export const tierOf = (route) =>
  refSet.has(route) ? 'ref' : coreSet.has(route) ? 'core' : 'ext'

export const TIER_META = {
  core: { label: '主干', mark: '●', desc: '课程主线, 建议按顺序读完' },
  ext: { label: '扩展', mark: '○', desc: '深水区或分支, 跳过不影响主线' },
  ref: { label: '速查', mark: '◆', desc: '工具页, 需要时再来' },
}

// 阅读档位 —— 速成路线页和侧栏筛选共用
export const LEVELS = [
  { id: 'all', label: '全部', hint: '一个都不落' },
  { id: 'core', label: '主干', hint: '完整主线' },
  { id: 'sprint', label: '冲刺', hint: '只有一天' },
]

// 某个档位下该不该显示这个路由
export const inLevel = (route, level) =>
  level === 'all' ? true : level === 'core' ? tierOf(route) === 'core' : isSprint(route)


