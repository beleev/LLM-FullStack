// 粗估每章的阅读量。数字来自真实内容长度, 不是拍脑袋填的:
//   数据驱动的章节 → 数 topicPages 里的正文字数
//   手写视图的章节 → 读 .vue 源码, 去掉 script/style/标签后数正文字数
// 按每分钟 300 字算 (技术文, 比小说慢), 每个实验台另加 4 分钟动手时间。
import { agentModules, inferModules, stageBy, topicPages, trainModules, timeline } from '@/data/models.js'
import { labMap } from '@/data/labMap.js'

const CHARS_PER_MIN = 300
const MIN_PER_LAB = 4

// 手写视图的章节 → 源文件
const VIEW_OF = {
  home: 'Home', basic: 'Basic', models: 'Models', attention: 'Attention',
  position: 'Position', blocks: 'Blocks', moe: 'MoE', diffusion: 'Diffusion',
  train: 'Train', finetune: 'Finetune', infer: 'Infer', agent: 'Agent',
  compare: 'Compare', glossary: 'Glossary',
}
const rawViews = import.meta.glob('@/views/*.vue', { query: '?raw', import: 'default', eager: true })

// CJK 字算 1, 连续的拉丁词算 1 —— 中英混排时长度才不会被低估
const textLen = (s) => {
  const cjk = (s.match(/[一-龥]/g) || []).length
  const latin = (s.match(/[A-Za-z][A-Za-z0-9_.]*/g) || []).length
  return cjk + latin
}

const viewChars = (name) => {
  const src = Object.entries(rawViews).find(([p]) => p.endsWith(`/${name}.vue`))?.[1]
  if (!src) return 0
  const body = src
    .replace(/<script[\s\S]*?<\/script>/g, '')
    .replace(/<style[\s\S]*?<\/style>/g, '')
    .replace(/<[^>]+>/g, ' ')          // 去标签, 只留给人读的字
  return textLen(body)
}

// 阶段总览页渲染的是 stages / 模块表 / 时间轴这些数据, 不在 .vue 源码里
const MODULE_TABLE = { train: trainModules, infer: inferModules, agent: agentModules }
const dataChars = (route) => {
  const s = stageBy[route]
  if (!s) return route === 'compare' || route === 'models' ? textLen(timeline.map((m) => m.blurb + m.name + m.kind).join(' ')) : 0
  const rows = MODULE_TABLE[route] || []
  return textLen([
    s.oneliner,
    ...(s.chapters || []).flatMap((c) => [c.label, c.hint]),
    ...rows.flatMap((m) => [m.name, m.concept, m.link]),
  ].filter(Boolean).join(' '))
}

// 手写视图直接 import 实验台组件, 不走 labMap, 所以要从源码里数
const viewLabs = (name) => {
  const src = Object.entries(rawViews).find(([p]) => p.endsWith(`/${name}.vue`))?.[1] || ''
  return new Set([...src.matchAll(/components\/labs\/(\w+)\.vue/g)].map((m) => m[1])).size
}

const pageChars = (page) => {
  const parts = [
    page.subtitle, page.tldr, page.question,
    ...(page.points || []).flatMap((p) => [p.title, p.body]),
    ...(page.links || []).map((l) => l.body),
    ...(page.sourceRows || []).map((r) => r.takeaway),
    page.snippet,
  ]
  return textLen(parts.filter(Boolean).join(' '))
}

const cache = new Map()

/** 这一章挂了几个实验台 (含手写视图里直接 import 的)。 */
export function labsOf(route) {
  const view = VIEW_OF[route]
  return new Set([...(topicPages[route]?.widgets || []), ...(labMap[route] || [])]).size + (view ? viewLabs(view) : 0)
}

/** 某一章的粗估分钟数 (至少 2 分钟)。 */
export function minutesOf(route) {
  if (cache.has(route)) return cache.get(route)
  const view = VIEW_OF[route]
  const chars = (topicPages[route] ? pageChars(topicPages[route]) : 0)
    + (view ? viewChars(view) + dataChars(route) : 0)
  const labs = new Set([...(topicPages[route]?.widgets || []), ...(labMap[route] || [])]).size
    + (view ? viewLabs(view) : 0)
  const m = Math.max(2, Math.round(chars / CHARS_PER_MIN) + labs * MIN_PER_LAB)
  cache.set(route, m)
  return m
}

/** 一串路由合计多久, 返回 "3 小时 20 分" 这样的说法。 */
export function totalTime(routes) {
  const m = routes.reduce((s, r) => s + minutesOf(r), 0)
  if (m < 60) return `约 ${m} 分钟`
  const h = Math.floor(m / 60)
  const rest = m % 60
  return rest ? `约 ${h} 小时 ${rest} 分` : `约 ${h} 小时`
}
