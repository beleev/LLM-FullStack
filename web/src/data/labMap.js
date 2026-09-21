// 章节路由 -> 该章挂载的实验台 (components/labs/ 下的文件名, 不带 .vue)。
// 每个阶段一个文件放在 data/labmap/ 下, 形如 export default { 'train-model-parallel': ['PipelineLab'] }
// 与 topicPages[route].widgets 合并去重, 顺序即页面上的展示顺序。
const files = import.meta.glob('./labmap/*.js', { eager: true, import: 'default' })
export const labMap = {}
for (const m of Object.values(files)) {
  for (const [route, names] of Object.entries(m || {})) labMap[route] = [...(labMap[route] || []), ...names]
}
