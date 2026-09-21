// 按阶段拆分的章节扩展: 本目录下每个阶段一个文件, 形如
//   export default {
//     stage: 'train',                                   // basic | models | train | finetune | infer | agent
//     chapters: [{ route: 'train-muon', label: 'Muon 优化器', hint: '…' }],   // 追加到该阶段侧栏末尾
//     pages: { 'train-muon': { title, subtitle, tldr, question, code, points, links, sourceRows,
//                              snippetTitle, snippet, source, run, widgets } },
//   }
// 路由 (/阶段/章节名)、侧栏、learningPath 全部由这里自动生成, 新增一章不需要再改 router 和 models.js。
// pages 里也可以写已有章节的 route 名 —— 会按字段覆盖 models.js 里的同名页 (用来补 source / 修正文字)。
const files = import.meta.glob('./*.js', { eager: true, import: 'default' })
const mods = Object.entries(files).filter(([f]) => !f.endsWith('/index.js')).map(([, m]) => m).filter(Boolean)

export const extraChapters = {}
export const extraPages = {}
for (const m of mods) {
  extraChapters[m.stage] = [...(extraChapters[m.stage] || []), ...(m.chapters || [])]
  Object.assign(extraPages, m.pages || {})
}
