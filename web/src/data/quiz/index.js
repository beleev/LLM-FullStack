// 自测题库: 本目录下每个阶段一个文件, 形如
//   export default { 'route-name': [{ q, options: [..], answer: 0, why: '...' }] }
// 这里用 glob 自动合并, 新增文件不需要改任何注册表。
const files = import.meta.glob('./*.js', { eager: true, import: 'default' })
export const quizBank = Object.assign(
  {},
  ...Object.entries(files).filter(([f]) => !f.endsWith('/index.js')).map(([, m]) => m || {}),
)
