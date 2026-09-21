// 校验教程里所有 `source: ['llm_x/file.py:symbol']` 引用: 文件必须存在, 符号必须能在文件里找到。
// Python 侧改名后页面只会静默显示"找不到", 所以放进 CI 把关。  用法: npm run check:sources
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { dirname, join } from 'node:path'

const web = join(dirname(fileURLToPath(import.meta.url)), '..')
const repo = join(web, '..')
const topicsDir = join(web, 'src/data/topics')

const refs = [] // { ref, where }
for (const f of readdirSync(topicsDir).filter((f) => f.endsWith('.js') && f !== 'index.js')) {
  const mod = (await import(pathToFileURL(join(topicsDir, f)))).default
  for (const [route, page] of Object.entries(mod.pages || {})) {
    for (const ref of [].concat(page.source || [])) refs.push({ ref, where: `topics/${f} › ${route}` })
  }
}
// models.js 依赖 Vite 的 import.meta.glob, node 里不能直接 import, 用正则取字面量
const base = readFileSync(join(web, 'src/data/models.js'), 'utf8')
for (const m of base.matchAll(/source:\s*\[([^\]]*)\]/g)) {
  for (const s of m[1].matchAll(/'([^']+)'/g)) refs.push({ ref: s[1], where: 'models.js' })
}

let bad = 0
for (const { ref, where } of refs) {
  const [path, symbol = ''] = ref.split(':')
  const file = join(repo, path)
  let problem = ''
  if (!existsSync(file)) problem = '文件不存在'
  else if (symbol) {
    const name = symbol.split('.').pop()
    const re = new RegExp(`^\\s*(?:async\\s+)?(?:def|class)\\s+${name}\\b`, 'm') // 与 src/utils/sources.js 的规则一致
    if (!re.test(readFileSync(file, 'utf8'))) problem = `找不到 def/class ${name}`
  }
  if (problem) { bad++; console.error(`✗ ${ref}  (${where}): ${problem}`) }
}
console.log(`${refs.length - bad} / ${refs.length} 个 source 引用有效`)
process.exit(bad ? 1 : 0)
