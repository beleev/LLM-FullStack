// 真源码加载: 构建期用 Vite glob 把仓库里的 .py 以纯文本懒加载。
// 页面展示的代码直接来自 Python 文件本身, 不再手抄, 所以不会和源码漂移。
const files = import.meta.glob('../../../llm_*/**/*.py', { query: '?raw', import: 'default' })

const keyOf = (path) => `../../../${path.replace(/^\/+/, '')}`
export const hasSource = (path) => keyOf(path) in files

// "llm_infer/m07/demo.py:spec_decode_greedy" -> { path, symbol }
export const parseRef = (ref) => {
  const [path, symbol = ''] = String(ref).trim().split(':')
  return { path, symbol }
}

// 按缩进截出一个 def / class 的完整定义 (含上方的装饰器)
export function extractSymbol(text, symbol) {
  const lines = text.split('\n')
  if (!symbol) return { code: text.trimEnd(), start: 1, end: lines.length }
  const name = symbol.split('.').pop()
  const head = new RegExp(`^(\\s*)(?:async\\s+)?(?:def|class)\\s+${name}\\b`)
  let i = lines.findIndex((l) => head.test(l))
  if (i < 0) return null
  const indent = lines[i].match(head)[1].length
  let start = i
  while (start > 0 && lines[start - 1].trim().startsWith('@')) start--
  let end = i + 1
  // 签名可能跨多行: 先走到以 ":" 结尾的那一行
  while (end < lines.length && !/:\s*(#.*)?$/.test(lines[end - 1])) end++
  while (end < lines.length) {
    const l = lines[end]
    if (l.trim() && l.match(/^\s*/)[0].length <= indent) break
    end++
  }
  while (end > start && !lines[end - 1].trim()) end--
  const body = lines.slice(start, end).map((l) => l.slice(indent))
  return { code: body.join('\n'), start: start + 1, end }
}

export async function loadSource(ref) {
  const { path, symbol } = parseRef(ref)
  const loader = files[keyOf(path)]
  if (!loader) return { path, symbol, error: `找不到源文件 ${path}` }
  const text = await loader()
  const hit = extractSymbol(text, symbol)
  if (!hit) return { path, symbol, error: `${path} 里找不到 ${symbol}` }
  return { path, symbol, ...hit }
}
