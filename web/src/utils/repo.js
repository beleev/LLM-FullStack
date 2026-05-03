// 仓库源码链接工具 — 把页面里 "对应代码: llm_basic/model.py" 这种引用
// 一键变成可跳转到 GitHub 的链接。
//
// 改 fork 时只需改这两个常量。

export const REPO_BASE = 'https://github.com/beleev/LLM-FullStack'
export const BRANCH = 'main'

// 解析 "llm_basic/model.py:42" / "llm_basic/model.py#L42" / "llm_basic/model.py"
// 返回 { path, line, label }
export const parseRef = (ref) => {
  if (!ref) return { path: '', line: null, label: '' }
  const trimmed = String(ref).trim()
  const hashLine = trimmed.match(/^(.*)#L(\d+)$/)
  if (hashLine) return { path: hashLine[1], line: Number(hashLine[2]), label: trimmed }

  const numberLine = trimmed.match(/^(.*):(\d+)$/)
  if (numberLine) return { path: numberLine[1], line: Number(numberLine[2]), label: trimmed }

  const symbolRef = trimmed.match(/^(.+\.(?:py|md|txt|npz|bin)):.+$/)
  if (symbolRef) return { path: symbolRef[1], line: null, label: trimmed }

  return { path: trimmed, line: null, label: trimmed }
}

export const looksLikeRepoRef = (ref) => {
  if (!ref) return false
  const s = String(ref).trim()
  if (!s || /\s/.test(s) && !/[\w/.-]+\.(py|md|txt|npz|bin)/.test(s)) return false
  return /^(llm_[\w_]+\/|(?:models|layers|training|methods|data|utils|run_finetune|core|full_loop|full_engine|m\d\d[\w_-]*|run_all\.py|README\.md)(?:\/|$)|[\w.-]+\.(?:py|md|txt|npz|bin)(?::|$))/.test(s)
}

export const normalizeRepoRef = (ref, base = '') => {
  const parsed = parseRef(ref)
  let path = parsed.path.trim()

  if (base && path && !path.startsWith('llm_') && !path.startsWith('/')) {
    const cleanBase = String(base).replace(/\/?$/, '/')
    path = `${cleanBase}${path}`
  }

  return { ...parsed, path: path.replace(/^\/+/, '') }
}

export const concreteRepoPath = (path) => {
  let p = String(path || '').trim().replace(/^\/+/, '')
  if (!p) return ''

  const braceIdx = p.search(/[{\*]/)
  if (braceIdx >= 0) p = p.slice(0, braceIdx)

  const rangeSegment = p.split('/').findIndex(part => part.includes('..'))
  if (rangeSegment >= 0) p = p.split('/').slice(0, rangeSegment).join('/')

  if (p.includes(',')) p = p.slice(0, p.indexOf(','))
  p = p.replace(/[:#].*$/, '')
  p = p.replace(/\/?$/, '')

  if (/\.[\w]+$/.test(p)) return p
  return p ? `${p}/` : ''
}

// 生成 GitHub blob 链接;  path 为空则返回仓库根链接。
export const repoUrl = (path = '', line = null) => {
  if (!path) return `${REPO_BASE}/tree/${BRANCH}`
  const cleanPath = concreteRepoPath(path)
  if (!cleanPath) return `${REPO_BASE}/tree/${BRANCH}`
  // 路径以 "/" 结尾或没有扩展名时视为目录, 用 tree/ 形式
  const isDir = cleanPath.endsWith('/') || !/\.[\w]+$/.test(cleanPath)
  if (isDir) return `${REPO_BASE}/tree/${BRANCH}/${cleanPath.replace(/\/$/, '')}`
  const lineSuffix = line ? `#L${line}` : ''
  return `${REPO_BASE}/blob/${BRANCH}/${cleanPath}${lineSuffix}`
}

// 拆分 "llm_basic/ · llm_models/ · llm_train/" 这种字符串为多个 ref。
export const splitRefs = (str, sep = '·') => {
  const text = String(str || '')
  const parts = sep === 'auto'
    ? text.split(/(?:\s\+\s|\s·\s|,(?![^{]*\}))/)
    : text.split(sep)
  return parts
    .map(s => s.trim())
    .filter(Boolean)
    .map(parseRef)
}
