// 仓库源码链接工具 — 把页面里 "对应代码: llm_basic/model.py" 这种引用
// 一键变成可跳转到 GitHub 的链接。
//
// 改 fork 时只需改这两个常量。

export const REPO_BASE = 'https://github.com/beleev/llm-arch'
export const BRANCH = 'main'

// 解析 "llm_basic/model.py:42" / "llm_basic/model.py#L42" / "llm_basic/model.py"
// 返回 { path, line, label }
export const parseRef = (ref) => {
  if (!ref) return { path: '', line: null, label: '' }
  const trimmed = String(ref).trim()
  const m = trimmed.match(/^([^#]+?)(?::(\d+))?(?:#L(\d+))?$/)
  if (!m) return { path: trimmed, line: null, label: trimmed }
  const path = m[1]
  const line = m[2] ? Number(m[2]) : (m[3] ? Number(m[3]) : null)
  return { path, line, label: trimmed }
}

// 生成 GitHub blob 链接;  path 为空则返回仓库根链接。
export const repoUrl = (path = '', line = null) => {
  if (!path) return `${REPO_BASE}/tree/${BRANCH}`
  const cleanPath = String(path).replace(/^\/+/, '')
  // 路径以 "/" 结尾视为目录, 用 tree/ 形式
  const isDir = cleanPath.endsWith('/')
  if (isDir) return `${REPO_BASE}/tree/${BRANCH}/${cleanPath.replace(/\/$/, '')}`
  const lineSuffix = line ? `#L${line}` : ''
  return `${REPO_BASE}/blob/${BRANCH}/${cleanPath}${lineSuffix}`
}

// 拆分 "llm_basic/ · llm_models/ · llm_train/" 这种字符串为多个 ref。
export const splitRefs = (str, sep = '·') =>
  String(str || '')
    .split(sep)
    .map(s => s.trim())
    .filter(Boolean)
    .map(parseRef)
