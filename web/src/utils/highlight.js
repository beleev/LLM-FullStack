// 极简 Python 语法高亮 —— 一个正则单趟扫描, 不引第三方库。
// 输出的 class (kw / fn / num / str / cm) 已在 main.css 的 pre.code 下定义。
const KW = 'def|class|return|if|elif|else|for|while|in|not|and|or|is|import|from|as|with|try|except|finally|raise|assert|lambda|yield|pass|break|continue|global|nonlocal|None|True|False|self'
const RE = new RegExp(
  [
    '(#[^\\n]*)',                                                      // 1 注释
    '("""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'|"(?:\\\\.|[^"\\\\\\n])*"|\'(?:\\\\.|[^\'\\\\\\n])*\')', // 2 字符串
    '\\b(\\d+(?:\\.\\d+)?(?:e[+-]?\\d+)?)\\b',                         // 3 数字
    `\\b(${KW})\\b`,                                                   // 4 关键字
    '\\b([A-Za-z_]\\w*)(?=\\()',                                       // 5 函数调用 / 定义
  ].join('|'),
  'g',
)
const CLS = [null, 'cm', 'str', 'num', 'kw', 'fn']
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

export function highlightPython(src) {
  let out = ''
  let last = 0
  for (const m of String(src).matchAll(RE)) {
    out += esc(src.slice(last, m.index))
    const g = CLS.findIndex((_, i) => i && m[i] !== undefined)
    out += `<span class="${CLS[g]}">${esc(m[0])}</span>`
    last = m.index + m[0].length
  }
  return out + esc(src.slice(last))
}
