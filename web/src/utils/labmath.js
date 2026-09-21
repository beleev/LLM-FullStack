// 实验台共用的小数学工具 —— 所有 lab 从这里取, 不要各自重抄。

// 确定性伪随机: 同一个 seed 永远得到同一串数, 拖滑杆时图形不会乱跳
export const mulberry32 = (seed) => {
  let a = seed | 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// 标准正态 (Box-Muller), rand 传入 mulberry32(seed)
export const randn = (rand) =>
  Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand())

// 数值稳定 softmax: 先减最大值再取指数; -Infinity 表示被 mask 的位置
export const softmax = (xs, T = 1) => {
  const s = xs.map((x) => x / T)
  const m = Math.max(...s)
  const e = s.map((x) => (x === -Infinity ? 0 : Math.exp(x - m)))
  const z = e.reduce((a, b) => a + b, 0) || 1
  return e.map((x) => x / z)
}

export const entropy = (ps) => ps.reduce((h, p) => (p > 0 ? h - p * Math.log(p) : h), 0)
export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x))
export const lerp = (a, b, t) => a + (b - a) * t
export const sum = (xs) => xs.reduce((a, b) => a + b, 0)
export const range = (n) => Array.from({ length: n }, (_, i) => i)
export const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0)

// 1234567 -> "1.23M";  字节数用 fmtBytes
export const fmtNum = (n) => {
  const a = Math.abs(n)
  if (a >= 1e9) return (n / 1e9).toFixed(2) + 'B'
  if (a >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (a >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return Number.isInteger(n) ? String(n) : n.toFixed(2)
}
export const fmtBytes = (b) => {
  const u = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++ }
  return b.toFixed(b >= 100 || i === 0 ? 0 : 1) + ' ' + u[i]
}

// 热力图配色: t∈[0,1] -> 以主题强调色为底的透明度, 明暗主题都能用
export const heat = (t, color = 'var(--accent)') =>
  `color-mix(in srgb, ${color} ${Math.round(clamp(t, 0, 1) * 100)}%, transparent)`
