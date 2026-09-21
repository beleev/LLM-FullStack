// 拖拽: 基于 Pointer Events, 鼠标 / 触屏 / 触控笔一套代码。
//   const { start } = useDrag()
//   <circle @pointerdown="start($event, { onMove: ({ x, y }) => ... })" />
// 在 SVG 里拖动时传 svg 根节点, x/y 会换算成 viewBox 坐标。
export function useDrag() {
  const start = (e, { onMove, onEnd, svg = null } = {}) => {
    e.preventDefault()
    const el = e.currentTarget
    el.setPointerCapture?.(e.pointerId)
    const x0 = e.clientX
    const y0 = e.clientY

    const toLocal = (ev) => {
      if (!svg) return { x: ev.clientX, y: ev.clientY }
      const pt = svg.createSVGPoint()
      pt.x = ev.clientX
      pt.y = ev.clientY
      const p = pt.matrixTransform(svg.getScreenCTM().inverse())
      return { x: p.x, y: p.y }
    }
    const move = (ev) => onMove?.({ ...toLocal(ev), dx: ev.clientX - x0, dy: ev.clientY - y0, event: ev })
    const end = (ev) => {
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', end)
      el.removeEventListener('pointercancel', end)
      onEnd?.({ ...toLocal(ev), event: ev })
    }
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', end)
    el.addEventListener('pointercancel', end)
  }
  return { start }
}
