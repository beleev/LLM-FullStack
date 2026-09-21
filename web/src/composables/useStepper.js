// 步进播放器: 调度 / 流水线 / 解码这类"随时间推进"的实验台共用。
//   const s = useStepper(() => frames.value.length)
//   s.step.value 是当前帧; 配合 <StepPlayer :stepper="s" /> 得到 播放/暂停/单步/拖动 控件。
import { computed, onBeforeUnmount, ref, unref, watch } from 'vue'

export function useStepper(total, { interval = 600, loop = false } = {}) {
  const n = computed(() => Math.max(1, typeof total === 'function' ? total() : unref(total)))
  const step = ref(0)
  const playing = ref(false)
  const speed = ref(1)
  let timer = null

  const stop = () => { clearInterval(timer); timer = null; playing.value = false }
  const next = () => {
    if (step.value < n.value - 1) step.value++
    else if (loop) step.value = 0
    else stop()
  }
  const prev = () => { if (step.value > 0) step.value-- }
  const reset = () => { stop(); step.value = 0 }
  const play = () => {
    if (step.value >= n.value - 1) step.value = 0
    stop()
    playing.value = true
    timer = setInterval(next, interval / speed.value)
  }
  const toggle = () => (playing.value ? stop() : play())

  watch(speed, () => { if (playing.value) play() })
  // 参数变了导致总帧数变短时, 把当前帧夹回合法范围
  watch(n, (v) => { if (step.value > v - 1) step.value = v - 1 })
  onBeforeUnmount(stop)

  return { step, total: n, playing, speed, play, pause: stop, toggle, next, prev, reset }
}
