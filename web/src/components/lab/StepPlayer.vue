<!-- 步进控件: 配合 useStepper 使用。 <StepPlayer :stepper="s" :label="`t = ${s.step.value}`" /> -->
<template>
  <div class="step-player">
    <button type="button" @click="stepper.reset()" aria-label="回到开头">⏮</button>
    <button type="button" @click="stepper.prev()" aria-label="上一步">◀</button>
    <button type="button" class="active" @click="stepper.toggle()">
      {{ stepper.playing.value ? '暂停' : '播放' }}
    </button>
    <button type="button" @click="stepper.next()" aria-label="下一步">▶</button>
    <input
      type="range" min="0" :max="stepper.total.value - 1" step="1"
      :value="stepper.step.value" aria-label="时间轴"
      @input="stepper.step.value = Number($event.target.value)"
    />
    <span class="mono step-label">{{ label || `${stepper.step.value + 1} / ${stepper.total.value}` }}</span>
    <select v-model.number="stepper.speed.value" aria-label="播放速度">
      <option :value="0.5">0.5×</option>
      <option :value="1">1×</option>
      <option :value="2">2×</option>
      <option :value="4">4×</option>
    </select>
  </div>
</template>

<script setup>
defineProps({
  stepper: { type: Object, required: true },
  label: { type: String, default: '' },
})
</script>
