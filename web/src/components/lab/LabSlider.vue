<!-- 一行滑杆: 标签 + range + 当前值。 <LabSlider v-model="T" label="温度 T" :min="0.1" :max="5" :step="0.1" /> -->
<template>
  <div class="ctl">
    <label :for="id">{{ label }}</label>
    <input
      :id="id" type="range" :min="min" :max="max" :step="step"
      :value="modelValue" @input="$emit('update:modelValue', Number($event.target.value))"
    />
    <span class="val mono">{{ format ? format(modelValue) : modelValue }}{{ unit }}</span>
  </div>
</template>

<script setup>
import { useId } from 'vue'
defineProps({
  modelValue: { type: Number, required: true },
  label: { type: String, required: true },
  min: { type: Number, default: 0 },
  max: { type: Number, default: 100 },
  step: { type: Number, default: 1 },
  unit: { type: String, default: '' },
  format: { type: Function, default: null },
})
defineEmits(['update:modelValue'])
const id = useId()
</script>
