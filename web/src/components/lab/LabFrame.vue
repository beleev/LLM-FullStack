<!--
  实验台外框 —— 所有 lab 统一用它, 保证标题 / 控件 / 图 / 读数 / 挑战题的版式一致。
  插槽: controls(滑杆按钮) · default(可视化主体) · stats(右侧读数) · footer
  challenge: 先让读者预测再动手, 比直接看答案记得牢。
-->
<template>
  <section class="lab card">
    <header class="lab-head">
      <h3>{{ title }} <span v-if="module" class="tag mono">{{ module }}</span></h3>
      <p v-if="sub" class="lab-sub">{{ sub }}</p>
    </header>

    <div v-if="$slots.controls" class="lab-controls"><slot name="controls" /></div>

    <div class="lab-body" :class="{ single: !$slots.stats }">
      <div class="lab-viz"><slot /></div>
      <aside v-if="$slots.stats" class="lab-stats"><slot name="stats" /></aside>
    </div>

    <details v-if="challenge" class="lab-challenge">
      <summary>🎯 试一试: {{ challenge.ask }}</summary>
      <p>{{ challenge.answer }}</p>
    </details>
    <p v-if="run" class="lab-run">对应可运行代码: <code class="inline">{{ run }}</code></p>
    <slot name="footer" />
  </section>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
  sub: { type: String, default: '' },
  module: { type: String, default: '' },     // 例如 "llm_infer/m02"
  run: { type: String, default: '' },        // 例如 "python -m llm_infer.m02_paged_attention.demo"
  challenge: { type: Object, default: null }, // { ask, answer }
})
</script>
