<template>
  <div class="flow">
    <template v-for="(step, i) in steps" :key="i">
      <!-- 起点 / 终点 -->
      <div v-if="step.type === 'input'" class="flow-node input">
        <div class="node-label mono">{{ step.label }}</div>
        <div class="shape-row">
          <ShapeTuple :shape="step.shape" :ctx="ctx" />
          <SizeBar :shape="step.shape" :ctx="ctx" :max="maxNumel" />
        </div>
        <div v-if="step.note" class="node-note">{{ step.note }}</div>
      </div>

      <div v-else-if="step.type === 'output'" class="flow-node output">
        <div class="node-label mono">{{ step.label }}</div>
        <div class="shape-row">
          <ShapeTuple :shape="step.shape" :ctx="ctx" />
          <SizeBar :shape="step.shape" :ctx="ctx" :max="maxNumel" />
        </div>
      </div>

      <!-- 单步 op -->
      <div v-else-if="step.type === 'op'"
           :class="['flow-node', 'op', step.kind || 'matmul',
                    { highlight: step.highlight, lit: usesParam(step) }]">
        <div class="op-line">
          <code class="op-expr">{{ step.op }}</code>
          <span v-if="step.out" class="arrow">→</span>
          <span v-if="step.out" class="out-name mono">{{ step.out }}</span>
        </div>
        <div v-if="step.shape" class="shape-row">
          <ShapeTuple :shape="step.shape" :ctx="ctx" />
          <SizeBar :shape="step.shape" :ctx="ctx" :max="maxNumel" />
        </div>
        <div v-if="step.note" class="node-note">{{ step.note }}</div>
      </div>

      <!-- 并行分支 -->
      <div v-else-if="step.type === 'branch'" class="flow-branch">
        <div v-if="step.note" class="branch-note">↓ {{ step.note }}</div>
        <div class="branch-row">
          <div v-for="(b, j) in step.items" :key="j"
               :class="['flow-node', 'op', 'branch-item', b.kind || 'matmul',
                        { lit: usesParam(b) }]">
            <div class="op-line">
              <code class="op-expr">{{ b.op }}</code>
              <span v-if="b.out" class="arrow">→</span>
              <span v-if="b.out" class="out-name mono">{{ b.out }}</span>
            </div>
            <div v-if="b.shape" class="shape-row">
              <ShapeTuple :shape="b.shape" :ctx="ctx" />
              <SizeBar :shape="b.shape" :ctx="ctx" :max="maxNumel" />
            </div>
            <div v-if="b.note" class="node-note">{{ b.note }}</div>
          </div>
        </div>
      </div>

      <!-- 连接线: 除了最后一步外, 每步后面都画一条 -->
      <div v-if="i < steps.length - 1" class="flow-connector"></div>
    </template>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import ShapeTuple from './ShapeTuple.vue'
import SizeBar from './SizeBar.vue'
import { numelOf } from '@/data/inspector.js'

const props = defineProps({
  steps: { type: Array, required: true },
  ctx:   { type: Object, required: true },
  // 鼠标停在左边哪个权重上 —— 用到它的那几步会亮起来
  activeParam: { type: String, default: '' },
})

// 全流程里最大的那个张量; 每条的长度都相对它, 所以 [B,H,T,T] 会明显压过别人
const maxNumel = computed(() => {
  let m = 0
  for (const s of props.steps) {
    for (const x of s.type === 'branch' ? s.items : [s]) {
      if (x.shape) m = Math.max(m, numelOf(x.shape, props.ctx))
    }
  }
  return m || 1
})

const usesParam = (step) =>
  !!props.activeParam && String(step.op || '').includes(props.activeParam)
</script>

<style scoped>
.flow {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 0;
}

.flow-node {
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 10px 14px;
  background: var(--bg-elev);
  position: relative;
}

.flow-node.input,
.flow-node.output {
  background: var(--bg-card);
  border: 1px solid var(--border-strong);
  border-left: 3px solid var(--accent);
}
.flow-node.output { border-left-color: var(--left); }

.flow-node .node-label {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 4px;
  color: var(--text);
}

.shape-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
/* 被左边权重表点亮的那几步 */
.flow-node.op.lit {
  background: color-mix(in srgb, var(--warn) 12%, var(--bg-elev));
  border-color: var(--warn);
}

/* 分类颜色 (左边框) */
.flow-node.op.matmul      { border-left: 3px solid #60a5fa; }
.flow-node.op.activation  { border-left: 3px solid #3dd68c; }
.flow-node.op.reshape     { border-left: 3px solid #9ca3af; }
.flow-node.op.attn        { border-left: 3px solid #c084fc; }
.flow-node.op.cond        { border-left: 3px solid #f5a623; }
.flow-node.op.route       { border-left: 3px solid #ec4899; }

.flow-node.op.highlight {
  background: color-mix(in srgb, var(--accent) 9%, var(--bg-elev));
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent-soft), 0 2px 12px rgba(124, 107, 241, 0.16);
}

.op-line {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
  font-family: "SF Mono", "JetBrains Mono", Menlo, Monaco, monospace;
  font-size: 12.5px;
  color: var(--text);
  line-height: 1.5;
}
.op-expr {
  background: transparent;
  padding: 0;
  border: 0;
  color: var(--text);
  font-family: inherit;
  font-size: inherit;
}
.op-line .arrow  { color: var(--text-dim); font-size: 11px; }
.op-line .out-name { color: var(--accent); font-weight: 500; }

.node-note {
  margin-top: 6px;
  font-size: 11.5px;
  color: var(--text-muted);
  line-height: 1.55;
  padding-left: 8px;
  border-left: 2px solid var(--border);
}

/* 分支: 几个 item 并排 */
.flow-branch {
  display: flex;
  flex-direction: column;
}
.branch-note {
  color: var(--text-dim);
  font-size: 11px;
  padding: 4px 0 6px 4px;
  letter-spacing: 0.3px;
}
.branch-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
}
.branch-row .flow-node.op {
  margin: 0;
}

/* 连接线: 中间竖线 */
.flow-connector {
  width: 2px;
  height: 18px;
  background: linear-gradient(to bottom, var(--border-strong), var(--border));
  margin: 0 auto;
  border-radius: 1px;
}
</style>
