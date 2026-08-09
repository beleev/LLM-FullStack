<template>
  <section class="architecture-lab" aria-labelledby="architecture-lab-title">
    <header class="lab-header">
      <div>
        <span class="lab-kicker">MODEL STRUCTURE EXPLORER</span>
        <h2 id="architecture-lab-title">模型结构与运行态实验台</h2>
        <p>拖动画布追踪组件组合，展开节点进入内部；切换运行态，观察同一结构上的权重、激活、梯度与缓存。</p>
      </div>
      <RepoLink :path="currentModel.source" label="当前模型源码" tiny />
    </header>

    <div class="model-tabs" role="tablist" aria-label="选择模型结构">
      <button
        v-for="model in modelArchitectures"
        :key="model.id"
        type="button"
        role="tab"
        :aria-selected="model.id === selectedModelId"
        :class="{ active: model.id === selectedModelId }"
        @click="selectedModelId = model.id"
      >
        <span>{{ model.name }}</span>
        <small>{{ model.badge }}</small>
      </button>
    </div>

    <div class="model-intro" aria-live="polite" aria-atomic="true">
      <div>
        <span class="model-badge">{{ currentModel.badge }}</span>
        <strong>{{ currentModel.name }}</strong>
      </div>
      <p>{{ currentModel.description }}</p>
    </div>

    <div class="runtime-toolbar">
      <div class="runtime-modes" role="group" aria-label="选择模型运行状态">
        <button
          v-for="item in runtimeModes"
          :key="item.id"
          type="button"
          :class="{ active: runtimeMode === item.id }"
          :aria-pressed="runtimeMode === item.id"
          @click="runtimeMode = item.id"
        >
          <span aria-hidden="true">{{ item.icon }}</span>
          {{ item.label }}
        </button>
      </div>
      <div class="runtime-legend" aria-label="运行态颜色图例">
        <span><i class="weight"></i>权重</span>
        <span><i class="activation"></i>激活</span>
        <span><i class="gradient"></i>梯度</span>
        <span><i class="cache"></i>持久状态</span>
      </div>
    </div>

    <section class="runtime-readout" :data-mode="runtimeMode" aria-live="polite" aria-atomic="true">
      <div class="runtime-copy">
        <span>{{ currentModeMeta.label }} · 当前路径</span>
        <strong>{{ currentRuntime.headline }}</strong>
        <p>{{ currentRuntime.note }}</p>
      </div>
      <dl class="runtime-stats">
        <div v-for="stat in currentRuntime.stats" :key="stat.label" :data-tone="stat.tone || 'neutral'">
          <dt>{{ stat.label }}</dt>
          <dd>{{ stat.value }}</dd>
        </div>
      </dl>
    </section>

    <div class="diagram-shell">
      <div class="canvas-column">
        <div class="canvas-toolbar">
          <div class="canvas-hint">
            <span aria-hidden="true">↔</span>
            空白处拖拽 · 滚轮缩放 · 点击节点查看细节
          </div>
          <div class="canvas-actions">
            <button type="button" title="展开全部组件" aria-label="展开全部组件" @click="expandAll">展开全部</button>
            <button type="button" title="收起全部组件" aria-label="收起全部组件" @click="collapseAll">收起</button>
            <span class="toolbar-divider" aria-hidden="true"></span>
            <button type="button" class="icon-button" title="缩小" aria-label="缩小结构图" @click="zoomBy(-0.12)">−</button>
            <output class="zoom-value" aria-label="当前缩放比例">{{ Math.round(zoom * 100) }}%</output>
            <button type="button" class="icon-button" title="放大" aria-label="放大结构图" @click="zoomBy(0.12)">+</button>
            <button type="button" title="适应画布" @click="fitToView">适应</button>
          </div>
        </div>

        <div
          ref="viewport"
          class="diagram-viewport"
          :class="{ dragging: isDragging }"
          tabindex="0"
          role="application"
          :aria-label="`${currentModel.name} 可拖拽结构图。使用方向键平移，加减键缩放，数字 0 适应画布。`"
          @wheel.prevent="onWheel"
          @pointerdown="onPointerDown"
          @pointermove="onPointerMove"
          @pointerup="onPointerUp"
          @pointercancel="onPointerUp"
          @keydown="onViewportKeydown"
        >
          <div class="canvas-grid" aria-hidden="true"></div>
          <div class="diagram" :style="diagramStyle">
            <svg
              class="edge-layer"
              :width="currentModel.canvas.width"
              :height="currentModel.canvas.height"
              aria-hidden="true"
            >
              <defs>
                <marker id="arch-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" />
                </marker>
                <marker id="arch-grad-arrow" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 10 0 L 0 5 L 10 10 z" />
                </marker>
              </defs>
              <g v-for="(item, index) in visibleEdges" :key="`${item.from}-${item.to}-${index}`">
                <path
                  :d="edgePath(item)"
                  class="architecture-edge"
                  :class="{ detail: item.detail, flowing: runtimeMode === 'prefill' || runtimeMode === 'decode' }"
                  marker-end="url(#arch-arrow)"
                />
                <path
                  v-if="runtimeMode === 'training' && !item.detail"
                  :d="edgePath(item, 5)"
                  class="gradient-edge"
                  marker-end="url(#arch-grad-arrow)"
                />
                <text
                  v-if="item.label"
                  class="edge-label"
                  :x="edgeLabelPosition(item).x"
                  :y="edgeLabelPosition(item).y"
                  text-anchor="middle"
                >{{ item.label }}</text>
              </g>
            </svg>

            <div
              v-for="item in visibleNodes"
              :key="item.id"
              class="arch-node"
              :class="[
                `category-${item.category}`,
                `mode-${runtimeMode}`,
                { selected: selectedNodeId === item.id, expandable: hasChildren(item.id) },
              ]"
              :style="{ left: `${item.x}px`, top: `${item.y}px` }"
            >
              <button
                type="button"
                class="node-select"
                :aria-pressed="selectedNodeId === item.id"
                @click.stop="selectNode(item.id)"
              >
                <span class="node-topline">
                  <span class="category-label"><i aria-hidden="true"></i>{{ categoryLabel(item.category) }}</span>
                  <span class="runtime-chip">{{ nodeRuntimeChip(item) }}</span>
                </span>
                <strong>{{ item.label }}</strong>
                <span class="node-shape mono">{{ item.shape || '结构节点' }}</span>
              </button>
              <button
                v-if="hasChildren(item.id)"
                type="button"
                class="expand-button"
                :title="expanded.has(item.id) ? '收起内部组件' : '展开内部组件'"
                :aria-label="`${expanded.has(item.id) ? '收起' : '展开'} ${item.label} 内部组件`"
                @click.stop="toggleExpanded(item.id)"
              >
                <span aria-hidden="true">{{ expanded.has(item.id) ? '−' : '+' }}</span>
              </button>
            </div>
          </div>

          <div class="viewport-status" aria-hidden="true">
            <span>{{ visibleNodes.length }} components</span>
            <span>{{ currentModeMeta.short }}</span>
          </div>
        </div>
      </div>

      <aside class="node-inspector" aria-labelledby="node-inspector-title">
        <div class="inspector-heading">
          <span>{{ currentModeMeta.label }} · COMPONENT INSPECTOR</span>
          <h3 id="node-inspector-title">{{ selectedNode?.label || '选择一个组件' }}</h3>
        </div>

        <template v-if="selectedNode">
          <p class="node-summary">{{ selectedNode.summary }}</p>

          <div class="runtime-callout" :data-mode="runtimeMode">
            <span>此刻发生什么</span>
            <p>{{ nodeRuntimeDescription(selectedNode) }}</p>
          </div>

          <dl class="node-facts">
            <div>
              <dt>组件类型</dt>
              <dd>{{ categoryLabel(selectedNode.category) }}</dd>
            </div>
            <div>
              <dt>张量形状</dt>
              <dd class="mono">{{ selectedNode.shape || '随上游保持不变' }}</dd>
            </div>
          </dl>

          <section v-if="selectedNode.weights?.length" class="inspector-section">
            <h4><i class="legend-dot weight"></i>权重参数</h4>
            <div v-for="weight in selectedNode.weights" :key="weight.name" class="tensor-row">
              <div>
                <strong class="mono">{{ weight.name }}</strong>
                <span class="mono">{{ weight.shape }}</span>
              </div>
              <p>{{ weight.note }}</p>
            </div>
          </section>

          <section v-if="selectedNode.activations" class="inspector-section">
            <h4><i class="legend-dot activation"></i>激活生命周期</h4>
            <p>{{ selectedNode.activations }}</p>
          </section>

          <section v-if="selectedNode.formula || selectedNode.detail" class="inspector-section">
            <h4>内部机制</h4>
            <code>{{ selectedNode.formula || selectedNode.detail }}</code>
          </section>

          <div v-if="selectedNode.source" class="source-row">
            <span>对应源码</span>
            <RepoLink :path="selectedNode.source" tiny />
          </div>
        </template>

        <p v-else class="inspector-empty">点击画布中的组件，查看权重、激活、运行态和源码入口。</p>
      </aside>
    </div>
  </section>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import RepoLink from '@/components/RepoLink.vue'
import { architectureById, modelArchitectures } from '@/data/modelArchitectures.js'

const NODE_WIDTH = 196
const NODE_HEIGHT = 92
const MIN_ZOOM = 0.38
const MAX_ZOOM = 1.55

const runtimeModes = [
  { id: 'structure', label: '结构', short: 'STRUCTURE', icon: '◇' },
  { id: 'training', label: '训练态', short: 'FORWARD + BACKWARD', icon: '↔' },
  { id: 'prefill', label: '并行前向', short: 'FULL-SEQUENCE', icon: '▦' },
  { id: 'decode', label: '迭代推理', short: 'STEP-BY-STEP', icon: '▷' },
]

const categoryLabels = {
  input: '输入',
  embedding: '嵌入',
  stack: '重复层',
  norm: '归一化',
  attention: '注意力',
  ffn: '前馈网络',
  router: '路由器',
  expert: '专家',
  state: '持久状态',
  weight: '投影权重',
  position: '位置编码',
  condition: '条件调制',
  merge: '合并',
  output: '输出层',
  result: '结果',
}

const selectedModelId = ref('llama')
const runtimeMode = ref('structure')
const selectedNodeId = ref('blocks')
const expanded = ref(new Set())
const viewport = ref(null)
const zoom = ref(0.72)
const pan = ref({ x: 24, y: 20 })
const isDragging = ref(false)
const dragStart = ref({ x: 0, y: 0, panX: 0, panY: 0 })

let resizeObserver

const currentModel = computed(() => architectureById[selectedModelId.value] || modelArchitectures[0])
const currentModeMeta = computed(() => runtimeModes.find((item) => item.id === runtimeMode.value))
const currentRuntime = computed(() => currentModel.value.runtime[runtimeMode.value])
const nodeMap = computed(() => new Map(currentModel.value.nodes.map((item) => [item.id, item])))

const hasChildren = (id) => currentModel.value.nodes.some((item) => item.parent === id)

const isNodeVisible = (item) => {
  let parentId = item.parent
  const seen = new Set()
  while (parentId) {
    if (seen.has(parentId) || !expanded.value.has(parentId)) return false
    seen.add(parentId)
    parentId = nodeMap.value.get(parentId)?.parent
  }
  return true
}

const visibleNodes = computed(() => currentModel.value.nodes.filter(isNodeVisible))
const visibleNodeIds = computed(() => new Set(visibleNodes.value.map((item) => item.id)))
const visibleEdges = computed(() => currentModel.value.edges.filter(
  (item) => visibleNodeIds.value.has(item.from) && visibleNodeIds.value.has(item.to),
))
const selectedNode = computed(() => nodeMap.value.get(selectedNodeId.value) || visibleNodes.value[0] || null)

const diagramStyle = computed(() => ({
  width: `${currentModel.value.canvas.width}px`,
  height: `${currentModel.value.canvas.height}px`,
  transform: `translate3d(${pan.value.x}px, ${pan.value.y}px, 0) scale(${zoom.value})`,
}))

const categoryLabel = (category) => categoryLabels[category] || '组件'

const nodeRuntimeChip = (item) => {
  if (runtimeMode.value === 'structure') return hasChildren(item.id) ? '可展开' : '组件'
  if (runtimeMode.value === 'training') {
    if (item.category === 'result') return 'LOSS 起点'
    if (item.weights?.length) return 'W + dW'
    if (item.category === 'state') return '保存状态'
    return '保存激活'
  }
  if (runtimeMode.value === 'prefill') {
    if (item.category === 'attention') return '写缓存'
    if (item.category === 'state') return '建立状态'
    if (item.weights?.length) return '只读 W'
    return '全序列'
  }
  if (item.category === 'attention') return '读 / 追加'
  if (item.category === 'state') return '跨步保留'
  if (item.category === 'input') return '1 STEP'
  if (item.weights?.length) return '只读 W'
  return '临时激活'
}

const genericRuntimeDescriptions = {
  structure: (item) => item.detail || `它在结构主干中承担“${categoryLabel(item.category)}”角色。`,
  training: (item) => item.weights?.length
    ? '前向读取权重产生激活；反向根据上游梯度计算 dW，优化器随后更新参数。'
    : '前向结果需保留到 backward 或由 activation checkpoint 重算；梯度沿相反方向通过此节点。',
  prefill: (item) => item.category === 'attention' || item.category === 'state'
    ? '整段输入并行计算，并建立后续迭代会复用的持久状态。'
    : '权重只读，整段输入并行通过；临时激活在下游消费后即可释放。',
  decode: (item) => item.category === 'attention' || item.category === 'state'
    ? '读取之前步骤留下的状态，处理本轮新输入并把新状态追加或覆盖。'
    : '只处理当前 step 的小激活；权重重复使用，中间张量不跨 step 保留。',
}

const nodeRuntimeDescription = (item) => (
  item.runtime?.[runtimeMode.value]
  || genericRuntimeDescriptions[runtimeMode.value](item)
)

const selectNode = (id) => {
  selectedNodeId.value = id
}

const toggleExpanded = (id) => {
  const next = new Set(expanded.value)
  if (next.has(id)) {
    next.delete(id)
    let parentId = selectedNode.value?.parent
    while (parentId) {
      if (parentId === id) {
        selectedNodeId.value = id
        break
      }
      parentId = nodeMap.value.get(parentId)?.parent
    }
  } else next.add(id)
  expanded.value = next
}

const expandAll = () => {
  expanded.value = new Set(currentModel.value.nodes.filter((item) => hasChildren(item.id)).map((item) => item.id))
  nextTick(fitToView)
}

const collapseAll = () => {
  expanded.value = new Set()
  if (selectedNode.value?.parent) selectedNodeId.value = currentModel.value.nodes.find((item) => !item.parent)?.id || ''
  nextTick(fitToView)
}

const edgeAnchors = (item, offset = 0) => {
  const from = nodeMap.value.get(item.from)
  const to = nodeMap.value.get(item.to)
  if (!from || !to) return null

  if (item.detail || Math.abs(to.x - from.x) < 70) {
    return {
      start: { x: from.x + NODE_WIDTH / 2 + offset, y: from.y + NODE_HEIGHT },
      end: { x: to.x + NODE_WIDTH / 2 + offset, y: to.y },
      vertical: true,
    }
  }

  return {
    start: { x: from.x + NODE_WIDTH, y: from.y + NODE_HEIGHT / 2 + offset },
    end: { x: to.x, y: to.y + NODE_HEIGHT / 2 + offset },
    vertical: false,
  }
}

const edgePath = (item, offset = 0) => {
  const anchors = edgeAnchors(item, offset)
  if (!anchors) return ''
  const { start, end, vertical } = anchors
  if (vertical) {
    const midY = start.y + (end.y - start.y) / 2
    return `M ${start.x} ${start.y} C ${start.x} ${midY}, ${end.x} ${midY}, ${end.x} ${end.y}`
  }
  const midX = start.x + (end.x - start.x) / 2
  return `M ${start.x} ${start.y} C ${midX} ${start.y}, ${midX} ${end.y}, ${end.x} ${end.y}`
}

const edgeLabelPosition = (item) => {
  const anchors = edgeAnchors(item)
  if (!anchors) return { x: 0, y: 0 }
  return {
    x: (anchors.start.x + anchors.end.x) / 2,
    y: (anchors.start.y + anchors.end.y) / 2 - 7,
  }
}

const visibleBounds = () => {
  if (!visibleNodes.value.length) return { minX: 0, minY: 0, width: 1, height: 1 }
  const minX = Math.min(...visibleNodes.value.map((item) => item.x))
  const minY = Math.min(...visibleNodes.value.map((item) => item.y))
  const maxX = Math.max(...visibleNodes.value.map((item) => item.x + NODE_WIDTH))
  const maxY = Math.max(...visibleNodes.value.map((item) => item.y + NODE_HEIGHT))
  return { minX, minY, width: maxX - minX, height: maxY - minY }
}

const fitToView = () => {
  if (!viewport.value) return
  const bounds = visibleBounds()
  const padding = 56
  const nextZoom = Math.min(
    1,
    (viewport.value.clientWidth - padding * 2) / bounds.width,
    (viewport.value.clientHeight - padding * 2) / bounds.height,
  )
  zoom.value = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, nextZoom))
  pan.value = {
    x: (viewport.value.clientWidth - bounds.width * zoom.value) / 2 - bounds.minX * zoom.value,
    y: (viewport.value.clientHeight - bounds.height * zoom.value) / 2 - bounds.minY * zoom.value,
  }
}

const zoomAround = (nextZoom, clientX, clientY) => {
  if (!viewport.value) return
  const bounded = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, nextZoom))
  const rect = viewport.value.getBoundingClientRect()
  const pointerX = clientX - rect.left
  const pointerY = clientY - rect.top
  const worldX = (pointerX - pan.value.x) / zoom.value
  const worldY = (pointerY - pan.value.y) / zoom.value
  pan.value = {
    x: pointerX - worldX * bounded,
    y: pointerY - worldY * bounded,
  }
  zoom.value = bounded
}

const zoomBy = (delta) => {
  if (!viewport.value) return
  const rect = viewport.value.getBoundingClientRect()
  zoomAround(zoom.value + delta, rect.left + rect.width / 2, rect.top + rect.height / 2)
}

const onWheel = (event) => {
  zoomAround(zoom.value * (event.deltaY > 0 ? 0.9 : 1.1), event.clientX, event.clientY)
}

const onPointerDown = (event) => {
  if (event.button !== 0 || event.target.closest('.arch-node')) return
  isDragging.value = true
  dragStart.value = { x: event.clientX, y: event.clientY, panX: pan.value.x, panY: pan.value.y }
  viewport.value?.setPointerCapture(event.pointerId)
}

const onPointerMove = (event) => {
  if (!isDragging.value) return
  pan.value = {
    x: dragStart.value.panX + event.clientX - dragStart.value.x,
    y: dragStart.value.panY + event.clientY - dragStart.value.y,
  }
}

const onPointerUp = (event) => {
  if (!isDragging.value) return
  isDragging.value = false
  if (viewport.value?.hasPointerCapture(event.pointerId)) viewport.value.releasePointerCapture(event.pointerId)
}

const onViewportKeydown = (event) => {
  if (event.target.closest('.arch-node')) return
  const step = event.shiftKey ? 80 : 32
  const actions = {
    ArrowLeft: () => { pan.value = { ...pan.value, x: pan.value.x + step } },
    ArrowRight: () => { pan.value = { ...pan.value, x: pan.value.x - step } },
    ArrowUp: () => { pan.value = { ...pan.value, y: pan.value.y + step } },
    ArrowDown: () => { pan.value = { ...pan.value, y: pan.value.y - step } },
    '+': () => zoomBy(0.12),
    '=': () => zoomBy(0.12),
    '-': () => zoomBy(-0.12),
    '0': fitToView,
  }
  if (!actions[event.key]) return
  event.preventDefault()
  actions[event.key]()
}

watch(selectedModelId, () => {
  expanded.value = new Set(currentModel.value.defaultExpanded || [])
  selectedNodeId.value = currentModel.value.defaultExpanded?.[0]
    || currentModel.value.nodes.find((item) => !item.parent)?.id
    || ''
  nextTick(fitToView)
}, { immediate: true })

onMounted(() => {
  resizeObserver = new ResizeObserver(() => fitToView())
  if (viewport.value) resizeObserver.observe(viewport.value)
  nextTick(fitToView)
})

onBeforeUnmount(() => resizeObserver?.disconnect())
</script>

<style scoped>
.architecture-lab {
  --runtime-weight: #f59e0b;
  --runtime-activation: #38bdf8;
  --runtime-gradient: #f472b6;
  --runtime-cache: #34d399;
  margin-top: 24px;
  border: 1px solid var(--border-strong);
  border-radius: calc(var(--radius) + 4px);
  background: var(--bg-card);
  overflow: hidden;
  box-shadow: 0 18px 52px color-mix(in srgb, var(--text) 8%, transparent);
}

.lab-header {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: flex-start;
  padding: 22px 24px 18px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(110deg, color-mix(in srgb, var(--accent) 8%, var(--bg-elev)), var(--bg-card) 62%);
}
.lab-kicker,
.inspector-heading > span,
.runtime-copy > span {
  color: var(--accent);
  font-family: "SF Mono", Menlo, monospace;
  font-size: 9.5px;
  font-weight: 700;
  letter-spacing: 1px;
}
.lab-header h2 { margin-top: 3px; font-size: 20px; text-wrap: balance; }
.lab-header p { margin-top: 5px; max-width: 780px; color: var(--text-muted); font-size: 12.5px; line-height: 1.7; text-wrap: pretty; }

.model-tabs {
  display: flex;
  gap: 1px;
  padding: 8px;
  overflow-x: auto;
  border-bottom: 1px solid var(--border);
  background: var(--bg-elev);
}
.model-tabs button {
  flex: 1 0 128px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  min-height: 52px;
  padding: 8px 11px;
  border-color: transparent;
  background: transparent;
  text-align: left;
}
.model-tabs button:hover { background: var(--bg-card); }
.model-tabs button.active {
  border-color: var(--border-strong);
  background: var(--bg-card);
  color: var(--text);
  box-shadow: 0 3px 10px color-mix(in srgb, var(--text) 6%, transparent);
}
.model-tabs button span { font-size: 12px; font-weight: 650; }
.model-tabs button small { margin-top: 1px; color: var(--text-muted); font-size: 9.5px; }
.model-tabs button.active small { color: var(--accent); }

.model-intro {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 18px;
  align-items: center;
  min-height: 66px;
  padding: 12px 24px;
  border-bottom: 1px solid var(--border);
}
.model-intro > div { display: flex; flex-direction: column; align-items: flex-start; }
.model-intro strong { margin-top: 2px; font-size: 16px; }
.model-badge { color: var(--accent); font-size: 9.5px; font-weight: 700; letter-spacing: 0.7px; text-transform: uppercase; }
.model-intro p { color: var(--text-muted); font-size: 12.5px; line-height: 1.65; text-wrap: pretty; }

.runtime-toolbar {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: center;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--code-bg);
}
.runtime-modes { display: flex; flex-wrap: wrap; gap: 6px; }
.runtime-modes button { min-height: 40px; padding: 6px 12px; font-size: 11.5px; }
.runtime-modes button span { margin-right: 4px; font-family: "SF Mono", Menlo, monospace; }
.runtime-legend { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px 12px; color: var(--text-muted); font-size: 10px; }
.runtime-legend span { display: inline-flex; align-items: center; gap: 5px; }
.runtime-legend i,
.legend-dot { width: 8px; height: 8px; border-radius: 50%; }
.weight { background: var(--runtime-weight); }
.activation { background: var(--runtime-activation); }
.gradient { background: var(--runtime-gradient); }
.cache { background: var(--runtime-cache); }

.runtime-readout {
  display: grid;
  grid-template-columns: minmax(280px, 1fr) minmax(420px, 0.9fr);
  gap: 20px;
  align-items: center;
  padding: 14px 20px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--accent) 4%, var(--bg-card));
}
.runtime-readout[data-mode="training"] { background: color-mix(in srgb, var(--runtime-gradient) 6%, var(--bg-card)); }
.runtime-readout[data-mode="prefill"] { background: color-mix(in srgb, var(--runtime-activation) 6%, var(--bg-card)); }
.runtime-readout[data-mode="decode"] { background: color-mix(in srgb, var(--runtime-cache) 6%, var(--bg-card)); }
.runtime-copy strong { display: block; margin-top: 3px; font-size: 13.5px; text-wrap: balance; }
.runtime-copy p { margin-top: 3px; color: var(--text-muted); font-size: 11.5px; line-height: 1.65; text-wrap: pretty; }
.runtime-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; }
.runtime-stats > div { min-width: 0; padding: 8px 10px; border-left: 2px solid var(--border-strong); background: var(--bg-elev); }
.runtime-stats > div[data-tone="weight"] { border-left-color: var(--runtime-weight); }
.runtime-stats > div[data-tone="activation"] { border-left-color: var(--runtime-activation); }
.runtime-stats > div[data-tone="gradient"] { border-left-color: var(--runtime-gradient); }
.runtime-stats > div[data-tone="cache"] { border-left-color: var(--runtime-cache); }
.runtime-stats dt { color: var(--text-dim); font-size: 9px; letter-spacing: 0.5px; text-transform: uppercase; }
.runtime-stats dd { margin-top: 2px; overflow-wrap: anywhere; color: var(--text); font-family: "SF Mono", Menlo, monospace; font-size: 10.5px; font-variant-numeric: tabular-nums; }

.diagram-shell { display: grid; grid-template-columns: minmax(0, 1fr) 330px; min-height: 650px; }
.canvas-column { min-width: 0; border-right: 1px solid var(--border); }
.canvas-toolbar {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  min-height: 50px;
  padding: 6px 10px 6px 14px;
  border-bottom: 1px solid var(--border);
}
.canvas-hint { color: var(--text-muted); font-size: 10.5px; }
.canvas-hint span { margin-right: 6px; color: var(--accent); }
.canvas-actions { display: flex; align-items: center; gap: 5px; }
.canvas-actions button { min-height: 38px; padding: 5px 9px; font-size: 10.5px; }
.canvas-actions .icon-button { width: 38px; padding: 0; font-size: 16px; }
.toolbar-divider { width: 1px; height: 24px; margin: 0 2px; background: var(--border); }
.zoom-value { min-width: 42px; color: var(--text-muted); font-family: "SF Mono", Menlo, monospace; font-size: 10px; font-variant-numeric: tabular-nums; text-align: center; }

.diagram-viewport {
  position: relative;
  height: 600px;
  overflow: hidden;
  outline: none;
  background: var(--bg);
  cursor: grab;
  touch-action: none;
  user-select: none;
}
.diagram-viewport:focus-visible { box-shadow: inset 0 0 0 3px var(--accent); }
.diagram-viewport.dragging { cursor: grabbing; }
.canvas-grid {
  position: absolute;
  inset: 0;
  pointer-events: none;
  opacity: 0.46;
  background-image:
    linear-gradient(var(--border) 1px, transparent 1px),
    linear-gradient(90deg, var(--border) 1px, transparent 1px),
    radial-gradient(circle, var(--border-strong) 1px, transparent 1px);
  background-size: 48px 48px, 48px 48px, 12px 12px;
}
.diagram {
  position: absolute;
  left: 0;
  top: 0;
  transform-origin: 0 0;
  transition: transform 80ms ease-out;
}
.dragging .diagram { transition-duration: 0ms; }
.edge-layer { position: absolute; inset: 0; overflow: visible; pointer-events: none; }
.architecture-edge,
.gradient-edge {
  fill: none;
  stroke: var(--border-strong);
  stroke-width: 2;
  vector-effect: non-scaling-stroke;
}
.architecture-edge.detail { stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 5 5; opacity: 0.72; }
.architecture-edge.flowing:not(.detail) { stroke: var(--runtime-activation); stroke-dasharray: 8 6; animation: flow-dashes 900ms linear infinite; }
.gradient-edge { stroke: var(--runtime-gradient); stroke-width: 1.6; stroke-dasharray: 3 6; opacity: 0.9; animation: reverse-flow 1.1s linear infinite; }
#arch-arrow path { fill: var(--border-strong); }
#arch-grad-arrow path { fill: var(--runtime-gradient); }
.edge-label {
  fill: var(--text-muted);
  paint-order: stroke;
  stroke: var(--bg);
  stroke-width: 5px;
  stroke-linejoin: round;
  font-family: "SF Mono", Menlo, monospace;
  font-size: 9px;
}
@keyframes flow-dashes { to { stroke-dashoffset: -28; } }
@keyframes reverse-flow { to { stroke-dashoffset: 27; } }

.arch-node {
  position: absolute;
  width: 196px;
  height: 92px;
  padding: 0;
  border: 1px solid var(--border-strong);
  border-top: 3px solid var(--node-color, var(--accent));
  border-radius: 10px;
  background: color-mix(in srgb, var(--node-color, var(--accent)) 5%, var(--bg-card));
  box-shadow: 0 7px 18px color-mix(in srgb, var(--text) 7%, transparent);
  cursor: pointer;
  transition-property: border-color, box-shadow, transform, background-color;
  transition-duration: 150ms;
  transition-timing-function: ease-out;
}
.arch-node:hover { transform: translateY(-2px); border-color: var(--node-color, var(--accent)); }
.arch-node:focus-visible { outline: 4px solid color-mix(in srgb, var(--node-color, var(--accent)) 68%, transparent); outline-offset: 3px; }
.arch-node.selected {
  border-color: var(--node-color, var(--accent));
  background: color-mix(in srgb, var(--node-color, var(--accent)) 12%, var(--bg-card));
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--node-color, var(--accent)) 20%, transparent), 0 12px 25px color-mix(in srgb, var(--text) 11%, transparent);
}
.category-input { --node-color: #64748b; }
.category-embedding, .category-position { --node-color: #8b5cf6; }
.category-stack { --node-color: #2563eb; }
.category-norm { --node-color: #0ea5e9; }
.category-attention { --node-color: #06b6d4; }
.category-ffn { --node-color: #f59e0b; }
.category-router { --node-color: #f97316; }
.category-expert { --node-color: #ef4444; }
.category-state { --node-color: #10b981; }
.category-weight { --node-color: #eab308; }
.category-condition { --node-color: #ec4899; }
.category-merge { --node-color: #14b8a6; }
.category-output { --node-color: #6366f1; }
.category-result { --node-color: #22c55e; }
.arch-node.mode-training { box-shadow: inset 0 -3px 0 color-mix(in srgb, var(--runtime-gradient) 48%, transparent), 0 7px 18px color-mix(in srgb, var(--text) 7%, transparent); }
.arch-node.mode-prefill { box-shadow: inset 0 -3px 0 color-mix(in srgb, var(--runtime-activation) 58%, transparent), 0 7px 18px color-mix(in srgb, var(--text) 7%, transparent); }
.arch-node.mode-decode { box-shadow: inset 0 -3px 0 color-mix(in srgb, var(--runtime-cache) 58%, transparent), 0 7px 18px color-mix(in srgb, var(--text) 7%, transparent); }
.node-select {
  width: 100%;
  height: 100%;
  min-height: 0;
  padding: 10px 11px;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: inherit;
  text-align: left;
}
.node-select:hover { border-color: transparent; }
.node-select:active:not(:disabled) { transform: none; }
.node-select:focus-visible { outline: 4px solid color-mix(in srgb, var(--node-color, var(--accent)) 68%, transparent); outline-offset: 3px; }
.node-topline { display: flex; justify-content: space-between; gap: 5px; align-items: center; }
.category-label { display: inline-flex; align-items: center; min-width: 0; color: var(--text-muted); font-size: 8.5px; font-weight: 700; letter-spacing: 0.5px; }
.category-label i { flex: 0 0 auto; width: 6px; height: 6px; margin-right: 4px; border-radius: 50%; background: var(--node-color, var(--accent)); }
.runtime-chip { max-width: 84px; overflow: hidden; color: var(--text-dim); font-family: "SF Mono", Menlo, monospace; font-size: 7.5px; text-overflow: ellipsis; white-space: nowrap; }
.mode-training .runtime-chip { color: var(--runtime-gradient); }
.mode-prefill .runtime-chip { color: var(--runtime-activation); }
.mode-decode .runtime-chip { color: var(--runtime-cache); }
.node-select > strong { display: block; margin-top: 7px; overflow: hidden; color: var(--text); font-size: 12.5px; font-weight: 680; line-height: 1.25; text-overflow: ellipsis; white-space: nowrap; }
.node-shape { display: block; margin-top: 5px; overflow: hidden; color: var(--text-muted); font-size: 8.5px; line-height: 1.25; text-overflow: ellipsis; white-space: nowrap; }
.expand-button {
  position: absolute;
  right: -10px;
  bottom: -10px;
  width: 30px;
  min-height: 30px;
  padding: 0;
  border: 2px solid var(--bg);
  border-radius: 50%;
  background: var(--node-color, var(--accent));
  color: #fff;
  font-family: "SF Mono", Menlo, monospace;
  font-size: 15px;
  line-height: 1;
  box-shadow: 0 4px 10px color-mix(in srgb, var(--text) 15%, transparent);
}
.expand-button:hover { border-color: var(--bg); filter: brightness(1.08); }
.viewport-status {
  position: absolute;
  left: 12px;
  bottom: 10px;
  display: flex;
  gap: 7px;
  pointer-events: none;
}
.viewport-status span { padding: 3px 7px; border: 1px solid var(--border); border-radius: 4px; background: color-mix(in srgb, var(--bg-elev) 88%, transparent); color: var(--text-muted); font-family: "SF Mono", Menlo, monospace; font-size: 8.5px; }

.node-inspector { min-width: 0; padding: 18px; background: var(--bg-elev); }
.inspector-heading { padding-bottom: 12px; border-bottom: 1px solid var(--border); }
.inspector-heading h3 { margin-top: 4px; font-size: 17px; line-height: 1.35; text-wrap: balance; }
.node-summary { margin-top: 13px; color: var(--text-muted); font-size: 12px; line-height: 1.7; text-wrap: pretty; }
.runtime-callout { margin-top: 14px; padding: 10px 12px; border-left: 3px solid var(--accent); background: var(--bg-card); }
.runtime-callout[data-mode="training"] { border-left-color: var(--runtime-gradient); }
.runtime-callout[data-mode="prefill"] { border-left-color: var(--runtime-activation); }
.runtime-callout[data-mode="decode"] { border-left-color: var(--runtime-cache); }
.runtime-callout span { color: var(--text-dim); font-size: 9px; font-weight: 700; letter-spacing: 0.7px; }
.runtime-callout p { margin-top: 3px; color: var(--text); font-size: 11px; line-height: 1.65; text-wrap: pretty; }
.node-facts { display: grid; grid-template-columns: 1fr; gap: 1px; margin-top: 14px; border: 1px solid var(--border); background: var(--border); }
.node-facts > div { padding: 8px 10px; background: var(--bg-card); }
.node-facts dt { color: var(--text-dim); font-size: 8.5px; letter-spacing: 0.5px; text-transform: uppercase; }
.node-facts dd { margin-top: 2px; overflow-wrap: anywhere; color: var(--text); font-size: 10.5px; }
.inspector-section { margin-top: 16px; }
.inspector-section h4 { display: flex; align-items: center; gap: 6px; margin-bottom: 7px; color: var(--text); font-size: 11px; }
.tensor-row { padding: 8px 0; border-top: 1px solid var(--border); }
.tensor-row > div { display: flex; justify-content: space-between; gap: 10px; }
.tensor-row strong { min-width: 0; overflow-wrap: anywhere; color: var(--text); font-size: 9.5px; }
.tensor-row span { flex: 0 0 auto; color: var(--runtime-weight); font-size: 9px; }
.tensor-row p,
.inspector-section > p { margin-top: 3px; color: var(--text-muted); font-size: 10.5px; line-height: 1.6; text-wrap: pretty; }
.inspector-section code { display: block; overflow-wrap: anywhere; padding: 9px 10px; border: 1px solid var(--border); border-radius: 5px; background: var(--code-bg); color: var(--code-text); font-size: 9.5px; line-height: 1.6; white-space: normal; }
.source-row { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; align-items: center; margin-top: 18px; padding-top: 13px; border-top: 1px solid var(--border); }
.source-row > span { color: var(--text-dim); font-size: 9px; letter-spacing: 0.5px; }
.inspector-empty { margin-top: 16px; color: var(--text-muted); font-size: 12px; line-height: 1.7; }

@media (max-width: 1080px) {
  .runtime-readout { grid-template-columns: 1fr; }
  .diagram-shell { grid-template-columns: 1fr; }
  .canvas-column { border-right: 0; border-bottom: 1px solid var(--border); }
  .node-inspector { display: grid; grid-template-columns: minmax(180px, 0.7fr) minmax(260px, 1fr); gap: 0 24px; }
  .inspector-heading,
  .node-summary,
  .runtime-callout { grid-column: 1; }
  .node-facts,
  .inspector-section,
  .source-row { grid-column: 2; }
  .node-facts { grid-row: 1 / span 2; margin-top: 0; }
}

@media (max-width: 720px) {
  .lab-header { flex-direction: column; padding: 18px; }
  .model-intro { grid-template-columns: 1fr; gap: 5px; padding: 12px 18px; }
  .runtime-toolbar { align-items: flex-start; flex-direction: column; }
  .runtime-legend { justify-content: flex-start; }
  .runtime-readout { padding: 14px 16px; }
  .runtime-stats { grid-template-columns: 1fr; }
  .canvas-toolbar { align-items: flex-start; flex-direction: column; padding: 9px 10px; }
  .canvas-actions { width: 100%; overflow-x: auto; }
  .canvas-hint { padding-left: 3px; }
  .diagram-viewport { height: 520px; }
  .node-inspector { display: block; padding: 16px; }
  .node-facts { margin-top: 14px; }
}
</style>
