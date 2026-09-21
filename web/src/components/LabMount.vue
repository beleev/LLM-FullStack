<!--
  实验台挂载点。labs/ 目录下的 .vue 自动注册 (文件名即 lab 名), 新增 lab 不需要改任何注册表。
  某章挂哪些 lab: topicPages[route].widgets 与 data/labMap.js 两处合并。
-->
<template>
  <component :is="lab" v-for="(lab, i) in list" :key="names[i]" />
</template>

<script setup>
import { computed, defineAsyncComponent } from 'vue'
import { useRoute } from 'vue-router'
import { labMap } from '@/data/labMap.js'
import { topicPages } from '@/data/models.js'

const modules = import.meta.glob('@/components/labs/*.vue')
const LABS = Object.fromEntries(
  Object.entries(modules).map(([file, loader]) => [
    file.split('/').pop().replace('.vue', ''),
    defineAsyncComponent(loader),
  ]),
)

const props = defineProps({ names: { type: Array, default: null } })
const route = useRoute()
const names = computed(() => {
  const wanted = props.names
    || [...(topicPages[route.name]?.widgets || []), ...(labMap[route.name] || [])]
  return [...new Set(wanted)].filter((n) => LABS[n])
})
const list = computed(() => names.value.map((n) => LABS[n]))
defineExpose({ count: computed(() => list.value.length) })
</script>
