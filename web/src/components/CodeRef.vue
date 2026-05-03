<template>
  <RepoLink
    v-if="isRef"
    :path="resolved.path"
    :line="resolved.line"
    :label="label || text"
    :tiny="tiny"
    :variant="variant"
  />
  <code v-else class="inline"><slot>{{ text }}</slot></code>
</template>

<script setup>
import { computed } from 'vue'
import RepoLink from '@/components/RepoLink.vue'
import { looksLikeRepoRef, normalizeRepoRef } from '@/utils/repo.js'

const props = defineProps({
  value:   { type: String, required: true },
  base:    { type: String, default: '' },
  label:   { type: String, default: '' },
  tiny:    { type: Boolean, default: false },
  variant: { type: String, default: 'chip' },
})

const text = computed(() => String(props.value || '').trim())
const isRef = computed(() => looksLikeRepoRef(text.value))
const resolved = computed(() => normalizeRepoRef(text.value, props.base))
</script>
