# 怎么给教程加一个交互实验台

目标：读者**拖一下、点一下，就看见这个技术点为什么存在**。一个 lab 只讲一件事。

范例：`src/components/labs/PipelineLab.vue`（约 130 行，GPipe vs 1F1B）。先读它再动手。

## 加内容只需要新增文件，不改任何注册表

| 想加什么 | 新增 / 编辑哪个文件 | 自动发生什么 |
|---|---|---|
| 实验台 | `src/components/labs/<Name>Lab.vue` | 文件名即 lab 名，自动注册 |
| 把 lab 挂到某章 | `src/data/labmap/<stage>.js` → `{ 'route-name': ['NameLab'] }` | 该章出现"动手实验台"一节 |
| 新章节 | `src/data/topics/<stage>.js` → `{ stage, chapters, pages }` | 路由 `/stage/name`、侧栏、上一章/下一章全部自动生成 |
| 章末自测 | `src/data/quiz/<stage>.js` → `{ 'route-name': [{ q, options, answer, why }] }` | 章末出现自测；全对后侧栏打 ✓ |
| 术语 | `src/data/glossary/<stage>.js` → `[{ term, aka?, stage, oneliner, number?, route }]` | 出现在"术语速查"页 |
| 真源码 | 章节 page 里写 `source: ['llm_x/path/file.py:函数或类名']` | 构建期直接读 Python 文件，不会和仓库漂移 |

`<stage>` 取 `basic | models | train | finetune | infer | agent`。章节字段说明见 `src/data/topics/index.js` 顶部注释，现有章节写法见 `src/data/models.js` 的 `baseTopicPages`。

**不要手抄 Python 代码到页面里**——用 `source`。`snippet` 只放刻意简化过的骨架/伪代码。

## 工具箱（都已写好，直接用）

```js
import LabFrame from '@/components/lab/LabFrame.vue'     // 外框: title / sub / module / run / challenge; 插槽 controls · default · stats · footer
import LabSlider from '@/components/lab/LabSlider.vue'   // <LabSlider v-model="x" label="…" :min :max :step unit format />
import StepPlayer from '@/components/lab/StepPlayer.vue' // 播放/暂停/单步/拖动时间轴
import { useStepper } from '@/composables/useStepper.js' // const s = useStepper(() => frames.value.length)
import { useDrag } from '@/composables/useDrag.js'       // 指针拖拽, 鼠标+触屏; SVG 内自动换算成 viewBox 坐标
import { mulberry32, randn, softmax, entropy, clamp, lerp, sum, range, argmax, fmtNum, fmtBytes, heat } from '@/utils/labmath.js'
```

共用样式在 `src/styles/main.css` 的 "Lab kit" 一节：`.lab-controls .row`、`.ctl`、`.lab-stats .kv > b(.good|.bad)`、`.lab-note`、`.cells` + `.cell(.on|.hot|.ok|.bad|.dim)`、`.draggable`。lab 自己的 `<style scoped>` 只写这个 lab 特有的东西。

## 写法约定

1. **状态在 `ref`，计算在 `computed`，模板只管画。** 模拟逻辑写成纯函数式的 `computed`，滑杆一动全部重算。不要在模板里写逻辑。
2. **模拟必须和 Python 模块算的是同一件事**，关键数字要对得上（例：PP=4、M=8 时 1F1B 峰值是 `[4,3,2,1]`）。拿不准就去跑对应的 `python -m …demo`。
3. **随机数用 `mulberry32(seed)`**，给一个"换一组"按钮改 seed。拖滑杆时图形不能乱跳。
4. **至少两种交互**：滑杆/按钮之外，优先考虑直接操作——拖动图上的点、点击格子切换状态、悬停高亮关联元素、步进播放。"读者的手放在被解释的那个量上"。
5. **右侧 stats 给 2–4 个会变的数字**，好的变绿 `.good`、坏的变红 `.bad`。数字比形容词有说服力。
6. **每个 lab 配一个 `challenge`**：先让读者预测，再展开看答案。答案要点破这个技术的本质取舍。
7. **颜色只用 CSS 变量**（`--accent --left --eye --right --warn --danger --text-*` …），明暗主题才都能看。不要写死色值。不要给会随滑杆变化的颜色加 `transition`（颜色滞后于数字会出现一瞬间的错误画面）。
8. **可访问性**：可点击的东西用 `<button>`；SVG 里的可交互节点加 `tabindex="0"`、`role`、`@keydown.enter`；滑杆用 `LabSlider`（自带 label 关联）。
9. **窄屏**：宽图放在默认插槽里（`.lab-viz` 自带横向滚动）；不要给容器写死像素宽度。390px 宽时页面不能出现横向滚动条。
10. 一个 lab 控制在 **120–250 行**。超了说明它在讲两件事，拆开。
11. 注释用中文、短、讲"为什么"。关键的那一行用 `★` 标出来。

## 自测

```bash
cd web && npm run dev          # 手动点一遍
npx vite build --outDir /tmp/llm-dist --emptyOutDir   # 必须零报错 (共享卷上直接 build 到 dist/ 偶尔会因清目录失败, 与代码无关)
```
