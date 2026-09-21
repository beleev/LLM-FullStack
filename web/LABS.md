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

`<stage>` 取 `basic | models | train | finetune | infer | agent`。章节字段说明见 `src/data/topics/index.js` 顶部注释，现有章节写法见 `src/data/topics/<stage>.js`——所有正文都住在那里，`models.js` 只留结构性数据（阶段、目录、模块表、时间轴）。

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
4. **至少两种交互**：滑杆和按钮之外，再给一种直接操作——拖动图上的点、点击格子切换状态、悬停高亮关联元素、步进播放。标准是"读者的手放在被解释的那个量上"。
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

## 画图的共用件

除了 `components/lab/` 那套 lab 外框，还有三个共享件给章节内的示意图用：

| 组件 | 干什么 |
|---|---|
| `components/FlowDiagram.vue` | 计算流。每一步自带一条与张量元素数成比例的 `SizeBar`，所以 `[B,H,T,T]` 随 T 长出来的样子是看得见的。传 `active-param` 可以让用到某个权重的步骤高亮 |
| `components/SizeBar.vue` | 一条正比于张量大小的横条，batch 维不计，只比形状本身 |
| `components/ParamsTable.vue` | 权重表。`@hover` 抛出权重名，配合 `FlowDiagram` 的 `active-param` 做两栏联动 |

画结构图时先问一句：**这张图有没有把"大小"画出来？** 两个形状写出来一样长，但元素数差一百倍——这种差别只有画成面积或长度才会被看见。

## 划重点：主干 / 扩展 / 冲刺

`src/data/tiers.js` 把每一章分成三层，**只影响阅读建议，不影响内容和可达性**：

| 层 | 标记 | 含义 |
|---|---|---|
| 冲刺 | ★ | 12 章。只有一天就读这些，每个阶段不读就接不上下一阶段的那几页 |
| 主干 | ● | 37 章（含冲刺）。完整课程主线 |
| 扩展 | ○ | 36 章。深水区与分支；跳过不影响主线 |

分层出现在三个地方：侧栏的档位筛选、面包屑的标记、`/fast-track` 速成路线页。新增章节默认归为扩展——主线是要守住的，加内容不该让它变长。要改归类就改 `tiers.js` 里的两个数组，其余全是算出来的。

**章内划重点**：每个章节的 `points` 里，给最该带走的那一条加 `key: true`，页面会渲染一个「重点」徽章。一章只标一条。

```js
points: [
  { title: 'cache 是手写 autograd tape', body: '…', key: true },
  { title: '形状先行', body: '…' },
]
```

## 写文案的规矩（说人话）

这是教程，不是文档。读者读一遍就要懂。

1. **先说具体的，再说抽象的。** 先讲发生了什么，再给它起名字。
2. **拆掉名词堆。** 「每个 forward 返回反向需要的输入、权重、归一化统计」→ 谁把什么交给谁，写成有动词的句子。
3. **一个数字胜过一个形容词。** 「显著更省」→「省 56.9×」。数字必须来自真实跑出来的 demo。
4. **先说为什么疼，再说怎么治。** 每个技术都是因为上一代出了问题才存在的。
5. **短句。** 不要「值得注意的是」「我们可以看到」「综上所述」。
6. `tldr` 是一句能背下来的话；`question` 是读者真会问的问题；`subtitle` 是读完这页你能做什么。
7. **诚实的结论比好看的结论重要。** 实测不符合流行说法就照实写，并说明是在什么规模下测的。
