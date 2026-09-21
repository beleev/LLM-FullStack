// 术语速查: 每个阶段一个文件放在 data/glossary/ 下, 形如
//   export default [{ term, aka?, stage, oneliner, number?, route }]
// stage 取 basic | models | train | finetune | infer | agent; route 是该词条对应章节的路由名。
const files = import.meta.glob('./glossary/*.js', { eager: true, import: 'default' })
export const glossary = Object.values(files).flatMap((m) => m || [])
