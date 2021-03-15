/**
 * 小数点取位 向下取整
 * @param num 源值
 * @param fix 位数
 */
export function toFixed(num: number, fix: number): number {
  const amount = 10 ** fix
  return ~~(num * amount) / amount
}