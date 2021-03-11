import type { Point } from "./types"

/**
 * 对比坐标
 * @param cond1 
 * @param cond2 
 */
export function contrastPoint(cond1: Point, cond2: Point) {
  return cond1[0] === cond2[0] && cond1[1] === cond2[1]
}


/**
 * 小数点取位 向下取整
 * @param num 源值
 * @param fix 位数
 */
export function toFixed(num: number, fix: number): number {
  const amount = 10 ** fix
  return ~~(num * amount) / amount
}