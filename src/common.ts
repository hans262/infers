import { Matrix } from "./matrix";

/**
 * 小数点取位 向下取整
 * @param num 源值
 * @param fix 位数
 */
export function toFixed(num: number, fix: number): number {
  const amount = 10 ** fix
  return ~~(num * amount) / amount
}

/**
 * 打乱特征和标签矩阵，
 * 两个矩阵行数必须统一
 */
export function upset(xs: Matrix, ys: Matrix) {
  let xss = xs.dataSync()
  let yss = ys.dataSync()
  for (let i = 1; i < ys.shape[0]; i++) {
    let random = Math.floor(Math.random() * (i + 1));
    [xss[i], xss[random]] = [xss[random], xss[i]];
    [yss[i], yss[random]] = [yss[random], yss[i]];
  }
  return { xs: new Matrix(xss), ys: new Matrix(yss) }
}