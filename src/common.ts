import { ActivationFunction } from "./types";
import { Matrix } from "./matrix";

/**
 * 按小数位向下取整
 * @param num 源
 * @param fix 位
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

/**
 * 激活函数求值
 * @param x 目标
 * @param rows 该层的多个输出值
 * @param af 激活函数类型
 */
export function afn(x: number, rows: number[], af?: ActivationFunction) {
  switch (af) {
    case 'Sigmoid':
      return 1 / (1 + Math.exp(-x))
    case 'Relu':
      return x >= 0 ? x : 0
    case 'Tanh':
      return Math.tanh(x)
    case 'Softmax':
      let d = Math.max(...rows) //防止指数过大
      return Math.exp(x - d) / rows.reduce((p, c) => p + Math.exp(c - d), 0)
    default:
      return x
  }
}

/**
 * 激活函数求导
 */
export function afd(x: number, af?: ActivationFunction) {
  switch (af) {
    case 'Sigmoid':
      return x * (1 - x)
    case 'Relu':
      return x >= 0 ? 1 : 0
    case 'Tanh':
      return 1 - Math.tanh(x) ** 2
    case 'Softmax':
    // only last-layer, y must is 0/1
    // d = (y=1) ? hy - 1; (y=0) ? hy - y; d = hy - y
    default:
      return 1
  }
}

export enum Channel { r, g, b, a }

/**
 * ImageData转Matrix
 */
export function imageDataToMatrix(d: ImageData, ch: keyof typeof Channel) {
  let channel = Channel[ch]
  let n: number[][] = []
  for (let i = 0; i < d.height; i++) {
    let m: number[] = []
    for (let j = 0; j < d.width; j++) {
      let k = (i * d.width + j) * 4
      m.push(d.data[k] + channel)
    }
    n.push(m)
  }
  return new Matrix(n)
}