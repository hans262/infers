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

export function canvasToMatrix(d: ImageData) {
  let m = Matrix.generate(d.width, d.height, 0)

  for (let i = 0; i < d.height; i++) {
    for (let j = 0; j < d.width * 4; j++) {
      let k = i + j * 4
      let red = d.data[k]
      let green = d.data[k + 1]
      let blue = d.data[k + 2]
      m.update(i, j, red)
    }
  }
  return m
}