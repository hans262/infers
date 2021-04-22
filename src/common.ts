import { Matrix } from "./matrix";
import { BPNet } from "./BPNet";

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
 * 加载json化的模型
 * @param modelJson 模型json字符串
 * @returns BPNet
 */
export function loadBPNet(modelJson: string) {
  let tmp = JSON.parse(modelJson)
  let w: Matrix[] = tmp.w.map((w: any) => new Matrix(w))
  let b: Matrix[] = tmp.b.map((b: any) => new Matrix(b))
  let scale = tmp.scale ? new Matrix(tmp.scale) : undefined
  return new BPNet(tmp.shape, {
    mode: tmp.mode,
    rate: tmp.mode,
    w, b, scale
  })
}