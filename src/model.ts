import { Matrix } from "./matrix"

class Model {
  weights: Matrix
  inputs: Matrix
  outputs: Matrix
  /** 缩放比例*/
  scalem: Matrix
  M: number
  rate = 0.01
  constructor(xs: Matrix, ys: Matrix) {
    if (xs.shape[1] !== ys.shape[1]) {
      throw new Error('输入输出矩阵行数不统一')
    }

    //特征归一化 -0.5 ～ 0.5
    const [inp, scalem] = xs.normalization()
    this.scalem = scalem

    //特征新增一列 X0 = 1
    this.inputs = inp.expansion(1)
    this.outputs = ys
    this.M = this.inputs.shape[0]
    this.weights = this.initWeights()
  }

  /**
   * 更新学习率
   * @param rate 
   */
  setRate(rate: number) {
    this.rate = rate
  }

  /**
   * 初始化权重
   * - row = 特征个数
   * - col = 输出个数
   * @returns 特征矩阵
   */
  initWeights() {
    const F = this.inputs.shape[1]
    const y = this.outputs.shape[1]
    return Matrix.generate(F, y, 0)
  }

  /**
   * 假设函数
   * - H(X) = θ0 * X0 + θ1 * X1 + θ2 * X2 + ... + θn * Xn
   * @param xs 输入矩阵
   * @returns 输出矩阵
   */
  hypothetical(xs: Matrix) {
    return xs.multiply(this.weights)
  }

  /**
   * 代价函数
   * - J(θ0, θ1, ..., θn) = 1 / 2 * m * ∑m(H(X[i]) - Y[i]) ** 2 
   * @returns number[]
   */
  cost() {
    const M = this.M
    let h = this.hypothetical(this.inputs)
    let n = []
    for (let i = 0; i < h.shape[1]; i++) {
      let sum = 0
      for (let j = 0; j < h.shape[0]; j++) {
        sum += (h.get(j, i) - this.outputs.get(j, i)) ** 2
      }
      n.push((1 / (2 * M)) * sum)
    }
    return n
  }

  /**
   * 梯度下降
   * - X[i][0] = 1
   * - θj = θj - rate * 1 / m ∑m(H(X[i]) - Y[i]) * X[i][j]
   */
  gradientDescent() {
    const M = this.M
    let h = this.hypothetical(this.inputs)
    const temps = this.initWeights()
    for (let i = 0; i < temps.shape[0]; i++) {
      for (let j = 0; j < temps.shape[1]; j++) {
        let sum = 0;
        for (let k = 0; k < h.shape[0]; k++) {
          sum += (h.get(k, j) - this.outputs.get(k, j)) * this.inputs.get(k, i)
        }
        let nw = this.weights.get(i, j) - this.rate * (1 / M) * sum
        temps.update(i, j, nw)
      }
    }
    this.weights = temps
  }

  fit(batch: number, callback?: (batch: number) => void) {
    for (let i = 0; i < batch; i++) {
      this.gradientDescent()
      if (callback) { callback(i) }
    }
  }

  /**
   * 对新的特征进行归一化
   * @param xs 
   * @returns 
   */
  reductionScale(xs: Matrix) {
    let n = []
    for (let i = 0; i < xs.shape[0]; i++) {
      let m = []
      for (let j = 0; j < xs.shape[1]; j++) {
        m.push(
          (xs.get(i, j) - this.scalem.get(0, j)) / this.scalem.get(1, j)
        )
      }
      n.push(m)
    }
    return new Matrix(n)
  }

  predict(xs: Matrix) {
    let a = this.reductionScale(xs)
    return this.hypothetical(a.expansion(1))
  }
}

/**
 * 回归模型
 */
export class RegressionModel extends Model { }

/**
 * 分类模型
 */
export class LogisticModel extends Model {
  constructor(xs: Matrix, ys: Matrix) {
    super(xs, ys)
    this.verifYs(ys)
  }

  /**
   * 验证输出矩阵
   * @param ys 
   */
  verifYs(ys: Matrix) {
    for (let i = 0; i < ys.shape[0]; i++) {
      if (ys.getLine(i).reduce((p, c) => p + c) !== 1)
        throw new Error('输出矩阵每行和必须等0')
      for (let j = 0; j < ys.shape[1]; j++) {
        if (ys.get(i, j) !== 0 && ys.get(i, j) !== 1)
          throw new Error('输出矩阵属于域 ∈ (0, 1)')
      }
    }
  }

  /**
   * 代价函数
   * - J(θ) = 1 / m * ∑m Cost
   * - y = 1 ? Cost = - Math.log(H(X[i]))
   * - y = 0 ? Cost = -Math.log(1 - H(X[i]))
   * @returns number[]
   */
  cost() {
    const M = this.M
    let h = this.hypothetical(this.inputs)
    let n = []
    for (let j = 0; j < h.shape[1]; j++) {
      let sum = 0
      for (let i = 0; i < h.shape[0]; i++) {
        let y = this.outputs.get(i, 0)
        let hy = h.get(i, 0)
        if (y === 1) {
          sum += -Math.log(hy)
        }
        if (y === 0) {
          sum += -Math.log(1 - hy)
        }
      }
      n.push((1 / M) * sum)
    }
    return n
  }

  sigmoid(x: number) {
    return 1 / (1 + Math.E ** -x)
  }

  hypothetical(xs: Matrix) {
    let a = xs.multiply(this.weights)
    let n = []
    for (let i = 0; i < a.shape[0]; i++) {
      let m = []
      for (let j = 0; j < a.shape[1]; j++) {
        m.push(this.sigmoid(a.get(i, j)))
      }
      n.push(m)
    }
    return new Matrix(n)
  }
}