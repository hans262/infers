import { Matrix } from "./matrix"

class Model {
  weights: Matrix
  inputs: Matrix
  outputs: Matrix
  /** 缩放比例*/
  scalem: Matrix
  /** 样本数 */
  m: number
  rate = 0.01
  constructor(xs: Matrix, ys: Matrix) {
    if (xs.shape[0] !== ys.shape[0]) {
      throw new Error('输入输出矩阵行数不统一')
    }
    //归一化 -0.5 ～ 0.5
    const [inputs, scalem] = xs.normalization()
    this.scalem = scalem

    //左侧扩增一列 X0 = 1
    this.inputs = inputs.expand(1, 'L')
    this.outputs = ys
    this.m = this.inputs.shape[0]
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
   */
  initWeights() {
    const f = this.inputs.shape[1]
    const y = this.outputs.shape[1]
    return Matrix.generate(f, y)
  }

  /**
   * 假设函数
   * - H(X) = θ0 * X0 + θ1 * X1 + θ2 * X2 + ... + θn * Xn
   * @param xs inputs
   */
  hypothetical(xs: Matrix) {
    return xs.multiply(this.weights)
  }

  /**
   * 代价函数
   * - J(θ0, θ1, ..., θn) = 1 / 2 * m * ∑m(H(X[i]) - Y[i]) ** 2 
   * @returns 多输出拥有多个损失值
   */
  cost() {
    let h = this.hypothetical(this.inputs)
    let sub = h.subtraction(this.outputs).atomicOperation(item => item ** 2).columnSum()
    return sub.getRow(0).map(v => (1 / (2 * this.m)) * v)
  }

  /**
   * 梯度下降
   * - X[i][0] = 1
   * - θj = θj - rate * 1 / m * ∑m(H(X[i]) - Y[i]) * X[i][j]
   */
  gradientDescent() {
    let h = this.hypothetical(this.inputs)
    const temps = this.initWeights()
    let hsub = h.subtraction(this.outputs)
    for (let i = 0; i < temps.shape[0]; i++) {
      for (let j = 0; j < temps.shape[1]; j++) {
        let sum = 0
        for (let k = 0; k < hsub.shape[0]; k++) {
          sum += hsub.get(k, j) * this.inputs.get(k, i)
        }
        let nw = this.weights.get(i, j) - this.rate * (1 / this.m) * sum
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
   * 特征按均值空间缩放
   * @param xs 
   */
  zoomScale(xs: Matrix) {
    return xs.atomicOperation((item, _, j) => {
      return this.scalem.get(1, j) === 0 ? 0 : (item - this.scalem.get(0, j)) / this.scalem.get(1, j)
    })
  }

  predict(xs: Matrix) {
    if (xs.shape[1] !== this.inputs.shape[1] - 1) {
      throw new Error('与预期特征数不符合')
    }
    let inputs = this.zoomScale(xs).expand(1, 'L')
    return this.hypothetical(inputs)
  }
}

export class RegressionModel extends Model { }

export class LogisticModel extends Model {
  constructor(xs: Matrix, ys: Matrix) {
    super(xs, ys)
    this.verifOutput(ys)
  }

  /**
   * 验证输出矩阵
   * @param ys 
   */
  verifOutput(ys: Matrix) {
    for (let i = 0; i < ys.shape[0]; i++) {
      if (ys.shape[1] > 1 && ys.getRow(i).reduce((p, c) => p + c) !== 1)
        throw new Error('输出矩阵每行求和必须等0')
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
   * @returns 多输出拥有多个损失值
   */
  cost() {
    let h = this.hypothetical(this.inputs)
    let t = h.atomicOperation((hy, i, j) => {
      let y = this.outputs.get(i, j)
      return y === 1 ? -Math.log(hy) : -Math.log(1 - hy)
    }).columnSum()
    return t.getRow(0).map(v => (1 / this.m) * v)
  }

  /**
   * 激活函数
   * @param x 
   * @returns 
   */
  sigmoid(x: number) {
    return 1 / (1 + Math.E ** -x)
  }

  hypothetical(xs: Matrix) {
    let a = xs.multiply(this.weights)
    return a.atomicOperation(item => this.sigmoid(item))
  }
}