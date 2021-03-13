import { Matrix } from "./matrix"

/**
 * 回归模型
 */
export class RegressionModel {
  weights: Matrix
  inputs: Matrix
  outputs: Matrix
  scalem: Matrix //缩放比例矩阵
  M: number
  rate = 0.01
  constructor(inputs: Matrix, outputs: Matrix) {
    //把特征进行缩放 -0.5 ～ 0.5之间
    const [inp, scalem] = inputs.normalization()
    this.scalem = scalem

    //左侧加入一个默认特征1
    this.inputs = inp.expansion(1)
    this.outputs = outputs
    this.M = this.inputs.shape[0]
    this.weights = this.initWeights()
  }
  setRate(rate: number) {
    this.rate = rate
  }
  /**
   * 初始化权重 = 特征个数
   * @returns 
   */
  initWeights() {
    const F = this.inputs.shape[1]
    return Matrix.generate(F, 1, 0)
  }

  /**
   * 假设函数  
   * `H(X) = θ0 * 1 + θ1 * X1 + θ2 * X2 + ... + θn * Xn`
   * @param xs 
   * @returns Matrix
   */
  hypothetical(xs: Matrix) {
    return xs.multiply(this.weights)
  }

  /**
   * 代价函数  
   * `J(θ0, θ1, ..., θn) = 1 / 2 * m * ∑m(H(X[i]) - Y[i]) ** 2`
   * @returns number
   */
  cost(): number {
    const M = this.M
    let h = this.hypothetical(this.inputs)
    let sum = 0
    for (let i = 0; i < h.shape[0]; i++) {
      sum += (h.get(i, 0) - this.outputs.get(i, 0)) ** 2
    }
    return (1 / (2 * M)) * sum
  }

  /**
   * 梯度下降  
   * `X[i][0] = 0`  
   * `θ0 = θ0 - rate * 1/m ∑m(H(X[i]) - Y[i]) * X[i][0]`  
   * `θj = θj - rate * 1/m ∑m(H(X[i]) - Y[i]) * X[i][j]`
   */
  gradientDescent() {
    const M = this.M
    const F = this.weights.shape[0]
    const temps = this.initWeights()
    for (let i = 0; i < F; i++) {
      let h = this.hypothetical(this.inputs)
      let sum = 0;
      for (let j = 0; j < h.shape[0]; j++) {
        sum += (h.get(j, 0) - this.outputs.get(j, 0)) * this.inputs.get(j, i)
      }
      temps.update(i, 0,
        this.weights.get(i, 0) - this.rate * (1 / M) * sum
      )
    }
    this.weights = temps
  }

  fit(batch: number, callback?: (batch: number) => void) {
    for (let i = 0; i < batch; i++) {
      this.gradientDescent()
      if (callback) { callback(i) }
    }
  }
  //按照已经缩放的特征预测
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
 * 分类模型
 */
export class LogisticModel {
  weights: Matrix
  inputs: Matrix
  outputs: Matrix
  rate = 0.01
  M: number
  constructor(xs: Matrix, ys: Matrix) {
    this.inputs = xs.expansion(1)
    this.outputs = ys
    this.weights = this.initWeights()
    this.M = this.inputs.shape[0]
  }
  setRate(rate: number) {
    this.rate = rate
  }
  initWeights() {
    const F = this.inputs.shape[1]
    return Matrix.generate(F, 1, 0)
  }

  /**
   * 代价函数
   *  J(θ) = 1 / m * ∑mCost(H(X[i]), Y[i])
   *  y = 0 | 1
   *  Cost(H(X[i]), Y[i]) = -y * log(H(x[i])) - ( 1- y) * log(1 - H(X[i]))
   * @returns number
   */
  cost() {
    const M = this.M
    let h = this.hypothetical(this.inputs)
    let sum = 0
    for (let i = 0; i < h.shape[0]; i++) {
      let y = this.outputs.get(i, 0)
      let hy = h.get(i, 0)
      if (y === 1 && hy !== 0) {
        sum += -Math.log(hy)
      }
      if (y === 0 && hy !== 1) {
        sum += -Math.log(1 - hy)
      }
    }
    return (1 / M) * sum
  }

  /**
   * 梯度下降  
   * `X[i][0] = 0`  
   * `θ0 = θ0 - rate * 1/m ∑m(H(X[i]) - Y[i]) * X[i][0]`  
   * `θj = θj - rate * 1/m ∑m(H(X[i]) - Y[i]) * X[i][j]`
   */
  gradientDescent() {
    const M = this.M
    const F = this.weights.shape[0]
    const temps = this.initWeights()
    for (let i = 0; i < F; i++) {
      let h = this.hypothetical(this.inputs)
      let sum = 0;
      for (let j = 0; j < h.shape[0]; j++) {
        sum += (h.get(j, 0) - this.outputs.get(j, 0)) * this.inputs.get(j, i)
      }
      temps.update(i, 0,
        this.weights.get(i, 0) - this.rate * (1 / M) * sum
      )
    }
    this.weights = temps
  }

  fit(batch: number, callback?: (batch: number) => void) {
    for (let i = 0; i < batch; i++) {
      this.gradientDescent()
      if (callback) { callback(i) }
    }
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

  predict(xs: Matrix) {
    return this.hypothetical(xs.expansion(1))
  }
}