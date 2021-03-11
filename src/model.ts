import { Matrix } from "./matrix"

export class Model {
  weights: number[]
  inputs: Matrix
  outputs: Matrix
  normal: Matrix
  M: number
  rate = 0.01
  constructor(inputs: Matrix, outputs: Matrix) {
    this.inputs = inputs
    this.outputs = outputs
    this.M = inputs.shape[0]
    this.weights = this.initWeights()
    this.normal = this.inputs.normalization()
  }
  setRate(rate: number) {
    this.rate = rate
  }
  /**
   * 初始化权重 = 特征个数 + 1
   * @returns 
   */
  initWeights() {
    const f = this.inputs.shape[1]
    return new Array(f + 1).fill(0)
  }

  /**
   * 假设函数  
   * `H(X) = θ0 * 1 + θ1 * X1 + θ2 * X2 + ... + θn * Xn`
   * @param xs 
   * @returns Matrix
   */
  hypothetical(xs: Matrix) {
    let m = xs.self.map(x => {
      x = [1, ...x]
      return [this.weights.reduce((p, c, i) => p + c * x[i], 0)]
    })
    return new Matrix(m)
  }

  /**
   * 代价函数  
   * `J(θ0, θ1, ..., θn) = 1 / 2 * m * ∑m(H(X[i]) - Y[i]) ** 2`
   * @returns number
   */
  cost(): number {
    const M = this.M
    let h = this.hypothetical(this.inputs)
    const sum = h.self.reduce((p, c, i) => {
      return p + (c[0] - this.outputs.get(i)[0]) ** 2
    }, 0)
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
    const F = this.weights.length
    const temps = this.initWeights()
    for (let i = 0; i < F; i++) {
      temps[i] = 0
      let h = this.hypothetical(this.inputs)
      let sum = h.self.reduce((p, c, j) => {
        let xs = this.inputs.get(j)
        xs = [1, ...xs]
        return p + (c[0] - this.outputs.get(j)[0]) * xs[i]
      }, 0)
      temps[i] = this.weights[i] - this.rate * (1 / M) * sum
    }
    for (let k = 0; k < F; k++) {
      this.weights[k] = temps[k]
    }
  }

  fit(batch: number, callback?: (batch: number) => void) {
    for (let i = 0; i < batch; i++) {
      this.gradientDescent()
      if (callback) { callback(i) }
    }
  }
  predict(xs: Matrix) {
    return this.hypothetical(xs)
  }
}