import { Matrix } from "./matrix"

export class Model {
  weights: Matrix
  inputs: Matrix
  outputs: Matrix
  M: number
  rate = 0.01
  constructor(inputs: Matrix, outputs: Matrix) {
    //左侧加入一个默认特征1
    this.inputs = inputs.expansion(1)
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
  
  predict(xs: Matrix) {
    return this.hypothetical(xs.expansion(1))
  }
}