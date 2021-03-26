import { Matrix } from "./matrix"

export class SeqModel {
  weights: Matrix
  scalem?: Matrix //缩放比例
  rate = 0.01
  constructor(
    /**网络形状*/
    public readonly shape: number[],
    /**激活函数类型*/
    public readonly af?: 'Sigmoid'
  ) {
    if (shape.length !== 2) {
      throw new Error('该模型只支持两层结构')
    }
    this.weights = this.initw()
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
   * - row = 输入个数
   * - col = 输出个数
   */
  initw() {
    const f = this.shape[0] + 1
    const y = this.shape[1]
    return Matrix.generate(f, y)
  }

  /**
   * 假设函数
   * - H(X) = θ0 * X0 + θ1 * X1 + θ2 * X2 + ... + θn * Xn
   * @param xs inputs
   */
  hypothetical(xs: Matrix) {
    let a = xs.multiply(this.weights)
    return this.af === 'Sigmoid' ? a.atomicOperation(i => this.sigmoid(i)) : a
  }

  /**
   * 代价函数
   * - J(θ0, θ1, ..., θn) = 1 / 2 * m * ∑m(H(X[i]) - Y[i]) ** 2 
   * @returns 多输出拥有多个损失值
   */
  cost(hys: Matrix, ys: Matrix) {
    let m = hys.shape[0]
    let sub = hys.subtraction(ys).atomicOperation(item => item ** 2).columnSum()
    return sub.getRow(0).map(v => (1 / (2 * m)) * v)
  }

  /**
   * Sigmoid代价函数
   * - J(θ) = 1 / m * ∑m Cost
   * - y = 1 ? Cost = - Math.log(H(X[i]))
   * - y = 0 ? Cost = -Math.log(1 - H(X[i]))
   * @returns 多输出拥有多个损失值
   */
  sigmoidCost(hys: Matrix, ys: Matrix) {
    let m = hys.shape[0]
    let t = hys.atomicOperation((hy, i, j) => {
      let y = ys.get(i, j)
      return y === 1 ? -Math.log(hy) : -Math.log(1 - hy)
    }).columnSum()
    return t.getRow(0).map(v => (1 / m) * v)
  }

  /**
   * 梯度下降
   * - X[i][0] = 1
   * - θj = θj - rate * 1 / m * ∑m(H(X[i]) - Y[i]) * X[i][j]
   */
  fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void) {
    if (xs.shape[0] !== ys.shape[0]) {
      throw new Error('输入输出矩阵行数不统一')
    }
    if (xs.shape[1] !== this.shape[0]) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`)
    }
    if (ys.shape[1] !== this.shape[1]) {
      throw new Error(`标签与网络输出不符合，output num -> ${this.shape[1]}`)
    }
    if (this.af === 'Sigmoid') {
      this.verifOutput(ys)
    }

    //归一化 -0.5 ～ 0.5
    let [inputs, scalem] = xs.normalization()
    this.scalem = scalem

    //左侧扩增一列 X0 = 1
    xs = inputs.expand(1, 'L')
    let m = xs.shape[0]

    for (let i = 0; i < batch; i++) {
      let hys = this.hypothetical(xs)
      const temps = this.initw()
      let hsub = hys.subtraction(ys)
      for (let i = 0; i < temps.shape[0]; i++) {
        for (let j = 0; j < temps.shape[1]; j++) {
          let sum = 0
          for (let k = 0; k < hsub.shape[0]; k++) {
            sum += hsub.get(k, j) * xs.get(k, i)
          }
          let nw = this.weights.get(i, j) - this.rate * (1 / m) * sum
          temps.update(i, j, nw)
        }
      }
      this.weights = temps
      let loss = this.af === 'Sigmoid' ? this.sigmoidCost(hys, ys)[0] : this.cost(hys, ys)[0]
      if (callback) callback(i, loss)
    }
  }

  /**
   * 激活函数
   * @param x 
   */
  sigmoid(x: number) {
    return 1 / (1 + Math.E ** -x)
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
   * 特征按均值空间缩放
   * @param xs 
   */
  zoomScalem(xs: Matrix) {
    return xs.atomicOperation((item, _, j) => {
      if (!this.scalem) return item
      return this.scalem.get(1, j) === 0 ? 0 : (item - this.scalem.get(0, j)) / this.scalem.get(1, j)
    })
  }

  predict(xs: Matrix) {
    if (xs.shape[1] !== this.shape[0]) {
      throw new Error('与预期特征数不符合')
    }
    let inputs = this.zoomScalem(xs).expand(1, 'L')
    return this.hypothetical(inputs)
  }
}