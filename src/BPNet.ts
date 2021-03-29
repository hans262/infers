import { Matrix } from "./matrix"

/**Activation function type*/
export type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh'
/**Network shape*/
export type NetShape = (number | [number, ActivationFunction])[]
/**Model configuration*/
export interface NetConfig {
  optimizer: 'SGD' | 'BGD'
}

export class BPNet {
  /**Weight matrix*/
  w: Matrix[]
  /**Partial value matrix*/
  b: Matrix[]
  nlayer: number
  /**Learning rate*/
  rate = 0.001
  /**Scaling*/
  scalem?: Matrix
  constructor(
    public readonly shape: NetShape,
    public netconf?: NetConfig
  ) {
    if (shape.length < 2) {
      throw new Error('The network has at least two layers')
    }
    this.nlayer = shape.length
    const [w, b] = this.initwb()
    this.w = w
    this.b = b
  }

  /**
   * Get the number of neurons in the current layer
   * @param l 
   */
  nOfLayer(l: number) {
    let n = this.shape[l]
    return Array.isArray(n) ? n[0] : n
  }

  /**
   * Gets the active function type of the current layer
   * @param l 
   */
  afOfLayer(l: number) {
    let n = this.shape[l]
    return Array.isArray(n) ? n[1] : undefined
  }

  /**
   * Initialization weight and bias  
   * default value -0.5 ~ 0.5
   * @returns [w, b]
   */
  initwb(v?: number) {
    let w: Matrix[] = []
    let b: Matrix[] = []
    for (let l = 1; l < this.shape.length; l++) {
      w[l] = Matrix.generate(this.nOfLayer(l), this.nOfLayer(l - 1), v)
      b[l] = Matrix.generate(1, this.nOfLayer(l), v)
    }
    return [w, b]
  }
  
  /**
   * Update learning rate
   * @param rate 
   */
  setRate(rate: number) {
    this.rate = rate
  }

  /**
   * Get activation function for current layer
   */
  afn(x: number, l: number) {
    let af = this.afOfLayer(l)
    switch (af) {
      case 'Sigmoid':
        return 1 / (1 + Math.exp(-x))
      case 'Relu':
        return x >= 0 ? x : 0
      case 'Tanh':
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))
      default:
        return x
    }
  }

  /**
   * Gets the active function derivative of the current layer
   */
  afd(x: number, l: number) {
    let af = this.afOfLayer(l)
    switch (af) {
      case 'Sigmoid':
        return x * (1 - x)
      case 'Relu':
        return x >= 0 ? 1 : 0
      case 'Tanh':
        return 1 - ((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))) ** 2
      default:
        return 1
    }
  }
  
  /**
   * Calculate the output of the whole network
   * - hy =  θ1 * X1 + θ2 * X2 + ... + θn * Xn + b
   * @param xs inputs
   */
  calcnet(xs: Matrix) {
    let hy: Matrix[] = []
    for (let l = 0; l < this.nlayer; l++) {
      if (l === 0) {
        hy[l] = xs
        continue;
      }
      hy[l] = hy[l - 1].multiply(this.w[l].T).atomicOperation((item, _, j) =>
        this.afn(item + this.b[l].get(0, j), l)
      )
    }
    return hy
  }

  /**
   * Scaling features in mean space
   * @param xs 
   */
  zoomScalem(xs: Matrix) {
    return xs.atomicOperation((item, _, j) => {
      if (!this.scalem) return item
      return this.scalem.get(1, j) === 0 ? 0 : (item - this.scalem.get(0, j)) / this.scalem.get(1, j)
    })
  }

  predict(xs: Matrix) {
    if (xs.shape[1] !== this.nOfLayer(0)) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.nOfLayer(0)}`)
    }
    return this.calcnet(this.zoomScalem(xs))
  }

  /**
   * The derivative of the valence function with respect to each neuron 
   * and the derivative of each weight are calculated.
   * It's for a single sample.
   * - J = 1 / 2 * (hy - y)^2
   * - ∂J / ∂hy = (1 / 2) * 2 * (hy - y) = hy - y  Derivation of the last node
   * - ∂J / ∂w = Value of input node * derivative of output node
   * 
   * Derivation follows the chain rule: branch nodes add, link nodes multiply.
   * If there is an activation function, it needs to be multiplied by the derivative of the activation function.
   * @returns [Neuron derivative matrix, Weighted derivative matrix]
   */
  calcDerivative(hy: Matrix[], ys: Matrix, n: number) {
    const [dw, dy] = this.initwb(0)
    for (let l = this.nlayer - 1; l > 0; l--) {
      if (l === this.nlayer - 1) {
        for (let j = 0; j < this.nOfLayer(l); j++) {
          dy[l].update(0, j, (hy[l].get(n, j) - ys.get(n, j)) * this.afd(hy[l].get(n, j), l))
          for (let k = 0; k < this.nOfLayer(l - 1); k++) {
            dw[l].update(j, k, hy[l - 1].get(n, k) * dy[l].get(0, j))
          }
        }
        continue;
      }
      for (let j = 0; j < this.nOfLayer(l); j++) {
        for (let i = 0; i < this.nOfLayer(l + 1); i++) {
          dy[l].update(0, j, dy[l + 1].get(0, i) * this.w[l + 1].get(i, j), '+=')
        }
        dy[l].update(0, j, this.afd(hy[l].get(n, j), l), '*=')
        for (let k = 0; k < this.nOfLayer(l - 1); k++) {
          dw[l].update(j, k, hy[l - 1].get(n, k) * dy[l].get(0, j))
        }
      }
    }
    return { dy, dw }
  }

  /**
   * update weight and bias matrix
   * - w = w - α * (∂J / ∂w)
   * - b = b - α * (∂J / ∂hy)
   */
  update(dy: Matrix[], dw: Matrix[]) {
    for (let l = 1; l < this.nlayer; l++) {
      this.w[l] = this.w[l].subtraction(dw[l].numberMultiply(this.rate))
      this.b[l] = this.b[l].subtraction(dy[l].numberMultiply(this.rate))
    }
  }

  /**
   * Quadratic cost function
   * Multiple outputs average multiple loss values  
   * - J = 1 / 2 * m * ∑m(hy - ys) ** 2 
   */
  cost(hy: Matrix[], ys: Matrix) {
    let m = ys.shape[0]
    let sub = hy[this.nlayer - 1].subtraction(ys).atomicOperation(item => item ** 2).columnSum()
    let tmp = sub.getRow(0).map(v => (1 / (2 * m)) * v)
    return tmp.reduce((p, c) => p + c) / tmp.length
  }

  /**
   * Cross entropy cost function
   * To simulate the last layer is the sigmoid activation function
   * Multiple outputs average multiple loss values  
   * 输出值域必须是 {0, 1}
   * - J = 1 / m * ∑m Cost
   * - y = 1 ? Cost = - Math.log(H(X[i]))
   * - y = 0 ? Cost = -Math.log(1 - H(X[i]))
   */
   crossCost(hy: Matrix[], ys: Matrix) {
    let m = ys.shape[0]
    let t = hy[this.nlayer - 1].atomicOperation((h, i, j) => {
      let y = ys.get(i, j)
      return y === 1 ? -Math.log(h) : -Math.log(1 - h)
    }).columnSum()
    let tmp = t.getRow(0).map(v => (1 / m) * v)
    return tmp.reduce((p, c) => p + c) / tmp.length
  }

  /**
   * 批量梯度下降，
   * 求所有样本导数的平均值，
   * 整个样本迭代一次在更新权值偏值  
   * 实用于小批量的数据，数据量过大时较慢
   */
  bgd(hy: Matrix[], ys: Matrix) {
    let m = ys.shape[0]
    const [ndw, ndy] = this.initwb(0)
    for (let n = 0; n < m; n++) {
      const { dy, dw } = this.calcDerivative(hy, ys, n)
      for (let l = 1; l < this.nlayer; l++) {
        ndw[l] = ndw[l].addition(dw[l])
        ndy[l] = ndy[l].addition(dy[l])
      }
    }
    for (let l = 1; l < this.nlayer; l++) {
      ndw[l] = ndw[l].atomicOperation(item => item / m)
      ndy[l] = ndy[l].atomicOperation(item => item / m)
    }
    this.update(ndy, ndw)
  }

  /**
   * 随机梯度下降，
   * 单个样本的导数更新权值
   */
  sgd(hy: Matrix[], ys: Matrix) {
    let m = ys.shape[0]
    for (let n = 0; n < m; n++) {
      const { dy, dw } = this.calcDerivative(hy, ys, n)
      this.update(dy, dw)
    }
  }

  fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void) {
    if (xs.shape[0] !== ys.shape[0]) {
      throw new Error('输入输出矩阵行数不统一')
    }
    if (xs.shape[1] !== this.nOfLayer(0)) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.nOfLayer(0)}`)
    }
    if (ys.shape[1] !== this.nOfLayer(this.nlayer - 1)) {
      throw new Error(`标签与网络输出不符合，output num -> ${this.nOfLayer(this.nlayer - 1)}`)
    }
    //normalization -0.5 ～ 0.5
    const [nxs, scalem] = xs.normalization()
    this.scalem = scalem
    xs = nxs
    for (let p = 0; p < batch; p++) {
      let hy = this.calcnet(xs)
      if (this.netconf && this.netconf.optimizer === 'BGD') {
        this.bgd(hy, ys)
      } else {
        this.sgd(hy, ys)
      }
      let loss = this.cost(hy, ys)
      if (callback) callback(p, loss)
    }
  }
}