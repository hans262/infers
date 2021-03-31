import { Matrix } from "./matrix"

/**Activation function type*/
export type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh'

/**Network shape*/
export type NetShape = (number | [number, ActivationFunction])[]

/**Model configuration*/
export interface NetConfig {
  optimizer: 'sgd' | 'bgd' | 'mbgd'
}

/**Fit configuration*/
export interface FitConf {
  epochs: number
  batchSize?: number
  onBatch?: (batch: number, size: number, loss: number) => void
  onEpoch?: (epoch: number, loss: number) => void
}

export class BPNet {
  /**Weight matrix*/
  w: Matrix[]
  /**Partial value matrix*/
  b: Matrix[]
  nlayer: number
  /**Learning rate*/
  rate = 0.001
  /**Scaling matrix*/
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
    return this.calcnet(this.zoomScalem(xs))[this.nlayer - 1]
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
  calcDerivative(hy: Matrix[], ys: Matrix) {
    const [dw, dy] = this.initwb(0)
    for (let l = this.nlayer - 1; l > 0; l--) {
      if (l === this.nlayer - 1) {
        for (let j = 0; j < this.nOfLayer(l); j++) {
          dy[l].update(0, j, (hy[l].get(0, j) - ys.get(0, j)) * this.afd(hy[l].get(0, j), l))
          for (let k = 0; k < this.nOfLayer(l - 1); k++) {
            dw[l].update(j, k, hy[l - 1].get(0, k) * dy[l].get(0, j))
          }
        }
        continue;
      }
      for (let j = 0; j < this.nOfLayer(l); j++) {
        for (let i = 0; i < this.nOfLayer(l + 1); i++) {
          dy[l].update(0, j, dy[l + 1].get(0, i) * this.w[l + 1].get(i, j), '+=')
        }
        dy[l].update(0, j, this.afd(hy[l].get(0, j), l), '*=')
        for (let k = 0; k < this.nOfLayer(l - 1); k++) {
          dw[l].update(j, k, hy[l - 1].get(0, k) * dy[l].get(0, j))
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
   * 梯度下降，
   * 求全部样本导数平均值
   */
  bgd(xs: Matrix, ys: Matrix, conf: FitConf) {
    let m = ys.shape[0]
    for (let ep = 0; ep < conf.epochs; ep++) {
      let eploss = 0
      const [dws, dys] = this.initwb(0)
      for (let n = 0; n < m; n++) {
        let xss = new Matrix([xs.getRow(n)])
        let yss = new Matrix([ys.getRow(n)])
        let hy = this.calcnet(xss)
        const { dy, dw } = this.calcDerivative(hy, yss)
        for (let l = 1; l < this.nlayer; l++) {
          dws[l] = dws[l].addition(dw[l])
          dys[l] = dys[l].addition(dy[l])
        }
        let loss = hy[this.nlayer - 1].subtraction(yss)
          .atomicOperation(item => (item ** 2) / 2)
          .getMeanOfRow(0)
        eploss += loss
      }
      for (let l = 1; l < this.nlayer; l++) {
        dws[l] = dws[l].atomicOperation(item => item / m)
        dys[l] = dys[l].atomicOperation(item => item / m)
      }
      this.update(dys, dws)
      if (conf.onEpoch) conf.onEpoch(ep, eploss / m)
    }
  }

  /**
   * 随机梯度下降，
   * 求单个样本导数
   */
  sgd(xs: Matrix, ys: Matrix, conf: FitConf) {
    let m = ys.shape[0]
    for (let ep = 0; ep < conf.epochs; ep++) {
      let eploss = 0
      for (let n = 0; n < m; n++) {
        let xss = new Matrix([xs.getRow(n)])
        let yss = new Matrix([ys.getRow(n)])
        let hy = this.calcnet(xss)
        const { dy, dw } = this.calcDerivative(hy, yss)
        this.update(dy, dw)
        let loss = hy[this.nlayer - 1].subtraction(yss)
          .atomicOperation(item => (item ** 2) / 2)
          .getMeanOfRow(0)
        eploss += loss
      }
      if (conf.onEpoch) conf.onEpoch(ep, eploss / m)
    }
  }

  /**
   * 批次梯度下降，
   * 求多个样本导数平均值
   */
  mbgd(xs: Matrix, ys: Matrix, conf: FitConf) {
    let batchSize = conf.batchSize ? conf.batchSize : 10
    let nbatch = 0
    let batch = 0
    let m = ys.shape[0]
    let [dws, dys] = this.initwb(0)
    for (let ep = 0; ep < conf.epochs; ep++) {
      let eploss = 0
      for (let n = 0; n < m; n++) {
        batch += 1
        let xss = new Matrix([xs.getRow(n)])
        let yss = new Matrix([ys.getRow(n)])
        let hy = this.calcnet(xss)
        const { dy, dw } = this.calcDerivative(hy, yss)
        for (let l = 1; l < this.nlayer; l++) {
          dws[l] = dws[l].addition(dw[l])
          dys[l] = dys[l].addition(dy[l])
        }
        let loss = hy[this.nlayer - 1].subtraction(yss)
          .atomicOperation(item => (item ** 2) / 2)
          .getMeanOfRow(0)
        eploss += loss
        //如果满足批次或最后一次迭代有余 则更新
        if (batch === batchSize || (ep === conf.epochs - 1 && n === m - 1 && batch !== 0)) {
          nbatch += 1
          for (let l = 1; l < this.nlayer; l++) {
            dws[l] = dws[l].atomicOperation(item => item / batch)
            dys[l] = dys[l].atomicOperation(item => item / batch)
          }
          this.update(dys, dws)
          if (conf.onBatch) conf.onBatch(nbatch, batch, loss)
          let [dwt, dyt] = this.initwb(0)
          dws = dwt
          dys = dyt
          batch = 0
        }
      }
      if (conf.onEpoch) conf.onEpoch(ep, eploss / m)
    }
  }

  fit(xs: Matrix, ys: Matrix, conf: FitConf) {
    if (xs.shape[0] !== ys.shape[0]) {
      throw new Error('输入输出矩阵行数不统一')
    }
    if (xs.shape[1] !== this.nOfLayer(0)) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.nOfLayer(0)}`)
    }
    if (ys.shape[1] !== this.nOfLayer(this.nlayer - 1)) {
      throw new Error(`标签与网络输出不符合，output num -> ${this.nOfLayer(this.nlayer - 1)}`)
    }
    if (conf.batchSize && conf.batchSize > ys.shape[0] * conf.epochs) {
      throw new Error(`批次大小不能大于 epochs * m`)
    }
    const [nxs, scalem] = xs.normalization()
    this.scalem = scalem
    xs = nxs
    let optimizer = this.netconf ? this.netconf.optimizer : undefined
    switch (optimizer) {
      case 'bgd':
        return this.bgd(xs, ys, conf)
      case 'mbgd':
        return this.mbgd(xs, ys, conf)
      case 'sgd':
      default:
        return this.sgd(xs, ys, conf)
    }
  }
}