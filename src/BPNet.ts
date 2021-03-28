import { Matrix } from "./matrix"

/**激活函数类型*/
export type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh'
/**网络形状*/
export type NetShape = (number | [number, ActivationFunction])[]
/**模型配置项*/
export interface NetConfig {
  optimizer: 'SGD' | 'BGD'
}

export class BPNet {
  /**权值矩阵*/
  w: Matrix[]
  /**偏值矩阵*/
  b: Matrix[]
  nlayer: number
  /**学习率*/
  rate = 0.001
  /**缩放比例*/
  scalem?: Matrix
  constructor(
    public readonly shape: NetShape,
    public netconf?: NetConfig
  ) {
    if (shape.length < 3) {
      throw new Error('BP网络至少有三层结构')
    }
    this.nlayer = shape.length
    const [w, b] = this.initwb()
    this.w = w
    this.b = b
  }

  /**
   * 获取当前层的神经元个数
   * @param l 
   */
  nOfLayer(l: number) {
    let n = this.shape[l]
    return Array.isArray(n) ? n[0] : n
  }

  /**
   * 获取当前层的激活函数类型
   * @param l 
   */
  afOfLayer(l: number) {
    let n = this.shape[l]
    return Array.isArray(n) ? n[1] : undefined
  }

  /**
   * 初始化权值、偏值
   * @param shape 
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

  setRate(rate: number) {
    this.rate = rate
  }

  /**
   * 激活函数
   * @param x 
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
   * 激活函数对应求导
   * @param x 
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
    if (xs.shape[1] !== this.nOfLayer(0)) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.nOfLayer(0)}`)
    }
    return this.calcnet(this.zoomScalem(xs))
  }

  /**
   * 求误差相对于每一个节点的导数  
   * 这是关于单个样本的导数  
   * - E = 1 / 2 (hy - y)^2
   * - ∂E / ∂hy = (1 / 2) * 2 * (hy - y) = hy - y  最后一个节点的导数
   * - ∂E / ∂w = 当前权重输入节点的值 * 输出节点的导数
   * 
   * 求导遵循链式法则：分支节点相加，链路节点相乘。
   * 有激活函数，还需乘以激活函数的导数。
   * @returns [节点导数矩阵，权重导数矩阵]
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
   * 调整权值和偏值矩阵
   * - w = w - α * (∂E / ∂w)  导数项 = 误差相对于当前权重的偏导数
   * - b = b - α * (∂E / ∂hy) 导数项 = 误差相对于节点值的偏导数
   */
  update(dy: Matrix[], dw: Matrix[]) {
    for (let l = 1; l < this.nlayer; l++) {
      this.w[l] = this.w[l].subtraction(dw[l].numberMultiply(this.rate))
      this.b[l] = this.b[l].subtraction(dy[l].numberMultiply(this.rate))
    }
  }

  /**
   * 代价函数
   * J = 1 / 2 * m * ∑m(hy - ys) ** 2 
   * @param hy 
   * @param ys 
   */
  cost(hy: Matrix[], ys: Matrix) {
    let m = ys.shape[0]
    let sub = hy[this.nlayer - 1].subtraction(ys).atomicOperation(item => item ** 2).columnSum()
    let tmp = sub.getRow(0).map(v => (1 / (2 * m)) * v)
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
    //归一化 -0.5 ～ 0.5
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