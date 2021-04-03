import { Matrix } from "./matrix"

/**激活函数类型*/
export type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh' | 'Softmax'

/**梯度更新方式*/
export type Mode = 'sgd' | 'bgd' | 'mbgd'

/**网络形状*/
export type NetShape = (number | [number, ActivationFunction])[]

/**模型配置*/
export interface NetConfig {
  mode?: Mode
  rate?: number
  w?: Matrix[]
  b?: Matrix[]
  scalem?: Matrix
}

/**训练配置*/
export interface FitConf {
  epochs: number
  batchSize?: number
  onBatch?: (batch: number, size: number, loss: number) => void
  onEpoch?: (epoch: number, loss: number) => void
}

export class BPNet {
  /**权值*/
  w: Matrix[]
  /**偏值*/
  b: Matrix[]
  nlayer: number
  /**缩放比*/
  scalem?: Matrix
  mode: Mode = 'sgd'
  /**学习率*/
  rate: number = 0.01
  constructor(
    public readonly shape: NetShape,
    conf?: NetConfig
  ) {
    if (shape.length < 2) {
      throw new Error('The network has at least two layers')
    }
    this.nlayer = shape.length
    const [w, b] = this.initwb()
    this.w = w
    this.b = b
    if (conf) {
      if (conf.mode) this.mode = conf.mode
      if (conf.rate) this.rate = conf.rate
      if (conf.w) this.w = conf.w
      if (conf.b) this.b = conf.b
      if (conf.scalem) this.scalem = conf.scalem
    }
  }

  /**
   * 获取当前层神经元
   */
  nOfLayer(l: number) {
    let n = this.shape[l]
    return Array.isArray(n) ? n[0] : n
  }

  /**
   * 获取当前层激活函数
   */
  afOfLayer(l: number) {
    let n = this.shape[l]
    return Array.isArray(n) ? n[1] : undefined
  }

  /**
   * 初始化权值偏值，
   * 默认值-0.5 ~ 0.5
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
   * 获取当前层激活函数
   */
  afn(x: number, l: number, rows: number[]) {
    let af = this.afOfLayer(l)
    switch (af) {
      case 'Sigmoid':
        return 1 / (1 + Math.exp(-x))
      case 'Relu':
        return x >= 0 ? x : 0
      case 'Tanh':
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))
      case 'Softmax':
        let d = Math.max(...rows) //防止指数过大
        return Math.exp(x - d) / rows.reduce((p, c) => p + Math.exp(c - d), 0)
      default:
        return x
    }
  }

  /**
   * 获取当前层激活函数求导
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
      case 'Softmax':
      default:
        return 1
    }
  }

  /**
   * 计算整个网络输出
   * - layer[hy][t-1] * w[t] + b
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
      let a = hy[l - 1].multiply(this.w[l].T).atomicOperation((item, _, j) => item + this.b[l].get(0, j))
      hy[l] = a.atomicOperation((item, i) => this.afn(item, l, a.getRow(i)))
    }
    return hy
  }

  /**
   * 按比例缩放矩阵
   */
  zoomScalem(xs: Matrix) {
    return xs.atomicOperation((item, _, j) => {
      if (!this.scalem) return item
      return this.scalem.get(1, j) === 0 ? 0 : (item - this.scalem.get(0, j)) / this.scalem.get(1, j)
    })
  }

  /**
   * 预测函数，输出最后一层求值
   */
  predict(xs: Matrix) {
    if (xs.shape[1] !== this.nOfLayer(0)) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.nOfLayer(0)}`)
    }
    return this.calcnet(this.zoomScalem(xs))[this.nlayer - 1]
  }

  /**
   * 多样本求导，求平均导数
   */
  calcDerivativeMul(hy: Matrix[], ys: Matrix) {
    let m = ys.shape[0]
    let dws: Matrix[] | null = null
    let dys: Matrix[] | null = null
    for (let n = 0; n < m; n++) {
      let nhy = hy.map(item => new Matrix([item.getRow(n)]))
      let nys = new Matrix([ys.getRow(n)])
      let { dw, dy } = this.calcDerivative(nhy, nys)
      dws = dws ? dws.map((d, l) => d.addition(dw[l])) : dw
      dys = dys ? dys.map((d, l) => d.addition(dy[l])) : dy
    }
    dws = dws!.map(d => d.atomicOperation(item => item / m))
    dys = dys!.map(d => d.atomicOperation(item => item / m))
    return { dy: dys, dw: dws }
  }

  /**
   * 单样本求导，对每个输出单元的求导，对每个权重的求导
   * - J = 1 / 2 * (hy - y)^2
   * - ∂J / ∂hy = (1 / 2) * 2 * (hy - y) = hy - y  最后一层节点求导
   * - ∂J / ∂w = 输入节点 * 输出节点导数   
   * 遵循链式求导法则：分支节点相加；链路节点相乘法；反向的计算过程
   * 如果有激活函数，需乘激活函数的导数
   * @returns [输出单元导数, 权重导数]
   */
  calcDerivative(hy: Matrix[], ys: Matrix) {
    let dw = this.w.map(w => w.zeroed())
    let dy = this.b.map(b => b.zeroed())
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
   * 更新权值偏值
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
   * 平方差带价函数
   * 多输出求平均值
   * - J = 1 / 2 * m * ∑m(hy - ys) ** 2 
   */
  cost(hy: Matrix, ys: Matrix) {
    let m = ys.shape[0]
    let sub = hy.subtraction(ys).atomicOperation(item => item ** 2).columnSum()
    let tmp = sub.getRow(0).map(v => v / (2 * m))
    return tmp.reduce((p, c) => p + c) / tmp.length
  }

  /**
   * 标准梯度下降，
   * 全部样本导数的平均值
   */
  bgd(xs: Matrix, ys: Matrix, conf: FitConf) {
    for (let ep = 0; ep < conf.epochs; ep++) {
      let hy = this.calcnet(xs)
      let { dy, dw } = this.calcDerivativeMul(hy, ys)
      this.update(dy, dw)
      if (conf.onEpoch) conf.onEpoch(ep, this.cost(hy[this.nlayer - 1], ys))
    }
  }

  /**
   * 随机梯度下降，
   * 单个样本的导数
   */
  sgd(xs: Matrix, ys: Matrix, conf: FitConf) {
    let m = ys.shape[0]
    for (let ep = 0; ep < conf.epochs; ep++) {
      let hys: Matrix | null = null
      for (let n = 0; n < m; n++) {
        let xss = new Matrix([xs.getRow(n)])
        let yss = new Matrix([ys.getRow(n)])
        let hy = this.calcnet(xss)
        const { dy, dw } = this.calcDerivative(hy, yss)
        this.update(dy, dw)
        //把当前结果缓存起来
        hys = hys ? hys.connect(hy[this.nlayer - 1]) : hy[this.nlayer - 1]
      }
      if (conf.onEpoch) conf.onEpoch(ep, this.cost(hys!, ys))
    }
  }

  /**
   * 批量梯度下降，
   * 多个样本导数的平均值
   */
  mbgd(xs: Matrix, ys: Matrix, conf: FitConf) {
    let m = ys.shape[0]
    let batchSize = conf.batchSize ? conf.batchSize : 10
    let batch = Math.ceil(m / batchSize) //总批次
    for (let ep = 0; ep < conf.epochs; ep++) {
      let { xs: xst, ys: yst } = this.upset(xs, ys)
      //必须每次打乱数据
      let eploss = 0
      for (let b = 0; b < batch; b++) {
        let start = b * batchSize
        let end = start + batchSize
        end = end > m ? m : end
        let size = end - start
        let xss = xst.slice(start, end)
        let yss = yst.slice(start, end)
        let hy = this.calcnet(xss)
        const { dy, dw } = this.calcDerivative(hy, yss)
        this.update(dy, dw)
        let bloss = this.cost(hy[this.nlayer - 1], yss)
        eploss += bloss
        if (conf.onBatch) conf.onBatch(b, size, bloss)
      }
      if (conf.onEpoch) conf.onEpoch(ep, eploss / batch)
    }
  }

  /**
   * 打乱数据
   */
  upset(xs: Matrix, ys: Matrix) {
    let xss = xs.dataSync()
    let yss = ys.dataSync()
    for (let i = 1; i < ys.shape[0]; i++) {
      let random = Math.floor(Math.random() * (i + 1));
      [xss[i], xss[random]] = [xss[random], xss[i]];
      [yss[i], yss[random]] = [yss[random], yss[i]];
    }
    return { xs: new Matrix(xss), ys: new Matrix(yss) }
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
    if (conf.batchSize && conf.batchSize > ys.shape[0]) {
      throw new Error(`批次大小不能大于样本数`)
    }
    const [nxs, scalem] = xs.normalization()
    this.scalem = scalem
    xs = nxs
    switch (this.mode) {
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