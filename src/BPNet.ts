import { upset } from "./common"
import { Matrix } from "./matrix"
import { ActivationFunction, FitConf, Mode, NetConfig, NetShape } from "./types"

export class BPNet {
  /**权值*/
  w: Matrix[]
  /**偏值*/
  b: Matrix[]
  /**隐藏层层数*/
  hlayer: number
  /**缩放比*/
  scale?: Matrix
  /**梯度更新模式*/
  mode: Mode = 'sgd'
  /**学习率*/
  rate: number = 0.01
  constructor(
    public readonly shape: NetShape,
    conf?: NetConfig
  ) {
    this.hlayer = shape.length - 1
    if (this.hlayer < 1) {
      throw new Error('The network has at least two layers')
    }
    //初始化权值偏值
    this.w = []
    this.b = []
    for (let l = 0; l < this.hlayer; l++) {
      this.w[l] = Matrix.generate(this.unit(l), this.unit(l - 1))
      this.b[l] = Matrix.generate(1, this.unit(l))
    }
    if (conf) {
      if (conf.mode) this.mode = conf.mode
      if (conf.rate) this.rate = conf.rate
      if (conf.w) this.w = conf.w
      if (conf.b) this.b = conf.b
      if (conf.scale) this.scale = conf.scale
    }
  }

  /**
   * 获取当前层单元数
   */
  unit(l: number) {
    let n = this.shape[l + 1]
    return Array.isArray(n) ? n[0] : n
  }

  /**
   * 获取当前层激活函数
   */
  af(l: number) {
    let n = this.shape[l + 1]
    return Array.isArray(n) ? n[1] : undefined
  }

  /**
   * 激活函数求值
   */
  afn(x: number, rows: number[], af?: ActivationFunction) {
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
   * 激活函数求导
   */
  afd(x: number, af?: ActivationFunction) {
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
   * @returns 模型JSON字符串
   */
  toJSON() {
    return JSON.stringify({
      mode: this.mode,
      shape: this.shape,
      rate: this.rate,
      scale: this.scale ? this.scale.dataSync() : undefined,
      w: this.w.map(w => w.dataSync()),
      b: this.b.map(b => b.dataSync()),
    })
  }

  /**
   * 计算整个网络输出
   * - layer[hy][t-1] * w[t] + b
   * - hy =  θ1 * X1 + θ2 * X2 + ... + θn * Xn + b
   * @param xs inputs
   */
  calcnet(xs: Matrix) {
    let hy: Matrix[] = []
    for (let l = 0; l < this.hlayer; l++) {
      let lastHy = l === 0 ? xs : hy[l - 1]
      let af = this.af(l)
      let tmp = lastHy.multiply(this.w[l].T).atomicOperation((item, _, j) => item + this.b[l].get(0, j))
      hy[l] = tmp.atomicOperation((item, i) => this.afn(item, tmp.getRow(i), af))
    }
    return hy
  }

  /**
   * 按照缩放比缩放新的特征
   */
  scaled(xs: Matrix) {
    if (!this.scale) return xs
    return xs.atomicOperation((item, _, j) => {
      let scale = this.scale!
      let range = scale.get(1, j)
      let average = scale.get(0, j)
      return range === 0 ? 0 : (item - average) / range
    })
  }

  /**
   * 预测函数，返回最后一层求值
   */
  predict(xs: Matrix) {
    this.checkInput(xs)
    xs = this.scaled(xs)
    let hy = this.calcnet(xs)
    return hy[hy.length - 1]
  }

  /**
   * 多样本求导，
   * 计算单个样本倒数求和，然后计算平均倒数
   */
  calcDerivativeMultiple(hy: Matrix[], xs: Matrix, ys: Matrix) {
    let m = ys.shape[0]
    let dws = this.w.map(w => w.zeroed())
    let dys = this.b.map(b => b.zeroed())
    for (let n = 0; n < m; n++) {
      let nhy = hy.map(item => new Matrix([item.getRow(n)]))
      let nxs = new Matrix([xs.getRow(n)])
      let nys = new Matrix([ys.getRow(n)])
      let { dw: ndw, dy: ndy } = this.calcDerivative(nhy, nxs, nys)
      dws = dws.map((d, l) => d.addition(ndw[l]))
      dys = dys.map((d, l) => d.addition(ndy[l]))
    }
    let dw = dws.map(d => d.atomicOperation(item => item / m))
    let dy = dys.map(d => d.atomicOperation(item => item / m))
    return { dy, dw }
  }


  /**
   * 单样本求导，对每个输出单元的求导，对每个权重的求导
   * - J = 1 / 2 * (hy - y)^2
   * - ∂J / ∂hy[last] = (1 / 2) * 2 * (hy - y) = (hy - y) * hy[激活函数求导]
   * - ∂J / ∂w = dy.T * lastHy
   * - ∂J / ∂hy[now] = (nextDy * nextW) * hy[激活函数求导]
   * 
   * 链式求导法则：
   *  - 反向的计算过程；
   *  - 分支节点相加；链路节点相乘法；
   *  - 如有激活函数需乘激活函数的导数；
   * @returns [神经单元导数, 权重导数]
   */
  calcDerivative(hy: Matrix[], xs: Matrix, ys: Matrix) {
    let dw: Matrix[] = [], dy: Matrix[] = []
    for (let l = this.hlayer - 1; l >= 0; l--) {
      let lastHy = hy[l - 1] ? hy[l - 1] : xs
      let af = this.af(l)
      if (l === this.hlayer - 1) {
        dy[l] = hy[l].atomicOperation((item, r, c) => (item - ys.get(r, c)) * this.afd(item, af))
      } else {
        dy[l] = dy[l + 1].multiply(this.w[l + 1]).atomicOperation((item, r, c) => item * this.afd(hy[l].get(r, c), af))
      }
      dw[l] = dy[l].T.multiply(lastHy)
    }
    return { dy, dw }
  }

  /**
   * 更新权值偏值
   * - w = w - α * (∂J / ∂w)
   * - b = b - α * (∂J / ∂hy)
   */
  update(dy: Matrix[], dw: Matrix[]) {
    this.w = this.w.map((w, l) => w.subtraction(dw[l].numberMultiply(this.rate)))
    this.b = this.b.map((b, l) => b.subtraction(dy[l].numberMultiply(this.rate)))
  }

  /**
   * 平方差带价函数
   * 多输出求平均值
   * - J = 1 / 2 * m * ∑m(hy - ys) ** 2 
   */
  cost(hy: Matrix, ys: Matrix) {
    let m = ys.shape[0]
    let sub = hy.subtraction(ys).atomicOperation(item => (item ** 2) / 2).columnSum()
    let tmp = sub.getRow(0).map(v => v / m)
    return tmp.reduce((p, c) => p + c) / tmp.length
  }

  /**
   * 计算一组样本的当前损失
   */
  calcLoss(xs: Matrix, ys: Matrix) {
    this.checkSample(xs, ys)
    let lastHy = this.predict(xs)
    return this.cost(lastHy, ys)
  }

  /**
   * 标准梯度下降，
   * 全部样本导数的平均值
   */
  async bgd(xs: Matrix, ys: Matrix, conf: FitConf) {
    for (let ep = 0; ep < conf.epochs; ep++) {
      let hy = this.calcnet(xs)
      let { dy, dw } = this.calcDerivativeMultiple(hy, xs, ys)
      this.update(dy, dw)
      if (conf.onEpoch) {
        conf.onEpoch(ep, this.cost(hy[hy.length - 1], ys))
        conf.async && await new Promise(resolve => setTimeout(resolve))
      }
      if (conf.onTrainEnd && ep === conf.epochs - 1) {
        conf.onTrainEnd(this.cost(hy[hy.length - 1], ys))
      }
    }
  }

  /**
   * 随机梯度下降，
   * 单个样本的导数
   */
  async sgd(xs: Matrix, ys: Matrix, conf: FitConf) {
    let m = ys.shape[0]
    for (let ep = 0; ep < conf.epochs; ep++) {
      let hys: Matrix | null = null
      for (let n = 0; n < m; n++) {
        let nxs = new Matrix([xs.getRow(n)])
        let nys = new Matrix([ys.getRow(n)])
        let hy = this.calcnet(nxs)
        const { dy, dw } = this.calcDerivative(hy, nxs, nys)
        this.update(dy, dw)
        hys = hys ? hys.connect(hy[hy.length - 1]) : hy[hy.length - 1]
      }
      if (conf.onEpoch) {
        conf.onEpoch(ep, this.cost(hys!, ys))
        conf.async && await new Promise(resolve => setTimeout(resolve))
      }
      if (conf.onTrainEnd && ep === conf.epochs - 1) {
        conf.onTrainEnd(this.cost(hys!, ys))
      }
    }
  }

  /**
   * 批量梯度下降，
   * 多个样本导数的平均值
   */
  async mbgd(xs: Matrix, ys: Matrix, conf: FitConf) {
    let m = ys.shape[0]
    let defaultBatchSize = m < 10 ? m : 10
    let batchSize = conf.batchSize ? conf.batchSize : defaultBatchSize
    let batch = Math.ceil(m / batchSize) //总批次
    for (let ep = 0; ep < conf.epochs; ep++) {
      let { xs: xst, ys: yst } = upset(xs, ys)
      //打乱数据加快拟合速度
      let eploss = 0
      for (let b = 0; b < batch; b++) {
        let start = b * batchSize
        let end = start + batchSize
        end = end > m ? m : end
        let size = end - start
        let bxs = xst.slice(start, end)
        let bys = yst.slice(start, end)
        let hy = this.calcnet(bxs)
        let lastHy = hy[hy.length - 1]
        const { dy, dw } = this.calcDerivativeMultiple(hy, bxs, bys)
        this.update(dy, dw)
        let bloss = this.cost(lastHy, bys)
        eploss += bloss
        if (conf.onBatch) conf.onBatch(b, size, bloss)
      }
      if (conf.onEpoch) {
        conf.onEpoch(ep, eploss / batch)
        conf.async && await new Promise(resolve => setTimeout(resolve))
      }
      if (conf.onTrainEnd && ep === conf.epochs - 1) {
        conf.onTrainEnd(eploss / batch)
      }
    }
  }

  checkInput(xs: Matrix) {
    if (xs.shape[1] !== this.unit(-1)) {
      throw new Error(`Input matrix column number error, input shape -> ${this.unit(-1)}.`)
    }
  }

  checkSample(xs: Matrix, ys: Matrix) {
    this.checkInput(xs)
    if (xs.shape[0] !== ys.shape[0]) {
      throw new Error('The row number of input and output matrix is not uniform.')
    }
    if (ys.shape[1] !== this.unit(this.hlayer - 1)) {
      throw new Error(`Output matrix column number error, output shape -> ${this.unit(this.hlayer - 1)}.`)
    }
  }

  fit(xs: Matrix, ys: Matrix, conf: FitConf) {
    this.checkSample(xs, ys)
    if (conf.batchSize && conf.batchSize > ys.shape[0]) {
      throw new Error(`The batch size cannot be greater than the number of samples.`)
    }
    const [nxs, scale] = xs.normalization()
    this.scale = scale
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