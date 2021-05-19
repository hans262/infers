import { upset, afd, afn } from "./common"
import { Matrix } from "./matrix"
import type {
  TrainingOptions, Mode, BPNetOptions, NetShape
} from "./types"

export const defaultTrainingOptions = (m: number): TrainingOptions => ({
  epochs: 100,
  batchSize: m > 10 ? 10 : m,
  async: false
})

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
    opt: BPNetOptions = {}
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
    if (opt.mode) this.mode = opt.mode
    if (opt.rate) this.rate = opt.rate
    if (opt.w) this.w = opt.w
    if (opt.b) this.b = opt.b
    if (opt.scale) this.scale = opt.scale
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
   * @returns json字符串
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
   * 加载json化的模型
   * @returns BPNet
   */
  static fromJSON(json: string) {
    let tmp = JSON.parse(json)
    let w: Matrix[] = tmp.w.map((w: any) => new Matrix(w))
    let b: Matrix[] = tmp.b.map((b: any) => new Matrix(b))
    let scale = tmp.scale ? new Matrix(tmp.scale) : undefined
    return new BPNet(tmp.shape, {
      mode: tmp.mode,
      rate: tmp.mode,
      w, b, scale
    })
  }

  /**
   * 计算整个网络输出
   * - layer[hy][t-1] * w[t] + b
   * - hy =  θ1 * X1 + θ2 * X2 + ... + θn * Xn + b
   * @param xs inputs
   */
  forwardPropagation(xs: Matrix) {
    let hy: Matrix[] = []
    for (let l = 0; l < this.hlayer; l++) {
      let lastHy = l === 0 ? xs : hy[l - 1]
      let af = this.af(l)
      let tmp = lastHy.multiply(this.w[l].T).atomicOperation((item, _, j) => item + this.b[l].get(0, j))
      hy[l] = tmp.atomicOperation((item, i) => afn(item, tmp.getRow(i), af))
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
   * 预测最后一层输出
   */
  predict(xs: Matrix) {
    let hy = this.predictNet(xs)
    return hy[hy.length - 1]
  }

  /**
   * 预测整个网络，包含输入层
   */
  predictNet(xs: Matrix) {
    this.checkInput(xs)
    xs = this.scaled(xs)
    let hy = this.forwardPropagation(xs)
    return [xs, ...hy]
  }

  /**
   * 多样本求导，
   * 计算单个样本倒数求和，然后计算平均倒数
   */
  backPropagationMultiple(hy: Matrix[], xs: Matrix, ys: Matrix) {
    let m = ys.shape[0]
    let dws = this.w.map(w => w.zeroed())
    let dys = this.b.map(b => b.zeroed())
    for (let n = 0; n < m; n++) {
      let nhy = hy.map(item => new Matrix([item.getRow(n)]))
      let nxs = new Matrix([xs.getRow(n)])
      let nys = new Matrix([ys.getRow(n)])
      let { dw: ndw, dy: ndy } = this.backPropagation(nhy, nxs, nys)
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
  backPropagation(hy: Matrix[], xs: Matrix, ys: Matrix) {
    let dw: Matrix[] = [], dy: Matrix[] = []
    for (let l = this.hlayer - 1; l >= 0; l--) {
      let lastHy = hy[l - 1] ? hy[l - 1] : xs
      let af = this.af(l)
      if (l === this.hlayer - 1) {
        dy[l] = hy[l].atomicOperation((item, r, c) => (item - ys.get(r, c)) * afd(item, af))
      } else {
        dy[l] = dy[l + 1].multiply(this.w[l + 1]).atomicOperation((item, r, c) => item * afd(hy[l].get(r, c), af))
      }
      dw[l] = dy[l].T.multiply(lastHy)
    }
    return { dy, dw }
  }

  /**
   * 调整权值偏值
   * - w = w - α * (∂J / ∂w)
   * - b = b - α * (∂J / ∂hy)
   */
  adjust(dy: Matrix[], dw: Matrix[]) {
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
  async bgd(xs: Matrix, ys: Matrix, opt: TrainingOptions) {
    for (let ep = 0; ep < opt.epochs; ep++) {
      let hy = this.forwardPropagation(xs)
      let { dy, dw } = this.backPropagationMultiple(hy, xs, ys)
      this.adjust(dy, dw)
      if (opt.onEpoch) {
        opt.onEpoch(ep, this.cost(hy[hy.length - 1], ys))
        opt.async && await new Promise(resolve => setTimeout(resolve))
      }
      if (opt.onTrainEnd && ep === opt.epochs - 1) {
        opt.onTrainEnd(this.cost(hy[hy.length - 1], ys))
      }
    }
  }

  /**
   * 随机梯度下降，
   * 单个样本的导数
   */
  async sgd(xs: Matrix, ys: Matrix, opt: TrainingOptions) {
    let m = ys.shape[0]
    for (let ep = 0; ep < opt.epochs; ep++) {
      let hys: Matrix | null = null
      for (let n = 0; n < m; n++) {
        let nxs = new Matrix([xs.getRow(n)])
        let nys = new Matrix([ys.getRow(n)])
        let hy = this.forwardPropagation(nxs)
        const { dy, dw } = this.backPropagation(hy, nxs, nys)
        this.adjust(dy, dw)
        hys = hys ? hys.connect(hy[hy.length - 1]) : hy[hy.length - 1]
      }
      if (opt.onEpoch) {
        opt.onEpoch(ep, this.cost(hys!, ys))
        opt.async && await new Promise(resolve => setTimeout(resolve))
      }
      if (opt.onTrainEnd && ep === opt.epochs - 1) {
        opt.onTrainEnd(this.cost(hys!, ys))
      }
    }
  }

  /**
   * 批量梯度下降，
   * 多个样本导数的平均值
   */
  async mbgd(xs: Matrix, ys: Matrix, opt: TrainingOptions) {
    let m = ys.shape[0]
    let batchSize = opt.batchSize
    let batch = Math.ceil(m / batchSize) //总批次
    for (let ep = 0; ep < opt.epochs; ep++) {
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
        let hy = this.forwardPropagation(bxs)
        let lastHy = hy[hy.length - 1]
        const { dy, dw } = this.backPropagationMultiple(hy, bxs, bys)
        this.adjust(dy, dw)
        let bloss = this.cost(lastHy, bys)
        eploss += bloss
        if (opt.onBatch) opt.onBatch(b, size, bloss)
      }
      if (opt.onEpoch) {
        opt.onEpoch(ep, eploss / batch)
        opt.async && await new Promise(resolve => setTimeout(resolve))
      }
      if (opt.onTrainEnd && ep === opt.epochs - 1) {
        opt.onTrainEnd(eploss / batch)
      }
    }
  }

  checkInput(xs: Matrix) {
    if (xs.shape[1] !== this.unit(-1)) {
      throw new Error(`Input matrix column number error, input shape -> ${this.unit(-1)}.`)
    }
  }

  checkOutput(ys: Matrix) {
    if (ys.shape[1] !== this.unit(this.hlayer - 1)) {
      throw new Error(`Output matrix column number error, output shape -> ${this.unit(this.hlayer - 1)}.`)
    }
  }

  checkSample(xs: Matrix, ys: Matrix) {
    if (xs.shape[0] !== ys.shape[0]) {
      throw new Error('The row number of input and output matrix is not uniform.')
    }
    this.checkInput(xs)
    this.checkOutput(ys)
  }

  fit(xs: Matrix, ys: Matrix, opt: Partial<TrainingOptions> = {}) {
    let m = ys.shape[0]
    let nopt = { ...defaultTrainingOptions(m), ...opt }
    this.checkSample(xs, ys)
    if (nopt.batchSize > m) {
      throw new Error(`The batch size cannot be greater than the number of samples.`)
    }
    const [nxs, scale] = xs.normalization()
    this.scale = scale
    xs = nxs
    switch (this.mode) {
      case 'bgd':
        return this.bgd(xs, ys, nopt)
      case 'mbgd':
        return this.mbgd(xs, ys, nopt)
      case 'sgd':
      default:
        return this.sgd(xs, ys, nopt)
    }
  }
}