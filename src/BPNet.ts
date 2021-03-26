import { Matrix } from "./matrix"

/**激活函数类型*/
export type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh'

export class BPNet {
  w: Matrix[] //权值矩阵
  b: Matrix[] //偏值矩阵
  nlayer: number
  rate = 0.001 //学习率
  scalem?: Matrix //缩放比例
  constructor(
    /**网络形状*/
    public readonly shape: number[],
    /**激活函数类型*/
    public readonly af?: ActivationFunction
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
   * 初始化权值、偏值
   * @param shape 
   * @returns [w, b]
   */
  initwb(v?: number) {
    let w: Matrix[] = []
    let b: Matrix[] = []
    for (let l = 1; l < this.shape.length; l++) {
      w[l] = Matrix.generate(this.shape[l], this.shape[l - 1], v)
      b[l] = Matrix.generate(1, this.shape[l], v)
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
  afn(x: number) {
    switch (this.af) {
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
  afd(x: number) {
    switch (this.af) {
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
      let w = this.w[l].T
      let b = this.b[l]
      hy[l] = hy[l - 1].multiply(w).atomicOperation((item, _, j) =>
        this.afn(item + b.get(0, j))
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
    if (xs.shape[1] !== this.shape[0]) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`)
    }
    return this.calcnet(this.zoomScalem(xs))
  }

  /**
   * 求误差相对于每一个节点的导数  
   * - E = 1 / 2 (hy - y)^2
   * - ∂E / ∂hy = (1 / 2) * 2 * (hy - y) = hy - y  最后一个节点的导数
   * - ∂E / ∂w = 当前权重输入节点的值 * 输出节点的导数
   * 
   * 求导遵循链式法则：分支节点相加，链路节点相乘。
   * 有激活函数，还需乘以激活函数的导数。
   * @returns [节点导数矩阵，权重导数矩阵]
   */
  calcDerivative(ys: number[], hy: Matrix[]) {
    const [dw, dy] = this.initwb(0)
    for (let l = this.nlayer - 1; l > 0; l--) {
      if (l === this.nlayer - 1) {
        for (let j = 0; j < this.shape[l]; j++) {
          let n = (hy[l].get(0, j) - ys[j]) * this.afd(hy[l].get(0, j))
          dy[l].update(0, j, n)
          for (let k = 0; k < this.shape[l - 1]; k++) {
            dw[l].update(j, k, hy[l - 1].get(0, k) * n)
          }
        }
        continue;
      }
      for (let j = 0; j < this.shape[l]; j++) {
        for (let i = 0; i < this.shape[l + 1]; i++) {
          dy[l].update(0, j, dy[l + 1].get(0, i) * this.w[l + 1].get(i, j), '+=')
        }
        dy[l].update(0, j, this.afd(hy[l].get(0, j)), '*=')
        for (let k = 0; k < this.shape[l - 1]; k++) {
          dw[l].update(j, k, hy[l - 1].get(0, k) * dy[l].get(0, j))
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
      this.w[l] = this.w[l].atomicOperation((w, i, j) => w - this.rate * dw[l].get(i, j))
      this.b[l] = this.b[l].atomicOperation((b, i, j) => b - this.rate * dy[l].get(i, j))
    }
  }

  fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void) {
    if (xs.shape[0] !== ys.shape[0]) {
      throw new Error('输入输出矩阵行数不统一')
    }
    if (xs.shape[1] !== this.shape[0]) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`)
    }
    if (ys.shape[1] !== this.shape[this.nlayer - 1]) {
      throw new Error(`标签与网络输出不符合，output num -> ${this.shape[this.nlayer - 1]}`)
    }
    //归一化 -0.5 ～ 0.5
    let [inputs, scalem] = xs.normalization()
    this.scalem = scalem
    xs = inputs

    for (let p = 0; p < batch; p++) {
      let loss = 0
      for (let n = 0; n < xs.shape[0]; n++) {
        let xss = new Matrix([xs.getRow(n)])
        let hy = this.calcnet(xss)
        const { dy, dw } = this.calcDerivative(ys.getRow(n), hy)
        this.update(dy, dw)
        let e = 0
        let l = this.nlayer - 1
        for (let j = 0; j < this.shape[l]; j++) {
          e += ((ys.get(n, j) - hy[l].get(0, j)) ** 2) / 2
        }
        loss += e / this.shape[l]
      }
      loss = loss / xs.shape[0]
      if (callback) callback(p, loss)
    }
  }
}