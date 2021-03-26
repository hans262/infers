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
      throw new Error('网络至少三层')
    }
    this.nlayer = shape.length
    const [w, b] = this.initwb(shape)
    this.w = w
    this.b = b
  }
  /**
   * 初始化权值、偏值
   * @param shape 
   * @returns [w, b]
   */
  initwb(shape: number[]) {
    let w: Matrix[] = []
    let b: Matrix[] = []
    for (let l = 1; l < shape.length; l++) {
      w[l] = Matrix.generate(shape[l], shape[l - 1])
      b[l] = Matrix.generate(1, shape[l])
    }
    return [w, b]
  }
  /**
   * 更新学习率
   * @param rate 
   */
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
    if (xs.shape[1] !== this.shape[0]) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`)
    }
    let ys: Matrix[] = []
    for (let l = 0; l < this.nlayer; l++) {
      if (l === 0) {
        ys[l] = xs
        continue;
      }
      let w = this.w[l].T
      let b = this.b[l]
      ys[l] = ys[l - 1].multiply(w).atomicOperation((item, _, j) =>
        this.afn(item + b.get(0, j))
      )
    }
    return ys
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
    return this.calcnet(this.zoomScalem(xs))
  }

  /**
   * 求误差相对于每一个节点的导数  
   * - E = 1 / 2 (hy - y)^2
   * - ∂E / ∂hy = (1 / 2) * 2 * (hy - y) = hy - y  
   * 求导遵循链式法则：分支节点相加，链路节点相乘。
   * 有激活函数的情况，还需乘以激活函数的导数
   */
  calcDerivative(ys: number[], hy: Matrix[]) {
    let dy = []
    for (let l = this.nlayer - 1; l > 0; l--) {
      if (l === this.nlayer - 1) {
        let n = []
        for (let j = 0; j < this.shape[l]; j++) {
          n[j] = (hy[l].get(0, j) - ys[j]) * this.afd(hy[l].get(0, j))
        }
        dy[l] = n
        continue;
      }
      let n = []
      for (let j = 0; j < this.shape[l]; j++) {
        n[j] = 0
        for (let i = 0; i < this.shape[l + 1]; i++) {
          n[j] += dy[l + 1][i] * this.w[l + 1].get(i, j)
        }
        n[j] *= this.afd(hy[l].get(0, j))
      }
      dy[l] = n
    }
    return dy
  }

  /**
   * 调整权值和阈值
   * @param dy 
   * @param hy 
   */
  update(dy: number[][], hy: Matrix[]) {
    for (let l = 1; l < this.nlayer; l++) {
      for (let j = 0; j < this.shape[l]; j++) {
        for (let i = 0; i < this.shape[l - 1]; i++) {
          this.w[l].update(j, i, this.rate * dy[l][j] * hy[l - 1].get(0, i), '-=')
          this.b[l].update(0, j, this.rate * dy[l][j], '-=')
        }
      }
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
        let dys = this.calcDerivative(ys.getRow(n), hy)
        this.update(dys, hy)

        let e = 0
        let l = this.nlayer - 1
        for (let j = 0; j < this.shape[l]; j++) {
          e += ((ys.get(n, j) - hy[l].get(0, j)) ** 2)
        }
        loss += e / this.shape[l]
      }
      loss = loss / (2 * xs.shape[0])
      if (callback) callback(p, loss)
    }
  }
}