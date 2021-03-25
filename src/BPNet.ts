import { Matrix } from "./matrix"

/**激活函数类型*/
type ActivationFunction = 'Sigmoid' | 'Relu'

export class BPNet {
  w: Matrix[] //权值矩阵
  b: Matrix[] //偏值矩阵
  layerNum: number
  rate = 0.001 //学习率
  constructor(
    /**网络形状*/
    public readonly shape: number[],
    /**激活函数类型*/
    public readonly af?: ActivationFunction
  ) {
    if (shape.length < 3) {
      throw new Error('网络至少三层')
    }
    this.layerNum = shape.length
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
      default:
        return 1
    }
  }

  predict(xs: Matrix) {
    if (xs.shape[1] !== this.shape[0]) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`)
    }
    let ys: Matrix[] = []
    for (let l = 0; l < this.layerNum; l++) {
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

  //计算单组特征
  calcNetwork(xs: number[]) {
    let ys: number[][] = []
    for (let l = 0; l < this.layerNum; l++) {
      if (l === 0) {
        ys[l] = xs
        continue;
      }
      ys[l] = []
      for (let j = 0; j < this.shape[l]; j++) {
        let u = 0
        for (let i = 0; i < this.shape[l - 1]; i++) {
          u += this.w[l].get(j, i) * ys[l - 1][i]
        }
        u += this.b[l].get(0, j)
        ys[l][j] = this.afn(u)
      }
    }
    return ys
  }

  //计算误差
  calcdelta(ys: number[], hy: number[][]) {
    let delta = []
    for (let l = this.layerNum - 1; l > 0; l--) {
      if (l === this.layerNum - 1) {
        let n = []
        for (let j = 0; j < this.shape[l]; j++) {
          n[j] = (ys[j] - hy[l][j]) * this.afd(hy[l][j])
        }
        delta[l] = n
        continue;
      }
      let n = []
      for (let j = 0; j < this.shape[l]; j++) {
        n[j] = 0
        for (let i = 0; i < this.shape[l + 1]; i++) {
          n[j] += delta[l + 1][i] * this.w[l + 1].get(i, j)
        }
        n[j] *= this.afd(hy[l][j])
      }
      delta[l] = n
    }
    return delta
  }

  //调整权值和阈值
  update(hy: number[][], delta: number[][]) {
    for (let l = 1; l < this.layerNum; l++) {
      for (let j = 0; j < this.shape[l]; j++) {
        for (let i = 0; i < this.shape[l - 1]; i++) {
          this.w[l].update(j, i, this.w[l].get(j, i) + this.rate * delta[l][j] * hy[l - 1][i])
          this.b[l].update(0, j, this.b[l].get(0, j) + this.rate * delta[l][j])
        }
      }
    }
  }

  fit(xs: Matrix, ys: Matrix, batch: number, callback?: (batch: number, loss: number) => void) {
    if (xs.shape[1] !== this.shape[0]) {
      throw new Error(`特征与网络输入不符合，input num -> ${this.shape[0]}`)
    }
    if (ys.shape[1] !== this.shape[this.layerNum - 1]) {
      throw new Error(`标签与网络输出不符合，output num -> ${this.shape[this.layerNum - 1]}`)
    }
    for (let p = 0; p < batch; p++) {
      let loss = 0
      for (let i = 0; i < xs.shape[0]; i++) {
        let hy = this.calcNetwork(xs.getRow(i))
        let delta = this.calcdelta(ys.getRow(i), hy)
        this.update(hy, delta)
        let n = 0
        let l1 = this.layerNum - 1
        for (let l = 0; l < this.shape[l1]; l++) {
          n += ((ys.get(i, l) - hy[l1][l]) ** 2)
        }
        loss += n / this.shape[l1]
      }
      loss = loss / (2 * xs.shape[0])
      if (callback) callback(p, loss)
    }
  }
}