import { Matrix } from "./matrix"

export class NeuralNetwork {
  w: number[][][] = [] //权值矩阵
  b: number[][] = [] //偏值矩阵
  readonly shape: number[] //网络形状
  layerNum: number //层数
  rate = 0.5 //学习率
  esp = 0.0001 //误差阈值
  constructor(shape: number[]) {
    if (shape.length < 3) {
      throw new Error('网络至少三层')
    }
    if (shape[0] < 2) {
      throw new Error('输入层至少两个特征')
    }

    this.shape = shape
    this.layerNum = shape.length

    // 初始化权值、偏值
    for (let l = 1; l < this.layerNum; l++) {
      let witem = []
      let bitem = []
      for (let j = 0; j < shape[l]; j++) {
        let n = []
        for (let i = 0; i < shape[l - 1]; i++) {
          n.push(0.5 - Math.random())
        }
        witem.push(n)
        bitem.push(0.5 - Math.random())
      }
      this.w[l] = witem
      this.b[l] = bitem
    }
  }
  setRate(rate: number) {
    this.rate = rate
  }
  sigmoid(x: number) {
    return 1 / (1 + Math.exp(-x))
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
      let w = new Matrix(this.w[l]).T
      let b = new Matrix([this.b[l]])
      ys[l] = ys[l - 1].multiply(w).atomicOperation((item, _, j) =>
        this.sigmoid(item + b.get(0, j))
      )
    }
    return ys
  }

  //计算单组特征 输出网络
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
          u += this.w[l][j][i] * ys[l - 1][i]
        }
        u += this.b[l][j]
        ys[l][j] = this.sigmoid(u)
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
          n[j] = (ys[j] - hy[l][j]) * hy[l][j] * (1 - hy[l][j])
        }
        delta[l] = n
        continue;
      }
      let n = []
      for (let j = 0; j < this.shape[l]; j++) {
        n[j] = 0
        for (let i = 0; i < this.shape[l + 1]; i++) {
          n[j] += delta[l + 1][i] * this.w[l + 1][i][j]
        }
        n[j] *= hy[l][j] * (1 - hy[l][j])
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
          this.w[l][j][i] += this.rate * delta[l][j] * hy[l - 1][i]
          this.b[l][j] += this.rate * delta[l][j]
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
        //正向求值
        let hy = this.calcNetwork(xs.getRow(i))
        //反向求误差
        let delta = this.calcdelta(ys.getRow(i), hy)
        this.update(hy, delta)
        //求损失平方和
        let n = 0
        let l1 = this.layerNum - 1
        for (let l = 0; l < this.shape[l1]; l++) {
          n += ((ys.get(i, l) - hy[l1][l]) ** 2)
        }
        loss += n / this.shape[l1]
      }
      loss = loss / (2 * xs.shape[0])
      if (callback) callback(p, loss)
      if (loss < this.esp) break;
    }
  }
}