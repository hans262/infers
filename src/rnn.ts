import { Matrix } from '../src'

type ActivationFunction = 'Tanh' | 'Softmax'

export class RNN {
  U: Matrix //隐藏层权值
  W: Matrix //上一时刻连接隐藏层权值
  V: Matrix //输出层权值

  indexWord: { [index: string]: number } = {}
  wordIndex: { [index: number]: string } = {}
  /**[['h', 'e', 'l', 'l', 'o'], ['h', 'u', 'a', 'h', 'u', 'a'], ...]*/
  trainData: string[][]

  inputSize: number // 输入层大小
  hideSize = 8 // 隐藏层大小

  constructor(data: string[]) {
    this.trainData = data.map(v => v.split(''))

    let temp = Array.from(new Set(this.trainData.flat(1)))
    for (let i = 0; i < temp.length; i++) {
      this.indexWord[temp[i]] = i
      this.wordIndex[i] = temp[i]
    }
    this.inputSize = temp.length
    this.wordIndex[temp.length] = '/n'
    this.indexWord['/n'] = temp.length

    let outputSize = this.inputSize + 1
    this.U = Matrix.generate(this.hideSize, this.inputSize)
    this.W = Matrix.generate(this.hideSize, this.hideSize)
    this.V = Matrix.generate(outputSize, this.hideSize)
  }

  afn(x: number, rows: number[], af?: ActivationFunction) {
    switch (af) {
      case 'Tanh':
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))
      case 'Softmax':
        let d = Math.max(...rows) //防止指数过大
        return Math.exp(x - d) / rows.reduce((p, c) => p + Math.exp(c - d), 0)
      default:
        return x
    }
  }

  afd(x: number, af?: ActivationFunction) {
    switch (af) {
      case 'Tanh':
        return 1 - ((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))) ** 2
      case 'Softmax':
      default:
        return 1
    }
  }

  //编码
  oneHotXs(inputIndex: number) {
    let xs = Matrix.generate(1, this.inputSize, 0)
    xs.update(0, inputIndex, 1)
    return xs
  }

  oneHotYs(outputIndex: number) {
    let ys = Matrix.generate(1, this.inputSize + 1, 0)
    if (outputIndex < 0) {
      ys.update(0, this.inputSize, 1)
    } else {
      ys.update(0, outputIndex, 1)
    }
    return ys
  }

  // 向前传播
  forwardPropagation(data: { xs: Matrix, ys: Matrix }[]) {
    //初始化st
    let lastSt = Matrix.generate(1, this.hideSize, 0)
    //求出每个时刻的值
    return data.map(v => {
      let { xs, ys } = v
      let st = xs.multiply(this.U.T).addition(lastSt.multiply(this.W.T))
      st = st.atomicOperation((item, i) => this.afn(item, st.getRow(i), 'Tanh'))
      let yt = st.multiply(this.V.T)
      yt = yt.atomicOperation((item, i) => this.afn(item, yt.getRow(i), 'Softmax'))
      let lastStCopy = lastSt
      lastSt = st //保存上一时刻st
      return { xs, ys, st, yt, lastSt: lastStCopy }
    })
  }

  // 反向传播
  backPropagation(hys: { xs: Matrix, ys: Matrix, st: Matrix, yt: Matrix, lastSt: Matrix }[]) {
    let dv = this.V.zeroed()
    let du = this.U.zeroed()
    let dw = this.W.zeroed()
    //求出每个时刻的倒数项目
    hys.forEach(hy => {
      let { xs, ys, st, yt, lastSt } = hy
      let dyt = yt.atomicOperation((item, r, c) => (item - ys.get(r, c)) * this.afd(item, 'Softmax'))

      let dst = dyt.multiply(this.V)
      dst = dst.atomicOperation((item, r, c) => item * this.afd(st.get(r, c), 'Tanh'))

      let ndv = dyt.T.multiply(st)
      let ndu = dst.T.multiply(xs)
      let ndw = dst.T.multiply(lastSt)

      dv = dv.addition(ndv)
      du = du.addition(ndu)
      dw = dw.addition(ndw)
    })

    // //更新
    let rate = 0.01
    this.U = this.U.subtraction(du.numberMultiply(rate))
    this.W = this.W.subtraction(dw.numberMultiply(rate))
    this.V = this.V.subtraction(dv.numberMultiply(rate))
  }

  predict() {
    let input = 'gou'.split('')
    let data = this.onehot(input)
    let hys = this.forwardPropagation(data)
    console.log(this.showWords(hys))
  }

  maxIndex(d: number[]) {
    var max = d[0]
    var index = 0
    for (var i = 0; i < d.length; i++) {
      if (d[i] > max) {
        max = d[i]
        index = i
      }
    }
    return index
  }

  showWords(hys: { xs: Matrix, ys: Matrix, st: Matrix, yt: Matrix, lastSt: Matrix }[]) {
    return hys.map(hy => {
      let index = this.maxIndex(hy.yt.getRow(0))
      if (!this.wordIndex[index]) debugger
      return this.wordIndex[index]
    })
  }

  cost(hys: { xs: Matrix, ys: Matrix, st: Matrix, yt: Matrix, lastSt: Matrix }[]) {
    let m = hys.map(hy => {
      let { yt, ys } = hy
      let tmp = yt.subtraction(ys).atomicOperation(item => (item ** 2) / 2).getRow(0)
      return tmp.reduce((a, b) => a + b) / tmp.length
    })
    return m.reduce((a, b) => a + b)
  }

  onehot(input: string[]) {
    return input.map((s, i) => {
      let inputIndex = this.indexWord[s]
      let nextWord = input[i + 1] ? input[i + 1] : '/n'
      let outoutIndex = this.indexWord[nextWord]
      let xs = this.oneHotXs(inputIndex)
      let ys = this.oneHotYs(outoutIndex)
      return { xs, ys }
    })
  }

  fit() {
    for (let i = 0; i < 5000; i++) {
      let e = 0
      for (let n = 0; n < this.trainData.length; n++) {
        let input = this.trainData[n]
        let data = this.onehot(input)
        let hys = this.forwardPropagation(data)
        this.backPropagation(hys)
        e += this.cost(hys)
      }
      if (i % 100 === 0) console.log('enpoch: ', i, 'loss: ', e / this.trainData.length)
    }
  }
}