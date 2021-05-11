import { Matrix } from '../src'
import type {
  ActivationFunction, RNNOptions,
  RNNTrainingOptions, RNNForwardResult
} from './types'

export class RNN {
  U: Matrix // input -> hiden
  W: Matrix // last hiden -> hiden
  V: Matrix // hiden -> output

  indexWord: { [index: string]: number } = {}
  wordIndex: { [index: number]: string } = {}
  trainData: string[][]
  inputSize: number
  hidenSize = 10

  firstSt: Matrix
  rate = 0.01
  constructor(opt: RNNOptions) {
    this.trainData = opt.trainData.map(v => v.split(''))
    if (opt.rate) this.rate = opt.rate

    let temp = Array.from(new Set(this.trainData.flat(1)))
    for (let i = 0; i < temp.length; i++) {
      this.indexWord[temp[i]] = i
      this.wordIndex[i] = temp[i]
    }
    this.inputSize = temp.length
    this.wordIndex[temp.length] = '/n'
    this.indexWord['/n'] = temp.length

    let outputSize = this.inputSize + 1
    this.U = Matrix.generate(this.hidenSize, this.inputSize)
    this.W = Matrix.generate(this.hidenSize, this.hidenSize)
    this.V = Matrix.generate(outputSize, this.hidenSize)

    this.firstSt = Matrix.generate(1, this.hidenSize, 0)
  }

  afn(x: number, rows: number[], af?: ActivationFunction) {
    switch (af) {
      case 'Tanh':
        return Math.tanh(x)
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
        return 1 - Math.tanh(x) ** 2
      case 'Softmax':
      default:
        return 1
    }
  }

  // encode xs
  oneHotX(inputIndex: number) {
    let xs = Matrix.generate(1, this.inputSize, 0)
    xs.update(0, inputIndex, 1)
    return xs
  }

  oneHotXs(input: string[]) {
    return input.map(s => {
      let nowIndex = this.indexWord[s]
      return this.oneHotX(nowIndex)
    })
  }

  // encode ys
  oneHotY(outputIndex: number) {
    let ys = Matrix.generate(1, this.inputSize + 1, 0)
    ys.update(0, outputIndex, 1)
    return ys
  }

  oneHotYs(input: string[]) {
    return input.map((_, i) => {
      let nextWord = input[i + 1] ? input[i + 1] : '/n'
      let nextIndex = this.indexWord[nextWord]
      return this.oneHotY(nextIndex)
    })
  }

  forwardPropagation(xs: Matrix[]): RNNForwardResult[] {
    let result: RNNForwardResult[] = []
    for (let i = 0; i < xs.length; i++) {
      let xst = xs[i]
      let lastSt = i === 0 ? this.firstSt : result[i - 1].st
      let { st, yt } = this.calcForward(xst, lastSt)
      result.push({ st, yt })
    }
    return result
  }

  calcForward(xs: Matrix, lastSt = this.firstSt) {
    let st = xs.multiply(this.U.T).addition(lastSt.multiply(this.W.T))
    st = st.atomicOperation((item, i) => this.afn(item, st.getRow(i), 'Tanh'))
    let yt = st.multiply(this.V.T)
    yt = yt.atomicOperation((item, i) => this.afn(item, yt.getRow(i), 'Softmax'))
    return { st, yt }
  }

  backPropagation(hy: RNNForwardResult[], xs: Matrix[], ys: Matrix[]) {
    let dv = this.V.zeroed()
    let du = this.U.zeroed()
    let dw = this.W.zeroed()
    //求出每个时刻的导数项目
    for (let i = 0; i < hy.length; i++) {
      let { st, yt } = hy[i]
      let xst = xs[i]
      let yst = ys[i]

      let lastSt = i === 0 ? this.firstSt : hy[i - 1].st
      let dyt = yt.atomicOperation((item, r, c) => (item - yst.get(r, c)) * this.afd(item, 'Softmax'))

      let dst = dyt.multiply(this.V)
      dst = dst.atomicOperation((item, r, c) => item * this.afd(st.get(r, c), 'Tanh'))

      let ndv = dyt.T.multiply(st)
      let ndu = dst.T.multiply(xst)
      let ndw = dst.T.multiply(lastSt)

      dv = dv.addition(ndv)
      du = du.addition(ndu)
      dw = dw.addition(ndw)
    }

    this.U = this.U.subtraction(du.numberMultiply(this.rate))
    this.W = this.W.subtraction(dw.numberMultiply(this.rate))
    this.V = this.V.subtraction(dv.numberMultiply(this.rate))
  }

  /**
   * @param input 
   * @param max 最大返回字符数
   */
  predict(input: string, max: number = 10) {
    let data = input.split('')
    let s = data.find(d => this.indexWord[d] === undefined)
    //检测 没有在词典中的单词
    if (s) {
      console.error(`检测到有未在词典中的字：${s}`)
      return undefined
    }

    let xs = this.oneHotXs(data)
    let hy = this.forwardPropagation(xs)
    let lastHy = hy[hy.length - 1]

    let nextIndex = lastHy.yt.argMax(0)
    let nextSt = lastHy.st

    let result = ''
    result += this.wordIndex[nextIndex]

    if (nextIndex === this.inputSize) return result

    for (let i = 0; i < max - 1; i++) {
      let nextXs = this.oneHotX(nextIndex)
      let hy = this.calcForward(nextXs, nextSt)
      nextIndex = hy.yt.argMax(0)
      nextSt = hy.st
      result += this.wordIndex[nextIndex]
      if (nextIndex === this.inputSize) break
    }
    return result
  }

  cost(hy: RNNForwardResult[], ys: Matrix[]) {
    let res = hy.map((nhy, i) => {
      let { yt } = nhy
      let yst = ys[i]
      let tmp = yt.subtraction(yst).atomicOperation(item => (item ** 2) / 2).getRow(0)
      return tmp.reduce((a, b) => a + b) / tmp.length
    })
    return res.reduce((a, b) => a + b)
  }

  fit(opt: RNNTrainingOptions = {}) {
    const { epochs = 1000, onEpochs } = opt
    for (let i = 0; i < epochs; i++) {
      let e = 0
      for (let n = 0; n < this.trainData.length; n++) {
        let input = this.trainData[n]
        let xs = this.oneHotXs(input)
        let ys = this.oneHotYs(input)
        let hy = this.forwardPropagation(xs)
        this.backPropagation(hy, xs, ys)
        e += this.cost(hy, ys)
      }
      if (onEpochs) {
        onEpochs(i, e / this.trainData.length)
      }
    }
  }
}