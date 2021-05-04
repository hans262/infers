export class MRNN {
  U: number = Math.random() //隐藏层权值
  W: number = Math.random() //上一时刻连接隐藏层权值
  V: number = Math.random() //输出层权值

  lastSt = 0 //上一时刻隐藏层输出值

  indexWord: { [index: string]: number } = {} // 词典索引
  wordIndex: { [index: number]: string } = {} // 索引词典
  trainData: string[] //训练数据

  constructor(data: string) {
    this.trainData = data.split('')
    let set = Array.from(new Set(this.trainData))
    for (let i = 0; i < set.length; i++) {
      this.indexWord[set[i]] = i
      this.wordIndex[i] = set[i]
    }
  }

  // 向前传播
  forwardPropagation(xs: number) {
    let st = xs * this.U + this.lastSt * this.W
    // st = 1 / (1 + Math.exp(-st))
    let yt = this.V * st
    //返回每一层的值
    return [xs, st, yt]
  }

  // 反向传播
  backPropagation(hy: number[], ys: number) {
    let dyt = hy[2] - ys
    let dst = dyt * this.V
    // dst = dst * (1 - dst)

    let dv = dyt * hy[1]
    let dw = dst * this.lastSt
    let du = dst * hy[0]

    //更新
    let rate = 0.001
    this.U = this.U - rate * du
    this.W = this.W - rate * dw
    this.V = this.V - rate * dv

    this.lastSt = hy[1]
  }

  predict() {
    let xs = this.indexWord['今']
    let hy = this.forwardPropagation(xs)
    console.log(hy[2])
  }

  fit() {
    for (let i = 0; i < 50000; i++) {
      this.lastSt = 0
      let e = 0
      for (let n = 0; n < this.trainData.length; n++) {
        let input = this.trainData[n]
        let output = this.trainData[n + 1]
        if (!output) break; //有可能是结尾

        let xs = this.indexWord[input]
        let ys = this.indexWord[output]

        let hy = this.forwardPropagation(xs)
        this.backPropagation(hy, ys)
        e += ((hy[2] - ys) ** 2) / 2
      }
      if (i % 100 === 0) console.log('enpoch: ', i, 'loss: ', e / this.trainData.length)
    }
  }
}