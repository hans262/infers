export class MRNN {
  U: number = Math.random() //上一时刻隐藏层权值
  W: number = Math.random() //输出层权值
  V: number = Math.random() //隐藏层权值
  lastDu = 0
  lastDw = 0
  lastDv = 0
  lastHt: number = 0 //上一时刻隐藏层的输出
  lastDHt: number = 0 //上一次的ht倒数

  dictWord: { [index: string]: number } = {} //词典索引
  trainData: string[] //训练数据
  constructor(trainData: string[]) {
    let temp0 = trainData.map(s => s.split(''))
    this.trainData = temp0.flat(1)
    let temp2 = Array.from(new Set(this.trainData))
    for (let i = 0; i < temp2.length; i++) {
      this.dictWord[temp2[i]] = i
    }
  }

  //输入字符的索引
  calcnet(xs: number) {
    let ht = xs * this.V + this.lastHt * this.U
    let yt = this.W * ht
    this.lastHt = ht
    return [xs, ht, yt]
  }

  backPropagation(hy: number[], ys: number) {
    //求导
    let dyt = hy[2] - ys
    let dht = this.W * dyt + this.lastDHt * this.U

    let dw = hy[1] * dyt + this.lastDw
    let du = this.lastHt * dht + this.lastDu
    let dv = hy[0] * dht + this.lastDv

    this.lastDHt = dht
    this.lastDw = dw
    this.lastDu = du
    this.lastDv = dv

    //更新
    let rate = 0.01
    this.U = this.U - rate * du
    this.W = this.W - rate * dw
    this.V = this.V - rate * dv

  }

  predict(input: string) {
    let xs = this.dictWord[input]
    let hy = this.calcnet(xs)
    console.log(hy[2])
  }

  fit() {
    for (let i = 0; i < 1; i++) {
      this.lastHt = 0
      this.lastDu = 0
      this.lastDw = 0
      this.lastDv = 0
      this.lastDHt = 0
      for (let n = 0; n < this.trainData.length; n++) {
        let input = this.trainData[n]
        let output = this.trainData[n + 1]
        let xs = this.dictWord[input]
        let ys = this.dictWord[output]
        let hy = this.calcnet(xs)
        this.backPropagation(hy, ys)

        let e = ((hy[2] - ys) ** 2) / 2
        console.log('e: ', e)
      }
    }
  }
}