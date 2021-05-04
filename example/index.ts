import { addition } from './addition'
import { iris, saveIris } from './iris'
import { xor } from './xor'
import { RNN } from './brnn/rnn'
import { MRNN } from './mrnn'

export namespace TestMatrix {
  function brnn() {
    const net = new RNN({
      hiddenLayers: [20, 16],
      learningRate: 0.3
    })
    const xs = [
      '今天天气怎么样',
      '你吃饭了吗',
      '下午要去银行一趟',
      '也许你真的不怎样呢',
      '你到低在哪儿'
    ]
    net.train(xs, { log: true, iterations: 300 })
    console.log(net.run('今天天气'))
    console.log(net.run('下午要去'))
    console.log(net.run('也许'))
    console.log(net.run('到底'))
  }

  function mrnn() {
    let data = '今天天气怎么样我发现你是真的很皮呢'
    let net = new MRNN(data)
    net.fit()
    net.predict()
  }

  export function run() {
    // addition()
    // xor()
    // iris()
    // saveIris()

    // brnn()

    mrnn()
  }
}
TestMatrix.run()