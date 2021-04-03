import { Matrix, BPNet, Model } from '../src'
import { data } from './SepalData'

export namespace TestModel {
  export function bpNet1() {
    let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
    let ys = new Matrix([[1], [1], [0], [0]])
    let model = new BPNet([2, [6, 'Tanh'], [1, 'Sigmoid']], { rate: 0.1 })
    model.fit(xs, ys, {
      epochs: 5000, onEpoch: (epoch, loss) => {
        if (epoch % 100 === 0) console.log('epoch = ' + epoch, loss)
      }
    })
    model.predict(xs).print()
  }
  export function bpNet2() {
    let xs = new Matrix([[1, 4], [3, 2], [6, 5], [4, 7]])
    let ys = new Matrix([[5], [5], [11], [11]])
    let model = new BPNet([2, 6, 6, 1], { mode: 'bgd', rate: 0.01 })
    model.fit(xs, ys, {
      epochs: 500, onEpoch: (epoch, loss) => {
        console.log('epoch = ' + epoch, loss)
      }
    })
    let xs2 = new Matrix([[5, 8], [22, 6], [-5, 9], [-5, -4]])
    model.predict(xs2).print()
  }
  export function bpNet3() {
    let xs = new Matrix(data.map(d => d[0]))
    let ys = new Matrix(data.map(d => [
      d[1][0] === 'setosa' ? 1 : 0,
      d[1][0] === "virginica" ? 1 : 0,
      d[1][0] === "versicolor" ? 1 : 0
    ]))
    let model = new BPNet(
      [4, [8, 'Relu'], [8, 'Sigmoid'], [3, 'Softmax']],
      { mode: 'mbgd', rate: 0.3 }
    )
    model.fit(xs, ys, {
      epochs: 100,
      batchSize: 10,
      onBatch: (batch, size, loss) => {
        console.log('batch = ' + batch, size, loss)
      },
      onEpoch: (epoch, loss) => {
        console.log('epoch = ' + epoch, loss)
      }
    })
    let xs2 = new Matrix([
      [4.4, 2.9, 1.4, 0.2], // Setosa     [1, 0, 0]
      [6.4, 3.2, 4.5, 1.5], // Versicolor [0, 0, 1]
      [5.8, 2.7, 5.1, 1.9], // Virginica  [0, 1, 0]
    ])
    model.predict(xs2).print()
  }
  function bpNet4(){
    let path = 'D://develop/infers/debug/model.json'
    let model = Model.loadFile(path)
    let xs2 = new Matrix([
      [4.4, 2.9, 1.4, 0.2], // Setosa     [1, 0, 0]
      [6.4, 3.2, 4.5, 1.5], // Versicolor [0, 0, 1]
      [5.8, 2.7, 5.1, 1.9], // Virginica  [0, 1, 0]
    ])
    model.predict(xs2).print()
  }
  export function run() {
    bpNet4()
  }
}
TestModel.run()