import { RegressionModel, Matrix, LogisticModel, BPNet } from '../src'

export namespace TestModel {
  export function regressionModel() {
    const xs = new Matrix([[1], [2], [3], [4]])
    const ys = new Matrix([[1], [3], [5], [7]])
    const model = new RegressionModel(xs, ys)
    model.setRate(0.01)
    model.fit(5000, (batch) => {
      if (batch % 500 === 0) {
        console.log(batch, model.cost()[0])
      }
    })
    const xs2 = new Matrix([[5], [20]])
    model.predict(xs2).print()
  }

  export function logisticModel() {
    const xs = new Matrix([[1], [2], [3], [4]])
    const ys = new Matrix([[0], [0], [1], [1]])
    const model = new LogisticModel(xs, ys)
    model.setRate(0.01)
    model.fit(50000, (batch) => {
      if (batch % 500 === 0) {
        console.log(batch, model.cost()[0])
      }
    })
    const xs2 = new Matrix([[20], [30], [-2], [0], [3], [2]])
    model.predict(xs2).print()
  }

  export function logisticModel2() {
    const xs = new Matrix([
      [58, 1],
      [62, 1],
      [48, 0],
      [52, 0]
    ])
    const ys = new Matrix([[0, 1], [1, 0], [0, 1], [1, 0]])
    const model = new LogisticModel(xs, ys)
    model.setRate(0.01)
    model.fit(100000, (batch) => {
      if (batch % 1000 === 0) {
        console.log(batch, model.cost()[0])
      }
    })
    const xs2 = new Matrix([[45, 1], [66, 1], [23, 0], [55, 0]])
    model.predict(xs2).print()
  }

  export function logisticModel3() {
    const xs = new Matrix([
      [-2], [-1], [1], [2], [3], [4]
    ])
    const ys = new Matrix([
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 1]
    ])
    const model = new LogisticModel(xs, ys)
    model.setRate(0.01)
    model.fit(100000, (batch) => {
      if (batch % 500 === 0) {
        console.log(batch, model.cost()[0])
      }
    })
    const xs2 = new Matrix([[-18], [1.5], [3.5]])
    model.predict(xs2).print()
  }

  function bpNet1() {
    let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
    let ys = new Matrix([[1], [1], [0], [0]])
    let model = new BPNet([2, 3, 1], 'Sigmoid')
    model.setRate(0.5)
    model.fit(xs, ys, 10000, (batch, loss) => {
      if (batch % 500 === 0) console.log(batch, loss)
    })
    model.predict(xs)[2].print()
  }

  function bpNet2() {
    let xs = new Matrix([[1, 4], [3, 2], [6, 5], [4, 7]])
    let ys = new Matrix([[5], [5], [11], [11]])
    let model = new BPNet([2, 5, 3, 1])
    model.setRate(0.001)
    model.fit(xs, ys, 1000, (batch, loss) => {
      if (batch % 10 === 0) console.log(batch, loss)
    })
    let xs2 = new Matrix([[5, 8], [22, 6]])
    model.predict(xs2)[3].print()
  }

  export function run() {
    bpNet2()
  }
}

TestModel.run()