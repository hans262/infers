import { RegressionModel, Matrix, LogisticModel } from '../src'

export namespace TestModel {
  export function regressionModel() {
    // data
    const xs = new Matrix([[1], [2], [3], [4]])
    const ys = new Matrix([[1], [3], [5], [7]])
    // create
    const model = new RegressionModel(xs, ys)
    model.setRate(0.01)
    // fit
    model.fit(5000, (batch) => {
      if (batch % 500 === 0) {
        console.log(batch, model.cost())
      }
    })
    // predict
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
        console.log(batch, model.cost())
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
    const ys = new Matrix([[0], [1], [0], [1]])
    const model = new LogisticModel(xs, ys)
    model.setRate(0.01)
    model.fit(1000000, (batch) => {
      if (batch % 1000 === 0) {
        console.log(batch, model.cost())
      }
    })
    const xs2 = new Matrix([[45, 1], [66, 1], [23, 0], [55, 0]])
    model.predict(xs2).print()
  }

  export function run() {
    logisticModel2()
  }
}

TestModel.run()