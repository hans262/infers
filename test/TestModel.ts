import { Model, Matrix } from '../src'

export namespace TestModel {
  export function model() {
    // data
    const xs = new Matrix([[1], [2], [3], [4]])
    const ys = new Matrix([[1], [3], [5], [7]])
    xs.print()
    ys.print()
    // create
    const model = new Model(xs, ys)
    model.setRate(0.01)
    // fit
    model.fit(1000, (batch) => {
      if (batch % 50 === 0) {
        console.log(batch, model.cost())
      }
    })
    // predict
    const xs2 = new Matrix([[5]])
    model.predict(xs2).print()
  }
  export function run() {
    model()
  }
}

TestModel.model()