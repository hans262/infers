import { SeqModel, Matrix, BPNet } from '../src'

export namespace TestModel {
  export function seq1() {
    const xs = new Matrix([[1], [2], [3], [4]])
    const ys = new Matrix([[1], [3], [5], [7]])
    const model = new SeqModel([1, 1])
    model.setRate(0.01)
    model.fit(xs, ys, 5000, (batch, loss) => {
      if (batch % 500 === 0) console.log(batch, loss)
    })
    const xs2 = new Matrix([[5], [20]])
    model.predict(xs2).print()
  }
  export function seq2() {
    const xs = new Matrix([[1], [2], [3], [4]])
    const ys = new Matrix([[0], [0], [1], [1]])
    const model = new SeqModel([1, 1], 'Sigmoid')
    model.setRate(0.01)
    model.fit(xs, ys, 50000, (batch, loss) => {
      if (batch % 500 === 0) console.log(batch, loss)
    })
    const xs2 = new Matrix([[20], [30], [-2], [0], [3], [2]])
    model.predict(xs2).print()
  }
  export function seq3() {
    const xs = new Matrix([[58, 1], [62, 1], [48, 0], [52, 0]])
    const ys = new Matrix([[0, 1], [1, 0], [0, 1], [1, 0]])
    const model = new SeqModel([2, 2], 'Sigmoid')
    model.setRate(0.01)
    model.fit(xs, ys, 100000, (batch, loss) => {
      if (batch % 1000 === 0) console.log(batch, loss)
    })
    const xs2 = new Matrix([[45, 1], [66, 1], [23, 0], [55, 0]])
    model.predict(xs2).print()
  }
  export function seq4() {
    const xs = new Matrix([[-2], [-1], [1], [2], [3], [4]])
    const ys = new Matrix([
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 1]
    ])
    const model = new SeqModel([1, 3], 'Sigmoid')
    model.setRate(0.01)
    model.fit(xs, ys, 100000, (batch, loss) => {
      if (batch % 1000 === 0) console.log(batch, loss)
    })
    const xs2 = new Matrix([[-18], [1.5], [3.5]])
    model.predict(xs2).print()
  }
  export function bpNet1() {
    let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
    let ys = new Matrix([[1], [1], [0], [0]])
    let model = new BPNet([2, [6, 'Tanh'], [1, 'Sigmoid']])
    model.setRate(0.1)
    model.fit(xs, ys, 10000, (batch, loss) => {
      if (batch % 500 === 0) console.log(batch, loss)
    })
    model.predict(xs)[2].print()
  }
  export function bpNet2() {
    let xs = new Matrix([[1, 4], [3, 2], [6, 5], [4, 7]])
    let ys = new Matrix([[5], [5], [11], [11]])
    let model = new BPNet([2, 6, 6, 1])
    model.setRate(0.01)
    model.fit(xs, ys, 10000, (batch, loss) => {
      if (batch % 500 === 0) console.log(batch, loss)
    })
    let xs2 = new Matrix([[5, 8], [22, 6], [-5, 9], [-5, -4]])
    model.predict(xs2)[3].print()
  }

  export function run() {
    bpNet2()
  }
}

TestModel.run()