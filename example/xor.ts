import { Matrix, BPNet } from '../src'

export function xor() {
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