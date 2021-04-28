import { Matrix, BPNet } from '../src'

export function xor() {
  let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
  let ys = new Matrix([[1, 0], [1, 0], [0, 1], [0, 1]])
  let model = new BPNet([2, [6, 'Tanh'], [6, 'Tanh'], [2, 'Softmax']], { rate: 0.01, mode: 'sgd' })
  model.fit(xs, ys, {
    epochs: 2000, onEpoch: (epoch, loss) => {
      if (epoch % 100 === 0) console.log('epoch:' + epoch, 'loss:', loss)
    }
  })
  model.predict(xs).print()
}