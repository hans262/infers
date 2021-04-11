import { Matrix, BPNet } from '../src'

export function addition() {
  let xs = new Matrix([[1, 4], [3, 2], [6, 5], [4, 7]])
  let ys = new Matrix([[5], [5], [11], [11]])
  let model = new BPNet([2, 6, 6, 1], { mode: 'bgd', rate: 0.01 })
  model.fit(xs, ys, {
    epochs: 500,
    onEpoch: (epoch, loss) => {
      console.log('epoch = ' + epoch, loss)
    },
    onTrainEnd: loss => {
      console.log('train end', loss)
    }
  })
  let xs2 = new Matrix([[5, 8], [22, 6], [-5, 9], [-5, -4]])
  model.predict(xs2).print()
}