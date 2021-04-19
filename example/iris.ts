import { Matrix, BPNet, loadBPNet } from '../src'
import { data } from './irisData'
import * as  fs from 'fs'

export function iris() {
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
    },
    onTrainEnd: loss => {
      console.log('train end', loss)
    }
  })
  let xs2 = new Matrix([
    [4.4, 2.9, 1.4, 0.2], // Setosa     [1, 0, 0]
    [6.4, 3.2, 4.5, 1.5], // Versicolor [0, 0, 1]
    [5.8, 2.7, 5.1, 1.9], // Virginica  [0, 1, 0]
  ])
  model.predict(xs2).print()
  return model
}

export function saveIris() {
  let path = '/desktop/develop/infers/example/model.json'
  let path2 = '/Users/macbookair/Desktop/develop/infers/example/model.json'
  //save
  let modelJson = iris().toJSON()
  fs.writeFileSync(path2, modelJson)
  //load
  let modelJson2 = fs.readFileSync(path2).toString()
  let model = loadBPNet(modelJson2)
  //test
  let xs2 = new Matrix([
    [4.4, 2.9, 1.4, 0.2], // Setosa     [1, 0, 0]
    [6.4, 3.2, 4.5, 1.5], // Versicolor [0, 0, 1]
    [5.8, 2.7, 5.1, 1.9], // Virginica  [0, 1, 0]
  ])
  model.predict(xs2).print()
}