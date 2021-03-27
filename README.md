# infers
The library of machine learning and matrix operation by Typescript.

The Chinese README is [here](https://gitee.com/badgua/infers/blob/main/cn.md).

## Installed
Make sure NPM is installed, Switch to the project directory and execute the following command.
```shell
$ npm install infers@latest
```
Then reference in the project: 
```ts
import { Matrix, BPNet, SeqModel } from 'infers'
```

## Examples
Matrix transpose: 
```ts
let m = new Matrix([
  [1, 5, 0],
  [2, 4 , -1],
  [0, -2, 0]
])
m.T.print()
```
BP neural network XOR example, three-layer network model: 
```ts
let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
let ys = new Matrix([[1], [1], [0], [0]])
let model = new BPNet([2, [6, 'Tanh'], [1, 'Sigmoid']])
model.setRate(0.1)
model.fit(xs, ys, 10000, (batch, loss) => {
  if (batch % 500 === 0) console.log(batch, loss)
})
model.predict(xs)[2].print()
// Matrix 4x1 [
//  0.9862025352830867, 
//  0.986128496195502, 
//  0.01443800549676924, 
//  0.014425871504885788, 
// ]
```
BP neural network addition example, four-layer network model: 
```ts
let xs = new Matrix([[1, 4], [3, 2], [6, 5], [4, 7]])
let ys = new Matrix([[5], [5], [11], [11]])
let model = new BPNet([2, 6, 6, [1, 'Relu']])
model.setRate(0.001)
model.fit(xs, ys, 10000, (batch, loss) => {
  if (batch % 500 === 0) console.log(batch, loss)
})
let xs2 = new Matrix([[5, 8], [22, 6], [-5, 9]])
model.predict(xs2)[3].print()
// Matrix 2x1 [
//  12.994745740521667, 
//  27.99134620596921, 
//  3.9987224114576856, 
// ]
```
Sequence model, only input layer and output layer model, support linear regression and logical classification.
```ts
const xs = new Matrix([[1], [2], [3], [4]])
const ys = new Matrix([[1], [3], [5], [7]])
const model = new SeqModel([1, 1])
model.setRate(0.01)
model.fit(xs, ys, 5000, (batch, loss) => {
  if (batch % 500 === 0) console.log(batch, loss)
})
const xs2 = new Matrix([[5], [20]])
model.predict(xs2).print()
```
Parameter introduction: 
 - **shape**: The network hierarchical structure of the model includes the number of neurons in each layer, the type of activation function in each layer and the total number of layers. The more complex the network structure is, the more computation is needed for a single training, and it is easy to cause over fitting.
 - **rate**: Learning rate is also known as training step. The lower the learning rate, the more training times are needed to achieve the optimal cost function. If the learning rate is too large, it may cross the optimal cost function due to too large span, resulting in the loss value approaching the positive infinite model and the problem that the model cannot converge.
 - **batch**: All the data of the whole training set are iterated once.

The selection of the above parameters is also the process of model optimization. The learning rate, training times and network shape needed to deal with different problems are different, which need to be adjusted according to each solution of the cost function.

## Export
- class Matrix
  - Add, multiply, multiply, transpose
  - Determinant, normalization
- class SeqModel
  - Two layer network model
  - Linear regression and logical classification
- class BPNet
  - Multi layer network model
  - Support multiple activation functions
  - Support classification and regression