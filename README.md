# infers
Matrix operation and machine learning library by TypesScript.

Here is an example of [XOR](https://badgua.gitee.io/infers/xor.html).

![net](https://github.com/ounana/infers/raw/main/browser/net.png)

## Installed
Make sure NPM is installed, Switch to the project directory then execute the following command.
```shell
$ npm install infers@latest
```
Reference in project:
```ts
import { Matrix, BPNet } from 'infers'
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
BP neural network example of XOR, three-layer network: 
```ts
let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
let ys = new Matrix([[1], [1], [0], [0]])
let model = new BPNet([2, [6, 'Tanh'], [1, 'Sigmoid']], { rate: 0.1 })
model.fit(xs, ys, {
  epochs: 5000, onEpoch: (epoch, loss) => {
    if (epoch % 100 === 0) console.log('epoch = ' + epoch, loss)
  }
})
model.predict(xs).print()
// Matrix 4x1 [
//  0.9862025352830867, 
//  0.986128496195502, 
//  0.01443800549676924, 
//  0.014425871504885788, 
// ]
```
BP neural network example of addition, four-layer network: 
```ts
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
// Matrix 2x1 [
//  12.994745740521667, 
//  27.99134620596921, 
//  3.9987224114576856, 
//  -9.000000644547901,
// ]
```
Parameter introduction: 
 - **shape**: The network hierarchical structure of the model includes the number of neurons in each layer, the type of activation function in each layer and the total number of layers. The more complex the network structure is, the more computation is needed for a single training, and it is easy to cause over fitting.
 - **rate**: Learning rate is also known as training step. The lower the learning rate, the more training times are needed to achieve the optimal cost function. If the learning rate is too large, it may cross the optimal cost function due to too large span, resulting in the loss value approaching the positive infinite model and the problem that the model cannot converge.
 - **epochs**: All the data of the whole training set are iterated once.
 - **mode**: Three gradient descent methods are SGD, BGD and MBGD.

The selection of the above parameters is also the process of model optimization. The learning rate, training times and network shape needed to deal with different problems are different, which need to be adjusted according to each solution of the cost function.

## Export
- class Matrix
  - addition, subtraction, multiply, transpose
  - determinant, normalization
- class BPNet
  - Multi-layer network model of CNN
  - Support multiple activation functions
  - Linear regression and Logical classification