# infers
Machine learning and Matrix operation library by TypeScript.
- [XOR EXAMPLE](https://badgua.gitee.io/infers)
- [API DOC](https://badgua.gitee.io/infers/api/)

![](https://gitee.com/badgua/infers/raw/main/docs/net.png)

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
// Matrix 3x3 [
//  1, 2, 0, 
//  5, 4, -2, 
//  0, -1, 0, 
// ]
```
BP neural network example of XOR, three-layer network: 
```ts
let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
let ys = new Matrix([[1], [1], [0], [0]])
let model = new BPNet([2, [6, 'Tanh'], [1, 'Sigmoid']], { rate: 0.1 })
model.fit(xs, ys, {
  epochs: 5000, onEpoch: (epoch, loss) => {
    if (epoch % 100 === 0) console.log('epoch:' + epoch, 'loss:', loss)
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
    console.log('epoch:' + epoch, 'loss:', loss)
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
RNN: Recurrent neural network example:
```ts
let trainData = ['hello rnn', 'good morning', 'I love 🍎!', 'I eat 🍊!']
let net = new RNN({ trainData })
net.fit({
  epochs: 1500, onEpochs: (epoch, loss) => {
    if (epoch % 10 === 0) console.log('epoch: ', epoch, 'loss: ', loss)
  }
})
console.log(net.predict('I love'))
console.log(net.predict('I eat'))
console.log(net.predict('hel'))
console.log(net.predict('good'))
//  🍊!/n
//  🍎!/n
// lo rnn/n
//  morning/n
```

## Parameter introduction: 
 - **shape**: The hierarchical structure of the network model, It includes the number of neurons in each layer, the type of activation function and the total number of layers.
 - **rate**: The learning rate is the update step of every gradient descent, generally between 0 and 1.
 - **epochs**: All the data of the whole training set are iterated once.
 - **mode**: training modes, sgd | bgd | mbgd.

The selection of the above。parameters is also the process of model optimization. The learning rate, training times and network shape needed to deal with different problems are different, which need to be adjusted according to each solution of the cost function.

## Export
- class Matrix
  - Mathematical operation of matrix
  - addition, subtraction, multiply, transpose, determinant
- class BPNet
  - Fully connected neural network
  - Multi-layer network model
- class RNN
  - Recurrent neural network
  - Used natural language processing