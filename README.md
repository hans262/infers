# infers
Matrix operation and machine learning library by TypesScript.

## Installed
确保npm已经安装，切换到项目目录执行以下命令。
```shell
$ npm install infers@latest
```

在项目中引用：
```ts
import { Matrix } from 'infers'
let m = new Matrix([
  [1, 5, 0],
  [2, 4 ,-1],
  [0, -2, 0]
])
//transposition
m.T.print()
```
BP网络
```ts
let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
let ys = new Matrix([[1], [1], [0], [0]])
let model = new NeuralNetwork([2, 3, 1])
model.fit(xs, ys, 50000, (batch, loss) => {
  if (batch % 1000 === 0) {
    console.log(batch, loss)
  }
})
//打印最后一层的输出
model.predict(xs)[2].print()
// Matrix 4x1 [
//  0.9862025352830867, 
//  0.986128496195502, 
//  0.01443800549676924, 
//  0.014425871504885788, 
// ]
```

线性回归模型：
```ts
// sample
const xs = new Matrix([[1], [2], [3], [4]])
const ys = new Matrix([[1], [3], [5], [7]])
// create model
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
```
逻辑分类模型：
```ts
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
```
多分类：
```ts
const xs = new Matrix([
  [-2], [-1], [1], [2], [3], [4]
])
const ys = new Matrix([
  [1, 0, 0],
  [1, 0, 0],
  [0, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [0, 0, 1]
])
const model = new LogisticModel(xs, ys)
// ...
```
学习率和训练次数的选择，需根据代价函数的每次求解来判定，每种模型所需要的学习率和训练次数各不相同，越低的学习率也就需要相对较多的训练次数才能达到代价函数最优，过高也可能造成跨度太大而越过最优点，造成损失值趋近于正无穷模型无法收敛的问题。

## Export
- class Matrix
  - 加法、乘法、数乘、转置
  - 行列式、归一化
- class Model
  - 线性回归模型
  - 分类模型