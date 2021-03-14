# infers
Matrix operation and machine learning library.

## Installed
使用npm的方式安装到项目中，可切换淘宝源下载。
```shell
$ npm install infers@latest
```

在项目中使用：
```ts
import { Matrix } from 'infers'
let m = new Matrix([
  [1, 5, 0],
  [2, 4 ,-1],
  [0, -2, 0]
])
m.print()
// solving det
console.log(m.det())
```
线性回归模型：
```ts
// data
const xs = new Matrix([[1], [2], [3], [4]])
const ys = new Matrix([[1], [3], [5], [7]])
// create
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
分类模型：
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
学习率和训练次数的设定，要根据代价函数的每次求解来判定，每种模型需要的学习率和训练次数各不相同，学习率越低也就需要相对较多的训练次数才能达到代价函数最优，学习率过高也可能造成跨度太大越过最优解，最后损失值趋近于正无穷，造成模型无法收敛的问题。

## Export
- class Matrix
  - 加法、乘法、数乘、转置
  - 行列式
- class Model
  - 线性回归模型
  - 分类模型