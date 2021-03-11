# infers
Matrix operation and machine learning library.

## Installed
本软件包采用npm源管理，也可将npm切换到淘宝源下载。
```shell
$ npm install infers@latest
```

在项目中使用：
```ts
import { Model, Matrix } from 'infers'
let a = new Matrix([
  [3, -7, 8, 9, -6],
  [0, 2, -5, 7, 3],
  [0, 0, 1, 5, 0],
  [0, 0, 2, 4, -1],
  [0, 0, 0, -2, 0]
])
a.print()
// solving det
console.log(a.det())
```
使用线性回归模型：
```ts
// data
const xs = new Matrix([[1], [2], [3], [4]])
const ys = new Matrix([[1], [3], [5], [7]])
// create model
const model = new Model(xs, ys)
model.setRate(0.001)
// fit
model.fit(10000, (batch) => {
  if (batch % 500 === 0) {
    console.log(batch, model.cost())
  }
})
//predict
const xs2 = new Matrix([[5]])
model.predict(xs2).print()
```

## Export
- class Matrix
  - 加法、乘法、数乘、转置
  - 行列式
- class Model
  - 线性回归模型