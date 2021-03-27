# infers
Matrix operation and machine learning library by TypesScript.

## Installed
确保npm已经安装，切换到项目目录执行以下命令。
```shell
$ npm install infers@latest
```
然后在项目中引用：
```ts
import { Matrix, BPNet, SeqModel } from 'infers'
```
矩阵转置：
```ts
let m = new Matrix([
  [1, 5, 0],
  [2, 4 , -1],
  [0, -2, 0]
])
m.T.print()
```
BP神经网络XOR示例，三层网络：
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
BP神经网络加法示例，四层网络：
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
序列模型，该模型只有输入层和输出层两层，支持线性回归和逻辑分类。
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
参数影响
 - **shape**：模型的网络层次结构，结构越复杂单次训练的计算量就相对较大，且容易造成过拟合。
 - **rate**：步长学习率，越低的学习率也就需要相对较多的训练次数才能达到代价函数最优，过大则可能因跨度太大而越过最优点造成损失值趋近于正无穷模型无法收敛的问题。
 - **batch**：训练集全部数据迭代一次的过程。

以上参数的选择也是就是模型的调优的过程，每种模型所需要的学习率、训练次数、模型结构各不相同，需根据代价函数的每次求解来判定。

## Export
- class Matrix
  - 加法、乘法、数乘、转置
  - 行列式、归一化
- class SeqModel
  - 两层模型
  - 线性回归、逻辑分类
- class BPNet
  - 多层网络模型
  - 支持多种激活函数
  - 支持分类和回归