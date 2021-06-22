import { Matrix } from "../src";

export function cnn() {
  //样本 | 标签
  let xs = Matrix.generate(5, 5, { range: [0, 10], integer: true })
  let ys = Matrix.generate(5, 5, { range: [0, 10], integer: true })
  xs.print()
  ys.print()

  //初始化 3 x 3 卷积核
  let h = Matrix.generate(3, 3)

  //处理样本 周围添加一层
  let m = xs.expand(0, 'L')
  m = m.expand(0, 'T')
  m = m.expand(0, 'R')
  m = m.expand(0, 'B')
  m.print()

  //利用卷积核大小，采集所有的样本数据
  let inputs: Matrix[][] = []
  for (let i = 0; i < xs.shape[0]; i++) {
    let n = []
    for (let j = 0; j < xs.shape[1]; j++) {
      let t = new Matrix([
        [m.get(i, j), m.get(i, j + 1), m.get(i, j + 2)],
        [m.get(i + 1, j), m.get(i + 1, j + 1), m.get(i + 1, j + 2)],
        [m.get(i + 2, j), m.get(i + 2, j + 1), m.get(i + 2, j + 2)]
      ])
      n.push(t)
    }
    inputs.push(n)
  }

  function forwardPropagation() {
    let outputs = Matrix.generate(inputs.length, inputs[0].length, 0)
    for (let i = 0; i < inputs.length; i++) {
      for (let j = 0; j < inputs[0].length; j++) {
        let v = inputs[i][j].coLocationOperation(h, 'mul').sum()
        outputs.update(i, j, v)
      }
    }
    return outputs
  }

  function calcLoss(outputs: Matrix) {
    let dy = outputs.subtraction(ys)
    let loss = dy.atomicOperation(item => (item ** 2) / 2).sum() / (ys.shape[0] * ys.shape[1])
    return { loss, dy }
  }

  //调整卷积核
  function adjust(dy: Matrix) {
    let dh = Matrix.generate(3, 3, 0)
    for (let i = 0; i < dy.shape[0]; i++) {
      for (let j = 0; j < dy.shape[1]; j++) {
        let input = inputs[i][j]
        let ndy = dy.get(i, j)
        for (let k = 0; k < dh.shape[0]; k++) {
          for (let h = 0; h < dh.shape[1]; h++) {
            dh.update(h, k, ndy * input.get(k, h), '+=')
          }
        }
      }
    }
    dh = dh.atomicOperation(item => item / (dy.shape[0] * dy.shape[1]))
    h = h.subtraction(dh.multiply(0.001))
  }

  for (let i = 0; i < 200; i++) {
    //计算结果
    let outputs = forwardPropagation()
    //计算误差
    let { loss, dy } = calcLoss(outputs)
    console.log(i, loss)
    //调整卷积核
    adjust(dy)
  }
}