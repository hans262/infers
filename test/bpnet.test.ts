import { BPNet, Matrix, toFixed } from '../src'

describe('BPNet -> 测试bp网络', () => {
  const model = new BPNet([2, 6, 6, 2])
  test('cost', () => {
    let hy = new Matrix([
      [1, 3, 4],
      [2, 1, 9]
    ])
    let ys = new Matrix([
      [3, 4, 1],
      [6, 3, 2]
    ])
    let loss = model.cost(hy, ys)
    loss = toFixed(loss, 4)
    expect(loss).toBe(6.9166)
  })
})