import { BPNet, Matrix, toFixed } from '../src'

describe('BPNet -> 测试bp网络', () => {
  test('cost', () => {
    const model = new BPNet([2, 6, 6, 2])
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

  test('addition', () => {
    let xs = new Matrix([[1, 4], [3, 2], [6, 5], [4, 7]])
    let ys = new Matrix([[5], [5], [11], [11]])
    let model = new BPNet([2, 6, 6, 1], { mode: 'sgd', rate: 0.01 })
    model.fit(xs, ys, { epochs: 500 })
    let xs2 = new Matrix([[5, 8], [22, 6], [-5, 9], [-5, -4]])
    let result = model.predict(xs2)
    // result.print()
    expect(result.get(0, 0)).toBeCloseTo(13)
    expect(result.get(1, 0)).toBeCloseTo(28)
    expect(result.get(2, 0)).toBeCloseTo(4)
    expect(result.get(3, 0)).toBeCloseTo(-9)
  })

  test('xor - Softmax', () => {
    let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
    let ys = new Matrix([[1, 0], [1, 0], [0, 1], [0, 1]])
    let model = new BPNet([2, [6, 'Tanh'], [6, 'Tanh'], [2, 'Softmax']], { rate: 0.01, mode: 'sgd' })
    model.fit(xs, ys, { epochs: 2000 })
    let result = model.predict(xs)
    // result.print()
    expect(result.get(0, 0)).toBeGreaterThan(0.8)
    expect(result.get(0, 0)).toBeLessThan(1)

    expect(result.get(1, 0)).toBeGreaterThan(0.8)
    expect(result.get(1, 0)).toBeLessThan(1)

    expect(result.get(2, 0)).toBeGreaterThan(0)
    expect(result.get(2, 0)).toBeLessThan(0.2)

    expect(result.get(3, 0)).toBeGreaterThan(0)
    expect(result.get(3, 0)).toBeLessThan(0.2)
  })

  test('xor - Sigmoid', () => {
    let xs = new Matrix([[1, 0], [0, 1], [0, 0], [1, 1]])
    let ys = new Matrix([[1], [1], [0], [0]])
    let model = new BPNet([2, [6, 'Tanh'], [6, 'Tanh'], [1, 'Sigmoid']], { rate: 0.1, mode: 'sgd' })
    model.fit(xs, ys, { epochs: 3000 })
    let result = model.predict(xs)
    // result.print()
    expect(result.get(0, 0)).toBeGreaterThan(0.8)
    expect(result.get(0, 0)).toBeLessThan(1)

    expect(result.get(1, 0)).toBeGreaterThan(0.8)
    expect(result.get(1, 0)).toBeLessThan(1)

    expect(result.get(2, 0)).toBeGreaterThan(0)
    expect(result.get(2, 0)).toBeLessThan(0.2)

    expect(result.get(3, 0)).toBeGreaterThan(0)
    expect(result.get(3, 0)).toBeLessThan(0.2)
  })
})