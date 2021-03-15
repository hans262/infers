import { Matrix } from '../src'

describe('test -> Matrix', () => {
  test('转置', () => {
    let m = new Matrix([
      [1, 2, 3],
      [4, 5, 6]
    ])
    let mt = new Matrix([
      [1, 4],
      [2, 5],
      [3, 6]
    ])
    expect(m.T.equals(mt)).toBeTruthy()
  })

  test('行列式', () => {
    let m = new Matrix([
      [1, 5, 0],
      [2, 4, -1],
      [0, -2, 0]
    ])
    expect(m.det()).toBe(-2)
  })

  test('余子式', () => {
    let m = new Matrix([
      [1, 5, 0],
      [2, 4, -1],
      [0, -2, 0]
    ])
    let c12 = new Matrix([
      [1, 5],
      [0, -2]
    ])
    expect(m.cominor(1, 2).equals(c12)).toBeTruthy()
  })

  test('乘法', () => {
    let a = new Matrix([
      [1, 2, 3],
      [4, 5, 6]
    ])
    let b = new Matrix([
      [-1, -2],
      [0, 1],
      [1, 2]
    ])
    let amb = new Matrix([
      [1 * -1 + 2 * 0 + 3 * 1, 1 * -2 + 2 * 1 + 3 * 2],
      [4 * -1 + 5 * 0 + 6 * 1, 4 * -2 + 5 * 1 + 6 * 2]
    ])
    expect(a.multiply(b).equals(amb)).toBeTruthy()
  })

  test('数乘', () => {
    let a = new Matrix([
      [1, 2, 3],
      [4, 5, 6]
    ])
    let amb = new Matrix([
      [1 * 2, 2 * 2, 3 * 2],
      [4 * 2, 5 * 2, 6 * 2]
    ])
    expect(a.numberMultiply(2).equals(amb)).toBeTruthy()
  })

  test('矩阵增列', () => {
    let a = new Matrix([
      [1, 2, 3],
      [4, 5, 6]
    ])
    let aexp = new Matrix([
      [1, 2, 3, 0],
      [4, 5, 6, 0]
    ])
    expect(a.expand(0, 'R').equals(aexp)).toBeTruthy()
  })

  test('加法', () => {
    let a = new Matrix([
      [1, 2, 3],
      [4, 5, 6]
    ])
    let b = new Matrix([
      [1, 1, 1],
      [1, 1, 1]
    ])
    let ab = new Matrix([
      [2, 3, 4],
      [5, 6, 7]
    ])
    expect(a.addition(b).equals(ab)).toBeTruthy()
  })

})