import { toFixed, imageDataToMatrix, Matrix } from '../src'

describe('test -> 工具函数', () => {
  test('toFixed', () => {
    expect(toFixed(3.1415926, 0)).toBe(3)
    expect(toFixed(3.1415926, 1)).toBe(3.1)
    expect(toFixed(3.1415926, 2)).toBe(3.14)
    expect(toFixed(3.1415926, 3)).toBe(3.141)

    expect(toFixed(3, 0)).toBe(3)
    expect(toFixed(3, 1)).toBe(3)
  })

  test('imageDataToMatrix', () => {
    let imageData = {
      width: 2,
      height: 3,
      data: new Array(2 * 3 * 4).fill(0).map((_, k) => k)
    } as any
    let mat = imageDataToMatrix(imageData, 'g')
    expect(mat.equals(new Matrix([
      [1, 5],
      [9, 13],
      [17, 21]
    ]))).toBeTruthy()
  })
})