import { toFixed } from '../src'
describe('test -> 工具函数', () => {
  test('toFixed', () => {
    expect(toFixed(3.1415926, 0)).toBe(3)
    expect(toFixed(3.1415926, 1)).toBe(3.1)
    expect(toFixed(3.1415926, 2)).toBe(3.14)
    expect(toFixed(3.1415926, 3)).toBe(3.141)

    expect(toFixed(3, 0)).toBe(3)
    expect(toFixed(3, 1)).toBe(3)
  })
})