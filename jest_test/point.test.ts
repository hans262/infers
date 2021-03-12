import { Point } from '../src'

describe('test -> point', () => {
  test('点重合验证', () => {
    let p = new Point(1, 2)
    let p2 = new Point(1, 2)
    let p3 = new Point(5, 2)
    expect(p.contrast(p2)).toBeTruthy()
    expect(p.contrast(p3)).toBeFalsy()
  })
})