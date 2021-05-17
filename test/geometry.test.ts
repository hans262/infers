import { Point, Edge, Polygon } from '../src'

describe('test -> point', () => {
  test('点重合验证', () => {
    let p = new Point(1, 2)
    let p2 = new Point(1, 2)
    let p3 = new Point(5, 2)
    expect(p.contrast(p2)).toBeTruthy()
    expect(p.contrast(p3)).toBeFalsy()
  })
})

describe('test -> edge', () => {
  test('点在边的斜率上', () => {
    let edge = new Edge([0, 0], [3, 3])
    expect(edge.testPointIn(new Point(0, 0))).toBeTruthy()
    expect(edge.testPointIn(new Point(5, 5))).toBeTruthy()
    expect(edge.testPointIn(new Point(5, 0))).toBeFalsy()

    let edge2 = new Edge([0, 0], [0, 10])
    expect(edge2.testPointIn(new Point(0, 5))).toBeTruthy()
    expect(edge2.testPointIn(new Point(10, 10))).toBeFalsy()
  })

  test('点在边内', () => {
    let edge = new Edge([0, 0], [3, 3])
    expect(edge.testPointInside(new Point(0, 0))).toBeTruthy()
    expect(edge.testPointInside(new Point(2, 2))).toBeTruthy()
    expect(edge.testPointInside(new Point(5, 5))).toBeFalsy()
    expect(edge.testPointInside(new Point(5, 0))).toBeFalsy()

    let edge2 = new Edge([0, 0], [0, 10])
    expect(edge2.testPointInside(new Point(0, 5))).toBeTruthy()
    expect(edge2.testPointInside(new Point(0, 20))).toBeFalsy()
    expect(edge2.testPointInside(new Point(10, 10))).toBeFalsy()
  })
})

describe('test -> polygon', () => {
  test('多边形点包含测试', () => {
    const p = new Polygon( [
      [100, 100], [200, 100],
      [200, 200], [100, 200]
    ])
    expect(p.testPointInsidePolygon(new Point(100, 150))).toBe(-1)
    expect(p.testPointInsidePolygon(new Point(100, 201))).toBe(0)
    expect(p.testPointInsidePolygon(new Point(50, 150))).toBe(0)
    expect(p.testPointInsidePolygon(new Point(150, 150))).toBe(1)
    expect(p.testPointInsidePolygon(new Point(101, 199))).toBe(1)
  })
})