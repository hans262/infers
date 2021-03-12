import { Point, Edge } from '../src'

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



// describe('test -> testIntersectEdge', () => {
//   test('不相交的边', () => {
//     const edge: Edge = [[10, 8], [200, 75]]
//     const edge2: Edge = [[100, 50], [167, 200]]
//     expect(testIntersectEdge(edge, edge2)).toBeFalsy()
//   })
//   test('相交的边', () => {
//     const edge: Edge = [[10, 8], [200, 75]]
//     const edge2: Edge = [[100, 20], [167, 200]]
//     expect(testIntersectEdge(edge, edge2)).toBeTruthy()
//   })
// })

// describe('test -> polygon', () => {
//   test('正多边形点包含关系', () => {
//     const polygon: Polygon = [
//       [100, 100], [200, 100],
//       [200, 200], [100, 200]
//     ]
//     const points: Point[] = [
//       [100, 150], //在多边形边界上
//       [100, 201], //不在多边形内
//       [50, 150], //不在多边形内
//       [150, 150], //在多边形内
//       [101, 199], //在多边形内
//     ]
//     expect(testPointInsidePolygon(points[0], polygon)).toBe(-1)
//     expect(testPointInsidePolygon(points[1], polygon)).toBe(0)
//     expect(testPointInsidePolygon(points[2], polygon)).toBe(0)
//     expect(testPointInsidePolygon(points[3], polygon)).toBe(1)
//     expect(testPointInsidePolygon(points[4], polygon)).toBe(1)
//   })
// })