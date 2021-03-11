import { testPointInsidePolygon, testPointInsideEdge, testIntersectEdge } from '../src'
import type { Point, Polygon, Edge } from '../src'

describe('test -> testPointInsideEdge', () => {
  test('45度斜边', () => {
    const edge: Edge = [[10, 10], [50, 50]]
    expect(testPointInsideEdge([10, 10], edge)).toBeTruthy()
    expect(testPointInsideEdge([51, 7], edge)).toBeFalsy()
    expect(testPointInsideEdge([30, 30], edge)).toBeTruthy()
  })
})

describe('test -> testIntersectEdge', () => {
  test('不相交的边', () => {
    const edge: Edge = [[10, 8], [200, 75]]
    const edge2: Edge = [[100, 50], [167, 200]]
    expect(testIntersectEdge(edge, edge2)).toBeFalsy()
  })
  test('相交的边', () => {
    const edge: Edge = [[10, 8], [200, 75]]
    const edge2: Edge = [[100, 20], [167, 200]]
    expect(testIntersectEdge(edge, edge2)).toBeTruthy()
  })
})

describe('test -> polygon', () => {
  test('正多边形点包含关系', () => {
    const polygon: Polygon = [
      [100, 100], [200, 100],
      [200, 200], [100, 200]
    ]
    const points: Point[] = [
      [100, 150], //在多边形边界上
      [100, 201], //不在多边形内
      [50, 150], //不在多边形内
      [150, 150], //在多边形内
      [101, 199], //在多边形内
    ]
    expect(testPointInsidePolygon(points[0], polygon)).toBe(-1)
    expect(testPointInsidePolygon(points[1], polygon)).toBe(0)
    expect(testPointInsidePolygon(points[2], polygon)).toBe(0)
    expect(testPointInsidePolygon(points[3], polygon)).toBe(1)
    expect(testPointInsidePolygon(points[4], polygon)).toBe(1)
  })
})
