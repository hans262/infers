import { testIsPath, testIsPolygon, testIsEdge } from '../src'
import type { Point } from '../src'

describe('test -> 类型合法性', () => {
  test('testIsEdge', () => {
    const edges: Point[][] = [
      [[1, 2], [2, 3], [2, 4]], //点个数不对
      [[1, 2], [1, 2]], //点重合
      [[1, 2], [2, 3]], //满足
    ]
    expect(testIsEdge(edges[0])).toBeFalsy()
    expect(testIsEdge(edges[1])).toBeFalsy()
    expect(testIsEdge(edges[2])).toBeTruthy()
  })

  test('testIsPath', () => {
    const paths: Point[][] = [
      [[1, 2]], //点不足
      [[1, 3], [1, 2], [1, 2]], // 连续重合的点
      [[1, 2], [1, 3], [1, 2]], //非连续重合点
      [[1, 2], [1, 3], [2, 3]] //正常
    ]
    expect(testIsPath(paths[0])).toBeFalsy()
    expect(testIsPath(paths[1])).toBeFalsy()
    expect(testIsPath(paths[2])).toBeTruthy()
    expect(testIsPath(paths[3])).toBeTruthy()
  })

  test('testIsPolygon', () => {
    const polygon: Point[][] = [
      [[1, 2], [1, 2]], //点不足情况
      [[1, 1], [1, 5], [1, 5]], //有相同点情况
      [[1, 2], [1, 3], [1, 6]], //三点一线情况，X坐标相同
      [[1, 2], [2, 2], [3, 2]], //三点一线情况，Y坐标相同
    ]
    expect(testIsPolygon(polygon[0])).toBeFalsy()
    expect(testIsPolygon(polygon[1])).toBeFalsy()
    expect(testIsPolygon(polygon[2])).toBeFalsy()
    expect(testIsPolygon(polygon[3])).toBeFalsy()
  })
})