import type { Path, Polygon, Edge } from './types'
import { contrastPoint } from './common'

/**
 * 边合法性检查
 * * 必须是两个点
 * * 且两个点不能重合
 * @param pts 
 */
export function testIsEdge(pts: Point[]): pts is Edge {
  return pts.length === 2 && !contrastPoint(pts[0], pts[1])
}

/**
 * 路径合法性检查
 * * 至少有两个点
 * * 不能有连续重合的点
 * @param pts 点集
 */
export function testIsPath(pts: Point[]): pts is Path {
  if (pts.length < 2) return false
  //找出连续重合的点
  const dd = pts.find((pt, i) => {
    const next = pts[i + 1] as Point | undefined
    if (!next) return false
    return contrastPoint(pt, next)
  })
  if (dd) return false
  return true
}

/**
 * 多边形合法性检查
 * * 至少三个点
 * * 不能有相同的点
 * * 所有点不能在一条线上
 * @param pts 点集
 */
export function testIsPolygon(pts: Path): pts is Polygon {
  const len = pts.length
  //至少有三个点
  if (len < 3) return false
  //不能有相同的点
  const r0 = pts.map(p => p[0].toString() + p[1].toString()).sort()
  const r1 = r0.find((x, i) => x === r0[i + 1])
  if (r1) return false
  /**
   * 所有点不在一条线上
   * 利用斜率相等，验证是否在一条线上
   * 判断后续点与第一个点所构成直线的 斜率是否相等
   */
  const firstPoint = pts[0]
  const r3 = pts.slice(1)
  //先排出 所有x坐标相等的情况
  const r4 = new Set(
    pts.map(p => p[0])
  )
  if (r4.size === 1) return false
  const r5 = r3.map(p => (p[1] - firstPoint[1]) / p[0] - firstPoint[0])
  const r6 = new Set(r5).size
  if (r6 === 1) return false
  return true
}