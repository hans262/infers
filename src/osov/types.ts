/**
 * 点
 */
type Point = [number, number]

/**
 * 边，包括两个不同的点
 */
type Edge = [Point, Point]

/**
 * 路径
 */
type Path = Point[]

/**
 * 多边形
 */
type Polygon = Point[]

/**
 * 矩形
 */
interface Rect {
  point: Point,
  width: number,
  height: number
}

export type {
  Point,
  Edge,
  Path,
  Polygon,
  Rect
}