export {
  testIsEdge,
  testIsPath,
  testIsPolygon
} from './typeCheck'

export {
  testPointInsidePolygon,
  testPointInsideEdge,
  testIntersectEdge
} from './booleanCheck'

export {
  toFixed
} from './common'

export type {
  Point,
  Edge,
  Path,
  Polygon
} from './types'