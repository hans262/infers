import { contrastPoint, toFixed } from './common'
import type { Edge, Point, Polygon } from './types'

/**
 * 测试点是否在边内，不包括在边的延长线上，
 * 算法：测试点到两点距离是否等于边距离
 * @param pt 
 * @param edge 
 */
export function testPointInsideEdge(pt: Point, edge: Edge): boolean {
  const [cond1, cond2] = edge
  const [maxX, maxY, minX, minY] = [
    Math.max(cond1[0], cond2[0]),
    Math.max(cond1[1], cond2[1]),
    Math.min(cond1[0], cond2[0]),
    Math.min(cond1[1], cond2[1]),
  ]
  //排除测试点不在边所构成的矩形框内
  if (pt[0] < minX || pt[0] > maxX || pt[1] < minY || pt[1] > maxY) {
    return false
  }
  //是否跟边顶点重合
  if (contrastPoint(pt, cond1) || contrastPoint(pt, cond2)) {
    return true
  }
  //测试点到边两端点的距离和是否等于边的长度，来判断是否在线段内
  const s1 = Math.sqrt(Math.abs(pt[0] - cond1[0]) ** 2 + Math.abs(pt[1] - cond1[1]) ** 2)
  const s2 = Math.sqrt(Math.abs(pt[0] - cond2[0]) ** 2 + Math.abs(pt[1] - cond2[1]) ** 2)
  const s0 = Math.sqrt(Math.abs(cond1[0] - cond2[0]) ** 2 + Math.abs(cond1[1] - cond2[1]) ** 2)
  //误差处理 取小数点后一位
  return toFixed(s1 + s2, 1) === toFixed(s0, 1)
}

/**
 * 测试点是否在边所构成的斜率上
 * @param pt 
 * @param edge 
 */
export function testPointInEdge(pt: Point, edge: Edge): boolean {
  const [cond1, cond2] = edge
  //排除测试点与边端点重合
  if (contrastPoint(pt, cond1) || contrastPoint(pt, cond2)) {
    return true
  }
  //垂直边
  if (cond1[0] === cond2[0]) {
    return pt[0] === cond1[0]
  }
  //水平边
  if (cond1[1] === cond2[1]) {
    return pt[1] === cond1[1]
  }
  //测试点与某一端点垂直
  if (pt[0] === cond1[0] || pt[0] === cond2[0]) {
    return false
  }
  //斜边 直接求斜率是否相等 斜率取小数点后1位
  let x1 = Math.abs((pt[1] - cond1[1]) / (pt[0] - cond1[0]))
  let x2 = Math.abs((pt[1] - cond2[1]) / (pt[0] - cond2[0]))
  return toFixed(x1, 1) === toFixed(x2, 1)
}

/**
 * 测试两条线段是否相交
 * * 排斥实验
 * * 跨立实验
 * @param edge1 
 * @param edge2 
 */
export function testIntersectEdge(edge1: Edge, edge2: Edge) {
  const [p1, p2] = edge1
  const [q1, q2] = edge2
  //如果两个矩形都不想交，那么线段肯定不想交
  if (
    !(
      Math.min(p1[0], p2[0]) <= Math.max(q1[0], q1[0]) &&
      Math.min(q1[0], q2[0]) <= Math.max(p1[0], p2[0]) &&
      Math.min(p1[1], p2[1]) <= Math.max(q1[1], q1[1]) &&
      Math.min(q1[1], q2[1]) <= Math.max(p1[1], p2[1])
    )
  ) return false
  const c1 = ((p1[0] - q1[0]) * (q2[1] - q1[1]) - (p1[1] - q1[1]) * (q2[0] - q1[0])) * ((q2[0] - q1[0]) * (p2[1] - q1[1]) - (q2[1] - q1[1]) * (p2[0] - q1[0]))
  const c2 = ((q1[0] - p1[0]) * (p1[1] - p2[1]) - (q1[1] - p1[1]) * (p2[0] - p1[0])) * ((p2[0] - p1[0]) * (q2[1] - p1[1]) - (p2[1] - p1[1]) * (q2[0] - p1[0]))
  console.log(c1)
  console.log(c2)
  if (
    c1 > 0 &&
    c2 > 0
  ) {
    return true
  }
  return false
}

/**
 * 测试点在多边形内部
 * 
 * 引射线法
 * 从目标点，水平向右发出一条射线，
 * 该射线与多边形相交点的个数为奇数，则目标点在对变形内部，反之在外。
 * 
 * 排除几种情况
 * 1. 点在多边形构成的矩形外部
 * 3. 点在多边形边上
 * 
 * 4. 射线经过多边形顶点
 * 5. 射线进过多边形的边
 * 
 */
export function testPointInPolygon(pt: Point, polygon: Polygon): boolean {
  const [polygonXs, polygonYs] = [
    polygon.map(p => p[0]),
    polygon.map(p => p[1])
  ]
  const [maxX, maxY, minX, minY] = [
    Math.max(...polygonXs),
    Math.max(...polygonYs),
    Math.min(...polygonXs),
    Math.min(...polygonYs),
  ]
  //排除测试点不在边所构成的矩形框内
  if (pt[0] < minX || pt[0] > maxX || pt[1] < minY || pt[1] > maxY) {
    return false
  }
  for (let i = 0; i < polygon.length; i++) {
    const cp = polygon[i]
    // const np = polygon[i + 1] ?? polygon[0]
    // const edge: Edge = [cp, np]


  }
  return true
}


export function testPointInsidePolygon(pt: Point, polygon: Polygon): number {
  //判断一个点是否在一个多边形里面
  /**
   * 算法
   * 引射线法：从目标点出发引一条射线，
   * 看这条射线和多边形所有边的交点数目。
   * 如果有奇数个交点，则说明在内部，如果有偶数个交点，则说明在外部。
   */
  //returns 0 if false, +1 if true, -1 if pt ON polygon boundary
  //See "The Point in Polygon Problem for Arbitrary Polygons" by Hormann & Agathos
  //http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.88.5498&rep=rep1&type=pdf

  var result = 0,
    cnt = polygon.length;
  if (cnt < 3)
    return 0;
  var ip = polygon[0];
  for (var i = 1; i <= cnt; ++i) {
    var ipNext = i === cnt ? polygon[0] : polygon[i]
    if (ipNext[1] === pt[1]) {
      if ((ipNext[0] === pt[0]) || (ip[1] === pt[1] && ((ipNext[0] > pt[0]) === (ip[0] < pt[0]))))
        return -1;
    }
    if ((ip[1] < pt[1]) !== (ipNext[1] < pt[1])) {
      if (ip[0] >= pt[0]) {
        if (ipNext[0] > pt[0])
          result = 1 - result;
        else {
          var d = (ip[0] - pt[0]) * (ipNext[1] - pt[1]) - (ipNext[0] - pt[0]) * (ip[1] - pt[1]);
          if (d === 0)
            return -1;
          else if ((d > 0) === (ipNext[1] > ip[1]))
            result = 1 - result;
        }
      }
      else {
        if (ipNext[0] > pt[0]) {
          var d = (ip[0] - pt[0]) * (ipNext[1] - pt[1]) - (ipNext[0] - pt[0]) * (ip[1] - pt[1]);
          if (d === 0)
            return -1;
          else if ((d > 0) === (ipNext[1] > ip[1]))
            result = 1 - result;
        }
      }
    }
    ip = ipNext;
  }
  return result;
}