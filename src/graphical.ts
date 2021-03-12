export class Point {
  constructor(public X: number, public Y: number) { }
  contrast(pt: Point) {
    return this.X === pt.X && this.Y === pt.Y
  }
}

export class Edge {
  start: Point
  end: Point
  constructor(cond1: [number, number], cond2: [number, number]) {
    this.start = new Point(cond1[0], cond1[1])
    this.end = new Point(cond2[0], cond2[1])
    if (this.start.contrast(this.end)) {
      throw new Error('两个点不能相同')
    }
  }
  minXY() {
    let X = Math.min(this.start.X, this.end.X)
    let Y = Math.min(this.start.Y, this.end.Y)
    return new Point(X, Y)
  }
  maxXY() {
    let X = Math.max(this.start.X, this.end.X)
    let Y = Math.max(this.start.Y, this.end.Y)
    return new Point(X, Y)
  }
  /**
   * 点是否在边的斜率上
   * @param pt 
   * @returns 
   */
  testPointIn(pt: Point): boolean {
    if (pt.contrast(this.start) || pt.contrast(this.end)) {
      return true
    }
    let slope1 = pt.X - this.start.X === 0 ? Infinity : (pt.Y - this.start.Y) / (pt.X - this.start.X)
    let slope2 = pt.X - this.end.X === 0 ? Infinity : (pt.Y - this.end.Y) / (pt.X - this.end.X)
    return slope1 === slope2
  }
  /**
   * 点内边内
   * @param pt 
   * @returns 
   */
  testPointInside(pt: Point): boolean {
    if (this.testPointIn(pt)) {
      let min = this.minXY()
      let max = this.maxXY()
      return (pt.X >= min.X && pt.X <= max.X) && (pt.Y >= min.Y && pt.Y <= max.Y)
    }
    return false
  }
  //测试边相交
  testIntersectEdge(edge2: Edge) {
    const [p1, p2] = [this.start, this.end]
    const [q1, q2] = [edge2.start, edge2.end]
    //如果两个矩形都不想交，那么线段肯定不想交
    if (
      !(
        Math.min(p1.X, p2.X) <= Math.max(q1.X, q1.X) &&
        Math.min(q1.X, q2.X) <= Math.max(p1.X, p2.X) &&
        Math.min(p1.Y, p2.Y) <= Math.max(q1.Y, q1.Y) &&
        Math.min(q1.Y, q2.Y) <= Math.max(p1.Y, p2.Y)
      )
    ) return false
    const c1 = ((p1.X - q1.X) * (q2.Y - q1.Y) - (p1.Y - q1.Y) * (q2.X - q1.X)) * ((q2.X - q1.X) * (p2.Y - q1.Y) - (q2.Y - q1.Y) * (p2.X - q1.X))
    const c2 = ((q1.X - p1.X) * (p1.Y - p2.Y) - (q1.Y - p1.Y) * (p2.X - p1.X)) * ((p2.X - p1.X) * (q2.Y - p1.Y) - (p2.Y - p1.Y) * (q2.X - p1.X))
    if (
      c1 > 0 &&
      c2 > 0
    ) {
      return true
    }
    return false
  }
}



export class Path {
  constructor(public pts: Point[]) {
    if (pts.length < 2) {
      throw new Error('至少两个点')
    }
    const n = pts.find((pt, i) => {
      const next = pts[i + 1] as Point | undefined
      if (!next) return false
      return pt.contrast(next)
    })
    if (n) {
      throw new Error('不能有连续重合的点')
    }
  }
}

export class Polygon {
  constructor(public pts: Point[]) {
    if (pts.length < 3) {
      throw new Error('至少三个点')
    }
    const r0 = pts.map(p => p.X.toString() + p.Y.toString()).sort()
    const r1 = r0.find((x, i) => x === r0[i + 1])
    if (r1) {
      throw new Error('不能有相同的点')
    }
    const first = pts[0]
    const r3 = pts.slice(1)
    let m = r3.map(p =>
      p.X === first.X ? Infinity //无穷大
        : (p.Y - first.Y) / (p.X - first.X)
    )
    if (new Set(m).size === 1) {
      throw new Error('所有点不能在一条线上')
    }
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
  testPointInsidePolygon(pt: Point): number {
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

    let polygon = this.pts
    var result = 0,
      cnt = polygon.length;
    if (cnt < 3)
      return 0;
    var ip = polygon[0];
    for (var i = 1; i <= cnt; ++i) {
      var ipNext = i === cnt ? polygon[0] : polygon[i]
      if (ipNext.Y === pt.Y) {
        if ((ipNext.X === pt.X) || (ip.Y === pt.Y && ((ipNext.X > pt.X) === (ip.X < pt.X))))
          return -1;
      }
      if ((ip.Y < pt.Y) !== (ipNext.Y < pt.Y)) {
        if (ip.X >= pt.X) {
          if (ipNext.X > pt.X)
            result = 1 - result;
          else {
            var d = (ip.X - pt.X) * (ipNext.Y - pt.Y) - (ipNext.X - pt.X) * (ip.Y - pt.Y);
            if (d === 0)
              return -1;
            else if ((d > 0) === (ipNext.Y > ip.Y))
              result = 1 - result;
          }
        }
        else {
          if (ipNext.X > pt.X) {
            var d = (ip.X - pt.X) * (ipNext.Y - pt.Y) - (ipNext.X - pt.X) * (ip.Y - pt.Y);
            if (d === 0)
              return -1;
            else if ((d > 0) === (ipNext.Y > ip.Y))
              result = 1 - result;
          }
        }
      }
      ip = ipNext;
    }
    return result;
  }
}