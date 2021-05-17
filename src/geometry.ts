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
}

export class Polygon {
  points: Point[]
  constructor(pts: [number, number][]) {
    this.points = []
    for (let i = 0; i < pts.length; i++) {
      this.points.push(new Point(pts[i][0], pts[i][1]))
    }

    if (this.points.length < 3) {
      throw new Error('至少三个点')
    }
    const r0 = this.points.map(p => p.X.toString() + p.Y.toString()).sort()
    const r1 = r0.find((x, i) => x === r0[i + 1])
    if (r1) {
      throw new Error('不能有相同的点')
    }
    const first = this.points[0]
    const r3 = this.points.slice(1)
    let m = r3.map(p =>
      p.X === first.X ? Infinity //正无穷
        : (p.Y - first.Y) / (p.X - first.X)
    )
    if (new Set(m).size === 1) {
      throw new Error('所有点不能在一条线上')
    }
  }

  testPointInsidePolygon(pt: Point): number {
    let polygon = this.points
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