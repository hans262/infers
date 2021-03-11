type Point = [number, number]

function drawPath(ctx: CanvasRenderingContext2D, path: Point[]) {
  const firstPoint = path[0]
  ctx.beginPath()
  ctx.moveTo(firstPoint[0], firstPoint[1])
  path.slice(1).forEach(p => ctx.lineTo(p[0], p[1]))
  ctx.closePath()
  ctx.stroke()
}

function drawPoint(ctx: CanvasRenderingContext2D, point: Point) {
  ctx.beginPath()
  ctx.fillStyle = 'red'
  ctx.arc(point[0], point[1], 2, 0, 2 * Math.PI)
  ctx.fill()
}

/**
 * 外扩路径
 * @param path 源
 * @param scale 缩放比例
 * 
 * void AddPath(Path path, JoinType jointype, EndType endtype);
 * JoinType 0 - 2
 * EndType  0 - 4
 */
// const scaleOut = (path: ClipperPath, scale: number): ClipperPath => {
//   var ret = new ClipperLib.Paths()
//   var co = new ClipperLib.ClipperOffset()
//   co.ArcTolerance = 0.25
//   co.AddPath(path, 1, 4)
//   co.Execute(ret, scale)
//   return ret[0]
// }

function main() {
  const canvas = document.querySelector('#canvas') as HTMLCanvasElement
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D
  canvas.width = 800
  canvas.height = 400

  // drawPath(ctx, [
  //   [100, 100], [200, 100],
  //   [200, 200], [100, 200]
  // ])
  // drawPoint(ctx, [100, 200])

  // drawPath(ctx, [
  //   [10, 8], [200, 75]
  // ])

  // drawPath(ctx, [
  //   [100, 20], [167, 200]
  // ])

  drawPath(ctx, [
    [100, 90], [300, 80],
    [350, 200], [250, 180],
    [180, 270], [100, 190]
  ])

}
main()