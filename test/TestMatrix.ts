import { Matrix } from '../src'

export namespace TestMatrix {
  export function run() {
    let a = new Matrix([[1, 5, 0], [2, 4, -1], [0, -2, 0]])
    let b = new Matrix([
      [3, -7, 8, 9, -6],
      [0, 2, -5, 7, 3],
      [0, 0, 1, 5, 0],
      [0, 0, 2, 4, -1],
      [0, 0, 0, -2, 0]
    ])
    a.print()
    b.print()
    console.log(b.det())
  }
}