import { Matrix } from '../src'

export namespace TestMatrix {
  export function det() {
    let b = new Matrix([
      [3, -7, 8, 9, -6],
      [0, 2, -5, 7, 3],
      [0, 0, 1, 5, 0],
      [0, 0, 2, 4, -1],
      [0, 0, 0, -2, 0]
    ])
    b.print()
    console.log(b.det())
  }
  export function expansion() {
    let a = new Matrix([
      [1, 5, 0],
      [2, 4, -1],
      [0, -2, 0]
    ])
    a.print()
    a.expand(1, 'L').print()
  }

  export function generate() {
    let a = Matrix.generate(3, 2, 1)
    a.print()
  }

  export function update() {
    let a = new Matrix([
      [1, 5, 0],
      [2, 4, -1],
      [0, -2, 0]
    ])
    a.update(2, 0, 100)
    a.print()
  }
  
  export function run() {
    let m = new Matrix([
      [1, 5, 0],
      [2, 4, -1],
      [0, -2, 0]
    ])
    m.cominor(1, 2).print()
  }
}

TestMatrix.run()