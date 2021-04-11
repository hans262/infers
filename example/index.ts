import { addition } from './addition'
import { iris } from './iris'
import { xor } from './xor'

export namespace TestMatrix {
  export function run() {
    addition()
    // xor()
    // iris()
  }
}
TestMatrix.run()