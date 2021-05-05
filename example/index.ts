import { addition } from './addition'
import { iris, saveIris } from './iris'
import { xor } from './xor'
import { rnn } from './rnn'

export namespace TestMatrix {
  export function run() {
    // addition()
    // xor()
    // iris()
    // saveIris()
    rnn()
  }
}
TestMatrix.run()