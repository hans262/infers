import { addition } from './addition'
import { iris, saveIris } from './iris'
import { xor } from './xor'
import { rnn } from './rnn'
import { testgru } from './grn'

export namespace TestMatrix {
  export function run() {
    // addition()
    // xor()
    // iris()
    // saveIris()
    // rnn()
    testgru()
  }
}
TestMatrix.run()