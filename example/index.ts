import { addition } from './addition'
import { iris, saveIris } from './iris'
import { xor } from './xor'
import { rnn, rnn2 } from './rnn'
import { gru } from './gru'
import { run } from './recurrent'
import { cnn } from './cnn'
import { Matrix } from '../src'

(() => {
  // addition()
  // xor()
  // iris()
  // saveIris()
  // rnn()
  // rnn2()
  // gru()
  // run()
  // cnn()
})();

(() => {
  let a = new Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ])

  let b = new Matrix([
    [-4.5, 7, -1.5],
    [-2, 4, -1],
    [1.5, -2, 0.5]
  ])

  a.inverse().print()

})();