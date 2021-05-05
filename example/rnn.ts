import { RNN } from '../src'

export function rnn() {
  let data = ['hello', 'huahua', 'goudan', 'name beichuan', 'is']
  let net = new RNN(data)
  net.fit()
  net.predict()
}
