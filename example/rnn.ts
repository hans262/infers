import { RNN } from '../src'

export function rnn() {
  let trainData = ['hello rnn', 'good morning', 'I love 🍎!', 'I eat 🍊!']
  let net = new RNN({ trainData })
  net.fit({
    epochs: 1500, onEpochs: (epoch, loss) => {
      if (epoch % 10 === 0) console.log('epoch: ', epoch, 'loss: ', loss)
    }
  })
  console.log(net.predict('I love'))
  console.log(net.predict('I eat'))
  console.log(net.predict('hel'))
  console.log(net.predict('good'))
}
