import { RNN } from '../src'

export function rnn() {
  let trainData = ['Hello RNN', 'morning', 'I love 🍎', '我的🍎在哪里', '我的🍊被人偷了']
  let net = new RNN({ trainData })
  net.fit({
    epochs: 1500,
    onEpochs: (epoch, loss) => {
      if (epoch % 10 === 0) console.log('epoch: ', epoch, 'loss: ', loss)
    }
  })
  console.log(net.predict('我的🍎'))
  console.log(net.predict('我的🍊'))
  console.log(net.predict('Hel'))
  console.log(net.predict('morn'))
  console.log(net.predict('I lo'))
}
