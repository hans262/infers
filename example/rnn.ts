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

export function rnn2() {
  let trainData = [
    '今天的任务做完了吗',
    '早餐你吃了吗',
    '世界上的山有哪些',
    '地球是圆的吗',
    '老人与海 海明威',
    '时间不会继续等我们',
    '你在车站的站台上等我',
    '有那么一次',
    '悲惨世界'
  ]
  let net = new RNN({ trainData })
  net.fit({
    epochs: 1500, onEpochs: (epoch, loss) => {
      console.log('epoch: ', epoch, 'loss: ', loss)
    }
  })
  console.log(net.predict('老人'))
}