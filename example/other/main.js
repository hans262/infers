const math = require('./math');
const seedrandom = require('seedrandom');
const RNN = require('./rnn');

let rng = seedrandom(1234);

function main() {

  const TRAIN_NUM = 30;  // time sequence
  const TEST_NUM = 10;

  const N_IN = 1;
  const N_HIDDEN = 4;
  const N_OUT = 1;
  const TRUNCATED_TIME = 4;
  const LEARNING_RATE = 0.01;
  const EPOCHS = 100;

  let classifier = new RNN(N_IN, N_HIDDEN, N_OUT, TRUNCATED_TIME, LEARNING_RATE, math.fn.tanh, rng);

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    if (epoch !== 0 && epoch % 10 === 0) {
      console.log(`epoch: ${epoch}`);
    }
    let _data = loadData(TRAIN_NUM);

    classifier.sgd(_data.x, _data.y);
  }

  let testX = loadData(TEST_NUM).x;
  let output = null;
  for (let i = 0; i < 50; i++) {
    output = classifier.predict(testX);
    testX.push(output[output.length - 1]);
  }

  for (let i = TEST_NUM; i < testX.length - 1; i++) {
    console.log(output[i][0]);
  }
}

function loadData(dataNum) {
  let x = [];  // sin wave + noise [0, t]
  let y = [];  // t + 1
  const TIME_STEP = 0.1;

  let noise = () => {
    return 0.1 * math.random.uniform(-1, 1, rng);
  }

  for (let i = 0; i < dataNum + 1; i++) {
    let _t = i * TIME_STEP;
    let _sin = Math.sin(_t * Math.PI);
    x[i] = [_sin + noise()];

    if (i !== 0) {
      y[i - 1] = x[i];
    }
  }
  x.pop();

  return { x, y }
}

main();
