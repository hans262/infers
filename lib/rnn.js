"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RNN = void 0;
const src_1 = require("../src");
class RNN {
    constructor(opt) {
        this.indexWord = {};
        this.wordIndex = {};
        this.hidenSize = 10;
        this.finis = '/n';
        this.rate = 0.01;
        this.trainData = opt.trainData;
        if (opt.rate)
            this.rate = opt.rate;
        let temp = Array.from(new Set(this.trainData.map(v => v.split('')).flat(1)));
        for (let i = 0; i < temp.length; i++) {
            this.indexWord[temp[i]] = i;
            this.wordIndex[i] = temp[i];
        }
        this.inputSize = temp.length;
        this.outputSize = this.inputSize + 1;
        this.wordIndex[temp.length] = this.finis;
        this.indexWord[this.finis] = temp.length;
        this.U = src_1.Matrix.generate(this.hidenSize, this.inputSize);
        this.W = src_1.Matrix.generate(this.hidenSize, this.hidenSize);
        this.V = src_1.Matrix.generate(this.outputSize, this.hidenSize);
        this.firstSt = src_1.Matrix.generate(1, this.hidenSize, 0);
    }
    oneHotXs(inputIndex) {
        let xs = src_1.Matrix.generate(1, this.inputSize, 0);
        xs.update(0, inputIndex, 1);
        return xs;
    }
    oneHotYs(outputIndex) {
        let ys = src_1.Matrix.generate(1, this.outputSize, 0);
        ys.update(0, outputIndex, 1);
        return ys;
    }
    generateXs(input) {
        let temp = input.split('');
        return temp.map(s => {
            let nowIndex = this.indexWord[s];
            if (isNaN(nowIndex))
                throw new Error(`checked word non-existent from dictionary is ${s}`);
            return this.oneHotXs(nowIndex);
        });
    }
    generateYs(input) {
        let temp = input.split('');
        return temp.map((_, i) => {
            let nextWord = temp[i + 1] ? temp[i + 1] : this.finis;
            let nextIndex = this.indexWord[nextWord];
            return this.oneHotYs(nextIndex);
        });
    }
    forwardPropagation(xs) {
        let result = [];
        for (let i = 0; i < xs.length; i++) {
            let xst = xs[i];
            let lastSt = i === 0 ? this.firstSt : result[i - 1].st;
            let { st, yt } = this.calcForward(xst, lastSt);
            result.push({ st, yt });
        }
        return result;
    }
    calcForward(xs, lastSt) {
        let st = xs.multiply(this.U.T).addition(lastSt.multiply(this.W.T));
        st = st.atomicOperation((item, i) => src_1.afn(item, st.getRow(i), 'Tanh'));
        let yt = st.multiply(this.V.T);
        yt = yt.atomicOperation((item, i) => src_1.afn(item, yt.getRow(i), 'Softmax'));
        return { st, yt };
    }
    backPropagation(hy, xs, ys) {
        let dv = this.V.zeroed();
        let du = this.U.zeroed();
        let dw = this.W.zeroed();
        for (let i = 0; i < hy.length; i++) {
            let { st, yt } = hy[i];
            let xst = xs[i];
            let yst = ys[i];
            let lastSt = i === 0 ? this.firstSt : hy[i - 1].st;
            let dyt = yt.atomicOperation((item, r, c) => (item - yst.get(r, c)) * src_1.afd(item, 'Softmax'));
            let dst = dyt.multiply(this.V);
            dst = dst.atomicOperation((item, r, c) => item * src_1.afd(st.get(r, c), 'Tanh'));
            let ndv = dyt.T.multiply(st);
            let ndu = dst.T.multiply(xst);
            let ndw = dst.T.multiply(lastSt);
            dv = dv.addition(ndv);
            du = du.addition(ndu);
            dw = dw.addition(ndw);
        }
        this.U = this.U.subtraction(du.multiply(this.rate));
        this.W = this.W.subtraction(dw.multiply(this.rate));
        this.V = this.V.subtraction(dv.multiply(this.rate));
    }
    predict(input, length = 10) {
        let xs = this.generateXs(input);
        let hy = this.forwardPropagation(xs);
        let lastHy = hy[hy.length - 1];
        let nextIndex = lastHy.yt.argMax(0);
        let nextLastSt = lastHy.st;
        let result = '';
        result += this.wordIndex[nextIndex];
        if (nextIndex === this.inputSize)
            return result;
        for (let i = 0; i < length - 1; i++) {
            let nextXs = this.oneHotXs(nextIndex);
            let hy = this.calcForward(nextXs, nextLastSt);
            nextIndex = hy.yt.argMax(0);
            nextLastSt = hy.st;
            result += this.wordIndex[nextIndex];
            if (nextIndex === this.inputSize)
                break;
        }
        return result;
    }
    cost(hy, ys) {
        let res = hy.map((nhy, i) => {
            let { yt } = nhy;
            let yst = ys[i];
            let tmp = yt.subtraction(yst).atomicOperation(item => (item ** 2) / 2).getRow(0);
            return tmp.reduce((a, b) => a + b) / tmp.length;
        });
        return res.reduce((a, b) => a + b);
    }
    fit(opt = {}) {
        const { epochs = 1000, onEpochs } = opt;
        for (let i = 0; i < epochs; i++) {
            let e = 0;
            for (let n = 0; n < this.trainData.length; n++) {
                let input = this.trainData[n];
                let xs = this.generateXs(input);
                let ys = this.generateYs(input);
                let hy = this.forwardPropagation(xs);
                this.backPropagation(hy, xs, ys);
                e += this.cost(hy, ys);
            }
            if (onEpochs)
                onEpochs(i, e / this.trainData.length);
        }
    }
}
exports.RNN = RNN;
