"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RNN = void 0;
const src_1 = require("../src");
class RNN {
    constructor(data) {
        this.indexWord = {};
        this.wordIndex = {};
        this.hideSize = 8;
        this.trainData = data.map(v => v.split(''));
        let temp = Array.from(new Set(this.trainData.flat(1)));
        for (let i = 0; i < temp.length; i++) {
            this.indexWord[temp[i]] = i;
            this.wordIndex[i] = temp[i];
        }
        this.inputSize = temp.length;
        this.wordIndex[temp.length] = '/n';
        this.indexWord['/n'] = temp.length;
        let outputSize = this.inputSize + 1;
        this.U = src_1.Matrix.generate(this.hideSize, this.inputSize);
        this.W = src_1.Matrix.generate(this.hideSize, this.hideSize);
        this.V = src_1.Matrix.generate(outputSize, this.hideSize);
    }
    afn(x, rows, af) {
        switch (af) {
            case 'Tanh':
                return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
            case 'Softmax':
                let d = Math.max(...rows);
                return Math.exp(x - d) / rows.reduce((p, c) => p + Math.exp(c - d), 0);
            default:
                return x;
        }
    }
    afd(x, af) {
        switch (af) {
            case 'Tanh':
                return 1 - ((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))) ** 2;
            case 'Softmax':
            default:
                return 1;
        }
    }
    oneHotXs(inputIndex) {
        let xs = src_1.Matrix.generate(1, this.inputSize, 0);
        xs.update(0, inputIndex, 1);
        return xs;
    }
    oneHotYs(outputIndex) {
        let ys = src_1.Matrix.generate(1, this.inputSize + 1, 0);
        if (outputIndex < 0) {
            ys.update(0, this.inputSize, 1);
        }
        else {
            ys.update(0, outputIndex, 1);
        }
        return ys;
    }
    forwardPropagation(data) {
        let lastSt = src_1.Matrix.generate(1, this.hideSize, 0);
        return data.map(v => {
            let { xs, ys } = v;
            let st = xs.multiply(this.U.T).addition(lastSt.multiply(this.W.T));
            st = st.atomicOperation((item, i) => this.afn(item, st.getRow(i), 'Tanh'));
            let yt = st.multiply(this.V.T);
            yt = yt.atomicOperation((item, i) => this.afn(item, yt.getRow(i), 'Softmax'));
            let lastStCopy = lastSt;
            lastSt = st;
            return { xs, ys, st, yt, lastSt: lastStCopy };
        });
    }
    backPropagation(hys) {
        let dv = this.V.zeroed();
        let du = this.U.zeroed();
        let dw = this.W.zeroed();
        hys.forEach(hy => {
            let { xs, ys, st, yt, lastSt } = hy;
            let dyt = yt.atomicOperation((item, r, c) => (item - ys.get(r, c)) * this.afd(item, 'Softmax'));
            let dst = dyt.multiply(this.V);
            dst = dst.atomicOperation((item, r, c) => item * this.afd(st.get(r, c), 'Tanh'));
            let ndv = dyt.T.multiply(st);
            let ndu = dst.T.multiply(xs);
            let ndw = dst.T.multiply(lastSt);
            dv = dv.addition(ndv);
            du = du.addition(ndu);
            dw = dw.addition(ndw);
        });
        let rate = 0.01;
        this.U = this.U.subtraction(du.numberMultiply(rate));
        this.W = this.W.subtraction(dw.numberMultiply(rate));
        this.V = this.V.subtraction(dv.numberMultiply(rate));
    }
    predict() {
        let input = 'gou'.split('');
        let data = this.onehot(input);
        let hys = this.forwardPropagation(data);
        console.log(this.showWords(hys));
    }
    maxIndex(d) {
        var max = d[0];
        var index = 0;
        for (var i = 0; i < d.length; i++) {
            if (d[i] > max) {
                max = d[i];
                index = i;
            }
        }
        return index;
    }
    showWords(hys) {
        return hys.map(hy => {
            let index = this.maxIndex(hy.yt.getRow(0));
            if (!this.wordIndex[index])
                debugger;
            return this.wordIndex[index];
        });
    }
    cost(hys) {
        let m = hys.map(hy => {
            let { yt, ys } = hy;
            let tmp = yt.subtraction(ys).atomicOperation(item => (item ** 2) / 2).getRow(0);
            return tmp.reduce((a, b) => a + b) / tmp.length;
        });
        return m.reduce((a, b) => a + b);
    }
    onehot(input) {
        return input.map((s, i) => {
            let inputIndex = this.indexWord[s];
            let nextWord = input[i + 1] ? input[i + 1] : '/n';
            let outoutIndex = this.indexWord[nextWord];
            let xs = this.oneHotXs(inputIndex);
            let ys = this.oneHotYs(outoutIndex);
            return { xs, ys };
        });
    }
    fit() {
        for (let i = 0; i < 5000; i++) {
            let e = 0;
            for (let n = 0; n < this.trainData.length; n++) {
                let input = this.trainData[n];
                let data = this.onehot(input);
                let hys = this.forwardPropagation(data);
                this.backPropagation(hys);
                e += this.cost(hys);
            }
            if (i % 100 === 0)
                console.log('enpoch: ', i, 'loss: ', e / this.trainData.length);
        }
    }
}
exports.RNN = RNN;
