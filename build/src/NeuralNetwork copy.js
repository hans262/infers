"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class NeuralNetwork {
    constructor(shape) {
        this.w = [];
        this.b = [];
        this.rate = 0.5;
        this.shape = shape;
        this.layerNum = shape.length;
        for (let l = 1; l < this.layerNum; l++) {
            let item = [];
            let bitem = [];
            for (let j = 0; j < shape[l]; j++) {
                let temp = [];
                for (let i = 0; i < shape[l - 1]; i++) {
                    temp.push(0.5 - Math.random());
                }
                item.push(temp);
                bitem.push(0.5 - Math.random());
            }
            this.w[l] = item;
            this.b[l] = bitem;
        }
    }
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    predict(xs) {
        let ys = [];
        for (let l = 0; l < this.layerNum; l++) {
            if (l === 0) {
                ys[l] = xs;
                continue;
            }
            ys[l] = [];
            for (let j = 0; j < this.shape[l]; j++) {
                let u = 0;
                for (let i = 0; i < this.shape[l - 1]; i++) {
                    u += this.w[l][j][i] * ys[l - 1][i];
                }
                u += this.b[l][j];
                ys[l][j] = this.sigmoid(u);
            }
        }
        return ys;
    }
    calcdelta(ys, hy) {
        let delta = [];
        for (let l = this.layerNum - 1; l > 0; l--) {
            if (l === this.layerNum - 1) {
                delta[l] = [(ys[0] - hy[l][0]) * hy[l][0] * (1 - hy[l][0])];
            }
            else {
                delta[l] = [];
                for (let j = 0; j < this.shape[l]; j++) {
                    delta[l][j] = delta[l + 1][0] * this.w[l + 1][0][j] * hy[l][j] * (1 - hy[l][j]);
                }
            }
        }
        return delta;
    }
    update(hy, delta) {
        for (let l = 1; l < this.layerNum; l++) {
            for (let j = 0; j < this.shape[l]; j++) {
                for (let i = 0; i < this.shape[l - 1]; i++) {
                    this.w[l][j][i] += this.rate * delta[l][j] * hy[l - 1][i];
                }
                this.b[l][j] += this.rate * delta[l][j];
            }
        }
    }
    fit(xs, ys, batch) {
        for (let p = 0; p < batch; p++) {
            let loss = 0;
            for (let i = 0; i < xs.shape[0]; i++) {
                let hy = this.predict(xs.getRow(i));
                let delta = this.calcdelta(ys.getRow(i), hy);
                this.update(hy, delta);
                loss += ((ys.get(i, 0) - hy[2][0]) ** 2);
            }
            loss = loss / (2 * xs.shape[0]);
            if (p % 1000 === 0) {
                console.log(p, loss);
            }
        }
    }
}
exports.NeuralNetwork = NeuralNetwork;
//# sourceMappingURL=NeuralNetwork copy.js.map