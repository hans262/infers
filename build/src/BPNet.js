"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class BPNet {
    constructor(shape) {
        this.w = [];
        this.b = [];
        this.batch = 100000;
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
    hypothetical(xs) {
        let y = [];
        y[0] = xs;
        for (let l = 1; l < this.layerNum; l++) {
            y[l] = [];
            for (let j = 0; j < this.shape[l]; j++) {
                let u = 0;
                for (let i = 0; i < this.shape[l - 1]; i++) {
                    u += this.w[l][j][i] * y[l - 1][i];
                }
                u += this.b[l][j];
                y[l][j] = this.sigmoid(u);
            }
        }
        return y;
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
    train(xs, ys) {
        for (let p = 0; p < this.batch; p++) {
            let loss = 0;
            for (let i = 0; i < xs.length; i++) {
                let hy = this.hypothetical(xs[i]);
                let delta = this.calcdelta(ys[i], hy);
                this.update(hy, delta);
                loss += ((ys[i][0] - hy[2][0]) ** 2);
            }
            loss = loss / (2 * xs.length);
            if (p % 1000 === 0) {
                console.log(p, loss);
            }
        }
    }
}
exports.BPNet = BPNet;
//# sourceMappingURL=BPNet.js.map