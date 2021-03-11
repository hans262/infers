"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Model = void 0;
const matrix_1 = require("./matrix");
class Model {
    constructor(inputs, outputs) {
        this.rate = 0.01;
        this.inputs = inputs;
        this.outputs = outputs;
        this.M = inputs.shape[0];
        this.weights = this.initWeights();
        this.normal = this.inputs.normalization();
    }
    setRate(rate) {
        this.rate = rate;
    }
    initWeights() {
        const f = this.inputs.shape[1];
        return new Array(f + 1).fill(0);
    }
    hypothetical(xs) {
        let m = xs.self.map(x => {
            x = [1, ...x];
            return [this.weights.reduce((p, c, i) => p + c * x[i], 0)];
        });
        return new matrix_1.Matrix(m);
    }
    cost() {
        const M = this.M;
        let h = this.hypothetical(this.inputs);
        const sum = h.self.reduce((p, c, i) => {
            return p + (c[0] - this.outputs.get(i)[0]) ** 2;
        }, 0);
        return (1 / (2 * M)) * sum;
    }
    gradientDescent() {
        const M = this.M;
        const F = this.weights.length;
        const temps = this.initWeights();
        for (let i = 0; i < F; i++) {
            temps[i] = 0;
            let h = this.hypothetical(this.inputs);
            let sum = h.self.reduce((p, c, j) => {
                let xs = this.inputs.get(j);
                xs = [1, ...xs];
                return p + (c[0] - this.outputs.get(j)[0]) * xs[i];
            }, 0);
            temps[i] = this.weights[i] - this.rate * (1 / M) * sum;
        }
        for (let k = 0; k < F; k++) {
            this.weights[k] = temps[k];
        }
    }
    fit(batch, callback) {
        for (let i = 0; i < batch; i++) {
            this.gradientDescent();
            if (callback) {
                callback(i);
            }
        }
    }
    predict(xs) {
        return this.hypothetical(xs);
    }
}
exports.Model = Model;
//# sourceMappingURL=model.js.map