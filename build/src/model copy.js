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
        const F = this.inputs.shape[1] + 1;
        let m = [];
        for (let i = 0; i < F; i++) {
            m.push([0]);
        }
        return m;
    }
    hypothetical(xs) {
        let mm = new matrix_1.Matrix(this.weights);
        return xs.expansion(1).multiply(mm);
    }
    cost() {
        const M = this.M;
        let h = this.hypothetical(this.inputs);
        const sum = h.self.reduce((p, c, i) => {
            return p + (c[0] - this.outputs.get(i, 0)) ** 2;
        }, 0);
        return (1 / (2 * M)) * sum;
    }
    gradientDescent() {
        const M = this.M;
        const F = this.weights.length;
        const temps = this.initWeights();
        for (let i = 0; i < F; i++) {
            let h = this.hypothetical(this.inputs);
            let sum = h.self.reduce((p, c, j) => {
                let xs = this.inputs.getLine(j);
                xs = [1, ...xs];
                return p + (c[0] - this.outputs.get(j, 0)) * xs[i];
            }, 0);
            temps[i][0] = this.weights[i][0] - this.rate * (1 / M) * sum;
        }
        this.weights = temps;
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
//# sourceMappingURL=model%20copy.js.map