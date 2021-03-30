"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Layer = void 0;
const matrix_1 = require("./matrix");
class Layer {
    constructor(index, unit, lastUnit, lastLayer, af) {
        this.index = index;
        this.unit = unit;
        this.lastUnit = lastUnit;
        this.lastLayer = lastLayer;
        this.af = af;
        this.w = matrix_1.Matrix.generate(unit, this.lastUnit);
        this.b = matrix_1.Matrix.generate(1, unit);
    }
    calchy(xs) {
        if (this.index === 1) {
            this.hy = xs;
        }
        else {
            this.hy = this.lastLayer.hy.multiply(this.w.T).atomicOperation((item, _, j) => this.afn(item + this.b.get(0, j)));
        }
    }
    afn(x) {
        switch (this.af) {
            case 'Sigmoid':
                return 1 / (1 + Math.exp(-x));
            case 'Relu':
                return x >= 0 ? x : 0;
            case 'Tanh':
                return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
            default:
                return x;
        }
    }
    afd(x) {
        switch (this.af) {
            case 'Sigmoid':
                return x * (1 - x);
            case 'Relu':
                return x >= 0 ? 1 : 0;
            case 'Tanh':
                return 1 - ((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x))) ** 2;
            default:
                return 1;
        }
    }
}
exports.Layer = Layer;
//# sourceMappingURL=Layer.js.map