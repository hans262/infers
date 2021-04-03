"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Model = void 0;
const BPNet_1 = require("./BPNet");
const fs = require("fs");
const matrix_1 = require("./matrix");
class Model {
    static saveLocalstorage(model) { }
    static saveFile(model, path) {
        const conf = {
            mode: model.mode,
            shape: model.shape,
            rate: model.rate,
            scalem: model.scalem ? model.scalem.dataSync() : null,
            w: model.w.map(w => w.dataSync()),
            b: model.b.map(b => b.dataSync()),
        };
        fs.writeFileSync(path, JSON.stringify(conf));
    }
    static loadFile(path) {
        let file = fs.readFileSync(path).toString();
        let mp = JSON.parse(file);
        let nlayer = mp.shape.length;
        let w = [];
        let b = [];
        for (let l = 1; l < nlayer; l++) {
            w[l] = new matrix_1.Matrix(mp.w[l]);
            b[l] = new matrix_1.Matrix(mp.b[l]);
        }
        let scalem = mp.scalem ? new matrix_1.Matrix(mp.scalem) : undefined;
        return new BPNet_1.BPNet(mp.shape, {
            mode: mp.mode,
            rate: mp.mode,
            w, b, scalem
        });
    }
}
exports.Model = Model;
//# sourceMappingURL=Model.js.map