"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.toFixed = exports.contrastPoint = void 0;
function contrastPoint(cond1, cond2) {
    return cond1[0] === cond2[0] && cond1[1] === cond2[1];
}
exports.contrastPoint = contrastPoint;
function toFixed(num, fix) {
    const amount = 10 ** fix;
    return ~~(num * amount) / amount;
}
exports.toFixed = toFixed;
//# sourceMappingURL=common.js.map