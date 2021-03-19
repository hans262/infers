"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function toFixed(num, fix) {
    const amount = 10 ** fix;
    return ~~(num * amount) / amount;
}
exports.toFixed = toFixed;
//# sourceMappingURL=common.js.map