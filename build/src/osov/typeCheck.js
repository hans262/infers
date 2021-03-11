"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.testIsPolygon = exports.testIsPath = exports.testIsEdge = void 0;
const common_1 = require("./common");
function testIsEdge(pts) {
    return pts.length === 2 && !common_1.contrastPoint(pts[0], pts[1]);
}
exports.testIsEdge = testIsEdge;
function testIsPath(pts) {
    if (pts.length < 2)
        return false;
    const dd = pts.find((pt, i) => {
        const next = pts[i + 1];
        if (!next)
            return false;
        return common_1.contrastPoint(pt, next);
    });
    if (dd)
        return false;
    return true;
}
exports.testIsPath = testIsPath;
function testIsPolygon(pts) {
    const len = pts.length;
    if (len < 3)
        return false;
    const r0 = pts.map(p => p[0].toString() + p[1].toString()).sort();
    const r1 = r0.find((x, i) => x === r0[i + 1]);
    if (r1)
        return false;
    const firstPoint = pts[0];
    const r3 = pts.slice(1);
    const r4 = new Set(pts.map(p => p[0]));
    if (r4.size === 1)
        return false;
    const r5 = r3.map(p => (p[1] - firstPoint[1]) / p[0] - firstPoint[0]);
    const r6 = new Set(r5).size;
    if (r6 === 1)
        return false;
    return true;
}
exports.testIsPolygon = testIsPolygon;
//# sourceMappingURL=typeCheck.js.map