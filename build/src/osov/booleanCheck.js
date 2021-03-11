"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.testPointInsidePolygon = exports.testPointInPolygon = exports.testIntersectEdge = exports.testPointInEdge = exports.testPointInsideEdge = void 0;
const common_1 = require("./common");
function testPointInsideEdge(pt, edge) {
    const [cond1, cond2] = edge;
    const [maxX, maxY, minX, minY] = [
        Math.max(cond1[0], cond2[0]),
        Math.max(cond1[1], cond2[1]),
        Math.min(cond1[0], cond2[0]),
        Math.min(cond1[1], cond2[1]),
    ];
    if (pt[0] < minX || pt[0] > maxX || pt[1] < minY || pt[1] > maxY) {
        return false;
    }
    if (common_1.contrastPoint(pt, cond1) || common_1.contrastPoint(pt, cond2)) {
        return true;
    }
    const s1 = Math.sqrt(Math.abs(pt[0] - cond1[0]) ** 2 + Math.abs(pt[1] - cond1[1]) ** 2);
    const s2 = Math.sqrt(Math.abs(pt[0] - cond2[0]) ** 2 + Math.abs(pt[1] - cond2[1]) ** 2);
    const s0 = Math.sqrt(Math.abs(cond1[0] - cond2[0]) ** 2 + Math.abs(cond1[1] - cond2[1]) ** 2);
    return common_1.toFixed(s1 + s2, 1) === common_1.toFixed(s0, 1);
}
exports.testPointInsideEdge = testPointInsideEdge;
function testPointInEdge(pt, edge) {
    const [cond1, cond2] = edge;
    if (common_1.contrastPoint(pt, cond1) || common_1.contrastPoint(pt, cond2)) {
        return true;
    }
    if (cond1[0] === cond2[0]) {
        return pt[0] === cond1[0];
    }
    if (cond1[1] === cond2[1]) {
        return pt[1] === cond1[1];
    }
    if (pt[0] === cond1[0] || pt[0] === cond2[0]) {
        return false;
    }
    let x1 = Math.abs((pt[1] - cond1[1]) / (pt[0] - cond1[0]));
    let x2 = Math.abs((pt[1] - cond2[1]) / (pt[0] - cond2[0]));
    return common_1.toFixed(x1, 1) === common_1.toFixed(x2, 1);
}
exports.testPointInEdge = testPointInEdge;
function testIntersectEdge(edge1, edge2) {
    const [p1, p2] = edge1;
    const [q1, q2] = edge2;
    if (!(Math.min(p1[0], p2[0]) <= Math.max(q1[0], q1[0]) &&
        Math.min(q1[0], q2[0]) <= Math.max(p1[0], p2[0]) &&
        Math.min(p1[1], p2[1]) <= Math.max(q1[1], q1[1]) &&
        Math.min(q1[1], q2[1]) <= Math.max(p1[1], p2[1])))
        return false;
    const c1 = ((p1[0] - q1[0]) * (q2[1] - q1[1]) - (p1[1] - q1[1]) * (q2[0] - q1[0])) * ((q2[0] - q1[0]) * (p2[1] - q1[1]) - (q2[1] - q1[1]) * (p2[0] - q1[0]));
    const c2 = ((q1[0] - p1[0]) * (p1[1] - p2[1]) - (q1[1] - p1[1]) * (p2[0] - p1[0])) * ((p2[0] - p1[0]) * (q2[1] - p1[1]) - (p2[1] - p1[1]) * (q2[0] - p1[0]));
    console.log(c1);
    console.log(c2);
    if (c1 > 0 &&
        c2 > 0) {
        return true;
    }
    return false;
}
exports.testIntersectEdge = testIntersectEdge;
function testPointInPolygon(pt, polygon) {
    const [polygonXs, polygonYs] = [
        polygon.map(p => p[0]),
        polygon.map(p => p[1])
    ];
    const [maxX, maxY, minX, minY] = [
        Math.max(...polygonXs),
        Math.max(...polygonYs),
        Math.min(...polygonXs),
        Math.min(...polygonYs),
    ];
    if (pt[0] < minX || pt[0] > maxX || pt[1] < minY || pt[1] > maxY) {
        return false;
    }
    for (let i = 0; i < polygon.length; i++) {
        const cp = polygon[i];
    }
    return true;
}
exports.testPointInPolygon = testPointInPolygon;
function testPointInsidePolygon(pt, polygon) {
    var result = 0, cnt = polygon.length;
    if (cnt < 3)
        return 0;
    var ip = polygon[0];
    for (var i = 1; i <= cnt; ++i) {
        var ipNext = i === cnt ? polygon[0] : polygon[i];
        if (ipNext[1] === pt[1]) {
            if ((ipNext[0] === pt[0]) || (ip[1] === pt[1] && ((ipNext[0] > pt[0]) === (ip[0] < pt[0]))))
                return -1;
        }
        if ((ip[1] < pt[1]) !== (ipNext[1] < pt[1])) {
            if (ip[0] >= pt[0]) {
                if (ipNext[0] > pt[0])
                    result = 1 - result;
                else {
                    var d = (ip[0] - pt[0]) * (ipNext[1] - pt[1]) - (ipNext[0] - pt[0]) * (ip[1] - pt[1]);
                    if (d === 0)
                        return -1;
                    else if ((d > 0) === (ipNext[1] > ip[1]))
                        result = 1 - result;
                }
            }
            else {
                if (ipNext[0] > pt[0]) {
                    var d = (ip[0] - pt[0]) * (ipNext[1] - pt[1]) - (ipNext[0] - pt[0]) * (ip[1] - pt[1]);
                    if (d === 0)
                        return -1;
                    else if ((d > 0) === (ipNext[1] > ip[1]))
                        result = 1 - result;
                }
            }
        }
        ip = ipNext;
    }
    return result;
}
exports.testPointInsidePolygon = testPointInsidePolygon;
//# sourceMappingURL=booleanCheck.js.map