"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.testPointInsidePolygon = exports.testPointInPolygon = void 0;
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