"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.toFixed = exports.testIntersectEdge = exports.testPointInsideEdge = exports.testPointInsidePolygon = exports.testIsPolygon = exports.testIsPath = exports.testIsEdge = void 0;
var typeCheck_1 = require("./typeCheck");
Object.defineProperty(exports, "testIsEdge", { enumerable: true, get: function () { return typeCheck_1.testIsEdge; } });
Object.defineProperty(exports, "testIsPath", { enumerable: true, get: function () { return typeCheck_1.testIsPath; } });
Object.defineProperty(exports, "testIsPolygon", { enumerable: true, get: function () { return typeCheck_1.testIsPolygon; } });
var booleanCheck_1 = require("./booleanCheck");
Object.defineProperty(exports, "testPointInsidePolygon", { enumerable: true, get: function () { return booleanCheck_1.testPointInsidePolygon; } });
Object.defineProperty(exports, "testPointInsideEdge", { enumerable: true, get: function () { return booleanCheck_1.testPointInsideEdge; } });
Object.defineProperty(exports, "testIntersectEdge", { enumerable: true, get: function () { return booleanCheck_1.testIntersectEdge; } });
var common_1 = require("./common");
Object.defineProperty(exports, "toFixed", { enumerable: true, get: function () { return common_1.toFixed; } });
//# sourceMappingURL=index.js.map