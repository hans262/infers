export declare class Point {
    X: number;
    Y: number;
    constructor(X: number, Y: number);
    contrast(pt: Point): boolean;
}
export declare class Edge {
    start: Point;
    end: Point;
    constructor(start: Point, end: Point);
    minXY(): Point;
    maxXY(): Point;
    testPointInEdge(pt: Point): boolean;
    testPointInsideEdge(pt: Point): boolean;
    testIntersectEdge(edge2: Edge): boolean;
}
export declare class Path {
    pts: Point[];
    constructor(pts: Point[]);
}
export declare class Polygon {
    pts: Point[];
    constructor(pts: Point[]);
    testPointInsidePolygon(pt: Point): number;
}
export declare function toFixed(num: number, fix: number): number;
export interface Rect {
    point: Point;
    width: number;
    height: number;
}
