export declare class Point {
    X: number;
    Y: number;
    constructor(X: number, Y: number);
    contrast(pt: Point): boolean;
}
export declare class Edge {
    start: Point;
    end: Point;
    constructor(cond1: [number, number], cond2: [number, number]);
    minXY(): Point;
    maxXY(): Point;
    testPointIn(pt: Point): boolean;
    testPointInside(pt: Point): boolean;
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
