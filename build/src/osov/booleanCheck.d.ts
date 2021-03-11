import type { Edge, Point, Polygon } from './types';
export declare function testPointInsideEdge(pt: Point, edge: Edge): boolean;
export declare function testPointInEdge(pt: Point, edge: Edge): boolean;
export declare function testIntersectEdge(edge1: Edge, edge2: Edge): boolean;
export declare function testPointInPolygon(pt: Point, polygon: Polygon): boolean;
export declare function testPointInsidePolygon(pt: Point, polygon: Polygon): number;
