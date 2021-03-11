import type { Path, Polygon, Edge } from './types';
export declare function testIsEdge(pts: Point[]): pts is Edge;
export declare function testIsPath(pts: Point[]): pts is Path;
export declare function testIsPolygon(pts: Path): pts is Polygon;
