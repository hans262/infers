declare type Point = [number, number];
declare type Edge = [Point, Point];
declare type Path = Point[];
declare type Polygon = Point[];
interface Rect {
    point: Point;
    width: number;
    height: number;
}
export type { Point, Edge, Path, Polygon, Rect };
