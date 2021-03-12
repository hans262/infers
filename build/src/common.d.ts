import type { Point } from './graphical';
export declare function toFixed(num: number, fix: number): number;
interface Rect {
    point: Point;
    width: number;
    height: number;
}
export type { Rect };
