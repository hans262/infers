import { ActivationFunction } from "./types";
import { Matrix } from "./matrix";
export declare function toFixed(num: number, fix: number): number;
export declare function upset(xs: Matrix, ys: Matrix): {
    xs: Matrix;
    ys: Matrix;
};
export declare function afn(x: number, rows: number[], af?: ActivationFunction): number;
export declare function afd(x: number, af?: ActivationFunction): number;
