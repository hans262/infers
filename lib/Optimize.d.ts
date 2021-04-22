import { Matrix } from "./matrix";
export declare class Optimize {
    crossCost(hy: Matrix, ys: Matrix): number;
    momentum(xs: Matrix[], ys: Matrix): void;
    adaGrad(xs: Matrix[], ys: Matrix): void;
    adaDelta(xs: Matrix[], ys: Matrix): void;
}
