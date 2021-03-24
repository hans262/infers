export declare class BPNet {
    w: number[][][];
    b: number[][];
    batch: number;
    rate: number;
    layerNum: number;
    shape: number[];
    constructor(shape: number[]);
    sigmoid(x: number): number;
    hypothetical(xs: number[]): number[][];
    calcdelta(ys: number[], hy: number[][]): number[][];
    update(hy: number[][], delta: number[][]): void;
    train(xs: number[][], ys: number[][]): void;
}
