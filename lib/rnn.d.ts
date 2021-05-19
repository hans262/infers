import { Matrix } from '../src';
import type { RNNOptions, RNNTrainingOptions, RNNForwardResult } from './types';
export declare class RNN {
    U: Matrix;
    W: Matrix;
    V: Matrix;
    indexWord: {
        [index: string]: number;
    };
    wordIndex: {
        [index: number]: string;
    };
    trainData: string[][];
    inputSize: number;
    hidenSize: number;
    firstSt: Matrix;
    rate: number;
    constructor(opt: RNNOptions);
    oneHotX(inputIndex: number): Matrix;
    oneHotXs(input: string[]): Matrix[];
    oneHotY(outputIndex: number): Matrix;
    oneHotYs(input: string[]): Matrix[];
    forwardPropagation(xs: Matrix[]): RNNForwardResult[];
    calcForward(xs: Matrix, lastSt?: Matrix): {
        st: Matrix;
        yt: Matrix;
    };
    backPropagation(hy: RNNForwardResult[], xs: Matrix[], ys: Matrix[]): void;
    predict(input: string, length?: number): string | undefined;
    cost(hy: RNNForwardResult[], ys: Matrix[]): number;
    fit(opt?: RNNTrainingOptions): void;
}
