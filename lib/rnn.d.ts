import { Matrix } from '../src';
declare type ActivationFunction = 'Tanh' | 'Softmax';
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
    hideSize: number;
    constructor(data: string[]);
    afn(x: number, rows: number[], af?: ActivationFunction): number;
    afd(x: number, af?: ActivationFunction): number;
    oneHotXs(inputIndex: number): Matrix;
    oneHotYs(outputIndex: number): Matrix;
    forwardPropagation(data: {
        xs: Matrix;
        ys: Matrix;
    }[]): {
        xs: Matrix;
        ys: Matrix;
        st: Matrix;
        yt: Matrix;
        lastSt: Matrix;
    }[];
    backPropagation(hys: {
        xs: Matrix;
        ys: Matrix;
        st: Matrix;
        yt: Matrix;
        lastSt: Matrix;
    }[]): void;
    predict(): void;
    maxIndex(d: number[]): number;
    showWords(hys: {
        xs: Matrix;
        ys: Matrix;
        st: Matrix;
        yt: Matrix;
        lastSt: Matrix;
    }[]): string[];
    cost(hys: {
        xs: Matrix;
        ys: Matrix;
        st: Matrix;
        yt: Matrix;
        lastSt: Matrix;
    }[]): number;
    onehot(input: string[]): {
        xs: Matrix;
        ys: Matrix;
    }[];
    fit(): void;
}
export {};
