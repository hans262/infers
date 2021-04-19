import { Matrix } from "./matrix";
export declare type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh' | 'Softmax';
export declare type Mode = 'sgd' | 'bgd' | 'mbgd';
export declare type NetShape = [number, (number | [number, ActivationFunction]), ...(number | [number, ActivationFunction])[]];
export interface NetConfig {
    mode?: Mode;
    rate?: number;
    w?: Matrix[];
    b?: Matrix[];
    scale?: Matrix;
}
export interface FitConf {
    epochs: number;
    batchSize?: number;
    async?: boolean;
    onBatch?: (batch: number, size: number, loss: number) => void;
    onEpoch?: (epoch: number, loss: number) => void;
    onTrainEnd?: (loss: number) => void;
}
