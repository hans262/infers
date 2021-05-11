import { Matrix } from "./matrix"

/**激活函数类型*/
export type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh' | 'Softmax'

/**梯度更新方式*/
export type Mode = 'sgd' | 'bgd' | 'mbgd'

/**网络形状*/
export type NetShape = [number, (number | [number, ActivationFunction]), ...(number | [number, ActivationFunction])[]]

/**模型配置*/
export interface BPNetOptions {
  mode?: Mode
  rate?: number
  w?: Matrix[]
  b?: Matrix[]
  scale?: Matrix
}

/**训练配置*/
export interface TrainingOptions {
  epochs: number
  batchSize: number
  /**是否异步训练*/
  async: boolean
  onBatch?: (batch: number, size: number, loss: number) => void
  onEpoch?: (epoch: number, loss: number) => void
  onTrainEnd?: (loss: number) => void
}

export interface RNNOptions {
  trainData: string[]
  rate?: number
}

export interface RNNTrainingOptions {
  epochs?: number
  onEpochs?: (epoch: number, loss: number) => void
}

export interface RNNForwardResult {
  st: Matrix
  yt: Matrix
}