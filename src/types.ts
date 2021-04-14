import { Matrix } from "./matrix"

/**激活函数类型*/
export type ActivationFunction = 'Sigmoid' | 'Relu' | 'Tanh' | 'Softmax'

/**梯度更新方式*/
export type Mode = 'sgd' | 'bgd' | 'mbgd'

/**网络形状*/
export type NetShape = (number | [number, ActivationFunction])[]

/**模型配置*/
export interface NetConfig {
  mode?: Mode
  rate?: number
  w?: Matrix[]
  b?: Matrix[]
  scale?: Matrix
}

/**训练配置*/
export interface FitConf {
  epochs: number
  batchSize?: number
  async?: boolean
  onBatch?: (batch: number, size: number, loss: number) => void
  onEpoch?: (epoch: number, loss: number) => void
  onTrainEnd?: (loss: number) => void
}