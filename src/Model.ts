import { BPNet } from "./BPNet";
import * as fs from 'fs'
import { Matrix } from "./matrix";

export class Model {
  static saveFile(model: BPNet, path: string) {
    const conf = {
      mode: model.mode,
      shape: model.shape,
      rate: model.rate,
      scale: model.scale ? model.scale.dataSync() : undefined,
      w: model.w.map(w => w.dataSync()),
      b: model.b.map(b => b.dataSync()),
    }
    fs.writeFileSync(path, JSON.stringify(conf))
  }
  static loadFile(path: string) {
    let file = fs.readFileSync(path).toString()
    let mp = JSON.parse(file)
    let nlayer = mp.shape.length
    let w: Matrix[] = []
    let b: Matrix[] = []
    for (let l = 1; l < nlayer; l++) {
      w[l] = new Matrix(mp.w[l])
      b[l] = new Matrix(mp.b[l])
    }
    let scale = mp.scale ? new Matrix(mp.scale) : undefined
    return new BPNet(mp.shape, {
      mode: mp.mode,
      rate: mp.mode,
      w, b, scale
    })
  }
}