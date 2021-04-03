import { BPNet } from "./BPNet";
export declare class Model {
    static saveLocalstorage(model: BPNet): void;
    static saveFile(model: BPNet, path: string): void;
    static loadFile(path: string): BPNet;
}
