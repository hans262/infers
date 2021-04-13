import { BPNet } from "./BPNet";
export declare class Model {
    static saveFile(model: BPNet, path: string): void;
    static loadFile(path: string): BPNet;
}
