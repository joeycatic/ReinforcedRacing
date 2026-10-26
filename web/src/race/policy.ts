import * as ort from "onnxruntime-web/wasm";
import { clamp } from "../sim/physics";
import type { Action } from "../sim/types";

export interface Prediction {
  action: Action;
  value: number;
}
export interface Policy {
  predict(observations: number[]): Promise<Prediction>;
}
export class OnnxPolicy implements Policy {
  private constructor(private readonly session: ort.InferenceSession) {}
  static async create(bytes: ArrayBuffer): Promise<OnnxPolicy> {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = new URL(
      `${import.meta.env.BASE_URL}runtime/`,
      location.href,
    ).href;
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new OnnxPolicy(session);
  }
  async predict(observations: number[]): Promise<Prediction> {
    const input = new ort.Tensor(
      "float64",
      Float64Array.from(observations),
      [1, 22],
    );
    const outputs = await this.session.run({ observations: input });
    try {
      const values = outputs.action.data as Float32Array;
      const value = Number(outputs.value.data[0]);
      if (values.length !== 2 || ![...values, value].every(Number.isFinite))
        throw new Error("Policy returned invalid controls");
      return {
        action: [clamp(values[0], -1, 1), clamp(values[1], -1, 1)],
        value,
      };
    } finally {
      input.dispose();
      Object.values(outputs).forEach((output) => output.dispose());
    }
  }
  async release(): Promise<void> {
    await this.session.release();
  }
}
