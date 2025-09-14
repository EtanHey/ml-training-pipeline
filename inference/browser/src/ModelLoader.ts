import * as tf from '@tensorflow/tfjs';
import * as ort from 'onnxruntime-web';

export interface ModelConfig {
  modelUrl: string;
  modelType: 'tensorflow' | 'onnx' | 'tflite';
  inputShape?: number[];
  outputShape?: number[];
  preprocess?: (input: any) => tf.Tensor | Float32Array;
  postprocess?: (output: tf.Tensor | ort.Tensor) => any;
}

export class ModelLoader {
  private model: tf.GraphModel | tf.LayersModel | ort.InferenceSession | null = null;
  private config: ModelConfig;
  private isLoaded: boolean = false;

  constructor(config: ModelConfig) {
    this.config = config;
  }

  async load(): Promise<void> {
    if (this.isLoaded) {
      console.warn('Model already loaded');
      return;
    }

    try {
      switch (this.config.modelType) {
        case 'tensorflow':
          await this.loadTensorFlowModel();
          break;
        case 'onnx':
          await this.loadONNXModel();
          break;
        case 'tflite':
          await this.loadTFLiteModel();
          break;
        default:
          throw new Error(`Unsupported model type: ${this.config.modelType}`);
      }
      this.isLoaded = true;
      console.log(`Model loaded successfully: ${this.config.modelUrl}`);
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  private async loadTensorFlowModel(): Promise<void> {
    try {
      this.model = await tf.loadGraphModel(this.config.modelUrl);
    } catch {
      this.model = await tf.loadLayersModel(this.config.modelUrl);
    }
  }

  private async loadONNXModel(): Promise<void> {
    this.model = await ort.InferenceSession.create(this.config.modelUrl);
  }

  private async loadTFLiteModel(): Promise<void> {
    throw new Error('TFLite support coming soon');
  }

  async predict(input: any): Promise<any> {
    if (!this.isLoaded || !this.model) {
      throw new Error('Model not loaded. Call load() first.');
    }

    let processedInput = input;
    if (this.config.preprocess) {
      processedInput = this.config.preprocess(input);
    }

    let output;
    if (this.config.modelType === 'tensorflow') {
      const tfModel = this.model as tf.GraphModel | tf.LayersModel;
      const inputTensor = processedInput instanceof tf.Tensor
        ? processedInput
        : tf.tensor(processedInput);

      output = tfModel.predict(inputTensor) as tf.Tensor;
    } else if (this.config.modelType === 'onnx') {
      const session = this.model as ort.InferenceSession;
      const feeds: Record<string, ort.Tensor> = {};

      if (processedInput instanceof Float32Array) {
        const inputName = session.inputNames[0];
        feeds[inputName] = new ort.Tensor(
          'float32',
          processedInput,
          this.config.inputShape || [1, processedInput.length]
        );
      }

      const results = await session.run(feeds);
      output = results[session.outputNames[0]];
    }

    if (this.config.postprocess && output) {
      return this.config.postprocess(output);
    }

    return output;
  }

  dispose(): void {
    if (this.config.modelType === 'tensorflow' && this.model) {
      const tfModel = this.model as tf.GraphModel | tf.LayersModel;
      tfModel.dispose();
    }
    this.model = null;
    this.isLoaded = false;
  }

  getModelInfo(): any {
    if (!this.isLoaded || !this.model) {
      return null;
    }

    if (this.config.modelType === 'tensorflow') {
      const tfModel = this.model as tf.GraphModel | tf.LayersModel;
      return {
        inputs: tfModel.inputs,
        outputs: tfModel.outputs,
        weights: tfModel.getWeights ? tfModel.getWeights().length : 'N/A'
      };
    } else if (this.config.modelType === 'onnx') {
      const session = this.model as ort.InferenceSession;
      return {
        inputNames: session.inputNames,
        outputNames: session.outputNames
      };
    }

    return null;
  }
}