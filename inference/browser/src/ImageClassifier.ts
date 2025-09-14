import * as tf from '@tensorflow/tfjs';
import { ModelLoader, ModelConfig } from './ModelLoader';

export interface ClassificationResult {
  className: string;
  probability: number;
  index: number;
}

export interface ImageClassifierConfig extends ModelConfig {
  imageSize: number;
  numClasses: number;
  classNames?: string[];
  topK?: number;
}

export class ImageClassifier {
  private modelLoader: ModelLoader;
  private config: ImageClassifierConfig;

  constructor(config: ImageClassifierConfig) {
    this.config = {
      ...config,
      preprocess: this.preprocessImage.bind(this),
      postprocess: this.postprocessPredictions.bind(this)
    };
    this.modelLoader = new ModelLoader(this.config);
  }

  async load(): Promise<void> {
    await this.modelLoader.load();
  }

  private preprocessImage(imageElement: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement): tf.Tensor {
    return tf.tidy(() => {
      let imageTensor = tf.browser.fromPixels(imageElement);

      imageTensor = tf.image.resizeBilinear(
        imageTensor as tf.Tensor3D,
        [this.config.imageSize, this.config.imageSize]
      );

      imageTensor = imageTensor.expandDims(0);

      imageTensor = imageTensor.toFloat().div(tf.scalar(255));

      return imageTensor;
    });
  }

  private async postprocessPredictions(predictions: tf.Tensor): Promise<ClassificationResult[]> {
    const topK = this.config.topK || 5;
    const values = await predictions.data();
    const valuesArray = Array.from(values);

    const indices = valuesArray
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value)
      .slice(0, topK);

    return indices.map(({ value, index }) => ({
      className: this.config.classNames?.[index] || `Class ${index}`,
      probability: value,
      index
    }));
  }

  async classifyImage(
    image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | string
  ): Promise<ClassificationResult[]> {
    let imageElement: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement;

    if (typeof image === 'string') {
      imageElement = await this.loadImage(image);
    } else {
      imageElement = image;
    }

    const predictions = await this.modelLoader.predict(imageElement);
    return predictions;
  }

  async classifyBatch(
    images: Array<HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | string>
  ): Promise<ClassificationResult[][]> {
    const results: ClassificationResult[][] = [];

    for (const image of images) {
      const prediction = await this.classifyImage(image);
      results.push(prediction);
    }

    return results;
  }

  private loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  }

  async warmup(): Promise<void> {
    const dummyImage = tf.zeros([this.config.imageSize, this.config.imageSize, 3]);
    const canvas = document.createElement('canvas');
    await tf.browser.toPixels(dummyImage as tf.Tensor3D, canvas);
    await this.classifyImage(canvas);
    dummyImage.dispose();
  }

  dispose(): void {
    this.modelLoader.dispose();
  }
}