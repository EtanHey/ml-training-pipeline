import * as tf from '@tensorflow/tfjs';
import { ModelLoader, ModelConfig } from './ModelLoader';

export interface DetectionResult {
  bbox: [number, number, number, number]; // [x, y, width, height]
  className: string;
  score: number;
}

export interface ObjectDetectorConfig extends ModelConfig {
  imageSize: number;
  numClasses: number;
  classNames?: string[];
  scoreThreshold?: number;
  maxDetections?: number;
  iouThreshold?: number;
}

export class ObjectDetector {
  private modelLoader: ModelLoader;
  private config: ObjectDetectorConfig;

  constructor(config: ObjectDetectorConfig) {
    this.config = {
      scoreThreshold: 0.5,
      maxDetections: 100,
      iouThreshold: 0.5,
      ...config,
      preprocess: this.preprocessImage.bind(this),
      postprocess: this.postprocessDetections.bind(this)
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

      if (this.config.modelType === 'tensorflow') {
        imageTensor = imageTensor.toFloat().div(tf.scalar(255));
      } else {
        imageTensor = imageTensor.toFloat();
      }

      return imageTensor;
    });
  }

  private async postprocessDetections(outputs: any): Promise<DetectionResult[]> {
    if (Array.isArray(outputs)) {
      return this.postprocessYOLOOutput(outputs);
    } else {
      return this.postprocessSSDOutput(outputs);
    }
  }

  private async postprocessYOLOOutput(outputs: tf.Tensor[]): Promise<DetectionResult[]> {
    const [boxes, scores, classes, validDetections] = outputs;

    const boxesData = await boxes.data();
    const scoresData = await scores.data();
    const classesData = await classes.data();
    const validDetectionsData = await validDetections.data();

    const numDetections = validDetectionsData[0];
    const detections: DetectionResult[] = [];

    for (let i = 0; i < numDetections; i++) {
      const score = scoresData[i];
      if (score >= this.config.scoreThreshold!) {
        const bbox = [
          boxesData[i * 4],
          boxesData[i * 4 + 1],
          boxesData[i * 4 + 2] - boxesData[i * 4],
          boxesData[i * 4 + 3] - boxesData[i * 4 + 1]
        ] as [number, number, number, number];

        const classIndex = Math.round(classesData[i]);
        const className = this.config.classNames?.[classIndex] || `Class ${classIndex}`;

        detections.push({
          bbox,
          className,
          score
        });
      }
    }

    return detections;
  }

  private async postprocessSSDOutput(output: tf.Tensor): Promise<DetectionResult[]> {
    const predictions = await output.array();
    const detections: DetectionResult[] = [];

    if (Array.isArray(predictions[0])) {
      for (const pred of predictions[0]) {
        const [classId, score, x1, y1, x2, y2] = pred;
        if (score >= this.config.scoreThreshold!) {
          const className = this.config.classNames?.[classId] || `Class ${classId}`;
          detections.push({
            bbox: [x1, y1, x2 - x1, y2 - y1],
            className,
            score
          });
        }
      }
    }

    return this.nonMaxSuppression(detections);
  }

  private nonMaxSuppression(detections: DetectionResult[]): DetectionResult[] {
    if (detections.length === 0) return [];

    detections.sort((a, b) => b.score - a.score);

    const selected: DetectionResult[] = [];
    const used = new Set<number>();

    for (let i = 0; i < detections.length; i++) {
      if (used.has(i)) continue;

      const current = detections[i];
      selected.push(current);

      if (selected.length >= this.config.maxDetections!) break;

      for (let j = i + 1; j < detections.length; j++) {
        if (used.has(j)) continue;

        const iou = this.calculateIOU(current.bbox, detections[j].bbox);
        if (iou > this.config.iouThreshold!) {
          used.add(j);
        }
      }
    }

    return selected;
  }

  private calculateIOU(box1: [number, number, number, number], box2: [number, number, number, number]): number {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;

    const xOverlap = Math.max(0, Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2));
    const yOverlap = Math.max(0, Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2));

    const intersectionArea = xOverlap * yOverlap;
    const unionArea = w1 * h1 + w2 * h2 - intersectionArea;

    return intersectionArea / unionArea;
  }

  async detect(
    image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | string
  ): Promise<DetectionResult[]> {
    let imageElement: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement;

    if (typeof image === 'string') {
      imageElement = await this.loadImage(image);
    } else {
      imageElement = image;
    }

    const detections = await this.modelLoader.predict(imageElement);
    return detections;
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

  drawDetections(
    canvas: HTMLCanvasElement,
    image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
    detections: DetectionResult[]
  ): void {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = image.width;
    canvas.height = image.height;

    ctx.drawImage(image, 0, 0);

    const scaleX = image.width / this.config.imageSize;
    const scaleY = image.height / this.config.imageSize;

    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = '#00FF00';

    for (const detection of detections) {
      const [x, y, width, height] = detection.bbox;
      const scaledX = x * scaleX;
      const scaledY = y * scaleY;
      const scaledWidth = width * scaleX;
      const scaledHeight = height * scaleY;

      ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

      const label = `${detection.className} (${(detection.score * 100).toFixed(1)}%)`;
      const textWidth = ctx.measureText(label).width;
      const textHeight = 20;

      ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
      ctx.fillRect(scaledX, scaledY - textHeight, textWidth + 4, textHeight);

      ctx.fillStyle = '#000000';
      ctx.fillText(label, scaledX + 2, scaledY - 4);
    }
  }

  dispose(): void {
    this.modelLoader.dispose();
  }
}