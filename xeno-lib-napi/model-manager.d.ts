/**
 * @xeno/lib Model Download Manager
 *
 * Downloads and manages ONNX model files from Cloudflare R2
 * (updates.xenostudio.ai) to ~/.xeno-lib/models/.
 */

/** Information about a single model from the manifest. */
export interface ModelInfo {
  /** ONNX filename (e.g., "birefnet-general.onnx") */
  file: string;
  /** Expected file size in bytes */
  size: number;
  /** SHA256 hash for verification (empty string if not yet computed) */
  sha256: string;
  /** Feature flags this model powers (e.g., ["background-removal"]) */
  features: string[];
  /** Apps that use this model (e.g., ["xeno-pixel", "xeno-hub"]) */
  apps: string[];
  /** Human-readable description */
  description: string;
}

/** The full model manifest fetched from R2 or bundled locally. */
export interface ModelManifest {
  /** Manifest schema version */
  version: string;
  /** Base URL for model downloads */
  baseUrl: string;
  /** Map of model name to model info */
  models: Record<string, ModelInfo>;
}

/** Progress information for a single model download. */
export interface DownloadProgress {
  /** Model name (key in manifest) */
  modelName: string;
  /** File name being downloaded */
  fileName: string;
  /** Bytes downloaded so far */
  downloaded: number;
  /** Total bytes expected (0 if unknown) */
  total: number;
  /** Download percentage (0-100, -1 if total unknown) */
  percent: number;
  /** Current status */
  status: 'downloading' | 'verifying' | 'complete' | 'error';
}

/** Status of a single model (downloaded or not). */
export interface ModelStatus {
  /** Model name (key in manifest) */
  name: string;
  /** ONNX filename */
  file: string;
  /** Whether the model file exists locally */
  downloaded: boolean;
  /** Expected size in bytes from manifest */
  expectedSize: number;
  /** Actual size on disk (0 if not downloaded) */
  actualSize: number;
  /** Features this model enables */
  features: string[];
  /** Apps that use this model */
  apps: string[];
  /** Human-readable description */
  description: string;
}

/** Progress for batch model downloads. */
export interface BatchDownloadProgress {
  /** Name of the model currently being downloaded */
  current: string;
  /** Zero-based index of the current model */
  currentIndex: number;
  /** Total number of models being downloaded */
  totalModels: number;
  /** Progress of the current model download */
  modelProgress: DownloadProgress;
}

/** Disk usage summary. */
export interface DiskUsage {
  /** Total bytes used by downloaded models */
  totalBytes: number;
  /** Number of downloaded models */
  modelCount: number;
}

export interface ModelManagerOptions {
  /** Override model directory (default: ~/.xeno-lib/models/) */
  modelDir?: string;
  /** Override manifest URL */
  manifestUrl?: string;
}

/**
 * Model download and management for xeno-lib AI features.
 *
 * Downloads models from Cloudflare R2 (updates.xenostudio.ai)
 * to ~/.xeno-lib/models/. Provides progress reporting, hash
 * verification, and status queries.
 */
export class ModelManager {
  constructor(options?: ModelManagerOptions);

  /**
   * Get the path to the local model directory.
   * Creates it if it does not exist.
   */
  getModelDir(): string;

  /**
   * Check if a specific model is available locally.
   * @param modelName - Model name (key in manifest, e.g., "birefnet")
   */
  isModelAvailable(modelName: string): Promise<boolean>;

  /**
   * Check if a model file exists by direct filename.
   * @param fileName - ONNX file name (e.g., "birefnet-general.onnx")
   */
  isModelFileAvailable(fileName: string): boolean;

  /** List all downloaded .onnx files in the model directory. */
  listDownloadedFiles(): string[];

  /** Get the full status of all models from the manifest. */
  getModelStatuses(): Promise<ModelStatus[]>;

  /**
   * Get models needed for a specific feature.
   * @param feature - Feature name (e.g., "background-removal", "upscale")
   */
  getModelsForFeature(feature: string): Promise<ModelStatus[]>;

  /**
   * Get models needed for a specific app.
   * @param app - App name (e.g., "xeno-pixel", "xeno-sound")
   */
  getModelsForApp(app: string): Promise<ModelStatus[]>;

  /** Get total disk usage of downloaded models. */
  getDiskUsage(): Promise<DiskUsage>;

  /**
   * Fetch the model manifest. Tries R2 first, falls back to bundled manifest.
   * @param forceRefresh - Force re-fetch from R2
   */
  getManifest(forceRefresh?: boolean): Promise<ModelManifest>;

  /**
   * Download a model by name with progress reporting.
   * @param modelName - Model name (key in manifest)
   * @param onProgress - Progress callback
   * @param signal - Optional abort signal to cancel download
   * @returns Path to the downloaded model file
   */
  downloadModel(
    modelName: string,
    onProgress?: (progress: DownloadProgress) => void,
    signal?: AbortSignal,
  ): Promise<string>;

  /**
   * Download multiple models in sequence with combined progress.
   * @param modelNames - Array of model names to download
   * @param onProgress - Progress callback with batch info
   * @param signal - Optional abort signal
   * @returns Paths to all downloaded model files
   */
  downloadModels(
    modelNames: string[],
    onProgress?: (progress: BatchDownloadProgress) => void,
    signal?: AbortSignal,
  ): Promise<string[]>;

  /**
   * Delete a downloaded model file.
   * @param modelName - Model name (key in manifest)
   * @returns true if the file was deleted
   */
  deleteModel(modelName: string): Promise<boolean>;
}
