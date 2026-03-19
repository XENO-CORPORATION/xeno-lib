// @xeno/lib Model Download Manager
//
// Provides model download, verification, and management for all apps
// that consume xeno-lib AI features. Downloads models from Cloudflare R2
// (updates.xenostudio.ai) to ~/.xeno-lib/models/.
//
// Usage:
//   const { ModelManager } = require('@xeno/lib/model-manager');
//   const manager = new ModelManager();
//   await manager.downloadModel('birefnet', (progress) => console.log(progress));

const { createWriteStream, existsSync, mkdirSync, statSync, readdirSync, unlinkSync, renameSync } = require('fs');
const { join } = require('path');
const { homedir } = require('os');
const { pipeline } = require('stream/promises');
const https = require('https');
const http = require('http');
const crypto = require('crypto');
const { createReadStream } = require('fs');

// Manifest URL on R2
const MANIFEST_URL = 'https://updates.xenostudio.ai/models/manifest.json';

// Fallback: bundled manifest from the repo
const BUNDLED_MANIFEST_PATH = join(__dirname, '..', 'models', 'manifest.json');

/**
 * @typedef {Object} ModelInfo
 * @property {string} file - ONNX filename
 * @property {number} size - Expected file size in bytes
 * @property {string} sha256 - SHA256 hash for verification (empty string if unknown)
 * @property {string[]} features - Feature flags this model powers
 * @property {string[]} apps - Apps that use this model
 * @property {string} description - Human-readable description
 */

/**
 * @typedef {Object} ModelManifest
 * @property {string} version - Manifest version
 * @property {string} baseUrl - Base URL for model downloads
 * @property {Object.<string, ModelInfo>} models - Map of model name to info
 */

/**
 * @typedef {Object} DownloadProgress
 * @property {string} modelName - Name of the model being downloaded
 * @property {string} fileName - File name being downloaded
 * @property {number} downloaded - Bytes downloaded so far
 * @property {number} total - Total bytes expected (0 if unknown)
 * @property {number} percent - Download percentage (0-100, -1 if unknown)
 * @property {'downloading' | 'verifying' | 'complete' | 'error'} status
 */

/**
 * @typedef {Object} ModelStatus
 * @property {string} name - Model name (key in manifest)
 * @property {string} file - ONNX filename
 * @property {boolean} downloaded - Whether the model file exists locally
 * @property {number} expectedSize - Expected size in bytes from manifest
 * @property {number} actualSize - Actual size on disk (0 if not downloaded)
 * @property {string[]} features - Features this model enables
 * @property {string[]} apps - Apps that use this model
 * @property {string} description - Human-readable description
 */

class ModelManager {
  /**
   * Create a new ModelManager instance.
   * @param {Object} [options]
   * @param {string} [options.modelDir] - Override model directory (default: ~/.xeno-lib/models/)
   * @param {string} [options.manifestUrl] - Override manifest URL
   */
  constructor(options = {}) {
    this._modelDir = options.modelDir || join(homedir(), '.xeno-lib', 'models');
    this._manifestUrl = options.manifestUrl || MANIFEST_URL;
    /** @type {ModelManifest | null} */
    this._cachedManifest = null;
  }

  /**
   * Get the path to the local model directory.
   * Creates it if it does not exist.
   * @returns {string} Absolute path to model directory
   */
  getModelDir() {
    if (!existsSync(this._modelDir)) {
      mkdirSync(this._modelDir, { recursive: true });
    }
    return this._modelDir;
  }

  /**
   * Check if a specific model is available locally.
   * @param {string} modelName - Model name (key in manifest, e.g., "birefnet")
   * @returns {Promise<boolean>} true if the model file exists
   */
  async isModelAvailable(modelName) {
    const manifest = await this.getManifest();
    const info = manifest.models[modelName];
    if (!info) return false;
    const filePath = join(this.getModelDir(), info.file);
    return existsSync(filePath);
  }

  /**
   * Check if a model file exists by direct filename.
   * @param {string} fileName - ONNX file name (e.g., "birefnet-general.onnx")
   * @returns {boolean} true if the file exists
   */
  isModelFileAvailable(fileName) {
    return existsSync(join(this.getModelDir(), fileName));
  }

  /**
   * List all downloaded model files.
   * @returns {string[]} Array of .onnx filenames present in the model directory
   */
  listDownloadedFiles() {
    const dir = this.getModelDir();
    if (!existsSync(dir)) return [];
    return readdirSync(dir).filter(f => f.endsWith('.onnx'));
  }

  /**
   * Get the full status of all models from the manifest.
   * @returns {Promise<ModelStatus[]>} Array of model statuses
   */
  async getModelStatuses() {
    const manifest = await this.getManifest();
    const dir = this.getModelDir();
    const statuses = [];

    for (const [name, info] of Object.entries(manifest.models)) {
      const filePath = join(dir, info.file);
      const downloaded = existsSync(filePath);
      let actualSize = 0;
      if (downloaded) {
        try {
          actualSize = statSync(filePath).size;
        } catch {
          // ignore stat errors
        }
      }
      statuses.push({
        name,
        file: info.file,
        downloaded,
        expectedSize: info.size,
        actualSize,
        features: info.features,
        apps: info.apps,
        description: info.description,
      });
    }

    return statuses;
  }

  /**
   * Get models needed for a specific feature.
   * @param {string} feature - Feature name (e.g., "background-removal", "upscale")
   * @returns {Promise<ModelStatus[]>} Models for that feature
   */
  async getModelsForFeature(feature) {
    const statuses = await this.getModelStatuses();
    return statuses.filter(s => s.features.includes(feature));
  }

  /**
   * Get models needed for a specific app.
   * @param {string} app - App name (e.g., "xeno-pixel", "xeno-sound")
   * @returns {Promise<ModelStatus[]>} Models for that app
   */
  async getModelsForApp(app) {
    const statuses = await this.getModelStatuses();
    return statuses.filter(s => s.apps.includes(app));
  }

  /**
   * Get total disk usage of downloaded models.
   * @returns {Promise<{ totalBytes: number, modelCount: number }>}
   */
  async getDiskUsage() {
    const statuses = await this.getModelStatuses();
    let totalBytes = 0;
    let modelCount = 0;
    for (const s of statuses) {
      if (s.downloaded) {
        totalBytes += s.actualSize;
        modelCount++;
      }
    }
    return { totalBytes, modelCount };
  }

  /**
   * Fetch the model manifest. Tries R2 first, falls back to bundled manifest.
   * Caches the result in memory.
   * @param {boolean} [forceRefresh=false] - Force re-fetch from R2
   * @returns {Promise<ModelManifest>}
   */
  async getManifest(forceRefresh = false) {
    if (this._cachedManifest && !forceRefresh) {
      return this._cachedManifest;
    }

    // Try fetching from R2
    try {
      const data = await this._fetchJson(this._manifestUrl);
      this._cachedManifest = data;
      return data;
    } catch {
      // Fall back to bundled manifest
    }

    // Load bundled manifest
    try {
      const data = require(BUNDLED_MANIFEST_PATH);
      this._cachedManifest = data;
      return data;
    } catch {
      throw new Error(
        'Failed to load model manifest from both R2 and bundled file. ' +
        'Ensure models/manifest.json exists in the xeno-lib package.'
      );
    }
  }

  /**
   * Download a model by name with progress reporting.
   * @param {string} modelName - Model name (key in manifest)
   * @param {(progress: DownloadProgress) => void} [onProgress] - Progress callback
   * @param {AbortSignal} [signal] - Optional abort signal to cancel download
   * @returns {Promise<string>} Path to the downloaded model file
   */
  async downloadModel(modelName, onProgress, signal) {
    const manifest = await this.getManifest();
    const info = manifest.models[modelName];
    if (!info) {
      throw new Error(`Unknown model: "${modelName}". Available: ${Object.keys(manifest.models).join(', ')}`);
    }

    const dir = this.getModelDir();
    const finalPath = join(dir, info.file);
    const tempPath = finalPath + '.downloading';

    // Check if already downloaded
    if (existsSync(finalPath)) {
      const stat = statSync(finalPath);
      // If size matches (or manifest size is approximate), consider it done
      if (info.size === 0 || Math.abs(stat.size - info.size) < info.size * 0.1) {
        if (onProgress) {
          onProgress({
            modelName,
            fileName: info.file,
            downloaded: stat.size,
            total: stat.size,
            percent: 100,
            status: 'complete',
          });
        }
        return finalPath;
      }
    }

    const url = `${manifest.baseUrl}/${info.file}`;

    if (onProgress) {
      onProgress({
        modelName,
        fileName: info.file,
        downloaded: 0,
        total: info.size,
        percent: 0,
        status: 'downloading',
      });
    }

    try {
      await this._downloadFile(url, tempPath, info.size, (downloaded, total) => {
        if (signal && signal.aborted) return;
        if (onProgress) {
          const percent = total > 0 ? Math.round((downloaded / total) * 100) : -1;
          onProgress({
            modelName,
            fileName: info.file,
            downloaded,
            total,
            percent,
            status: 'downloading',
          });
        }
      }, signal);

      // Verify hash if available
      if (info.sha256 && info.sha256.length > 0) {
        if (onProgress) {
          onProgress({
            modelName,
            fileName: info.file,
            downloaded: info.size,
            total: info.size,
            percent: 100,
            status: 'verifying',
          });
        }
        const hash = await this._hashFile(tempPath);
        if (hash !== info.sha256) {
          unlinkSync(tempPath);
          throw new Error(
            `Hash mismatch for ${info.file}: expected ${info.sha256}, got ${hash}`
          );
        }
      } else {
        console.warn(
          `[xeno-lib] SHA256 hash not available for model "${modelName}" (${info.file}) — skipping integrity verification. ` +
          `Populate the sha256 field in models/manifest.json to enable hash verification.`
        );
      }

      // Atomic rename
      renameSync(tempPath, finalPath);

      if (onProgress) {
        onProgress({
          modelName,
          fileName: info.file,
          downloaded: info.size,
          total: info.size,
          percent: 100,
          status: 'complete',
        });
      }

      return finalPath;
    } catch (err) {
      // Clean up temp file on error
      try {
        if (existsSync(tempPath)) unlinkSync(tempPath);
      } catch {
        // ignore cleanup errors
      }

      if (onProgress) {
        onProgress({
          modelName,
          fileName: info.file,
          downloaded: 0,
          total: info.size,
          percent: 0,
          status: 'error',
        });
      }

      throw err;
    }
  }

  /**
   * Download multiple models in sequence with combined progress.
   * @param {string[]} modelNames - Array of model names to download
   * @param {(progress: { current: string, currentIndex: number, totalModels: number, modelProgress: DownloadProgress }) => void} [onProgress]
   * @param {AbortSignal} [signal] - Optional abort signal
   * @returns {Promise<string[]>} Paths to all downloaded model files
   */
  async downloadModels(modelNames, onProgress, signal) {
    const paths = [];
    for (let i = 0; i < modelNames.length; i++) {
      if (signal && signal.aborted) {
        throw new Error('Download cancelled');
      }
      const name = modelNames[i];
      const path = await this.downloadModel(name, (modelProgress) => {
        if (onProgress) {
          onProgress({
            current: name,
            currentIndex: i,
            totalModels: modelNames.length,
            modelProgress,
          });
        }
      }, signal);
      paths.push(path);
    }
    return paths;
  }

  /**
   * Delete a downloaded model file.
   * @param {string} modelName - Model name (key in manifest)
   * @returns {Promise<boolean>} true if the file was deleted
   */
  async deleteModel(modelName) {
    const manifest = await this.getManifest();
    const info = manifest.models[modelName];
    if (!info) return false;

    const filePath = join(this.getModelDir(), info.file);
    if (existsSync(filePath)) {
      unlinkSync(filePath);
      return true;
    }
    return false;
  }

  // ---- Internal helpers ----

  /**
   * Fetch JSON from a URL.
   * @param {string} url
   * @returns {Promise<any>}
   * @private
   */
  _fetchJson(url) {
    return new Promise((resolve, reject) => {
      const client = url.startsWith('https') ? https : http;
      const req = client.get(url, { timeout: 10000 }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          // Follow redirect
          this._fetchJson(res.headers.location).then(resolve, reject);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} fetching ${url}`));
          return;
        }
        let data = '';
        res.on('data', chunk => { data += chunk; });
        res.on('end', () => {
          try {
            resolve(JSON.parse(data));
          } catch (e) {
            reject(new Error(`Invalid JSON from ${url}: ${e.message}`));
          }
        });
        res.on('error', reject);
      });
      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error(`Timeout fetching ${url}`));
      });
    });
  }

  /**
   * Download a file from a URL with progress.
   * @param {string} url
   * @param {string} destPath
   * @param {number} expectedSize
   * @param {(downloaded: number, total: number) => void} onProgress
   * @param {AbortSignal} [signal]
   * @returns {Promise<void>}
   * @private
   */
  _downloadFile(url, destPath, expectedSize, onProgress, signal) {
    return new Promise((resolve, reject) => {
      if (signal && signal.aborted) {
        reject(new Error('Download cancelled'));
        return;
      }

      const client = url.startsWith('https') ? https : http;
      const req = client.get(url, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          // Follow redirect
          this._downloadFile(res.headers.location, destPath, expectedSize, onProgress, signal)
            .then(resolve, reject);
          return;
        }
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} downloading ${url}`));
          return;
        }

        const total = parseInt(res.headers['content-length'], 10) || expectedSize;
        let downloaded = 0;

        const fileStream = createWriteStream(destPath);

        res.on('data', (chunk) => {
          if (signal && signal.aborted) {
            res.destroy();
            fileStream.close();
            reject(new Error('Download cancelled'));
            return;
          }
          downloaded += chunk.length;
          onProgress(downloaded, total);
        });

        res.pipe(fileStream);

        fileStream.on('finish', () => {
          fileStream.close();
          resolve();
        });

        fileStream.on('error', (err) => {
          fileStream.close();
          reject(err);
        });

        res.on('error', (err) => {
          fileStream.close();
          reject(err);
        });
      });

      req.on('error', reject);

      if (signal) {
        signal.addEventListener('abort', () => {
          req.destroy();
          reject(new Error('Download cancelled'));
        }, { once: true });
      }
    });
  }

  /**
   * Compute SHA256 hash of a file.
   * @param {string} filePath
   * @returns {Promise<string>} Hex-encoded SHA256
   * @private
   */
  _hashFile(filePath) {
    return new Promise((resolve, reject) => {
      const hash = crypto.createHash('sha256');
      const stream = createReadStream(filePath);
      stream.on('data', chunk => hash.update(chunk));
      stream.on('end', () => resolve(hash.digest('hex')));
      stream.on('error', reject);
    });
  }
}

module.exports = { ModelManager };
