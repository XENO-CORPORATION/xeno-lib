// @xeno/lib — N-API bindings loader with graceful degradation
//
// This file loads the platform-specific native addon built by napi-rs.
// After running `napi build --platform --release`, the .node file will
// be placed next to this file with a platform-specific name.
//
// If the native module is not available, exports a stub that throws
// descriptive errors, allowing apps to check availability before use.

const { existsSync } = require('fs');
const { join } = require('path');

const { platform, arch } = process;

let nativeBinding = null;
let loadError = null;

// Try loading the platform-specific binary
const triples = {
  'win32-x64': 'xeno-lib.win32-x64-msvc.node',
  'darwin-arm64': 'xeno-lib.darwin-arm64.node',
  'darwin-x64': 'xeno-lib.darwin-x64.node',
  'linux-x64': 'xeno-lib.linux-x64-gnu.node',
};

const tripleKey = `${platform}-${arch}`;
const binaryName = triples[tripleKey];

if (binaryName) {
  const binaryPath = join(__dirname, binaryName);
  try {
    if (existsSync(binaryPath)) {
      nativeBinding = require(binaryPath);
    } else {
      // Fallback: try loading from the generic name (debug builds)
      nativeBinding = require(join(__dirname, 'xeno-lib-napi.node'));
    }
  } catch (e) {
    loadError = e;
  }
} else {
  loadError = new Error(`Unsupported platform: ${platform}-${arch}`);
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError;
  }
  throw new Error(`Failed to load native binding for ${tripleKey}`);
}

// Expose a helper to check if the native module loaded successfully
nativeBinding.__isAvailable = true;
nativeBinding.__loadError = null;

// Model download URL base (Cloudflare R2 via updates.xenostudio.ai)
const MODEL_BASE_URL = 'https://updates.xenostudio.ai/models';

// Model file names required by each AI feature
// These match the actual filenames used by the Rust model loaders in src/
const MODEL_MANIFEST = {
  removeBackground: ['birefnet-general.onnx'],
  upscaleImage: ['realesrgan_x4plus.onnx'],
  restoreFaces: ['gfpgan.onnx'],
  colorize: ['ddcolor.onnx'],
  inpaint: ['lama.onnx'],
  detectFaces: ['scrfd_10g.onnx'],
  estimateDepth: ['depth_anything.onnx'],
  styleTransfer: ['style_mosaic.onnx', 'style_candy.onnx', 'style_rain_princess.onnx', 'style_udnie.onnx', 'style_pointillism.onnx'],
  extractText: ['paddle_det.onnx', 'paddle_rec.onnx'],
  detectPoses: ['movenet_lightning.onnx'],
  analyzeFaces: ['age_estimation.onnx', 'gender_classification.onnx', 'emotion_recognition.onnx'],
  transcribeAudio: ['whisper-base.onnx'],
  separateStems: ['demucs_hybrid.onnx'],
  interpolateFrames: ['rife-v4.6.onnx'],
  denoiseImage: [], // uses built-in spatial filter, no model needed
  denoiseAudio: [], // uses built-in limiter, no model needed
};

/**
 * Get the list of model files required for a specific AI function.
 * @param {string} functionName - Name of the AI function (e.g., "removeBackground")
 * @returns {string[]} Array of model file names needed
 */
nativeBinding.getRequiredModels = function getRequiredModels(functionName) {
  return MODEL_MANIFEST[functionName] || [];
};

/**
 * Get the download URL for a model file.
 * @param {string} modelName - Model file name (e.g., "birefnet.onnx")
 * @returns {string} Full download URL
 */
nativeBinding.getModelDownloadUrl = function getModelDownloadUrl(modelName) {
  return `${MODEL_BASE_URL}/${modelName}`;
};

/**
 * Check which models are missing for a given AI function.
 * @param {string} functionName - Name of the AI function
 * @returns {string[]} Array of missing model file names (empty if all present)
 */
nativeBinding.getMissingModels = function getMissingModels(functionName) {
  const required = MODEL_MANIFEST[functionName] || [];
  if (required.length === 0) return [];
  if (typeof nativeBinding.isModelAvailable !== 'function') return required;
  return required.filter(name => !nativeBinding.isModelAvailable(name));
};

// Attach the ModelManager class for model download/management
const { ModelManager } = require('./model-manager');
nativeBinding.ModelManager = ModelManager;

module.exports = nativeBinding;
