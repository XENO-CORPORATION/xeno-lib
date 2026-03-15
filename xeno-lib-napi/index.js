// @xeno/lib — N-API bindings loader
//
// This file loads the platform-specific native addon built by napi-rs.
// After running `napi build --platform --release`, the .node file will
// be placed next to this file with a platform-specific name.

const { existsSync, readFileSync } = require('fs');
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

module.exports = nativeBinding;
