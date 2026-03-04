// Copyright 2023-2024 Google LLC
// Copyright 2025- flacenc-rs developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let built_rs = PathBuf::from(out_dir).join("built.rs");

    // `built` intermittently returns Ok but leaves no output file on some Windows setups.
    // Keep using it first, then fall back to a minimal generated file when needed.
    let _ = built::write_built_file();
    if built_rs.exists() {
        return;
    }

    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    let pkg_version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".to_string());
    let rustc_version = detect_rustc_version();
    let features = detect_features();

    let fallback = format!(
        "// auto-generated fallback build info\n\
pub static PROFILE: &str = {profile:?};\n\
pub static PKG_VERSION: &str = {pkg_version:?};\n\
pub static FEATURES_LOWERCASE_STR: &str = {features:?};\n\
pub static RUSTC_VERSION: &str = {rustc_version:?};\n"
    );

    fs::write(&built_rs, fallback).expect("Failed to write fallback build info");
}

fn detect_rustc_version() -> String {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    Command::new(rustc)
        .arg("-V")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn detect_features() -> String {
    let mut features: Vec<String> = env::vars()
        .filter_map(|(k, _)| k.strip_prefix("CARGO_FEATURE_").map(ToOwned::to_owned))
        .map(|name| name.to_ascii_lowercase().replace('_', "-"))
        .collect();
    features.sort();
    features.join(",")
}
