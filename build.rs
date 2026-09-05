//! Record the resolved Rust dependency, independently of Python's installation.

use std::{env, error::Error, path::PathBuf, process::Command};

use serde_json::Value;

fn reducers_provenance() -> Result<(String, String), Box<dyn Error>> {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("no manifest directory")?)
        .join("Cargo.toml");
    let output = Command::new(env::var_os("CARGO").ok_or("no Cargo executable")?)
        .args(["metadata", "--format-version", "1", "--offline", "--locked"])
        .arg("--manifest-path")
        .arg(&manifest)
        .output()?;
    if !output.status.success() {
        return Err(format!("cargo metadata exited with {}", output.status).into());
    }
    let metadata: Value = serde_json::from_slice(&output.stdout)?;
    let workspace = metadata["workspace_root"]
        .as_str()
        .ok_or("no workspace root")?;
    println!(
        "cargo:rerun-if-changed={}",
        PathBuf::from(workspace).join("Cargo.lock").display()
    );
    let root = metadata["resolve"]["root"]
        .as_str()
        .ok_or("no resolved root")?;
    let node = metadata["resolve"]["nodes"]
        .as_array()
        .ok_or("no resolved nodes")?
        .iter()
        .find(|node| node["id"].as_str() == Some(root))
        .ok_or("root node missing")?;
    let dependency = node["deps"]
        .as_array()
        .ok_or("no root dependencies")?
        .iter()
        .find(|dep| dep["name"].as_str() == Some("reducers"))
        .ok_or("reducers dependency missing")?;
    // Cargo package IDs are opaque: compare them rather than parsing them.
    let package = metadata["packages"]
        .as_array()
        .ok_or("no packages")?
        .iter()
        .find(|package| package["id"] == dependency["pkg"])
        .ok_or("reducers package missing")?;
    if let Some(path) = package["manifest_path"].as_str() {
        println!("cargo:rerun-if-changed={path}");
    }
    let version = package["version"].as_str().ok_or("no reducers version")?;
    let source = package["source"].as_str().unwrap_or("path");
    Ok((version.to_owned(), source.to_owned()))
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    let (version, source) = reducers_provenance().unwrap_or_else(|error| {
        // Missing optional provenance must neither break a build nor be guessed.
        println!("cargo:warning=Rust reducers provenance unavailable: {error}");
        ("unknown".to_owned(), "unknown".to_owned())
    });
    println!("cargo:rustc-env=IMC_REDUCERS_VERSION={version}");
    println!("cargo:rustc-env=IMC_REDUCERS_SOURCE={source}");
}
