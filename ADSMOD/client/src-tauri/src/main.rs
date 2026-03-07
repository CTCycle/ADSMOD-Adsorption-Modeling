#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager, RunEvent, State};

const BACKEND_DEFAULT_HOST: &str = "127.0.0.1";
const BACKEND_PORT_ENV: &str = "ADSMOD_BACKEND_PORT";
const BACKEND_HOST_ENV: &str = "ADSMOD_BACKEND_HOST";
const BACKEND_BASE_DIR_ENV: &str = "ADSMOD_BASE_DIR";
const BACKEND_HEALTH_PATH: &str = "/health";
const BACKEND_BOOT_TIMEOUT: Duration = Duration::from_secs(90);
const BACKEND_SIDECAR_STEM: &str = "adsmod_backend";

const DEFAULT_CONFIG_JSON: &str = include_str!("../../../settings/configurations.json");
const DEFAULT_ENV_LOCAL: &str = include_str!("../../../settings/.env.local.example");

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeConfig {
    api_origin: String,
    backend_host: String,
    backend_port: u16,
    base_dir: String,
}

#[derive(Default)]
struct BackendRuntime {
    child: Mutex<Option<Child>>,
    config: Mutex<Option<RuntimeConfig>>,
}

#[tauri::command]
fn get_runtime_config(state: State<'_, BackendRuntime>) -> Result<RuntimeConfig, String> {
    let guard = state
        .config
        .lock()
        .map_err(|_| "Failed to acquire runtime config lock".to_string())?;
    guard
        .clone()
        .ok_or_else(|| "Runtime config is not available".to_string())
}

fn choose_backend_port() -> Result<u16, String> {
    if let Ok(raw) = std::env::var(BACKEND_PORT_ENV) {
        let parsed = raw
            .parse::<u16>()
            .map_err(|_| format!("Invalid {BACKEND_PORT_ENV}: {raw}"))?;
        return Ok(parsed);
    }

    let listener = TcpListener::bind((BACKEND_DEFAULT_HOST, 0))
        .map_err(|err| format!("Unable to pick a free localhost port: {err}"))?;
    let port = listener
        .local_addr()
        .map_err(|err| format!("Unable to resolve local port: {err}"))?
        .port();
    drop(listener);
    Ok(port)
}

fn ensure_runtime_layout(base_dir: &Path) -> Result<(), String> {
    let settings_dir = base_dir.join("settings");
    let resources_dir = base_dir.join("resources");
    let logs_dir = resources_dir.join("logs");
    let checkpoints_dir = resources_dir.join("checkpoints");

    fs::create_dir_all(&settings_dir)
        .map_err(|err| format!("Failed to create settings dir: {err}"))?;
    fs::create_dir_all(&logs_dir).map_err(|err| format!("Failed to create logs dir: {err}"))?;
    fs::create_dir_all(&checkpoints_dir)
        .map_err(|err| format!("Failed to create checkpoints dir: {err}"))?;

    let config_path = settings_dir.join("configurations.json");
    if !config_path.exists() {
        fs::write(&config_path, DEFAULT_CONFIG_JSON)
            .map_err(|err| format!("Failed to write default configurations.json: {err}"))?;
    }

    let env_path = settings_dir.join(".env");
    if !env_path.exists() {
        fs::write(&env_path, DEFAULT_ENV_LOCAL)
            .map_err(|err| format!("Failed to write default .env: {err}"))?;
    }

    Ok(())
}

fn is_sidecar_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }

    let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
        return false;
    };

    if !stem.starts_with(BACKEND_SIDECAR_STEM) {
        return false;
    }

    #[cfg(target_os = "windows")]
    {
        return path
            .extension()
            .and_then(|value| value.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("exe"))
            .unwrap_or(false);
    }

    #[cfg(not(target_os = "windows"))]
    {
        true
    }
}

fn find_sidecar(path: &Path, depth: usize) -> Option<PathBuf> {
    if depth == 0 || !path.exists() {
        return None;
    }

    let entries = fs::read_dir(path).ok()?;
    for entry in entries {
        let Ok(entry) = entry else {
            continue;
        };
        let candidate = entry.path();
        if is_sidecar_file(&candidate) {
            return Some(candidate);
        }

        if candidate.is_dir() {
            if let Some(found) = find_sidecar(&candidate, depth - 1) {
                return Some(found);
            }
        }
    }

    None
}

fn resolve_sidecar_binary(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(explicit) = std::env::var("ADSMOD_BACKEND_SIDECAR") {
        let sidecar = PathBuf::from(explicit);
        if sidecar.exists() {
            return Ok(sidecar);
        }
    }

    let mut search_roots: Vec<PathBuf> = Vec::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        search_roots.push(resource_dir);
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            search_roots.push(exe_dir.to_path_buf());
        }
    }

    for root in search_roots {
        if let Some(found) = find_sidecar(&root, 4) {
            return Ok(found);
        }
    }

    Err("Unable to resolve backend sidecar binary".to_string())
}

fn check_health(host: &str, port: u16) -> bool {
    let address = format!("{host}:{port}");
    let Ok(socket) = address.parse() else {
        return false;
    };

    let Ok(mut stream) = TcpStream::connect_timeout(&socket, Duration::from_secs(2)) else {
        return false;
    };

    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));

    let request = format!(
        "GET {BACKEND_HEALTH_PATH} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n"
    );

    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }

    let mut raw = String::new();
    if stream.read_to_string(&mut raw).is_err() {
        return false;
    }

    raw.starts_with("HTTP/1.1 200") || raw.starts_with("HTTP/1.0 200")
}

fn wait_for_backend(host: &str, port: u16, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;

    while Instant::now() < deadline {
        if check_health(host, port) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    Err(format!(
        "Backend health check timed out after {} seconds",
        timeout.as_secs()
    ))
}

fn stop_backend(app: &AppHandle) {
    if let Some(state) = app.try_state::<BackendRuntime>() {
        if let Ok(mut guard) = state.child.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

fn spawn_backend(app: &AppHandle) -> Result<(), String> {
    let backend_host = std::env::var(BACKEND_HOST_ENV).unwrap_or_else(|_| BACKEND_DEFAULT_HOST.to_string());
    let backend_port = choose_backend_port()?;

    let base_dir = app
        .path()
        .app_local_data_dir()
        .map_err(|err| format!("Unable to resolve app data directory: {err}"))?
        .join("runtime");

    ensure_runtime_layout(&base_dir)?;

    let sidecar_binary = resolve_sidecar_binary(app)?;

    let mut command = Command::new(sidecar_binary);
    command
        .env("FASTAPI_HOST", &backend_host)
        .env("FASTAPI_PORT", backend_port.to_string())
        .env(BACKEND_BASE_DIR_ENV, base_dir.as_os_str())
        .env(
            "MPLBACKEND",
            std::env::var("MPLBACKEND").unwrap_or_else(|_| "Agg".to_string()),
        )
        .env(
            "KERAS_BACKEND",
            std::env::var("KERAS_BACKEND").unwrap_or_else(|_| "torch".to_string()),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let mut child = command
        .spawn()
        .map_err(|err| format!("Failed to start backend sidecar: {err}"))?;

    if let Err(error) = wait_for_backend(&backend_host, backend_port, BACKEND_BOOT_TIMEOUT) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }

    let runtime_config = RuntimeConfig {
        api_origin: format!("http://{backend_host}:{backend_port}"),
        backend_host,
        backend_port,
        base_dir: base_dir.to_string_lossy().to_string(),
    };

    let state = app.state::<BackendRuntime>();

    {
        let mut config_guard = state
            .config
            .lock()
            .map_err(|_| "Failed to acquire runtime config lock".to_string())?;
        *config_guard = Some(runtime_config);
    }

    {
        let mut child_guard = state
            .child
            .lock()
            .map_err(|_| "Failed to acquire backend process lock".to_string())?;
        *child_guard = Some(child);
    }

    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(BackendRuntime::default())
        .invoke_handler(tauri::generate_handler![get_runtime_config])
        .setup(|app| {
            spawn_backend(&app.handle()).map_err(std::io::Error::other)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::ExitRequested { .. } = event {
                stop_backend(app);
            }
        });
}
