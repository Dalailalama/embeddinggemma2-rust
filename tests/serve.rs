//! `egemma2 serve` end to end: the stdio and HTTP transports against the library's own embeddings (CPU).
//!
//! Needs the model weights in `models/embeddinggemma-2` (or `EGEMMA2_MODEL_DIR`); skipped, loudly, without them.

use embeddinggemma2::{EmbeddingGemma2, Input, Part, media};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;

fn model_dir() -> PathBuf {
    std::env::var_os("EGEMMA2_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("models/embeddinggemma-2"))
}

fn model() -> Option<&'static EmbeddingGemma2> {
    static MODEL: OnceLock<Option<EmbeddingGemma2>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            if !model_dir().join("model.safetensors").exists() {
                eprintln!("SKIPPED: {} has no model.safetensors", model_dir().display());
                return None;
            }
            Some(EmbeddingGemma2::load(model_dir()).expect("load model"))
        })
        .as_ref()
}

fn image_path() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/image/img_b_96x64.png").to_string_lossy().into_owned()
}

fn spawn(args: &[&str]) -> Child {
    Command::new(env!("CARGO_BIN_EXE_egemma2"))
        .arg("--device")
        .arg("cpu")
        .arg("--model-dir")
        .arg(model_dir())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn egemma2")
}

fn floats(v: &Value) -> Vec<f32> {
    match v {
        Value::String(b) => {
            let bytes = base64::decode(b).expect("base64");
            bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
        }
        v => v.as_array().expect("embedding array").iter().map(|x| x.as_f64().unwrap() as f32).collect(),
    }
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let d = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    // Batched requests share packed passes, which may sum in a different order than a lone input.
    assert!(d < 2e-5, "{what}: max|d| = {d:e}");
}

fn expected(m: &EmbeddingGemma2, input: Input, dim: Option<usize>) -> Vec<f32> {
    embeddinggemma2::model::truncate_and_normalize(m.embed(&input).unwrap(), dim).unwrap()
}

#[test]
fn stdio_round_trip() {
    let Some(m) = model() else { return };
    let img = image_path();
    let q = m.prompt("SearchQuery").unwrap().to_string();
    // Request 2 is written out by hand: `json!` sorts object keys, and key order is what is being tested.
    let quoted = Value::String(img.clone()).to_string();
    let requests = [
        json!({"id": 1, "input": ["the moon", "a red planet"], "prompt_name": "SearchQuery"}).to_string(),
        format!(r#"{{"id": 2, "input": [{{"text": "a photo of", "image": {quoted}}}, {{"image": {quoted}, "text": "a photo of"}}], "dimensions": 256}}"#),
        json!({"id": 3, "input": {"parts": ["A:", {"image": img}, {"image": img}]}, "encoding_format": "base64"}).to_string(),
        json!({"id": 4, "input": {"imgae": img}}).to_string(),
        json!({"id": 5, "input": "plain", "model": "Document"}).to_string(),
    ];
    let mut child = spawn(&["serve"]);
    {
        let mut stdin = child.stdin.take().unwrap();
        for r in &requests {
            writeln!(stdin, "{r}").unwrap();
        }
        writeln!(stdin, "this is not json").unwrap();
    } // closing stdin ends the server
    let lines: Vec<Value> = BufReader::new(child.stdout.take().unwrap()).lines().map(|l| serde_json::from_str(&l.unwrap()).unwrap()).collect();
    let status = child.wait().unwrap();
    assert!(status.success(), "serve exited with {status}");
    assert_eq!(lines.len(), requests.len() + 1, "{lines:?}");
    for (i, l) in lines.iter().take(requests.len()).enumerate() {
        assert_eq!(l["id"], json!(i + 1), "responses keep request order: {l}");
    }

    let image = media::load_image(Path::new(&img)).unwrap();
    let r = &lines[0];
    assert_close(&floats(&r["data"][0]["embedding"]), &expected(m, Input::text(format!("{q}the moon")), None), "text 0");
    assert_close(&floats(&r["data"][1]["embedding"]), &expected(m, Input::text(format!("{q}a red planet")), None), "text 1");
    assert_eq!(r["usage"]["total_tokens"], json!(m.tokenize(&format!("{q}the moon")).unwrap().len() + m.tokenize(&format!("{q}a red planet")).unwrap().len()));

    let r = &lines[1];
    let ti = Input { parts: vec![Part::Text("a photo of".into()), Part::Image(image.clone())] };
    let it = Input { parts: vec![Part::Image(image.clone()), Part::Text("a photo of".into())] };
    assert_close(&floats(&r["data"][0]["embedding"]), &expected(m, ti, Some(256)), "text + image");
    assert_close(&floats(&r["data"][1]["embedding"]), &expected(m, it, Some(256)), "image + text (key order)");

    let two = Input { parts: vec![Part::Text("A:".into()), Part::Image(image.clone()), Part::Image(image)] };
    assert!(lines[2]["data"][0]["embedding"].is_string());
    assert_close(&floats(&lines[2]["data"][0]["embedding"]), &expected(m, two, None), "parts, base64");

    assert!(lines[3]["error"]["message"].as_str().unwrap().contains("imgae"), "{}", lines[3]);
    let d = m.prompt("Document").unwrap();
    assert_close(&floats(&lines[4]["data"][0]["embedding"]), &expected(m, Input::text(format!("{d}plain")), None), "model names a prompt");
    assert!(lines[5]["error"].is_object() && lines[5].get("id").is_none());
}

/// A minimal HTTP/1.1 client: returns (status, body).
fn http(port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: &str) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n", body.len());
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    req.push_str(body);
    s.write_all(req.as_bytes()).unwrap();
    let mut resp = String::new();
    s.read_to_string(&mut resp).unwrap();
    let status = resp.split(' ').nth(1).unwrap().parse().unwrap();
    let body = resp.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
    (status, body)
}

#[test]
fn http_endpoints() {
    let Some(m) = model() else { return };
    let mut child = spawn(&["serve", "--http", "127.0.0.1:0", "--api-key", "secret", "--allow-origin", "http://ok.example"]);
    // The server prints its address once it listens.
    let mut stderr = BufReader::new(child.stderr.take().unwrap());
    let port: u16 = loop {
        let mut line = String::new();
        assert!(stderr.read_line(&mut line).unwrap() > 0, "server exited before listening");
        if let Some(rest) = line.strip_prefix("listening on http://") {
            break rest.split_whitespace().next().unwrap().rsplit(':').next().unwrap().parse().unwrap();
        }
    };
    let auth = [("Authorization", "Bearer secret"), ("Content-Type", "application/json")];
    let result = std::panic::catch_unwind(|| {
        assert_eq!(http(port, "GET", "/health", &[], "").0, 200, "health checks need no key");
        assert_eq!(http(port, "GET", "/v1/models", &[], "").0, 401, "API key required");
        assert_eq!(http(port, "GET", "/v1/models", &[("Authorization", "Bearer wrong")], "").0, 401);
        let (st, body) = http(port, "GET", "/v1/models", &auth, "");
        assert_eq!(st, 200);
        assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["data"][0]["id"], "embeddinggemma-2");

        // What the openai Python SDK sends by default (base64), with a prompt passed in extra_body.
        let req = json!({"input": ["the moon", "a red planet"], "model": "embeddinggemma-2", "encoding_format": "base64", "prompt_name": "Document"});
        let (st, body) = http(port, "POST", "/v1/embeddings", &auth, &req.to_string());
        assert_eq!(st, 200, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["object"], "list");
        let d = m.prompt("Document").unwrap();
        assert_close(&floats(&v["data"][1]["embedding"]), &expected(m, Input::text(format!("{d}a red planet")), None), "http base64");

        let (st, body) = http(port, "POST", "/v1/embeddings", &auth, r#"{"input": [[1, 2, 3]]}"#);
        assert_eq!(st, 400);
        assert!(body.contains("check_embedding_ctx_length"), "{body}");
        assert_eq!(http(port, "POST", "/v1/embeddings", &auth, "{").0, 400);
        assert_eq!(http(port, "GET", "/v1/embeddings", &auth, "").0, 405);
        assert_eq!(http(port, "GET", "/nope", &auth, "").0, 404);
        let evil = [("Authorization", "Bearer secret"), ("Origin", "http://evil.example")];
        assert_eq!(http(port, "POST", "/v1/embeddings", &evil, r#"{"input": "x"}"#).0, 403, "foreign origins are refused");
        let ok = [("Authorization", "Bearer secret"), ("Origin", "http://ok.example")];
        assert_eq!(http(port, "POST", "/v1/embeddings", &ok, r#"{"input": "x"}"#).0, 200);
    });
    let _ = child.kill();
    let _ = child.wait();
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
