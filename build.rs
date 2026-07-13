fn main() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let asm_file = match arch.as_str() {
        "aarch64" => "asm/aarch64.s",
        "x86_64" => "asm/x86_64.s",
        _ => panic!("unsupported architecture: {arch}"),
    };
    cc::Build::new().file(asm_file).compile("context_switch");
    println!("cargo:rerun-if-changed={asm_file}");
}
