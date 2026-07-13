fn main() {
    println!("cargo:rerun-if-env-changed=MYTH_ROOT");
    println!("cargo:rerun-if-env-changed=ABT_ROOT");

    if std::env::var("CARGO_FEATURE_MASSIVETHREADS").is_ok() {
        if pkg_config::probe_library("massivethreads").is_err() {
            if let Ok(root) = std::env::var("MYTH_ROOT") {
                println!("cargo:rustc-link-search=native={root}/lib");
                println!("cargo:rustc-link-lib=myth");
            } else {
                eprintln!("massivethreads feature enabled but library not found");
                eprintln!("Set MYTH_ROOT or ensure pkg-config can locate massivethreads");
                std::process::exit(1);
            }
        }
    }

    if std::env::var("CARGO_FEATURE_ARGOBOTS").is_ok() {
        if pkg_config::probe_library("argobots").is_err() {
            if let Ok(root) = std::env::var("ABT_ROOT") {
                println!("cargo:rustc-link-search=native={root}/lib");
                println!("cargo:rustc-link-lib=abt");
            } else {
                eprintln!("argobots feature enabled but library not found");
                eprintln!("Set ABT_ROOT=/path/to/argobots-install or ensure pkg-config can locate argobots");
                eprintln!("Build: ./configure --prefix=$ABT_ROOT && make -j && make install");
                std::process::exit(1);
            }
        }
    }
}
