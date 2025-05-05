mod args;

// #[cfg(feature = "mkl")]
// extern crate intel_mkl_src;
// 
// #[cfg(feature = "accelerate")]
// extern crate accelerate_src;

use clap::Parser;
use crate::args::Args;

fn main(){
    let args = Args::parse();
    println!("{:?}", args);
}