use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result};

/// Determines the appropriate computational device (CPU or GPU) to use based on the input parameter and system capabilities.
///
/// # Parameters
/// - `cpu`: A boolean flag indicating whether to force the use of the CPU.
///   - `true`: Use the CPU.
///   - `false`: Attempt to use a GPU if available.
///
/// # Returns
/// - `Result<Device>`: A `Device` object representing the selected computational device. Defaults to `Device::Cpu` if no GPU is available.
///
/// # Behavior
/// - If `cpu` is `true`, returns `Device::Cpu`.
/// - If CUDA is available, returns a CUDA device (`Device::new_cuda(0)`).
/// - If Metal is available (on macOS), returns a Metal device (`Device::new_metal(0)`).
/// - If no GPU is available:
///   - On macOS with `aarch64`, prints a message about building with the `metal` feature.
///   - On other platforms, prints a message about building with the `cuda` feature.
///   - Defaults to `Device::Cpu`.
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}
