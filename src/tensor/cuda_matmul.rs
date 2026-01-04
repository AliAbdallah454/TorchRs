use crate::tensor::Tensor;

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
mod imp {
    use libc::c_int;
    use crate::tensor::Tensor;

    #[link(name = "mat_mul_cuda", kind = "static")]
    extern "C" {
        fn launch_mat_mul(a: *mut f32, b: *mut f32, c: *mut f32, m: c_int, k: c_int, n: c_int);
        fn launch_cuBLAS_mat_mul(a: *mut f32, b: *mut f32, c: *mut f32, m: c_int, k: c_int, n: c_int);
    }

    pub fn cuda_mat_mul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
        assert_eq!(a.data.len(), m * k, "Matrix A size mismatch");
        assert_eq!(b.data.len(), k * n, "Matrix B size mismatch");

        let mut c = vec![0.0f32; m * n];
        unsafe {
            launch_mat_mul(
                a.data.as_ptr() as *mut f32,
                b.data.as_ptr() as *mut f32,
                c.as_mut_ptr(),
                m as c_int,
                k as c_int,
                n as c_int,
            );
        }
        Tensor::new_2d(c, m, n)
    }

    pub fn cublas_mat_mul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
        assert_eq!(a.data.len(), m * k, "Matrix A size mismatch");
        assert_eq!(b.data.len(), k * n, "Matrix B size mismatch");

        let mut c = vec![0.0f32; m * n];
        unsafe {
            launch_cuBLAS_mat_mul(
                a.data.as_ptr() as *mut f32,
                b.data.as_ptr() as *mut f32,
                c.as_mut_ptr(),
                m as c_int,
                k as c_int,
                n as c_int,
            );
        }
        Tensor::new_2d(c, m, n)
    }
}

// Re-export when CUDA is available
#[cfg(all(feature = "cuda", not(target_os = "macos")))]
pub use imp::{cuda_mat_mul, cublas_mat_mul};

// Fallback stubs when CUDA isn't available
#[cfg(not(all(feature = "cuda", not(target_os = "macos"))))]
#[allow(dead_code)]
pub fn cuda_mat_mul(_a: &Tensor, _b: &Tensor, _m: usize, _k: usize, _n: usize) -> Tensor {
    panic!("CUDA is not enabled / not supported on this platform");
}

#[cfg(not(all(feature = "cuda", not(target_os = "macos"))))]
#[allow(dead_code)]
pub fn cublas_mat_mul(_a: &Tensor, _b: &Tensor, _m: usize, _k: usize, _n: usize) -> Tensor {
    panic!("CUDA is not enabled / not supported on this platform");
}