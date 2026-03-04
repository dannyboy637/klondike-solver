pub mod inference;
pub mod model;

#[cfg(test)]
mod tests {
    use tch::{Device, Tensor};

    fn auto_device() -> Device {
        if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else if tch::utils::has_mps() {
            Device::Mps
        } else {
            Device::Cpu
        }
    }

    #[test]
    fn test_tch_tensor_on_best_device() {
        let device = auto_device();
        eprintln!("tch device: {:?}", device);

        let t = Tensor::randn([2, 3], (tch::Kind::Float, device));
        assert_eq!(t.size(), &[2, 3]);
        assert_eq!(t.device(), device);

        // Basic math should work
        let sum = t.sum(tch::Kind::Float);
        assert!(sum.size().is_empty(), "scalar tensor should have empty shape");
        eprintln!("tensor sum: {:?}", f64::try_from(&sum).unwrap());
    }
}
