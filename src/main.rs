use burn::{
    Tensor,
    backend::{Candle, candle::CandleDevice},
};

type Backend = Candle;

fn main() {
    let device = CandleDevice::cuda(0);
    let tensor1 = Tensor::<Backend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let tensor2 = Tensor::<Backend, 2>::ones_like(&tensor1);
    println!("tensor1: {}", tensor1);
    println!("tensor2: {}", tensor2);
    let tensor3 = tensor1 + tensor2;
    println!("tensor3: {}", tensor3);
}
