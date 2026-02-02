use crate::mnist::model;
use burn::backend::{Candle, candle::CandleDevice};

mod mnist;

type Backend = Candle;

fn main() {
    let device = CandleDevice::cuda(0);
    let model = model::ModelConfig::new(10, 128).init::<Backend>(&device);
    println!("model: {}", model);
}
