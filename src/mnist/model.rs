use burn::{
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::backend::Backend,
};

/// MNIST model: conv1 -> bn1 -> relu -> conv2 -> bn2 -> pool -> fc1 -> relu -> fc2 -> softmax.
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    pool: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 4], [3, 3]).init(device),
            bn1: BatchNormConfig::new(4).init(device),
            conv2: Conv2dConfig::new([4, 8], [3, 3]).init(device),
            bn2: BatchNormConfig::new(8).init(device),
            pool: MaxPool2dConfig::new([2, 2]).init(),
            fc1: LinearConfig::new(8 * 12 * 12, self.hidden_size).init(device),
            fc2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            relu: Relu::new(),
        }
    }
}
