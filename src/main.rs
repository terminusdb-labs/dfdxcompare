mod vecmath;

use std::time::SystemTime;

use dfdx::tensor_ops::TryMatMul;
use dfdx::{
    shapes::Rank1,
    shapes::Rank2,
    tensor::{AsArray, Cpu, SampleTensor, Tensor},
};
use vecmath::{normalized_cosine_distance, Embedding};

fn cpu_compare(v1: &Embedding, v2: &Embedding) -> f32 {
    normalized_cosine_distance(v1, v2)
}

fn main() {
    // Creates a handle to device ordinal 0.
    let dev: Cpu = Default::default();
    const NUM_VECTORS: usize = 100000;
    const VECTOR_DIM: usize = 1536;
    let data: Tensor<Rank2<NUM_VECTORS, VECTOR_DIM>, f32, _> = dev.sample_uniform();
    let data2 = data.clone();
    let query: Tensor<Rank1<VECTOR_DIM>, f32, _> = dev.sample_uniform();
    let query2 = query.clone();
    let now = SystemTime::now();
    let r = data.matmul(query);
    let tensor_duration = now.elapsed().unwrap().as_millis();
    eprintln!("tensor duration: {tensor_duration}");
    let d: Vec<f32> = r.as_vec();
    eprintln!("d: {}", d.len());
    /*
    let data_array: [[f32; VECTOR_DIM]; NUM_VECTORS] = data2.array();

    let query: [f32; VECTOR_DIM] = query2.array();
    eprintln!("tensor duration: {tensor_duration}");
    let mut ds: Vec<f32> = Vec::with_capacity(NUM_VECTORS);
    let now = SystemTime::now();
    for v in data_array {
        ds.push(cpu_compare(&query, &v));
    }
    let cpu_duration = now.elapsed().unwrap().as_millis();
    eprintln!("cpu duration: {cpu_duration}");
    eprintln!("ds: {}", ds.len());
    */
}
