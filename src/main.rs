mod vecmath;

use std::ops::Sub;
use std::time::SystemTime;

use dfdx::tensor::{Cuda, TensorFromVec};
use dfdx::tensor_ops::{BroadcastTo, TryMatMul};
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
    //let dev: Cpu = Default::default();
    let dev: Cuda = Default::default();
    const NUM_VECTORS: usize = 10_000;
    const VECTOR_DIM: usize = 1536;
    let data: Tensor<Rank2<NUM_VECTORS, VECTOR_DIM>, f32, _> = dev.sample_uniform();
    let data2 = data.clone();
    let query: Tensor<Rank1<VECTOR_DIM>, f32, _> = dev.sample_uniform();
    let query2 = query.clone();
    let now = SystemTime::now();
    let r = data.matmul(query);
    //let r = (r - 1.0f32) / -2.0f32;
    //let d: Vec<f32> = r.as_vec();
    //eprintln!("d: {}", d.len());
    let tensor_duration = now.elapsed().unwrap().as_micros();
    eprintln!("tensor duration: {tensor_duration}");

    let now = SystemTime::now();
    let data_array: Vec<f32> = data2.as_vec();
    let data_extract_duration = now.elapsed().unwrap().as_micros();
    eprintln!("data extract duration: {data_extract_duration}");

    let cloned = data_array.clone();
    let now = SystemTime::now();
    let new_tensor = dev.tensor_from_vec(cloned, (NUM_VECTORS, VECTOR_DIM));
    let tensor_load_duration = now.elapsed().unwrap().as_micros();
    eprintln!("tensor load duration: {tensor_load_duration}");

    let query: [f32; VECTOR_DIM] = query2.array();
    let mut ds: Vec<f32> = Vec::with_capacity(NUM_VECTORS);
    let now = SystemTime::now();
    for v in data_array.chunks(VECTOR_DIM) {
        let array: &[f32; VECTOR_DIM] = unsafe { &*(v.as_ptr() as *const [f32; VECTOR_DIM]) };
        ds.push(cpu_compare(&query, array));
    }
    let cpu_duration = now.elapsed().unwrap().as_micros();
    eprintln!("cpu duration: {cpu_duration}");
    eprintln!("ds: {}", ds.len());
}
