use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

// mod vector_database;
// use vector_database::VectorDatabase;

/// Simple calculator for testing Rust integration
#[pyfunction]
fn add_numbers(a: f64, b: f64) -> f64 {
    a + b
}

/// Fast parallel distance calculation between face encodings
#[pyfunction]
fn calculate_distance(encoding1: Vec<f64>, encoding2: Vec<f64>) -> f64 {
    if encoding1.len() != encoding2.len() {
        return f64::INFINITY;
    }

    let sum_of_squares: f64 = encoding1
        .iter()
        .zip(encoding2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    sum_of_squares.sqrt()
}

/// Batch distance calculation for multiple faces (parallel processing)
#[pyfunction]
fn calculate_distances_batch(
    target_encoding: Vec<f64>,
    known_encodings: Vec<Vec<f64>>,
) -> Vec<f64> {
    known_encodings
        .par_iter()
        .map(|encoding| calculate_distance(target_encoding.clone(), encoding.clone()))
        .collect()
}

/// Fast face matching with confidence scoring
#[pyfunction]
#[pyo3(signature = (target_encoding, known_encodings, known_names, tolerance=None))]
fn find_best_match(
    target_encoding: Vec<f64>,
    known_encodings: Vec<Vec<f64>>,
    known_names: Vec<String>,
    tolerance: Option<f64>,
) -> (Option<String>, f64, usize) {
    let tolerance = tolerance.unwrap_or(0.6);

    if known_encodings.is_empty() {
        return (None, f64::INFINITY, 0);
    }

    // Parallel distance calculation
    let distances = calculate_distances_batch(target_encoding, known_encodings);

    // Find best match
    let (best_index, &best_distance) = distances
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    if best_distance < tolerance {
        let confidence = 1.0 - (best_distance / tolerance).min(1.0);
        (
            Some(known_names[best_index].clone()),
            confidence,
            best_index,
        )
    } else {
        (None, 0.0, best_index)
    }
}

/// Image preprocessing utilities
#[pyfunction]
fn normalize_encoding(mut encoding: Vec<f64>) -> Vec<f64> {
    let magnitude: f64 = encoding.iter().map(|x| x * x).sum::<f64>().sqrt();
    if magnitude > 0.0 {
        encoding.iter_mut().for_each(|x| *x /= magnitude);
    }
    encoding
}

/// Simple performance timer
#[pyclass]
struct PerformanceTimer {
    start_time: Option<std::time::Instant>,
    measurements: Vec<f64>,
}

#[pymethods]
impl PerformanceTimer {
    #[new]
    fn new() -> Self {
        PerformanceTimer {
            start_time: None,
            measurements: Vec::new(),
        }
    }

    fn start(&mut self) {
        self.start_time = Some(std::time::Instant::now());
    }

    fn stop(&mut self) -> PyResult<f64> {
        match self.start_time {
            Some(start) => {
                let duration = start.elapsed().as_secs_f64();
                self.measurements.push(duration);
                self.start_time = None;
                Ok(duration)
            }
            None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Timer not started",
            )),
        }
    }

    fn get_average(&self) -> f64 {
        if self.measurements.is_empty() {
            return 0.0;
        }
        self.measurements.iter().sum::<f64>() / self.measurements.len() as f64
    }

    fn reset(&mut self) {
        self.measurements.clear();
        self.start_time = None;
    }
}

/// Simple face recognizer
#[pyclass]
struct SimpleFaceRecognizer {
    known_faces: Vec<Vec<f64>>,
    known_names: Vec<String>,
    tolerance: f64,
}

#[pymethods]
impl SimpleFaceRecognizer {
    #[new]
    #[pyo3(signature = (tolerance=None))]
    fn new(tolerance: Option<f64>) -> Self {
        SimpleFaceRecognizer {
            known_faces: Vec::new(),
            known_names: Vec::new(),
            tolerance: tolerance.unwrap_or(0.6),
        }
    }

    fn add_known_face(&mut self, encoding: Vec<f64>, name: String) {
        self.known_faces.push(encoding);
        self.known_names.push(name);
    }

    fn recognize_face(&self, face_encoding: Vec<f64>) -> Option<String> {
        let mut best_match_index = None;
        let mut best_distance = f64::INFINITY;

        for (i, known_encoding) in self.known_faces.iter().enumerate() {
            let distance = calculate_distance(known_encoding.clone(), face_encoding.clone());
            if distance < self.tolerance && distance < best_distance {
                best_distance = distance;
                best_match_index = Some(i);
            }
        }

        match best_match_index {
            Some(index) => Some(self.known_names[index].clone()),
            None => None,
        }
    }

    fn get_known_faces_count(&self) -> usize {
        self.known_faces.len()
    }

    fn clear_all(&mut self) {
        self.known_faces.clear();
        self.known_names.clear();
    }

    fn get_all_names(&self) -> Vec<String> {
        self.known_names.clone()
    }
}

/// Simple Vector Database for face recognition
#[pyclass]
struct VectorDatabase {
    threshold: f64,
    faces: std::collections::HashMap<String, (String, Vec<f64>)>, // id -> (name, vector)
}

#[pymethods]
impl VectorDatabase {
    #[new]
    fn new() -> Self {
        VectorDatabase {
            threshold: 0.6,
            faces: std::collections::HashMap::new(),
        }
    }

    fn get_threshold(&self) -> f64 {
        self.threshold
    }

    fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
    
    fn get_face_count(&self) -> usize {
        self.faces.len()
    }
    
    fn add_face_vector(&mut self, id: String, name: String, vector: Vec<f64>) -> bool {
        self.faces.insert(id, (name, vector));
        true
    }
    
    fn search_face(&self, query_vector: Vec<f64>) -> Option<(String, String, f64)> {
        if self.faces.is_empty() || query_vector.is_empty() {
            return None;
        }
        
        let mut best_match: Option<(String, String, f64)> = None;
        let mut min_distance = f64::MAX;
        
        for (id, (name, vector)) in &self.faces {
            let distance = calculate_distance(query_vector.clone(), vector.clone());
            if distance < min_distance && distance < self.threshold {
                min_distance = distance;
                best_match = Some((id.clone(), name.clone(), 1.0 - distance / self.threshold));
            }
        }
        
        best_match
    }
    
    fn load_from_file(&mut self, _file_path: String) -> bool {
        // For now, just return true - implement file loading later if needed
        println!("⚠️ File loading not implemented yet, starting with empty database");
        true
    }
    
    fn save_to_file(&self, _file_path: String) -> bool {
        // For now, just return true - implement file saving later if needed
        println!("⚠️ File saving not implemented yet");
        true
    }
}

/*
/// Benchmark utilities for performance testing
#[pyclass]
struct PerformanceBenchmark {
    results: HashMap<String, Vec<f64>>,
}

#[pymethods]
impl PerformanceBenchmark {
    #[new]
    fn new() -> Self {
        PerformanceBenchmark {
            results: HashMap::new(),
        }
    }

    /// Benchmark distance calculation: Rust vs Python equivalent
    fn benchmark_distance_calculation(&mut self, iterations: usize) -> HashMap<String, f64> {
        let encoding1: Vec<f64> = (0..128).map(|i| (i as f64) * 0.01).collect();
        let encoding2: Vec<f64> = (0..128).map(|i| (i as f64) * 0.01 + 0.1).collect();

        // Rust benchmark
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = calculate_distance(encoding1.clone(), encoding2.clone());
        }
        let rust_time = start.elapsed().as_secs_f64();

        let mut result = HashMap::new();
        result.insert("rust_total_time".to_string(), rust_time);
        result.insert(
            "rust_per_operation".to_string(),
            rust_time / iterations as f64,
        );
        result.insert("iterations".to_string(), iterations as f64);

        result
    }

    /// Benchmark batch processing vs sequential
    fn benchmark_batch_processing(
        &mut self,
        num_faces: usize,
        encoding_size: usize,
    ) -> HashMap<String, f64> {
        let target: Vec<f64> = (0..encoding_size).map(|i| (i as f64) * 0.01).collect();
        let known_faces: Vec<Vec<f64>> = (0..num_faces)
            .map(|j| {
                (0..encoding_size)
                    .map(|i| (i as f64) * 0.01 + (j as f64) * 0.1)
                    .collect()
            })
            .collect();

        // Sequential benchmark
        let start = std::time::Instant::now();
        let _sequential: Vec<f64> = known_faces
            .iter()
            .map(|encoding| calculate_distance(target.clone(), encoding.clone()))
            .collect();
        let sequential_time = start.elapsed().as_secs_f64();

        // Parallel benchmark
        let start = std::time::Instant::now();
        let _parallel = calculate_distances_batch(target, known_faces);
        let parallel_time = start.elapsed().as_secs_f64();

        let mut result = HashMap::new();
        result.insert("sequential_time".to_string(), sequential_time);
        result.insert("parallel_time".to_string(), parallel_time);
        result.insert("speedup".to_string(), sequential_time / parallel_time);
        result.insert("num_faces".to_string(), num_faces as f64);

        result
    }

    /// Memory usage estimation
    fn estimate_memory_usage(
        &self,
        num_faces: usize,
        encoding_size: usize,
    ) -> HashMap<String, f64> {
        let bytes_per_face = encoding_size * 8 + 50; // 8 bytes per f64 + name overhead
        let total_rust_bytes = bytes_per_face * num_faces;

        let mut result = HashMap::new();
        result.insert("bytes_per_face".to_string(), bytes_per_face as f64);
        result.insert(
            "total_memory_kb".to_string(),
            total_rust_bytes as f64 / 1024.0,
        );
        result.insert("num_faces".to_string(), num_faces as f64);

        result
    }

    /// Get all benchmark results
    fn get_results(&self) -> HashMap<String, Vec<f64>> {
        self.results.clone()
    }
}
*/

/// Python module definition
#[pymodule]
fn face_recognition_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_distance, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_distances_batch, m)?)?;
    m.add_function(wrap_pyfunction!(find_best_match, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_encoding, m)?)?;
    m.add_class::<PerformanceTimer>()?;
    m.add_class::<SimpleFaceRecognizer>()?;
    // m.add_class::<PerformanceBenchmark>()?;  // Commented out temporarily
    m.add_class::<VectorDatabase>()?;
    Ok(())
}
