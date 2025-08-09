use pyo3::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;

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
        (Some(known_names[best_index].clone()), confidence, best_index)
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
    Ok(())
}
