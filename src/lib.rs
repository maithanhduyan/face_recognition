use pyo3::prelude::*;
use std::collections::HashMap;

/// Simple calculator for testing Rust integration
#[pyfunction]
fn add_numbers(a: f64, b: f64) -> f64 {
    a + b
}

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
    m.add_class::<PerformanceTimer>()?;
    m.add_class::<SimpleFaceRecognizer>()?;
    Ok(())
}
