use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceVector {
    pub id: String,
    pub name: String,
    pub vector: Vec<f64>,
    pub created_at: i64,
}

#[pyclass]
pub struct VectorDatabase {
    vectors: Arc<RwLock<HashMap<String, FaceVector>>>,
    threshold: f64,
}

impl VectorDatabase {
    fn calculate_distance(vec1: &[f64], vec2: &[f64]) -> f64 {
        if vec1.len() != vec2.len() {
            return f64::MAX;
        }

        let sum: f64 = vec1
            .iter()
            .zip(vec2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        sum.sqrt()
    }
}

#[pymethods]
impl VectorDatabase {
    #[new]
    #[pyo3(signature = (threshold=None))]
    pub fn new(threshold: Option<f64>) -> Self {
        VectorDatabase {
            vectors: Arc::new(RwLock::new(HashMap::new())),
            threshold: threshold.unwrap_or(0.6),
        }
    }

    /// Thêm vector khuôn mặt mới vào database
    pub fn add_face_vector(&self, id: String, name: String, vector: Vec<f64>) -> PyResult<bool> {
        let face_vector = FaceVector {
            id: id.clone(),
            name,
            vector,
            created_at: chrono::Utc::now().timestamp(),
        };

        let mut vectors = self.vectors.write().unwrap();
        vectors.insert(id, face_vector);
        Ok(true)
    }

    /// Tìm kiếm khuôn mặt trong database (parallel processing)
    pub fn search_face(&self, query_vector: Vec<f64>) -> PyResult<Option<(String, f64)>> {
        let vectors = self.vectors.read().unwrap();

        if vectors.is_empty() {
            return Ok(None);
        }

        // Parallel search for best match
        let best_match = vectors
            .par_iter()
            .map(|(_, face_vector)| {
                let distance = Self::calculate_distance(&query_vector, &face_vector.vector);
                (face_vector.name.clone(), distance)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        match best_match {
            Some((name, distance)) if distance < self.threshold => Ok(Some((name, distance))),
            _ => Ok(None),
        }
    }

    /// Batch search multiple faces
    pub fn batch_search(
        &self,
        query_vectors: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Option<(String, f64)>>> {
        let results: Vec<Option<(String, f64)>> = query_vectors
            .par_iter()
            .map(|vector| self.search_face(vector.clone()).unwrap_or(None))
            .collect();

        Ok(results)
    }

    /// Lấy số lượng faces trong database
    pub fn get_face_count(&self) -> PyResult<usize> {
        let vectors = self.vectors.read().unwrap();
        Ok(vectors.len())
    }

    /// Xóa face khỏi database
    pub fn remove_face(&self, id: String) -> PyResult<bool> {
        let mut vectors = self.vectors.write().unwrap();
        Ok(vectors.remove(&id).is_some())
    }

    /// Lấy danh sách tất cả faces
    pub fn list_faces(&self) -> PyResult<Vec<String>> {
        let vectors = self.vectors.read().unwrap();
        let names: Vec<String> = vectors.values().map(|v| v.name.clone()).collect();
        Ok(names)
    }

    /// Lưu database vào file
    pub fn save_to_file(&self, file_path: String) -> PyResult<bool> {
        let vectors = self.vectors.read().unwrap();
        let data: Vec<FaceVector> = vectors.values().cloned().collect();

        let json_data = serde_json::to_string_pretty(&data).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize error: {}", e))
        })?;

        fs::write(&file_path, json_data)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Write error: {}", e)))?;

        Ok(true)
    }

    /// Tải database từ file
    pub fn load_from_file(&self, file_path: String) -> PyResult<bool> {
        if !std::path::Path::new(&file_path).exists() {
            return Ok(false);
        }

        let json_data = fs::read_to_string(&file_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Read error: {}", e)))?;

        let data: Vec<FaceVector> = serde_json::from_str(&json_data).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Deserialize error: {}", e))
        })?;

        let mut vectors = self.vectors.write().unwrap();
        vectors.clear();

        for face_vector in data {
            vectors.insert(face_vector.id.clone(), face_vector);
        }

        Ok(true)
    }

    /// Cập nhật threshold
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }

    /// Lấy threshold hiện tại
    pub fn get_threshold(&self) -> f64 {
        self.threshold
    }
}

/// Performance timer for benchmarking
#[pyclass]
pub struct PerformanceTimer {
    start_time: std::time::Instant,
}

#[pymethods]
impl PerformanceTimer {
    #[new]
    pub fn new() -> Self {
        PerformanceTimer {
            start_time: std::time::Instant::now(),
        }
    }

    pub fn start(&mut self) {
        self.start_time = std::time::Instant::now();
    }

    pub fn stop(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}

/// Module Python
#[pymodule]
fn vector_database(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VectorDatabase>()?;
    m.add_class::<PerformanceTimer>()?;
    Ok(())
}
