# Security Hardening Report — Elimination of `allow_pickle=True`

**Document Status**: Official Security Remediation Report  
**Target Subsystem**: Vector Gallery Storage (`storage/vector_store.py`)  
**Audit & Remediation Date**: July 29, 2026  
**Security Engineer**: Senior Python Security Engineer  

---

## 1. Files Audited

A repository-wide audit was conducted for all occurrences of `np.load`, `numpy.load`, and `allow_pickle`.

- [storage/vector_store.py](storage/vector_store.py) — Core gallery vector index persistence.
- [cli.py](cli.py) — CLI system status inspection utility.

No other occurrences of `np.load` or `allow_pickle` exist in the repository codebase.

---

## 2. Every `np.load` Location Audited

| File Location | Target File | Previous Call | Hardened Call | Pickle Required? |
|---|---|---|---|---|
| [storage/vector_store.py](storage/vector_store.py#L102) | `gallery_features.npy` | `np.load(..., allow_pickle=True)` | `np.load(..., allow_pickle=False)` | **NO** (Float32 Matrix) |
| [storage/vector_store.py](storage/vector_store.py#L107) | `gallery_labels.npy` | `np.load(..., allow_pickle=True)` | `np.load(..., allow_pickle=False)` | **NO** (Unicode String Array) |
| [cli.py](cli.py#L669) | `gallery_features.npy` | `np.load(str(path))` | `np.load(str(path), allow_pickle=False)` | **NO** (Float32 Matrix) |
| [cli.py](cli.py#L670) | `gallery_labels.npy` | `np.load(str(path))` | `np.load(str(path), allow_pickle=False)` | **NO** (Unicode String Array) |

---

## 3. Previous Behavior vs. New Secure Behavior

### Previous Behavior
`VectorStore.load()` previously executed:
```python
features = np.load(self.features_file, allow_pickle=True)
labels = np.load(self.labels_file, allow_pickle=True)
```
- **Vulnerability**: If a malicious or corrupted `.npy` file containing pickled Python objects was placed in `models/gallery/` or `models/live_gallery/`, Python's `pickle.load` was automatically invoked during pipeline initialization or recognition, allowing **Arbitrary Code Execution (ACE)**.

### New Secure Behavior
`VectorStore.load()` now executes:
```python
features = np.load(self.features_file, allow_pickle=False)
labels = np.load(self.labels_file, allow_pickle=False)
```
Followed by strict runtime validation:
1. **Pickle Rejection**: `allow_pickle=False` instructs NumPy to reject any file requiring object deserialization. If an object array or pickled payload is detected, NumPy raises `ValueError`, which `VectorStore` intercepts and converts into a clear migration error.
2. **Object Dtype Check**: Rejects any array with `dtype == object` or `dtype.kind == 'O'`.
3. **Numeric Features Validation**: Confirms features array is numeric (`np.issubdtype(features.dtype, np.number)`).
4. **Dimension & Shape Validation**: Confirms features is 2D `(N, D)` and labels is 1D `(N,)`.
5. **Length Match Validation**: Confirms `len(features) == len(labels)`.

---

## 4. Why `allow_pickle=False` is Safe

- **Features Array**: Extracted 256-D gait embeddings are float32 numbers (`np.float32`). Pure float32 matrices save and load natively in `.npy` standard binary format without Python object wrapping.
- **Labels Array**: Subject identity labels (e.g. `'person_001'`, `'subject_alpha'`) are stored as NumPy fixed-width Unicode string arrays (`<U...` dtype). Standard string arrays in NumPy do **NOT** use pickle and load 100% natively under `allow_pickle=False`.

---

## 5. Backward Compatibility Assessment

- **Existing Numeric Galleries**: All existing gallery files containing float32 features and Unicode labels continue working without any changes.
- **Legacy Object Galleries**: If any legacy gallery file relied on pickled Python objects, attempting to load it now raises a descriptive exception:
  ```text
  ValueError: Gallery features file 'models/gallery/gallery_features.npy' contains object arrays or requires pickle deserialization (allow_pickle=True), which is prohibited for security reasons. Migration required.
  ```
- **No Silent Fallback**: The code does **NOT** fall back to `allow_pickle=True` under any circumstances.

---

## 6. Unit Tests Added

Added 8 comprehensive security test cases in [tests/unit/test_vector_store.py](tests/unit/test_vector_store.py):

1. `test_missing_file_returns_none`: Verifies `load()` returns `None` safely when gallery files do not exist.
2. `test_valid_float32_gallery_and_labels`: Verifies valid float32 features and Unicode labels load cleanly with `allow_pickle=False`.
3. `test_empty_gallery`: Verifies empty galleries `(0, 256)` and `(0,)` load without errors.
4. `test_object_array_rejection`: Verifies pickled object arrays trigger `allow_pickle=False` rejection and raise descriptive `ValueError`.
5. `test_object_dtype_rejection_if_passed_without_pickle`: Verifies object dtypes (`dtype=object`) are rejected.
6. `test_malformed_dimensions`: Verifies 1D feature arrays trigger 2D matrix shape validation error.
7. `test_mismatched_lengths`: Verifies feature count != label count raises `ValueError`.
8. `test_corrupted_file`: Verifies corrupted binary files raise descriptive `ValueError`.

---

## 7. Verification Results

All required verification suites were executed cleanly:

- `python -m compileall`: **PASSED** (0 errors)
- `ruff check .`: **PASSED** (`All checks passed!`, 0 errors)
- `pytest -q`: **PASSED** (**265 passed, 1 warning in 31.42s**; +8 new security unit tests passing)
- `git diff --check`: **PASSED** (0 formatting/whitespace issues)

---

## 8. Security Impact

- **Arbitrary Code Execution Risk**: **ELIMINATED**.
- **Gallery File Tampering Vulnerability**: **MITIGATED**.
- **Biometric Template Integrity**: **ENFORCED** (Strict type, dimension, and non-object shape validation).

---

## 9. Remaining Recommendations

1. **AES-256 Gallery File Encryption**: Add optional Fernet/AES-256 encryption wrapper for `.npy` gallery files at rest to prevent template theft.
2. **FAISS HNSW Index Migration**: Migrate linear NumPy vector search to FAISS HNSW binary index for galleries scaling beyond $N > 100,000$.
