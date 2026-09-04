import { db } from '../firebaseConfig';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { API_BASE, getAuthHeaders } from '../config/apiConfig';

/**
 * Poll the reference video processing job until completion or failure.
 *
 * @param {string} jobId - The job ID returned by /api/v1/cases/upload-reference
 * @param {function} onProgress - Callback receiving job progress updates
 * @param {number} timeoutMs - Maximum polling timeout in milliseconds
 * @returns {Promise<object>} - The completed job record or throws on error
 */
export async function pollReferenceJob(jobId, onProgress, timeoutMs = 180000) {
    const baseUrl = API_BASE || '';
    const pollUrl = `${baseUrl}/api/v1/cases/jobs/${jobId}`;
    const startTime = Date.now();

    while (Date.now() - startTime < timeoutMs) {
        try {
            const res = await fetch(pollUrl, {
                headers: getAuthHeaders(),
            });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(`Job status check error ${res.status}: ${text}`);
            }

            const jobData = await res.json();
            if (onProgress && typeof onProgress === 'function') {
                onProgress(jobData.progress || {}, jobData.status, jobData);
            }

            if (jobData.status === 'COMPLETED') {
                return jobData;
            }

            if (jobData.status === 'FAILED') {
                const diag = jobData.diagnostic_code ? `[${jobData.diagnostic_code}] ` : '';
                throw new Error(`${diag}${jobData.error_message || 'Reference processing failed'}`);
            }

            // Wait 1.0s between polls
            await new Promise((resolve) => setTimeout(resolve, 1000));
        } catch (pollErr) {
            if (pollErr.message.includes('[') || pollErr.message.includes('FAILED')) {
                throw pollErr;
            }
            console.warn(`[ARGUS] Poll retry notice for ${jobId}:`, pollErr.message);
            await new Promise((resolve) => setTimeout(resolve, 1500));
        }
    }

    throw new Error(`Reference video processing timed out after ${timeoutMs / 1000}s`);
}

/**
 * Helper to upload FormData with real XMLHttpRequest byte-level upload progress.
 *
 * @param {string} url - Destination URL
 * @param {FormData} formData - FormData payload
 * @param {function} onUploadProgress - Callback receiving { loaded, total, percent }
 * @param {object} options - Optional parameters (timeoutMs, headers, signal)
 * @returns {Promise<object>} - Parsed JSON response
 */
export function uploadWithProgress(url, formData, onUploadProgress, options = {}) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const timeoutMs = options.timeoutMs || 300000;

        xhr.open('POST', url, true);
        xhr.timeout = timeoutMs;

        // Apply auth headers
        const authHeaders = getAuthHeaders();
        for (const [key, value] of Object.entries(authHeaders)) {
            if (key.toLowerCase() !== 'content-type') {
                xhr.setRequestHeader(key, value);
            }
        }

        if (options.headers) {
            for (const [key, value] of Object.entries(options.headers)) {
                if (key.toLowerCase() !== 'content-type') {
                    xhr.setRequestHeader(key, value);
                }
            }
        }

        if (xhr.upload && onUploadProgress) {
            xhr.upload.onprogress = (event) => {
                if (event.lengthComputable && event.total > 0) {
                    const percent = Math.min(100, Math.round((event.loaded / event.total) * 100));
                    onUploadProgress({
                        loaded: event.loaded,
                        total: event.total,
                        percent: percent,
                    });
                }
            };
        }

        xhr.onload = () => {
            let responseData = null;
            try {
                responseData = xhr.responseText ? JSON.parse(xhr.responseText) : {};
            } catch {
                responseData = { text: xhr.responseText };
            }

            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(responseData);
            } else {
                const detail = responseData?.detail || responseData?.message || xhr.statusText || 'Upload failed';
                const error = new Error(`Upload error ${xhr.status}: ${typeof detail === 'object' ? JSON.stringify(detail) : detail}`);
                error.status = xhr.status;
                error.data = responseData;
                reject(error);
            }
        };

        xhr.onerror = () => {
            reject(new Error('Network error during file upload. Please check your connection.'));
        };

        xhr.ontimeout = () => {
            reject(new Error(`Upload timed out after ${timeoutMs / 1000}s`));
        };

        xhr.onabort = () => {
            reject(new Error('Upload aborted by user'));
        };

        if (options.signal) {
            options.signal.addEventListener('abort', () => {
                xhr.abort();
            });
        }

        xhr.send(formData);
    });
}

/**
 * Send media files (images and/or reference videos) to the ARGUS AI backend.
 *
 * Image Path:
 *   POST /api/v1/enroll (ByGaitLight 256D + OSNet 512D)
 *
 * Video Path:
 *   POST /api/v1/cases/upload-reference -> Asynchronous Camera-Independent Reference Processor
 *   Polls GET /api/v1/cases/jobs/{job_id} until completion, reporting real-time progress.
 *
 * @param {string} caseId - Person/case identifier
 * @param {string} name - Person name
 * @param {File[]} imageFiles - Uploaded image files
 * @param {File[]} videoFiles - Uploaded reference video files
 * @param {function} onProgress - Optional real-time progress callback
 */
/**
 * Upload a file using the resumable chunked upload subsystem.
 *
 * Slices the file into 2 MiB chunks, initializes a server upload session,
 * uploads each chunk with bounded exponential backoff retries, calculates
 * real transfer speed (MB/s) and ETA, and commits the session upon completion.
 *
 * @param {File} file - File to upload
 * @param {string} caseId - Person/case identifier
 * @param {function} onProgress - Callback receiving progress metrics
 * @param {object} options - Options { chunkSize, maxRetries, retryDelayBaseMs }
 * @returns {Promise<object>} - { upload_id, job_id, status, media_path }
 */
export async function uploadFileChunked(file, caseId, onProgress, options = {}) {
    const baseUrl = API_BASE || '';
    const chunkSize = options.chunkSize || 2 * 1024 * 1024; // 2 MiB default
    const maxRetries = options.maxRetries || 3;
    const retryDelayBaseMs = options.retryDelayBaseMs || 1000;
    const mediaType = file.type.startsWith('video') ? 'video' : 'image';
    const totalSize = file.size;

    // 1. Initialize upload session
    const initRes = await fetch(`${baseUrl}/api/v1/cases/upload-session/init`, {
        method: 'POST',
        headers: {
            ...getAuthHeaders(),
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            person_id: caseId,
            case_id: caseId,
            filename: file.name,
            total_size: totalSize,
            chunk_size: chunkSize,
            media_type: mediaType,
        }),
    });

    if (!initRes.ok) {
        const errText = await initRes.text();
        throw new Error(`Failed to initialize upload session (${initRes.status}): ${errText}`);
    }

    const initData = await initRes.json();
    const uploadId = initData.upload_id;
    const totalChunks = initData.total_chunks;

    // 2. Query status to support resumption if session existed
    let confirmedChunks = new Set();
    let bytesUploaded = 0;

    try {
        const statusRes = await fetch(`${baseUrl}/api/v1/cases/upload-session/${uploadId}/status`, {
            headers: getAuthHeaders(),
        });
        if (statusRes.ok) {
            const statusData = await statusRes.json();
            confirmedChunks = new Set(statusData.chunks_received || []);
            bytesUploaded = statusData.bytes_received || 0;
        }
    } catch (statusErr) {
        console.warn('[ARGUS] Upload session status check notice:', statusErr.message);
    }

    const startTime = Date.now();

    const reportProgress = (currentLoaded, connectionStatus = 'stable') => {
        if (!onProgress || typeof onProgress !== 'function') return;
        const now = Date.now();
        const elapsedSec = (now - startTime) / 1000;
        const speedBps = elapsedSec > 0.3 ? (currentLoaded / elapsedSec) : 0;
        const speedMBs = speedBps > 0 ? (speedBps / (1024 * 1024)).toFixed(2) : '0.00';
        const remainingBytes = Math.max(0, totalSize - currentLoaded);
        const etaSeconds = speedBps > 0 ? Math.ceil(remainingBytes / speedBps) : 0;
        const percent = totalSize > 0 ? Math.min(100, Math.round((currentLoaded / totalSize) * 100)) : 0;

        onProgress({
            phase: 'UPLOAD',
            mediaType,
            percent,
            loaded: currentLoaded,
            total: totalSize,
            speedMBs,
            etaSeconds,
            connectionStatus,
        }, 'UPLOADING');
    };

    reportProgress(bytesUploaded, 'stable');

    // 3. Upload missing chunks with bounded exponential backoff retry
    for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
        if (confirmedChunks.has(chunkIdx)) {
            continue;
        }

        const startByte = chunkIdx * chunkSize;
        const endByte = Math.min(totalSize, startByte + chunkSize);
        const chunkBlob = file.slice(startByte, endByte);

        let uploaded = false;
        let attempt = 0;

        while (!uploaded && attempt < maxRetries) {
            attempt++;
            try {
                const formData = new FormData();
                formData.append('chunk_index', chunkIdx.toString());
                formData.append('file', chunkBlob, file.name);

                const chunkRes = await fetch(`${baseUrl}/api/v1/cases/upload-session/${uploadId}/chunk`, {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: formData,
                });

                if (!chunkRes.ok) {
                    const errData = await chunkRes.text();
                    throw new Error(`Server returned status ${chunkRes.status}: ${errData}`);
                }

                const chunkData = await chunkRes.json();
                confirmedChunks.add(chunkIdx);
                bytesUploaded = chunkData.bytes_received;
                uploaded = true;
                reportProgress(bytesUploaded, 'stable');
            } catch (chunkErr) {
                console.warn(`[ARGUS] Chunk ${chunkIdx} upload failed (attempt ${attempt}/${maxRetries}):`, chunkErr.message);
                reportProgress(bytesUploaded, 'retrying');

                if (attempt >= maxRetries) {
                    throw new Error(`Upload failed on chunk ${chunkIdx}/${totalChunks} after ${maxRetries} retries: ${chunkErr.message}`);
                }

                const backoffDelay = retryDelayBaseMs * Math.pow(2, attempt - 1);
                await new Promise((resolve) => setTimeout(resolve, backoffDelay));
            }
        }
    }

    // 4. Commit upload session on server
    reportProgress(totalSize, 'committing');

    const commitRes = await fetch(`${baseUrl}/api/v1/cases/upload-session/${uploadId}/commit`, {
        method: 'POST',
        headers: getAuthHeaders(),
    });

    if (!commitRes.ok) {
        const commitErr = await commitRes.text();
        throw new Error(`Failed to commit upload session (${commitRes.status}): ${commitErr}`);
    }

    const commitData = await commitRes.json();

    if (onProgress && typeof onProgress === 'function') {
        onProgress({
            phase: 'UPLOAD_COMPLETE',
            mediaType,
            percent: 100,
            loaded: totalSize,
            total: totalSize,
            speedMBs: '0.00',
            etaSeconds: 0,
            connectionStatus: 'stable',
            jobId: commitData.job_id,
        }, 'UPLOAD_COMPLETE', commitData);
    }

    return commitData;
}

/**
 * Send media files (images and/or reference videos) to the ARGUS AI backend.
 *
 * Image Path:
 *   Decoupled async enrollment via POST /api/v1/enroll?async_mode=true
 *   Polls GET /api/v1/cases/jobs/{job_id} until completion.
 *
 * Video Path:
 *   Chunked resumable upload via uploadFileChunked
 *   Polls GET /api/v1/cases/jobs/{job_id} until completion.
 *
 * @param {string} caseId - Person/case identifier
 * @param {string} name - Person name
 * @param {File[]} imageFiles - Uploaded image files
 * @param {File[]} videoFiles - Uploaded reference video files
 * @param {function} onProgress - Optional real-time progress callback
 */
export async function sendMediaToModel(caseId, name, imageFiles, videoFiles, onProgress) {
    const errors = [];
    let imageEnrollResult = null;
    let videoJobResult = null;
    const baseUrl = API_BASE || '';

    const hasImages = Array.isArray(imageFiles) && imageFiles.length > 0;
    const hasVideos = Array.isArray(videoFiles) && videoFiles.length > 0;

    if (!hasImages && !hasVideos) {
        return { success: true, errors: [], status: 'NO_MEDIA' };
    }

    // 1. Process Reference Video(s) via Resumable Chunked Ingestion + Async Processing
    if (hasVideos) {
        for (let idx = 0; idx < videoFiles.length; idx++) {
            const videoFile = videoFiles[idx];
            try {
                console.log(`[ARGUS] Uploading reference video '${videoFile.name}' for case: ${caseId} (${(videoFile.size / (1024 * 1024)).toFixed(1)} MB)`);

                const commitData = await uploadFileChunked(videoFile, caseId, onProgress);
                const jobId = commitData.job_id;
                console.log(`[ARGUS] Video upload complete. Reference job created: ${jobId}. Polling execution...`);

                if (onProgress) {
                    onProgress({
                        phase: 'PROCESSING',
                        mediaType: 'video',
                        stage: 'QUEUED',
                        percent: 0,
                        jobId: jobId,
                    }, 'QUEUED');
                }

                // Poll job until COMPLETED or FAILED
                const completedJob = await pollReferenceJob(jobId, (progressData, status, jobData) => {
                    if (onProgress) {
                        onProgress({
                            phase: 'PROCESSING',
                            mediaType: 'video',
                            ...progressData,
                            jobId: jobId,
                        }, status, jobData);
                    }
                });
                videoJobResult = completedJob;
                console.log('[ARGUS] Reference video processing completed:', completedJob);
            } catch (vidErr) {
                console.error('[ARGUS] Reference video processing failed:', vidErr);
                errors.push(`Reference Video: ${vidErr.message}`);
            }
        }
    }

    // 2. Process Image Enrollments via Decoupled Asynchronous /api/v1/enroll
    if (hasImages) {
        try {
            const totalImageBytes = imageFiles.reduce((acc, f) => acc + (f.size || 0), 0);
            if (onProgress) {
                onProgress({
                    phase: 'UPLOAD',
                    mediaType: 'image',
                    stage: 'UPLOADING_IMAGES',
                    percent: 0,
                    loaded: 0,
                    total: totalImageBytes,
                    totalFiles: imageFiles.length,
                    speedMBs: '0.00',
                    etaSeconds: 0,
                    connectionStatus: 'stable',
                }, 'UPLOADING');
            }

            const imgFormData = new FormData();
            imgFormData.append('person_id', caseId);
            imgFormData.append('async_mode', 'true');
            imageFiles.forEach((file) => imgFormData.append('files', file));

            const enrollUrl = `${baseUrl}/api/v1/enroll`;
            const enrollData = await uploadWithProgress(enrollUrl, imgFormData, (progress) => {
                if (onProgress) {
                    onProgress({
                        phase: 'UPLOAD',
                        mediaType: 'image',
                        stage: 'UPLOADING_IMAGES',
                        percent: progress.percent,
                        loaded: progress.loaded,
                        total: progress.total,
                        totalFiles: imageFiles.length,
                        speedMBs: '0.00',
                        etaSeconds: 0,
                        connectionStatus: 'stable',
                    }, 'UPLOADING');
                }
            });

            if (onProgress) {
                onProgress({
                    phase: 'UPLOAD_COMPLETE',
                    mediaType: 'image',
                    percent: 100,
                    loaded: totalImageBytes,
                    total: totalImageBytes,
                    speedMBs: '0.00',
                    etaSeconds: 0,
                    connectionStatus: 'stable',
                }, 'UPLOAD_COMPLETE');
            }

            if (enrollData.job_id) {
                const photoJobId = enrollData.job_id;
                console.log(`[ARGUS] Photo enrollment job created: ${photoJobId}. Polling execution...`);
                const completedPhotoJob = await pollReferenceJob(photoJobId, (progressData, status, jobData) => {
                    if (onProgress) {
                        onProgress({
                            phase: 'PROCESSING',
                            mediaType: 'image',
                            ...progressData,
                            jobId: photoJobId,
                        }, status, jobData);
                    }
                });
                imageEnrollResult = {
                    ...enrollData,
                    gait_embeddings_added: completedPhotoJob?.result?.embeddings_committed || completedPhotoJob?.result?.gait_embeddings_committed || 0,
                    appearance_embeddings_added: completedPhotoJob?.result?.appearance_embeddings_committed || 0,
                };
            } else {
                imageEnrollResult = enrollData;
            }

            console.log('[ARGUS] Image enrollment complete:', imageEnrollResult);
        } catch (imgErr) {
            console.error('[ARGUS] Image enrollment failed:', imgErr);
            errors.push(`Images: ${imgErr.message}`);
        }
    }

    // 3. Save Unified Biometric Enrollment Lineage to Firestore
    const combinedResult = {
        status: errors.length === 0 ? 'ENROLLED' : 'PARTIAL_OR_FAILED',
        gait_embeddings_added:
            (imageEnrollResult?.gait_embeddings_added || 0) +
            (videoJobResult?.result?.embeddings_committed || 0),
        appearance_embeddings_added: imageEnrollResult?.appearance_embeddings_added || 0,
        firebase_status: errors.length === 0 ? 'CONFIRMED' : 'ERROR',
        video_job_id: videoJobResult?.job_id || null,
        video_result: videoJobResult?.result || null,
    };

    try {
        await saveEnrollmentResultToFirestore(caseId, combinedResult);
    } catch (fsErr) {
        console.error('[ARGUS] Firestore enrollment save failed:', fsErr);
        errors.push(`Firestore: ${fsErr.message}`);
    }

    return {
        success: errors.length === 0,
        errors,
        enrollResult: imageEnrollResult,
        videoJobResult,
        combinedResult,
    };
}

/**
 * Save enrollment results and embedding lineage to Firestore.
 */
export async function saveEnrollmentResultToFirestore(caseId, enrollResult) {
    const resultsRef = doc(db, 'biometric_cases', caseId);
    await setDoc(
        resultsRef,
        {
            caseId,
            status: enrollResult?.status || 'ENROLLED',
            gaitEmbeddingsCount: enrollResult?.gait_embeddings_added || 0,
            appearanceEmbeddingsCount: enrollResult?.appearance_embeddings_added || 0,
            firebasePersisted: enrollResult?.firebase_status || 'PENDING',
            videoJobId: enrollResult?.video_job_id || null,
            enrolledAt: serverTimestamp(),
            pipeline: 'ByGaitLight-256D + LiveGEI + ByteTrack (Offline Ref)',
        },
        { merge: true }
    );
    console.log(`[ARGUS] Enrollment metadata saved to Firestore for case: ${caseId}`);
}

/**
 * Legacy wrappers for backward compatibility
 */
export async function sendImagesToModel(imageFiles, caseId, name) {
    if (!imageFiles || imageFiles.length === 0) return { results: [], total_files: 0 };
    return sendMediaToModel(caseId, name, imageFiles, []);
}

export async function sendVideoToModel(videoFile, caseId, name) {
    if (!videoFile) return null;
    return sendMediaToModel(caseId, name, [], [videoFile]);
}
