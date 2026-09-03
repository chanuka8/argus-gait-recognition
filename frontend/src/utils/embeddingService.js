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

    // 1. Process Reference Video(s) via Camera-Independent Offline Pipeline
    if (hasVideos) {
        for (const videoFile of videoFiles) {
            try {
                console.log(`[ARGUS] Uploading reference video '${videoFile.name}' for case: ${caseId}`);
                if (onProgress) {
                    onProgress({ stage: 'UPLOADING_REFERENCE' }, 'UPLOADING');
                }

                const vidFormData = new FormData();
                vidFormData.append('person_id', caseId);
                vidFormData.append('case_id', caseId);
                vidFormData.append('file', videoFile);

                const uploadUrl = `${baseUrl}/api/v1/cases/upload-reference`;
                const uploadRes = await fetch(uploadUrl, {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: vidFormData,
                });

                if (!uploadRes.ok) {
                    const errText = await uploadRes.text();
                    throw new Error(`Upload error ${uploadRes.status}: ${errText}`);
                }

                const uploadData = await uploadRes.json();
                const jobId = uploadData.job_id;
                console.log(`[ARGUS] Reference job created: ${jobId}. Polling execution...`);

                // Poll job until COMPLETED or FAILED
                const completedJob = await pollReferenceJob(jobId, onProgress);
                videoJobResult = completedJob;
                console.log('[ARGUS] Reference video processing completed:', completedJob);
            } catch (vidErr) {
                console.error('[ARGUS] Reference video processing failed:', vidErr);
                errors.push(`Reference Video: ${vidErr.message}`);
            }
        }
    }

    // 2. Process Image Enrollments via /api/v1/enroll
    if (hasImages) {
        try {
            if (onProgress) {
                onProgress({ stage: 'ENROLLING_IMAGES' }, 'PROCESSING');
            }

            const imgFormData = new FormData();
            imgFormData.append('person_id', caseId);
            imageFiles.forEach((file) => imgFormData.append('files', file));

            const enrollUrl = `${baseUrl}/api/v1/enroll`;
            const response = await fetch(enrollUrl, {
                method: 'POST',
                headers: getAuthHeaders(),
                body: imgFormData,
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Enrollment API error ${response.status}: ${errText}`);
            }

            imageEnrollResult = await response.json();
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
