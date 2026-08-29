
import { db } from '../firebaseConfig';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { API_BASE } from '../config/apiConfig';

/**
 * Send media files (images + videos) to the ARGUS AI unified /api/v1/enroll endpoint
 * for gait + appearance embedding extraction.
 *
 * This replaces the legacy /process-images and /process-video endpoints with
 * the production enrollment pipeline that uses:
 *   - ByGaitLight CNN → 256D Gait Embedding
 *   - OSNet ReID → 512D Appearance Embedding
 *   - Firebase Durable Embedding Persistence
 */
export async function sendMediaToModel(caseId, name, imageFiles, videoFiles) {
    const errors = [];
    let enrollResult = null;

    const allFiles = [];
    if (imageFiles && imageFiles.length > 0) {
        allFiles.push(...imageFiles);
    }
    if (videoFiles && videoFiles.length > 0) {
        allFiles.push(...videoFiles);
    }

    if (allFiles.length === 0) {
        return { success: true, errors: [], status: 'NO_MEDIA' };
    }

    try {
        const formData = new FormData();
        formData.append('person_id', caseId);

        allFiles.forEach((file) => formData.append('files', file));

        const baseUrl = API_BASE || '';
        const url = `${baseUrl}/api/v1/enroll`;

        const response = await fetch(url, { method: 'POST', body: formData });
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`Enrollment API error: ${response.status} — ${errText}`);
        }
        enrollResult = await response.json();
        console.log('[ARGUS] Enrollment complete:', enrollResult);
    } catch (err) {
        console.error('[ARGUS] Enrollment failed:', err);
        errors.push(`Enrollment: ${err.message}`);
    }

    // Save enrollment metadata to Firestore
    if (enrollResult) {
        try {
            await saveEnrollmentResultToFirestore(caseId, enrollResult);
        } catch (err) {
            console.error('[ARGUS] Firestore enrollment save failed:', err);
            errors.push(`Firestore: ${err.message}`);
        }
    }

    return { success: errors.length === 0, errors, enrollResult };
}

/**
 * Save enrollment results and embedding lineage to Firestore.
 */
export async function saveEnrollmentResultToFirestore(caseId, enrollResult) {
    const resultsRef = doc(db, 'biometric_cases', caseId);
    await setDoc(resultsRef, {
        caseId,
        status: enrollResult?.status || 'ENROLLED',
        gaitEmbeddingsCount: enrollResult?.gait_embeddings_added || 0,
        appearanceEmbeddingsCount: enrollResult?.appearance_embeddings_added || 0,
        firebasePersisted: enrollResult?.firebase_status || 'PENDING',
        enrolledAt: serverTimestamp(),
        pipeline: 'ByGaitLight-256D + OSNet-512D',
    });
    console.log(`[ARGUS] Enrollment metadata saved to Firestore for case: ${caseId}`);
}

/**
 * Legacy: Send images to model (for backward compatibility).
 * Routes through the unified enrollment endpoint.
 */
export async function sendImagesToModel(imageFiles, caseId, name) {
    if (!imageFiles || imageFiles.length === 0) return { results: [], total_files: 0 };
    return sendMediaToModel(caseId, name, imageFiles, []);
}

/**
 * Legacy: Send video to model (for backward compatibility).
 * Routes through the unified enrollment endpoint.
 */
export async function sendVideoToModel(videoFile, caseId, name) {
    if (!videoFile) return null;
    return sendMediaToModel(caseId, name, [], [videoFile]);
}

/**
 * Legacy: Save model results to Firestore (for backward compatibility).
 */
export async function saveModelResultsToFirestore(caseId, imageResults, videoResults) {
    const resultsRef = doc(db, 'face_model_results', caseId);
    await setDoc(resultsRef, {
        caseId,
        imageResults: imageResults?.results || [],
        videoResults: (videoResults || []).filter(Boolean),
        total_image_files: imageResults?.total_files || 0,
        total_video_files: (videoResults || []).filter(Boolean).length,
        processedAt: serverTimestamp()
    });
    console.log(`[ARGUS] ML results saved to Firestore for case: ${caseId}`);
}
