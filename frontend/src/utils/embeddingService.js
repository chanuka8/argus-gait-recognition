/**
 * LEGACY / OPTIONAL: Face Recognition Embedding Service
 *
 * NOTE: The primary ARGUS biometric pipeline operates on Gait Biometrics (2D GEI + ByGaitLight).
 * This service is preserved for optional secondary face-recognition API integration
 * without breaking case reporting flows.
 */

import { db } from '../firebaseConfig';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { FACE_API_BASE_URL } from '../config/apiConfig';

export async function sendImagesToModel(imageFiles, caseId, name) {
    if (!imageFiles || imageFiles.length === 0) return { results: [], total_files: 0 };

    const formData = new FormData();
    imageFiles.forEach((file) => formData.append('files', file));
    formData.append('case_id', caseId);
    formData.append('name', name);

    const url = FACE_API_BASE_URL ? `${FACE_API_BASE_URL}/process-images` : '/process-images';
    const response = await fetch(url, { method: 'POST', body: formData });
    if (!response.ok) {
        const errText = await response.text();
        throw new Error(`API error (images): ${response.status} — ${errText}`);
    }
    return await response.json();
}

export async function sendVideoToModel(videoFile, caseId, name) {
    if (!videoFile) return null;

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('case_id', caseId);
    formData.append('name', name);

    const url = FACE_API_BASE_URL ? `${FACE_API_BASE_URL}/process-video` : '/process-video';
    const response = await fetch(url, { method: 'POST', body: formData });
    if (!response.ok) {
        const errText = await response.text();
        throw new Error(`API error (video "${videoFile.name}"): ${response.status} — ${errText}`);
    }
    return await response.json();
}

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

export async function sendMediaToModel(caseId, name, imageFiles, videoFiles) {
    const errors = [];
    let imageResults = null;
    const videoResults = [];

    if (imageFiles && imageFiles.length > 0) {
        try {
            imageResults = await sendImagesToModel(imageFiles, caseId, name);
            console.log('[ARGUS] Images registered:', imageResults);
        } catch (err) {
            console.error('[ARGUS] Image send failed:', err);
            errors.push(`Images: ${err.message}`);
        }
    }

    for (const videoFile of (videoFiles || [])) {
        try {
            const result = await sendVideoToModel(videoFile, caseId, name);
            if (result) videoResults.push(result);
            console.log(`[ARGUS] Video registered: ${videoFile.name}`, result);
        } catch (err) {
            console.error('[ARGUS] Video send failed "${videoFile.name}":', err);
            errors.push(`Video (${videoFile.name}): ${err.message}`);
        }
    }

    if (imageResults || videoResults.length > 0) {
        try {
            await saveModelResultsToFirestore(caseId, imageResults, videoResults);
        } catch (err) {
            console.error('[ARGUS] Firestore save failed:', err);
            errors.push(`Firestore: ${err.message}`);
        }
    }

    return { success: errors.length === 0, errors };
}
