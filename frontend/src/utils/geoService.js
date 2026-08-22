// Provides GPS coordinate capture, reverse geocoding via Nominatim, and Firestore detection alert logging for surveillance tracking.

import { db } from '../firebaseConfig';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';

export const getDeviceGPS = () => {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error("Geolocation is not supported by this browser/device."));
            return;
        }
        navigator.geolocation.getCurrentPosition(
            (position) => {
                resolve({
                    lat: parseFloat(position.coords.latitude.toFixed(6)),
                    lng: parseFloat(position.coords.longitude.toFixed(6)),
                    accuracy: position.coords.accuracy
                });
            },
            (error) => {
                reject(error);
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        );
    });
};

export const getCurrentDevicePosition = getDeviceGPS;

export const reverseGeocode = async (lat, lng) => {
    try {
        const response = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`, {
            headers: {
                'Accept-Language': 'en'
            }
        });
        if (!response.ok) throw new Error("Failed to reach OpenStreetMap Nominatim service");
        const data = await response.json();
        const city = data.address?.city || data.address?.town || data.address?.village || data.address?.suburb || "Sri Lanka Area";
        const district = data.address?.state_district || data.address?.county || data.address?.state || "Region";
        return {
            displayName: data.display_name || `${city}, ${district}`,
            city: city,
            district: district
        };
    } catch (error) {
        console.error("Reverse geocoding error:", error);
        return {
            displayName: `GPS: ${lat}, ${lng}`,
            city: "Custom GPS Coordinates",
            district: "GPS Data"
        };
    }
};

export const logCameraDetection = async (caseId, victimName, cameraId, locationName, lat, lng, confidenceScore = 0.92) => {
    try {
        const detectionRef = collection(db, 'detections');
        await addDoc(detectionRef, {
            caseId: caseId,
            victimName: victimName,
            cameraId: cameraId,
            locationName: locationName,
            coordinates: {
                lat: parseFloat(lat),
                lng: parseFloat(lng)
            },
            confidenceScore: confidenceScore,
            timestamp: serverTimestamp(),
            alertStatus: "ACTIVE_PURSUIT"
        });
        return true;
    } catch (error) {
        console.error("Error registering surveillance detection:", error);
        throw error;
    }
};
