// Provides surveillance zone definitions, CCTV camera node data, and simulated detection alert logging for the ARGUS CCTV network.

import { db } from '../firebaseConfig';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';

export const SURVEILLANCE_ZONES = [
    {
        id: 'Z01',
        name: 'Western Transit Corridor (Colombo)',
        center: [6.9315, 79.8590],
        radius: 4000,
        color: '#00E5FF',
        description: 'Primary terminal intersections, central railway hub, and harbor security perimeter.'
    },
    {
        id: 'Z02',
        name: 'Southern Gateway Checkpoint (Galle)',
        center: [6.0420, 80.2180],
        radius: 3500,
        color: '#00E5FF',
        description: 'Expressway interchange exits and southern transit junctions.'
    },
    {
        id: 'Z03',
        name: 'Central Highland Hub (Kandy)',
        center: [7.2880, 80.6250],
        radius: 4000,
        color: '#00E5FF',
        description: 'Municipal border crossings, central bus stand, and ring road bridge monitoring.'
    },
    {
        id: 'Z04',
        name: 'Northern Transit Sector (Jaffna)',
        center: [9.6630, 80.0180],
        radius: 4500,
        color: '#00E5FF',
        description: 'Northern railway gate checkpoints and expressway arterial routes.'
    }
];

export const CCTV_CAMERAS = [
    {
        id: 'CCTV-101',
        zoneId: 'Z01',
        name: 'Fort Railway Central Terminal Platform 1',
        lat: 6.9333,
        lng: 79.8601,
        status: 'ONLINE',
        resolution: '4K AI Stream',
        ip: '192.168.10.11'
    },
    {
        id: 'CCTV-102',
        zoneId: 'Z01',
        name: 'Pettah Main Interchange Gate',
        lat: 6.9358,
        lng: 79.8596,
        status: 'ONLINE',
        resolution: '1080p LPR & Face AI',
        ip: '192.168.10.12'
    },
    {
        id: 'CCTV-103',
        zoneId: 'Z01',
        name: 'Port City Security Access Gate 4',
        lat: 6.9389,
        lng: 79.8512,
        status: 'ONLINE',
        resolution: '4K IR PTZ Stream',
        ip: '192.168.10.15'
    },
    {
        id: 'CCTV-201',
        zoneId: 'Z02',
        name: 'Southern Expressway Toll Interchange',
        lat: 6.0535,
        lng: 80.2210,
        status: 'ONLINE',
        resolution: '4K Face & LPR Stream',
        ip: '192.168.20.04'
    },
    {
        id: 'CCTV-202',
        zoneId: 'Z02',
        name: 'Galle Central Roundabout Checkpoint',
        lat: 6.0351,
        lng: 80.2162,
        status: 'ONLINE',
        resolution: '1080p AI Stream',
        ip: '192.168.20.05'
    },
    {
        id: 'CCTV-301',
        zoneId: 'Z03',
        name: 'Kandy Clock Tower Intersection',
        lat: 7.2925,
        lng: 80.6350,
        status: 'ONLINE',
        resolution: '4K AI Surveillance',
        ip: '192.168.30.01'
    },
    {
        id: 'CCTV-302',
        zoneId: 'Z03',
        name: 'Peradeniya Junction Bridge View',
        lat: 7.2681,
        lng: 80.5966,
        status: 'ONLINE',
        resolution: '1080p AI Stream',
        ip: '192.168.30.02'
    },
    {
        id: 'CCTV-401',
        zoneId: 'Z04',
        name: 'Jaffna Railway Station Front Gate',
        lat: 9.6648,
        lng: 80.0175,
        status: 'ONLINE',
        resolution: '4K AI Stream',
        ip: '192.168.40.01'
    }
];

export const triggerSimulatedDetection = async (cameraObj, targetCase) => {
    try {
        const detectionRef = collection(db, 'detections');
        const detectionDoc = {
            caseId: targetCase?.caseId || targetCase?.id || 'TEST-CASE-999',
            victimName: targetCase?.name || 'Unassigned Test Subject',
            cameraId: cameraObj.id,
            locationName: cameraObj.name,
            coordinates: {
                lat: parseFloat(cameraObj.lat),
                lng: parseFloat(cameraObj.lng)
            },
            confidenceScore: 0.94,
            timestamp: serverTimestamp(),
            alertStatus: "ACTIVE_PURSUIT",
            zoneId: cameraObj.zoneId
        };
        await addDoc(detectionRef, detectionDoc);
        return true;
    } catch (error) {
        console.error("Failed to simulate AI CCTV match:", error);
        throw error;
    }
};
