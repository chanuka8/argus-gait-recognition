
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";



const firebaseConfig = {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "",
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "argus-17702.firebaseapp.com",
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "argus-17702",
    storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "argus-17702.firebasestorage.app",
    messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "178416740087",
    appId: import.meta.env.VITE_FIREBASE_APP_ID || "1:178416740087:web:bc4127e0e8e4d3de2a55f2"
};


const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);