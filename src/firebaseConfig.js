
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";



const firebaseConfig = {
    apiKey: "AIzaSyAhRtC8YU4pJ5g9KZ07cKXDqr4lLOCnNOs",
    authDomain: "argus-17702.firebaseapp.com",
    projectId: "argus-17702",
    storageBucket: "argus-17702.firebasestorage.app",
    messagingSenderId: "178416740087",
    appId: "1:178416740087:web:bc4127e0e8e4d3de2a55f2"
};


const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);