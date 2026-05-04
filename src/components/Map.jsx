import React from 'react';
import { MapContainer, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import './Map.css';

const Map = () => {
    // Default position (Sri Lanka)
    const position = [7.8731, 80.7718];
    
    // Restrict map to Sri Lanka bounds [SouthWest, NorthEast]
    const sriLankaBounds = [
        [5.8, 79.5],
        [9.9, 82.0]
    ];

    return (
        <div className="map-container">
            <MapContainer 
                center={position} 
                zoom={7} 
                minZoom={7}
                maxBounds={sriLankaBounds}
                maxBoundsViscosity={1.0}
                scrollWheelZoom={true}
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
            </MapContainer>
        </div>
    );
};

export default Map;