import { useState, useEffect, useRef, useCallback } from 'react';
import { loadLayoutPreferences, saveLayoutPreference } from '../utils/layoutStorage';

/**
 * Hook for high-performance, accessible, pointer-event-based panel resizing.
 *
 * @param {Object} options
 * @param {string} options.storageKey Key in layoutStorage (e.g. 'dashboardDockWidth')
 * @param {number} options.defaultSize Default dimension (in px)
 * @param {number} options.minSize Minimum dimension (in px)
 * @param {number} options.maxSize Maximum dimension (in px)
 * @param {'left' | 'right'} options.direction 'left' if panel is on the right side (dragging left increases width), 'right' if panel is on the left side (dragging right increases width)
 * @param {number} [options.step=15] Keyboard step in pixels
 */
export const useResizablePanel = ({
    storageKey,
    defaultSize = 420,
    minSize = 300,
    maxSize = 640,
    direction = 'left',
    step = 15,
}) => {
    const [size, setSize] = useState(() => {
        const stored = loadLayoutPreferences();
        return stored[storageKey] || defaultSize;
    });

    const [isResizing, setIsResizing] = useState(false);
    const sizeRef = useRef(size);

    useEffect(() => {
        sizeRef.current = size;
    }, [size]);

    const dragStartRef = useRef({
        startX: 0,
        startSize: size,
        rafId: null,
    });

    // Cleanup RAF and body class on unmount
    useEffect(() => {
        return () => {
            if (dragStartRef.current.rafId) {
                cancelAnimationFrame(dragStartRef.current.rafId);
            }
            document.body.classList.remove('argus-panel-resizing');
        };
    }, []);

    const onPointerDown = useCallback((e) => {
        // Only accept primary button
        if (e.button !== 0) return;

        e.preventDefault();
        e.stopPropagation();

        const startX = e.clientX;
        const startSize = sizeRef.current;

        dragStartRef.current = {
            startX,
            startSize,
            rafId: null,
        };

        setIsResizing(true);
        document.body.classList.add('argus-panel-resizing');

        const onPointerMove = (moveEvent) => {
            if (dragStartRef.current.rafId) {
                cancelAnimationFrame(dragStartRef.current.rafId);
            }

            dragStartRef.current.rafId = requestAnimationFrame(() => {
                const deltaX = moveEvent.clientX - dragStartRef.current.startX;
                const multiplier = direction === 'left' ? -1 : 1;
                const rawNewSize = dragStartRef.current.startSize + (deltaX * multiplier);

                // Check max constraint against viewport width as well
                const maxAvailable = typeof window !== 'undefined'
                    ? Math.min(maxSize, window.innerWidth * 0.65)
                    : maxSize;

                const clampedSize = Math.round(
                    Math.max(minSize, Math.min(maxAvailable, rawNewSize))
                );

                setSize(clampedSize);
            });
        };

        const onPointerUp = () => {
            if (dragStartRef.current.rafId) {
                cancelAnimationFrame(dragStartRef.current.rafId);
            }

            setIsResizing(false);
            document.body.classList.remove('argus-panel-resizing');

            // Persist the final size
            if (storageKey) {
                saveLayoutPreference(storageKey, sizeRef.current);
            }

            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('pointerup', onPointerUp);
            window.removeEventListener('pointercancel', onPointerUp);
        };

        window.addEventListener('pointermove', onPointerMove, { passive: true });
        window.addEventListener('pointerup', onPointerUp);
        window.addEventListener('pointercancel', onPointerUp);
    }, [direction, minSize, maxSize, storageKey]);

    const onKeyDown = useCallback((e) => {
        let delta = 0;
        if (e.key === 'ArrowLeft') {
            delta = direction === 'left' ? step : -step;
        } else if (e.key === 'ArrowRight') {
            delta = direction === 'left' ? -step : step;
        } else if (e.key === 'Home') {
            delta = -10000;
        } else if (e.key === 'End') {
            delta = 10000;
        } else {
            return;
        }

        e.preventDefault();
        const current = sizeRef.current;
        const maxAvailable = typeof window !== 'undefined'
            ? Math.min(maxSize, window.innerWidth * 0.65)
            : maxSize;

        const nextSize = Math.round(
            Math.max(minSize, Math.min(maxAvailable, current + delta))
        );

        setSize(nextSize);
        if (storageKey) {
            saveLayoutPreference(storageKey, nextSize);
        }
    }, [direction, minSize, maxSize, step, storageKey]);

    const resetSize = useCallback(() => {
        setSize(defaultSize);
        if (storageKey) {
            saveLayoutPreference(storageKey, defaultSize);
        }
    }, [defaultSize, storageKey]);

    return {
        size,
        isResizing,
        onPointerDown,
        onKeyDown,
        resetSize,
        minSize,
        maxSize,
    };
};
