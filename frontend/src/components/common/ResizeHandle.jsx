import React from 'react';
import './ResizeHandle.css';

/**
 * Accessible resize separator handle.
 *
 * @param {Object} props
 * @param {boolean} props.isResizing Whether resizing is currently active
 * @param {(e: React.PointerEvent) => void} props.onPointerDown Pointer down handler
 * @param {(e: React.KeyboardEvent) => void} props.onKeyDown Keyboard handler
 * @param {() => void} [props.onDoubleClick] Double click to reset
 * @param {number} props.currentSize Current panel size
 * @param {number} props.minSize Min size bound
 * @param {number} props.maxSize Max size bound
 * @param {string} [props.label="Resize panel"] Accessible label
 * @param {string} [props.className=""] Additional CSS classes
 */
export const ResizeHandle = ({
    isResizing = false,
    onPointerDown,
    onKeyDown,
    onDoubleClick,
    currentSize,
    minSize,
    maxSize,
    label = 'Resize panel',
    className = '',
}) => {
    return (
        <div
            className={`argus-resize-handle ${isResizing ? 'resizing' : ''} ${className}`}
            onPointerDown={onPointerDown}
            onKeyDown={onKeyDown}
            onDoubleClick={onDoubleClick}
            role="separator"
            aria-orientation="vertical"
            aria-valuenow={currentSize}
            aria-valuemin={minSize}
            aria-valuemax={maxSize}
            aria-label={label}
            tabIndex={0}
            title={`${label} (Drag or use Left/Right arrow keys. Double-click to reset)`}
        >
            <div className="argus-resize-handle-bar">
                <span className="resize-handle-dot" />
                <span className="resize-handle-dot" />
                <span className="resize-handle-dot" />
            </div>
        </div>
    );
};

export default ResizeHandle;
