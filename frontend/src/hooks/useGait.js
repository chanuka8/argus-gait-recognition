import { useContext } from 'react';
import { GaitContext } from '../contexts/gaitContextDef';

export const useGait = () => {
  const context = useContext(GaitContext);
  if (!context) {
    throw new Error('useGait must be used within a GaitProvider');
  }
  return context;
};

export default useGait;
