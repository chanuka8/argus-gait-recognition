import { useContext } from 'react';
import { AuthContext } from '../contexts/authContextDef';

export const useAuth = () => {
    return useContext(AuthContext);
};

export default useAuth;
