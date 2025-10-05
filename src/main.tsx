import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './home';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);