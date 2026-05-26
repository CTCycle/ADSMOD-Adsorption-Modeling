import { MachineLearningPage } from './pages/MachineLearningPage';
import './index.css';

function App() {
    return (
        <div className="app-container">
            <header className="app-header">
                <div className="header-content">
                    <div className="header-brand">
                        <img className="brand-logo" src="/favicon.png" alt="ADSMOD logo" />
                        <h1 className="brand-wordmark">ADSMOD ML</h1>
                    </div>
                </div>
            </header>
            <main className="app-main">
                <MachineLearningPage />
            </main>
        </div>
    );
}

export default App;
