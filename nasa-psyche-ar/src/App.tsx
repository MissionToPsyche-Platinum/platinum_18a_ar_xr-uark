import { useEffect, useState } from 'react';
// @ts-ignore
import init, { start_ar_session } from '../rust_engine/pkg/rust_engine';

const App = () => {
    const [gameState, setGameState] = useState('MENU'); // MENU, PLAYING
    const [score, setScore] = useState(0);
    const [testMode, setTestMode] = useState(false);
    const [scanPrompt, setScanPrompt] = useState(true);

    // Initialize Rust WASM on mount
    useEffect(() => {
        init().catch(console.error);
    }, []);

    const handleStart = async (mode: string) => {
        if (mode === 'desktop') {
            console.log("Starting DESKTOP TEST MODE");
            setTestMode(true);
            setGameState('PLAYING');
            // In a real optimized React app we would use state to toggle the A-Frame attributes,
            // but simpler to let the scene mount then modify it or use effects.
        } else {
            // Standard AR
            console.log("Starting AR MODE");
            setTestMode(false);
            try {
                await start_ar_session(mode);
                setGameState('PLAYING');
            } catch (e) {
                console.error("Failed to start AR session", e);
                // Fallback to playing state anyway for now so UI shows
                setGameState('PLAYING');
            }
        }
    };

    const moveCar = (direction: string) => {
        if (gameState !== 'PLAYING') return;

        const car = document.getElementById('car') as any; // Using direct DOM for A-Frame speed
        if (!car) return;

        const currentPos = car.getAttribute('position');
        const speed = 0.08;

        switch (direction) {
            case 'forward':
                currentPos.z -= speed;
                break;
            case 'left':
                currentPos.x -= speed;
                break;
            case 'right':
                currentPos.x += speed;
                break;
        }

        // Keep car on the road
        currentPos.x = Math.max(-0.2, Math.min(0.2, currentPos.x));
        currentPos.z = Math.max(-1.4, Math.min(0.7, currentPos.z));

        car.setAttribute('position', currentPos);
        checkCoinCollection(currentPos);
    };

    const checkCoinCollection = (carPos: any) => {
        ['coin1', 'coin2', 'coin3'].forEach(id => {
            const coin = document.getElementById(id) as any;
            if (!coin || coin.getAttribute('visible') === false) return;

            const coinPos = coin.getAttribute('position');
            const distance = Math.sqrt(
                Math.pow(carPos.x - coinPos.x, 2) +
                Math.pow(carPos.z - coinPos.z, 2)
            );

            if (distance < 0.15) {
                console.log("COIN COLLECTED!");
                coin.setAttribute('visible', false);
                setScore(s => s + 10);
            }
        });
    };

    // Setup event listeners for AR target when playing
    useEffect(() => {
        if (gameState === 'PLAYING') {
            const arTarget = document.getElementById('ar-target');
            if (arTarget) {
                arTarget.addEventListener('targetFound', () => {
                    console.log("AR Marker found!");
                    setScanPrompt(false);
                });
                arTarget.addEventListener('targetLost', () => {
                    console.log("AR Marker lost");
                    setScanPrompt(true);
                });

                if (testMode) {
                    arTarget.removeAttribute('mindar-image-target');
                    arTarget.setAttribute('position', '0 -0.5 -1.5');
                    arTarget.setAttribute('rotation', '-30 0 0');
                    setScanPrompt(false);
                }
            }
        }
    }, [gameState, testMode]);

    return (
        <div className="ar-container">
            {gameState === 'MENU' && (
                <div id="start-screen">
                    <h1>Psyche Mission</h1>
                    <p style={{ fontSize: '1.2rem' }}>Drive and collect resources!</p>
                    <button id="start-button" onClick={() => handleStart('story')}>Start AR Mode</button>
                    <button id="desktop-button" style={{ background: '#4CAF50', marginTop: '10px' }} onClick={() => handleStart('desktop')}>
                        Desktop Test Mode
                    </button>
                </div>
            )}

            {gameState === 'PLAYING' && (
                <>
                    {/* A-Frame Scene */}
                    <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: 0 }}>
                        {/* 
                 Note: We behave dangerously with A-Frame in React. 
                 Ideally use a wrapper, but for this port we render web components directly.
                 Using 'mindar-image' attribute requires strictly defined props types or ignore.
             */}
                        <a-scene
                            mindar-image="imageTargetSrc: https://cdn.jsdelivr.net/gh/hiukim/mind-ar-js@1.2.5/examples/image-tracking/assets/card-example/card.mind;"
                            color-space="sRGB"
                            renderer="colorManagement: true"
                            vr-mode-ui="enabled: false"
                            device-orientation-permission-ui="enabled: false"
                        >
                            <a-camera position="0 0 0" look-controls="enabled: false"></a-camera>

                            <a-entity id="ar-target" mindar-image-target="targetIndex: 0">
                                <a-plane position="0 0 0" rotation="0 0 0" width="1.5" height="2" color="#228B22"></a-plane>
                                <a-plane position="0 0.001 0" rotation="0 0 0" width="0.5" height="2" color="#333"></a-plane>

                                <a-plane position="0 0.002 -0.3" rotation="0 0 0" width="0.08" height="0.2" color="#FFFF00"></a-plane>
                                <a-plane position="0 0.002 -0.9" rotation="0 0 0" width="0.08" height="0.2" color="#FFFF00"></a-plane>

                                {/* Car */}
                                <a-entity id="car" position="0 0.08 0.6">
                                    <a-box position="0 0 0" width="0.15" height="0.08" depth="0.22" color="#FF0000"></a-box>
                                    <a-box position="0 0.06 -0.02" width="0.13" height="0.05" depth="0.1" color="#CC0000"></a-box>
                                    <a-box position="0 0.06 -0.08" width="0.13" height="0.04" depth="0.02" color="#87CEEB" opacity="0.7"></a-box>
                                    <a-cylinder position="-0.08 -0.04 0.08" radius="0.025" height="0.015" rotation="0 0 90" color="#000"></a-cylinder>
                                    <a-cylinder position="0.08 -0.04 0.08" radius="0.025" height="0.015" rotation="0 0 90" color="#000"></a-cylinder>
                                    <a-cylinder position="-0.08 -0.04 -0.08" radius="0.025" height="0.015" rotation="0 0 90" color="#000"></a-cylinder>
                                    <a-cylinder position="0.08 -0.04 -0.08" radius="0.025" height="0.015" rotation="0 0 90" color="#000"></a-cylinder>
                                </a-entity>

                                {/* Collectibles */}
                                <a-cylinder id="coin1" position="0.12 0.08 -0.3" radius="0.07" height="0.02" color="#FFD700"
                                    animation="property: rotation; to: 0 360 0; loop: true; dur: 2000"></a-cylinder>
                                <a-cylinder id="coin2" position="-0.12 0.08 -0.8" radius="0.07" height="0.02" color="#FFD700"
                                    animation="property: rotation; to: 0 360 0; loop: true; dur: 2000"></a-cylinder>
                                <a-cylinder id="coin3" position="0.08 0.08 -1.3" radius="0.07" height="0.02" color="#FFD700"
                                    animation="property: rotation; to: 0 360 0; loop: true; dur: 2000"></a-cylinder>
                            </a-entity>
                        </a-scene>
                    </div>

                    <div id="ui-overlay" style={{ display: 'block' }}>
                        {scanPrompt && !testMode && (
                            <div id="scan-prompt">
                                Point camera at marker
                            </div>
                        )}

                        <div id="score-display">
                            Score: <span id="score">{score}</span>
                        </div>

                        <div id="controls">
                            <div>
                                <div className="control-btn" onClick={() => moveCar('forward')}>▲</div>
                            </div>
                            <div>
                                <div className="control-btn" onClick={() => moveCar('left')}>◀</div>
                                <div className="control-btn" onClick={() => moveCar('right')}>▶</div>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

export default App;
