// Simple AR Driving Game
let score = 0;
let isPlaying = false;

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded");

    const startScreen = document.getElementById('start-screen');
    const startButton = document.getElementById('start-button');
    const desktopButton = document.getElementById('desktop-button');
    const uiOverlay = document.getElementById('ui-overlay');
    const scoreDisplay = document.getElementById('score');
    const scanPrompt = document.getElementById('scan-prompt');
    const arTarget = document.getElementById('ar-target');

    // Wait for AR scene to load
    const scene = document.querySelector('a-scene');
    scene.addEventListener('loaded', () => {
        console.log("A-Frame scene loaded");
    });

    // Desktop Test Mode - NO AR NEEDED FOR TESTING
    desktopButton.addEventListener('click', () => {
        console.log("Starting DESKTOP TEST MODE");
        startScreen.style.display = 'none';
        uiOverlay.style.display = 'block';
        isPlaying = true;

        // Disable AR tracking for desktop mode
        arTarget.removeAttribute('mindar-image-target');

        // Position content in front of camera (no marker needed)
        arTarget.setAttribute('position', '0 -0.5 -1.5');
        arTarget.setAttribute('rotation', '-30 0 0');

        console.log("Desktop mode - Test controls with mouse clicks!");
    });

    // Start AR Mode (requires mobile + marker)
    startButton.addEventListener('click', () => {
        console.log("Starting AR MODE");
        startScreen.style.display = 'none';
        uiOverlay.style.display = 'block';
        isPlaying = true;
        scanPrompt.style.display = 'none';
    });

    // Car Controls
    const controlButtons = document.querySelectorAll('.control-btn');
    controlButtons.forEach(btn => {
        // For touch devices
        btn.addEventListener('touchend', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log(" Touch: " + btn.dataset.dir);
            moveCar(btn.dataset.dir);
        });

        // For desktop testing
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            console.log("Click: " + btn.dataset.dir);
            moveCar(btn.dataset.dir);
        });
    });

    function moveCar(direction) {
        console.log("Moving car: " + direction);

        if (!isPlaying) {
            console.log("Game not playing yet");
            return;
        }

        const car = document.getElementById('car');
        if (!car) {
            console.log("Car not found!");
            return;
        }

        const currentPos = car.getAttribute('position');
        const speed = 0.08;

        console.log("Current position:", currentPos);

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
        console.log("New position:", currentPos);

        // Check coin collection
        checkCoinCollection(currentPos);
    }

    function checkCoinCollection(carPos) {
        const coins = [
            document.getElementById('coin1'),
            document.getElementById('coin2'),
            document.getElementById('coin3')
        ];

        coins.forEach((coin) => {
            if (!coin || coin.getAttribute('visible') === false) return;

            const coinPos = coin.getAttribute('position');
            const distance = Math.sqrt(
                Math.pow(carPos.x - coinPos.x, 2) +
                Math.pow(carPos.z - coinPos.z, 2)
            );

            if (distance < 0.15) {
                console.log("COIN COLLECTED! +10 points");
                coin.setAttribute('visible', false);
                score += 10;
                scoreDisplay.textContent = score;
            }
        });
    }

    // AR Target tracking
    if (arTarget) {
        arTarget.addEventListener('targetFound', () => {
            console.log("AR Marker found!");
            if (isPlaying) {
                scanPrompt.style.display = 'none';
            }
        });

        arTarget.addEventListener('targetLost', () => {
            console.log("AR Marker lost");
        });
    }
});
