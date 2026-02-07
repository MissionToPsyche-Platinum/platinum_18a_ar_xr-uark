# NASA Psyche AR Mission

## 🚀 Welcome to the Project!

This is an Augmented Reality (AR) web application designed to simulate the NASA Psyche mission. It allows users to drive a virtual rover on a physical surface using their phone's camera.

### 🧠 The Core Concept
This project uses a "hybrid" approach to give you the best of both worlds:
1.  **The Brain (Rust)**: We use the **Rust** programming language for the heavy lifting, like physics and pathfinding algorithms. It's extremely fast and safe.
2.  **The Face (React/TypeScript)**: We use **React** (a standard web framework) to build the user interface (buttons, menus) and handle the 3D rendering with **A-Frame**.

These two parts talk to each other using **WebAssembly (WASM)**, which lets the browser run the Rust code near-native speeds.

---

## 🛠️ Prerequisites (What you need to install first)

Before you start, you need to install a few tools on your computer.

### 1. Node.js (for the website)
*   **What is it?**: The runtime that lets you run JavaScript outside the browser. It includes `npm` (Node Package Manager) to install web libraries.
*   **How to install**: Download the "LTS" version from [nodejs.org](https://nodejs.org/).

### 2. Rust (for the engine)
*   **What is it?**: The systems programming language we use for high-performance logic.
*   **How to install**: 
    - **Windows**: Download `rustup-init.exe` from [rustup.rs](https://rustup.rs/).
    - **Mac/Linux**: Open a terminal and run `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

### 3. wasm-pack (the bridge)
*   **What is it?**: A tool that compiles your Rust code into a format the web browser can understand (WebAssembly).
*   **How to install**: Open your terminal/command prompt and run:
    ```bash
    cargo install wasm-pack
    ```

---

## ⚙️ Setting Up the Project

1.  **Open your terminal** (Command Prompt, PowerShell, or Terminal).
2.  **Navigate to the project folder**:
    ```bash
    cd path/to/nasa-psyche-ar
    ```
3.  **Install Web Dependencies**:
    This downloads all the JavaScript libraries React needs.
    ```bash
    npm install
    ```
    *(Note: This might take a minute. If you see warnings, that's usually okay.)*

---

## ▶️ Running the App

1.  **Start the Development Server**:
    This command compiles the Rust code *and* starts the local web server.
    ```bash
    npm run dev
    ```
2.  **Open in Browser**:
    You will see a URL like `http://localhost:5173/`. Ctrl+Click it (or copy-paste it) into your web browser.

### 📱 Testing AR (Important!)
*   **Desktop Mode**: Click "Desktop Test Mode" on the screen. You can use your mouse to simulate clicking buttons.
*   **Mobile Mode**: To see the AR in action, you need to open the app on your phone.
    *   Ensure your phone and computer are on the **same Wi-Fi**.
    *   Find your computer's local IP address (e.g., `192.168.1.5`).
    *   On your phone, go to `http://YOUR_COMPUTER_IP:5173`.
    *   *Note: AR features often require HTTPS. For local testing, you might need to enable specific browser flags or use a tunneling tool.*

---

## 📂 Project Structure (Where things live)

*   **`src/`**: The **Frontend**.
    *   `App.tsx`: The main controller. It decides if you are in the Menu or Playing the game.
    *   `index.css`: The styling (colors, fonts).
*   **`rust_engine/`**: The **Backend Logic**.
    *   `src/lib.rs`: The main entry point for the Rust code. This is where the heavy calculations happen.
*   **`rust_engine/pkg/`**: The **Bridge**.
    *   This folder is *automatically generated* when you build. It limits the compiled Rust code so the website can use it. **Do not edit files here manually.**

---

## 🐛 Troubleshooting

*   **"wasm-pack not found"**: Make sure you ran `cargo install wasm-pack` and restarted your terminal.
*   **"WebXR not supported"**: You are likely trying to run AR mode on a desktop browser. Use "Desktop Test Mode" or open it on a mobile device.
