/**
 * Reduces AR.js marker jitter: rolling position average + spike damping +
 * two-stage quaternion smoothing (content lives here, not under <a-marker>).
 */
function registerArSmoothAnchor(): void {
    const AFRAME = (window as unknown as { AFRAME?: any }).AFRAME;
    if (!AFRAME?.registerComponent) return;
    if (AFRAME.components['ar-smooth-anchor']) return;

    const THREE = AFRAME.THREE;

    AFRAME.registerComponent('ar-smooth-anchor', {
        schema: {
            target: { type: 'selector' },
            positionLerp: { default: 0.16 },
            rotationLerp: { default: 0.2 },
            scaleLerp: { default: 0.18 },
            /** If raw pose jumps farther than this (world units), extra damping is applied. */
            maxJump: { default: 0.1 },
            /**
             * Rolling average of this many raw positions before the main lerp (1 = off).
             * Higher = steadier, more lag when moving the device (try 6–10).
             */
            positionBlendFrames: { default: 8 },
            /** Same idea for scale (1 = off). */
            scaleBlendFrames: { default: 4 },
        },
        init(this: any) {
            this._pos = new THREE.Vector3();
            this._quat = new THREE.Quaternion();
            this._quatPre = new THREE.Quaternion();
            this._scale = new THREE.Vector3();
            this._rawPos = new THREE.Vector3();
            this._rawQuat = new THREE.Quaternion();
            this._rawScale = new THREE.Vector3();
            this._prevRawPos = new THREE.Vector3();
            this._prevRawQuat = new THREE.Quaternion();
            this._hasPrevRaw = false;
            this._initialized = false;
            this._avgPos = new THREE.Vector3();
            this._avgScale = new THREE.Vector3();
            this._posRing = [];
            this._scaleRing = [];
            this._posRingI = 0;
            this._scaleRingI = 0;
            this._posRingFill = 0;
            this._scaleRingFill = 0;
            this._nfPos = 0;
            this._nfScale = 0;
        },
        tick(this: any) {
            const el = this.data.target as { object3D?: any } | null;
            if (!el?.object3D) return;
            const src = el.object3D;
            if (!src.visible) {
                this.el.object3D.visible = false;
                this._initialized = false;
                this._hasPrevRaw = false;
                this._posRingFill = 0;
                this._scaleRingFill = 0;
                this._posRingI = 0;
                this._scaleRingI = 0;
                return;
            }
            this.el.object3D.visible = true;
            src.updateMatrixWorld(true);
            const mat = src.matrixWorld;
            mat.decompose(this._rawPos, this._rawQuat, this._rawScale);

            let pl = this.data.positionLerp;
            let rl = this.data.rotationLerp;
            let sl = this.data.scaleLerp;
            const maxJ = this.data.maxJump;

            if (this._hasPrevRaw && this._initialized && maxJ > 0) {
                const d = this._rawPos.distanceTo(this._prevRawPos);
                if (d > maxJ) {
                    pl = Math.min(pl, 0.022);
                    rl = Math.min(rl, 0.028);
                    sl = Math.min(sl, 0.03);
                }
                const dot = Math.min(1, Math.abs(this._prevRawQuat.dot(this._rawQuat)));
                const ang = 2 * Math.acos(dot);
                if (ang > 0.28) {
                    rl = Math.min(rl, 0.032);
                }
            }
            this._prevRawPos.copy(this._rawPos);
            this._prevRawQuat.copy(this._rawQuat);
            this._hasPrevRaw = true;

            const nfPos = Math.min(12, Math.max(1, Math.floor(Number(this.data.positionBlendFrames)) || 1));
            const nfScale = Math.min(8, Math.max(1, Math.floor(Number(this.data.scaleBlendFrames)) || 1));

            if (this._nfPos !== nfPos) {
                this._nfPos = nfPos;
                this._posRing = Array.from({ length: nfPos }, () => new THREE.Vector3());
                this._posRingI = 0;
                this._posRingFill = 0;
            }
            if (this._nfScale !== nfScale) {
                this._nfScale = nfScale;
                this._scaleRing = Array.from({ length: nfScale }, () => new THREE.Vector3());
                this._scaleRingI = 0;
                this._scaleRingFill = 0;
            }

            let posTarget = this._rawPos;
            if (nfPos > 1) {
                this._posRing[this._posRingI].copy(this._rawPos);
                this._posRingI = (this._posRingI + 1) % nfPos;
                this._posRingFill = Math.min(this._posRingFill + 1, nfPos);
                this._avgPos.set(0, 0, 0);
                for (let i = 0; i < this._posRingFill; i++) {
                    this._avgPos.add(this._posRing[i]);
                }
                this._avgPos.multiplyScalar(1 / this._posRingFill);
                posTarget = this._avgPos;
            }

            let scaleTarget = this._rawScale;
            if (nfScale > 1) {
                this._scaleRing[this._scaleRingI].copy(this._rawScale);
                this._scaleRingI = (this._scaleRingI + 1) % nfScale;
                this._scaleRingFill = Math.min(this._scaleRingFill + 1, nfScale);
                this._avgScale.set(0, 0, 0);
                for (let i = 0; i < this._scaleRingFill; i++) {
                    this._avgScale.add(this._scaleRing[i]);
                }
                this._avgScale.multiplyScalar(1 / this._scaleRingFill);
                scaleTarget = this._avgScale;
            }

            if (!this._initialized) {
                this._pos.copy(posTarget);
                this._quatPre.copy(this._rawQuat);
                this._quat.copy(this._rawQuat);
                this._scale.copy(scaleTarget);
                this._initialized = true;
            } else {
                this._pos.lerp(posTarget, pl);
                /* Extra rotation low-pass before display slerp (kills high-frequency tilt/spin noise). */
                this._quatPre.slerp(this._rawQuat, 0.14);
                this._quat.slerp(this._quatPre, rl);
                this._scale.lerp(scaleTarget, sl);
            }

            this.el.object3D.position.copy(this._pos);
            this.el.object3D.quaternion.copy(this._quat);
            this.el.object3D.scale.copy(this._scale);
        },
    });
}

registerArSmoothAnchor();
