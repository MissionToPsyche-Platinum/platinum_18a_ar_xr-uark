export type Difficulty = 'easy' | 'normal' | 'hard';

export interface ModeFeatures {
  spawnSamples: number;
  spawnObstacles: number;
  energyEnabled: boolean;
  energyDrainPerSec: number; // if energyEnabled
  samplePoints: number;           // points awarded per sample collected
  obstaclePenalty: number;        // points deducted per obstacle entry
  energyBonusEnabled: boolean;    // whether leftover energy gives bonus points at end
}

export const MODE_CONFIG: Record<Difficulty, ModeFeatures> = {
  easy:   { spawnSamples: 20, spawnObstacles: 1, energyEnabled: false, energyDrainPerSec: 0,     samplePoints: 100, obstaclePenalty: 0,  energyBonusEnabled: false },
  normal: { spawnSamples: 20, spawnObstacles: 2, energyEnabled: true,  energyDrainPerSec: 0.833, samplePoints: 100, obstaclePenalty: 0,  energyBonusEnabled: true  },
  hard:   { spawnSamples: 20, spawnObstacles: 4, energyEnabled: true,  energyDrainPerSec: 0.833, samplePoints: 200, obstaclePenalty: 100, energyBonusEnabled: true  },
};

export default MODE_CONFIG;
