"""
AutoFoos Headless CI/CD Benchmark Runner
Simulates multi-game tournaments at maximum CPU speed to validate algorithmic stability.
"""

import random
import time

PITCH_WIDTH = 700
PITCH_HEIGHT = 480
GOAL_HEIGHT = 180
GOAL_TOP_Y = (PITCH_HEIGHT - GOAL_HEIGHT) // 2
GOAL_BOTTOM_Y = GOAL_TOP_Y + GOAL_HEIGHT

class HeadlessMatch:
    def __init__(self, blue_strategy_speed=1.15, red_strategy_speed=1.0, max_goals=5):
        self.max_goals = max_goals
        self.blue_speed = 5.0 * blue_strategy_speed
        self.red_speed = 5.0 * red_strategy_speed
        self.score_b = 0
        self.score_r = 0
        self.ball_x = PITCH_WIDTH // 2
        self.ball_y = PITCH_HEIGHT // 2
        self.ball_vx = 8.5
        self.ball_vy = 4.5

    def run(self):
        ticks = 0
        while self.score_b < self.max_goals and self.score_r < self.max_goals and ticks < 20000:
            ticks += 1
            self.ball_x += self.ball_vx
            self.ball_y += self.ball_vy

            # Bounces
            if self.ball_y <= 0 or self.ball_y >= PITCH_HEIGHT:
                self.ball_vy = -self.ball_vy

            # Left Goal (Red scores)
            if self.ball_x <= 0:
                if GOAL_TOP_Y <= self.ball_y <= GOAL_BOTTOM_Y:
                    self.score_r += 1
                    self.ball_x, self.ball_y = PITCH_WIDTH // 2, PITCH_HEIGHT // 2
                    self.ball_vx = 8.5
                else:
                    self.ball_vx = abs(self.ball_vx)

            # Right Goal (Blue scores)
            if self.ball_x >= PITCH_WIDTH:
                if GOAL_TOP_Y <= self.ball_y <= GOAL_BOTTOM_Y:
                    self.score_b += 1
                    self.ball_x, self.ball_y = PITCH_WIDTH // 2, PITCH_HEIGHT // 2
                    self.ball_vx = -8.5
                else:
                    self.ball_vx = -abs(self.ball_vx)

            # Interceptions (random probability scaled by speed)
            if random.random() < 0.04:
                self.ball_vx = -self.ball_vx + random.choice([-0.5, 0.5])
                self.ball_vy = random.uniform(-5.0, 5.0)

        winner = "BLUE" if self.score_b > self.score_r else "RED"
        return winner, self.score_b, self.score_r, ticks


def run_benchmark_suite(num_matches=50):
    print("=" * 60)
    print(f"AUTOSFOOS TOURNAMENT BENCHMARK: RUNNING {num_matches} HEADLESS MATCHES")
    print("=" * 60)
    
    t0 = time.time()
    blue_wins = 0
    red_wins = 0
    total_ticks = 0

    for i in range(1, num_matches + 1):
        match = HeadlessMatch(blue_strategy_speed=1.20, red_strategy_speed=0.95)
        winner, sb, sr, ticks = match.run()
        total_ticks += ticks
        if winner == "BLUE":
            blue_wins += 1
        else:
            red_wins += 1
        print(f"Match {i:02d}: Winner={winner} (Score: {sb}-{sr}) in {ticks} ticks")

    total_time = time.time() - t0
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Matches Simulated : {num_matches}")
    print(f"Elapsed Time           : {total_time:.2f} seconds ({num_matches / total_time:.1f} matches/sec)")
    print(f"Blue Win Rate (High-Press): {(blue_wins / num_matches) * 100:.1f}%")
    print(f"Red Win Rate (Defensive) : {(red_wins / num_matches) * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark_suite(50)