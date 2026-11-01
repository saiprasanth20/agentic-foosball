import pygame
import sys
import threading
import time
import json
import os
import random
import math
import numpy as np
from datetime import datetime

# 1. Initialize Pygame & Audio Mixer
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# 2. Check for Gemini API Key
api_key = os.environ.get("GEMINI_API_KEY")
HAVE_GEMINI = False
client = None

if api_key:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        HAVE_GEMINI = True
        print("[SUCCESS] Gemini API Key connected! Live LLM coach active.")
    except Exception as e:
        print(f"[WARNING] Gemini init failed: {e}")
        HAVE_GEMINI = False
else:
    print("[INFO] No GEMINI_API_KEY found. Running with local heuristic coach.")

# 3. Dimensions & Display Config
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 730
FPS = 60

# Tournament Config
POINTS_TO_WIN = 10

# Enhanced Palette
PITCH_GREEN_LIGHT = (42, 155, 48)
PITCH_GREEN_DARK  = (30, 120, 36)
BORDER_WOOD_OUTER = (78, 48, 22)
BORDER_WOOD_INNER = (52, 32, 14)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BALL_BASE = (250, 250, 252)
BALL_SHADOW = (18, 40, 20, 90)

STEEL_BAR_HIGH = (220, 225, 235)
STEEL_BAR_MID  = (160, 165, 175)
STEEL_BAR_DARK = (90, 95, 105)

BLUE_PRIMARY = (35, 115, 235)
BLUE_DARK    = (15, 65, 160)
BLUE_LIGHT   = (90, 165, 255)

RED_PRIMARY  = (230, 45, 45)
RED_DARK     = (150, 20, 20)
RED_LIGHT    = (255, 105, 105)

TARGET_DOT = (255, 230, 0)
AIM_LASER = (0, 255, 220)
PASS_LASER = (60, 255, 80)
HUD_GLASS = (12, 16, 22, 230)
HIGHLIGHT_TEAL = (0, 255, 200)

# Pitch Dimensions
PITCH_X = 150
PITCH_Y = 75
PITCH_WIDTH = 700
PITCH_HEIGHT = 480

GOAL_HEIGHT = 180
GOAL_TOP_Y = PITCH_Y + (PITCH_HEIGHT - GOAL_HEIGHT) // 2
GOAL_BOTTOM_Y = GOAL_TOP_Y + GOAL_HEIGHT

FIGURE_WIDTH = 14
FIGURE_HEIGHT = 24

# Physics & Magnus Constants
BALL_FRICTION = 0.993
MIN_BALL_SPEED = 2.0
ROD_ACCELERATION = 0.16
SPIN_DECAY = 0.985
MAGNUS_COEFF = 0.015


# -------------------------------------------------------------
# AUDIO SYNTHESIS ENGINE
# -------------------------------------------------------------
class SoundSynth:
    @staticmethod
    def _create_sound(freq_start, freq_end, duration_s, decay=True, noise=0.0):
        sr = 22050
        n_samples = int(sr * duration_s)
        time_arr = np.linspace(0, duration_s, n_samples, False)
        freqs = np.linspace(freq_start, freq_end, n_samples)
        phase = 2 * np.pi * np.cumsum(freqs) / sr
        wave = np.sin(phase)

        if noise > 0:
            wave = (1 - noise) * wave + noise * np.random.uniform(-1, 1, n_samples)
        if decay:
            envelope = np.exp(-5.0 * time_arr / duration_s)
            wave = wave * envelope

        wave = np.clip(wave * 32767, -32767, 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        return pygame.sndarray.make_sound(stereo_wave)

    def __init__(self):
        try:
            self.kick_sound = self._create_sound(240, 80, 0.08, noise=0.25)
            self.wall_sound = self._create_sound(160, 90, 0.05, noise=0.35)
            self.trap_sound = self._create_sound(350, 200, 0.06, noise=0.1)
            self.goal_sound = self._create_sound(450, 180, 0.45, noise=0.4)
            self.match_sound = self._create_sound(600, 300, 0.70, noise=0.2)
            self.enabled = True
        except Exception:
            self.enabled = False

    def play_kick(self):
        if self.enabled: self.kick_sound.play()

    def play_wall(self):
        if self.enabled: self.wall_sound.play()

    def play_trap(self):
        if self.enabled: self.trap_sound.play()

    def play_goal(self):
        if self.enabled: self.goal_sound.play()

    def play_match_win(self):
        if self.enabled: self.match_sound.play()


audio = SoundSynth()


# -------------------------------------------------------------
# TOURNAMENT TELEMETRY TRACKER
# -------------------------------------------------------------
class TournamentTracker:
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()
        self.matches_played = 0
        self.series_wins_blue = 0
        self.series_wins_red = 0

        self.match_start = time.time()
        self.current_match_number = 1
        self.completed_matches = []

        self.shots_blue = 0
        self.shots_red = 0
        self.llm_inferences = []

    def record_shot(self, team):
        if team == "BLUE": self.shots_blue += 1
        else: self.shots_red += 1

    def record_llm_call(self, team, strategy, latency_ms):
        self.llm_inferences.append({
            "match": self.current_match_number,
            "team": team,
            "strategy": strategy,
            "latency_ms": round(latency_ms, 2)
        })

    def complete_match(self, winner, score_b, score_r, scouting_summary):
        self.matches_played += 1
        if winner == "BLUE":
            self.series_wins_blue += 1
        else:
            self.series_wins_red += 1

        match_record = {
            "match_number": self.current_match_number,
            "winner": winner,
            "final_score": f"{score_b}-{score_r}",
            "duration_seconds": round(time.time() - self.match_start, 2),
            "shots": {"blue": self.shots_blue, "red": self.shots_red},
            "scouting_summary": scouting_summary
        }
        self.completed_matches.append(match_record)

        with open("scouting_report.json", "w") as f:
            json.dump({
                "last_winner": winner,
                "matches_played": self.matches_played,
                "blue_wins": self.series_wins_blue,
                "red_wins": self.series_wins_red,
                "prior_scouting": scouting_summary
            }, f, indent=2)

        self.current_match_number += 1
        self.match_start = time.time()
        self.shots_blue = 0
        self.shots_red = 0

    def export(self):
        total_time = round(time.time() - self.session_start, 2)
        data = {
            "session_id": self.session_id,
            "total_duration_seconds": total_time,
            "tournament_summary": {
                "matches_played": self.matches_played,
                "blue_wins": self.series_wins_blue,
                "red_wins": self.series_wins_red,
                "blue_win_rate_pct": round((self.series_wins_blue / max(1, self.matches_played)) * 100, 1),
                "red_win_rate_pct": round((self.series_wins_red / max(1, self.matches_played)) * 100, 1)
            },
            "completed_matches": self.completed_matches,
            "llm_inferences_total": len(self.llm_inferences),
            "llm_audit_log": self.llm_inferences
        }
        filename = f"tournament_telemetry_{self.session_id}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[TELEMETRY EXPORTED] Tournament record saved to {filename}")


tournament = TournamentTracker()


# -------------------------------------------------------------
# SHOT ANALYTICS & STRUCTURED LLM COACH WITH CIRCUIT BREAKER
# -------------------------------------------------------------
class ShotAnalytics:
    def __init__(self):
        self.red_shots_top = 0
        self.red_shots_bottom = 0
        self.blue_shots_top = 0
        self.blue_shots_bottom = 0

    def record_shot_zone(self, shooter_team, target_y):
        mid = (GOAL_TOP_Y + GOAL_BOTTOM_Y) / 2
        if shooter_team == "RED":
            if target_y < mid: self.red_shots_top += 1
            else: self.red_shots_bottom += 1
        else:
            if target_y < mid: self.blue_shots_top += 1
            else: self.blue_shots_bottom += 1

    def get_summary_for_coach(self, defending_team):
        if defending_team == "BLUE":
            total = self.red_shots_top + self.red_shots_bottom
            if total == 0: return "Opponent shot patterns neutral."
            pct_top = int((self.red_shots_top / total) * 100)
            return f"Opponent targets: {pct_top}% TOP corner, {100-pct_top}% BOTTOM corner."
        else:
            total = self.blue_shots_top + self.blue_shots_bottom
            if total == 0: return "Opponent shot patterns neutral."
            pct_top = int((self.blue_shots_top / total) * 100)
            return f"Opponent targets: {pct_top}% TOP corner, {100-pct_top}% BOTTOM corner."


shot_analytics = ShotAnalytics()


class TacticalCoach:
    api_circuit_open = False

    def __init__(self, team_name):
        self.team = team_name
        self.active_tactic = "BALANCED"
        self.tactic_directive = "Maintain zone coverage."
        self.speed_multiplier = 1.0
        self.goalie_bias = 0.0
        self.shot_preference = "DYNAMIC"
        self.is_thinking = False
        self.last_call_time = 0

    def trigger_strategy_eval(self, score_self, score_opp):
        current_now = time.time()
        if self.is_thinking or (current_now - self.last_call_time < 15.0):
            return

        self.last_call_time = current_now
        thread = threading.Thread(target=self._run_coach_llm, args=(score_self, score_opp), daemon=True)
        thread.start()

    def _run_coach_llm(self, score_self, score_opp):
        self.is_thinking = True
        t0 = time.time()
        pattern_text = shot_analytics.get_summary_for_coach(self.team)

        prior_context = ""
        if os.path.exists("scouting_report.json"):
            try:
                with open("scouting_report.json", "r") as f:
                    scout_data = json.load(f)
                    prior_context = f"Historic Memory: {scout_data.get('prior_scouting', '')} "
            except Exception:
                pass

        prompt = (
            f"You are the Tactical Coach for {self.team} in an autonomous foosball championship. "
            f"Current Match: You {score_self} - Opponent {score_opp} (Race to 10 points). "
            f"{prior_context}Real-Time Telemetry: {pattern_text} "
            "Execute structured tactical parameters: 'strategy' ('HIGH_PRESS', 'DEFENSIVE_WALL', 'FAST_BREAK', 'BANK_SHOT_SNIPER'), "
            "'speed_multiplier' (0.9 to 1.25), 'goalie_bias' (-0.6 to 0.6), 'shot_preference' ('TOP', 'BOTTOM', 'BANK', 'DYNAMIC'), "
            "and 'directive' (max 6 words). Respond strictly in valid JSON."
        )

        used_live_api = False
        if HAVE_GEMINI and client and not TacticalCoach.api_circuit_open:
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )
                clean_json = response.text.strip().replace("```json", "").replace("```", "")
                data = json.loads(clean_json)
                self.active_tactic = data.get("strategy", "BALANCED")
                self.speed_multiplier = float(data.get("speed_multiplier", 1.0))
                self.goalie_bias = float(data.get("goalie_bias", 0.0))
                self.shot_preference = data.get("shot_preference", "DYNAMIC")
                self.tactic_directive = data.get("directive", "Lock open zones!")
                used_live_api = True
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    TacticalCoach.api_circuit_open = True
                    print("\n[CIRCUIT BREAKER] 429 Daily Quota Limit reached.")
                    print("[FALLBACK] Switching to local deterministic engine.\n")
                else:
                    print(f"[COACH ERROR] {err_str[:100]}")

        if not used_live_api:
            time.sleep(0.15)
            if score_self < score_opp:
                self.active_tactic = "HIGH_PRESS"
                self.speed_multiplier = 1.20
                self.goalie_bias = 0.2
                self.shot_preference = "BOTTOM"
                self.tactic_directive = "Snipe bottom corners!"
            elif score_self > score_opp:
                self.active_tactic = "DEFENSIVE_WALL"
                self.speed_multiplier = 0.95
                self.goalie_bias = -0.2
                self.shot_preference = "BANK"
                self.tactic_directive = "Lock goal corridors."
            else:
                self.active_tactic = "BANK_SHOT_SNIPER"
                self.speed_multiplier = 1.10
                self.goalie_bias = 0.0
                self.shot_preference = "DYNAMIC"
                self.tactic_directive = "Ricochet shots off walls."

        latency = (time.time() - t0) * 1000
        tournament.record_llm_call(self.team, self.active_tactic, latency)
        self.is_thinking = False


# -------------------------------------------------------------
# SCULPTED 2.5D ROD AGENT (WITH SHADOWS & ANATOMY)
# -------------------------------------------------------------
class Rod:
    def __init__(self, name, team, x, color_primary, color_dark, color_light, num_players, spacing, max_speed, direction_facing, accuracy=0.85):
        self.name = name
        self.team = team
        self.x = x
        self.color_primary = color_primary
        self.color_dark = color_dark
        self.color_light = color_light
        self.num_players = num_players
        self.spacing = spacing
        self.max_speed = max_speed
        self.current_max_speed = max_speed
        self.direction = direction_facing
        self.accuracy = accuracy

        self.state = "IDLE"
        self.is_tilted_up = False
        self.is_selected = False

        self.min_y = PITCH_Y + 10
        self.max_y = PITCH_Y + PITCH_HEIGHT - (self.num_players - 1) * self.spacing - FIGURE_HEIGHT - 10
        self.base_y = float((self.min_y + self.max_y) // 2)
        self.target_base_y = self.base_y
        self.prev_base_y = self.base_y

        self.stamina = 100.0
        self.fatigue_recovery = 0.35
        self.fatigue_drain = 0.55

        self.rotation_angle = 0.0
        self.target_rotation = 0.0

        self.trapping_frames = 0
        self.aim_target_pos = None
        self.is_passing_play = False
        self.predicted_y = None
        self.target_figure_idx = 0
        self.reaction_timer = random.randint(0, 5)
        self.shuffle_phase = random.uniform(0, 6.28)

    def apply_coach_strategy(self, multiplier):
        self.current_max_speed = self.max_speed * multiplier

    def reset_for_new_match(self):
        self.base_y = float((self.min_y + self.max_y) // 2)
        self.target_base_y = self.base_y
        self.stamina = 100.0
        self.trapping_frames = 0
        self.rotation_angle = 0.0
        self.target_rotation = 0.0
        self.aim_target_pos = None
        self.state = "IDLE"

    def trigger_kick_animation(self):
        self.rotation_angle = -45.0
        self.target_rotation = 60.0

    def update_agent(self, ball_x, ball_y, ball_vx, ball_vy, team_has_shot, coach_goalie_bias=0.0):
        self.rotation_angle += (self.target_rotation - self.rotation_angle) * 0.25
        if abs(self.rotation_angle - self.target_rotation) < 2.0 and self.target_rotation != 0.0:
            self.target_rotation = 0.0

        if team_has_shot:
            self.state = "CLEAR_LANE"
            self.is_tilted_up = True
            self.predicted_y = None
            self.aim_target_pos = None
            self.stamina = min(100.0, self.stamina + self.fatigue_recovery * 1.5)
            return

        self.is_tilted_up = False

        if self.trapping_frames > 0:
            self.trapping_frames -= 1
            self.state = "AIMING"
            self.stamina = min(100.0, self.stamina + self.fatigue_recovery)
            self.rotation_angle = -35.0
            return

        self.aim_target_pos = None
        self.is_passing_play = False
        self.reaction_timer += 1
        ball_approaching = (self.direction == 1 and ball_vx < 0 and ball_x > self.x) or \
                           (self.direction == -1 and ball_vx > 0 and ball_x < self.x)

        if ball_approaching:
            self.state = "TRACKING"
            if self.reaction_timer >= 5:
                self.reaction_timer = 0
                frames_to_reach = (self.x - ball_x) / ball_vx
                pred_y = ball_y + ball_vy * frames_to_reach

                while pred_y < PITCH_Y or pred_y > PITCH_Y + PITCH_HEIGHT:
                    if pred_y < PITCH_Y:
                        pred_y = PITCH_Y + (PITCH_Y - pred_y)
                    elif pred_y > PITCH_Y + PITCH_HEIGHT:
                        pred_y = (PITCH_Y + PITCH_HEIGHT) - (pred_y - (PITCH_Y + PITCH_HEIGHT))

                if "Goalie" in self.name:
                    pred_y += (coach_goalie_bias * 35.0)

                if random.random() > self.accuracy:
                    pred_y += random.choice([-20, 20])

                self.predicted_y = pred_y

                best_diff = 9999
                for i in range(self.num_players):
                    fig_center = self.base_y + i * self.spacing + FIGURE_HEIGHT // 2
                    diff = self.predicted_y - fig_center
                    if abs(diff) < abs(best_diff):
                        best_diff = diff
                        self.target_figure_idx = i

                self.target_base_y = self.base_y + best_diff
        else:
            self.state = "IDLE"
            self.predicted_y = None
            self.shuffle_phase += 0.05
            center_anchor = (self.min_y + self.max_y) // 2
            wiggle = math.sin(self.shuffle_phase) * 16.0
            self.target_base_y = center_anchor + wiggle

        stamina_efficiency = max(0.60, self.stamina / 100.0)
        effective_max_speed = self.current_max_speed * stamina_efficiency

        self.prev_base_y = self.base_y
        diff_y = self.target_base_y - self.base_y
        step = diff_y * ROD_ACCELERATION
        step = max(-effective_max_speed, min(step, effective_max_speed))
        self.base_y += step
        self.base_y = max(self.min_y, min(self.base_y, self.max_y))

        velocity = abs(self.base_y - self.prev_base_y)
        if velocity > 1.8:
            self.stamina = max(10.0, self.stamina - self.fatigue_drain)
        else:
            self.stamina = min(100.0, self.stamina + self.fatigue_recovery)

    def get_player_rects(self):
        if self.is_tilted_up:
            return []
        rects = []
        for i in range(self.num_players):
            fig_y = int(self.base_y) + i * self.spacing
            rects.append(pygame.Rect(self.x - FIGURE_WIDTH // 2, fig_y, FIGURE_WIDTH, FIGURE_HEIGHT))
        return rects

    def draw(self, surface, shadow_surf):
        # 1. Cast Rod Shadow onto table felt
        pygame.draw.line(shadow_surf, (15, 35, 18, 70), (self.x + 8, PITCH_Y - 10), (self.x + 8, PITCH_Y + PITCH_HEIGHT + 10), 6)

        # 2. Draw Metallic Chrome Steel Rod with highlight line
        rod_color = HIGHLIGHT_TEAL if self.is_selected else STEEL_BAR_MID
        pygame.draw.line(surface, STEEL_BAR_DARK, (self.x - 1, PITCH_Y - 20), (self.x - 1, PITCH_Y + PITCH_HEIGHT + 20), 6)
        pygame.draw.line(surface, rod_color, (self.x, PITCH_Y - 20), (self.x, PITCH_Y + PITCH_HEIGHT + 20), 4)
        pygame.draw.line(surface, STEEL_BAR_HIGH, (self.x + 1, PITCH_Y - 20), (self.x + 1, PITCH_Y + PITCH_HEIGHT + 20), 2)

        # 3. Dynamic Stamina Bar
        stam_h = int(35 * (self.stamina / 100.0))
        stam_color = (0, 255, 140) if self.stamina > 50 else (255, 190, 0) if self.stamina > 25 else (255, 60, 60)
        pygame.draw.line(surface, (40, 45, 55), (self.x - 5, PITCH_Y - 22), (self.x - 5, PITCH_Y + 15), 3)
        pygame.draw.line(surface, stam_color, (self.x - 5, PITCH_Y + 15 - stam_h), (self.x - 5, PITCH_Y + 15), 3)

        if self.predicted_y is not None:
            pygame.draw.circle(surface, TARGET_DOT, (self.x, int(self.predicted_y)), 4)

        if self.trapping_frames > 0 and self.aim_target_pos:
            fig_y = int(self.base_y) + self.target_figure_idx * self.spacing + FIGURE_HEIGHT // 2
            laser_color = PASS_LASER if self.is_passing_play else AIM_LASER
            pygame.draw.line(surface, laser_color, (self.x, fig_y), self.aim_target_pos, 2)
            pygame.draw.circle(surface, laser_color, self.aim_target_pos, 6, 1)

        # 4. Sculpted 2.5D Figures with Foreshortening & Cast Shadows
        cos_rot = math.cos(math.radians(self.rotation_angle))
        rendered_height = max(8, int(FIGURE_HEIGHT * cos_rot))
        y_offset = int((FIGURE_HEIGHT - rendered_height) / 2)

        for i in range(self.num_players):
            fig_y = int(self.base_y) + i * self.spacing + y_offset
            rect = pygame.Rect(self.x - FIGURE_WIDTH // 2, fig_y, FIGURE_WIDTH, rendered_height)

            # Cast Figure Shadow
            if not self.is_tilted_up:
                shadow_rect = pygame.Rect(rect.x + 6, rect.y + 6, rect.width, rect.height)
                pygame.draw.rect(shadow_surf, (15, 35, 18, 90), shadow_rect, border_radius=3)

            if self.is_tilted_up:
                # Wireframe indicating figures tilted horizontal
                pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=3)
            elif self.trapping_frames > 0 and i == self.target_figure_idx:
                # Trapping highlight
                pygame.draw.rect(surface, (255, 220, 40), rect, border_radius=4)
                pygame.draw.rect(surface, (255, 255, 255), (rect.x + 2, rect.y + 2, rect.width - 4, 4), border_radius=2)
            else:
                # Sculpted Foosball Figure Anatomy
                # Torso
                pygame.draw.rect(surface, self.color_primary, rect, border_radius=4)
                # Head/Collar
                pygame.draw.rect(surface, STEEL_BAR_HIGH, (rect.centerx - 3, rect.top - 2, 6, 3), border_radius=1)
                # Chest Highlight (Gradient look)
                pygame.draw.rect(surface, self.color_light, (rect.x + 2, rect.y + 2, rect.width - 4, 5), border_radius=2)
                # Darker Foot Wedge
                pygame.draw.rect(surface, self.color_dark, (rect.x + 1, rect.bottom - 6, rect.width - 2, 5), border_radius=2)
                # Chest Pin
                pygame.draw.circle(surface, WHITE, (rect.centerx, rect.y + 10), 2)


# -------------------------------------------------------------
# TABLE INITIALIZATION & VISUAL SURFACES
# -------------------------------------------------------------
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("AutoFoos - High-Fidelity Multi-Agent System Engine")
clock = pygame.time.Clock()

# Alpha Surfaces for Lighting, Shadows, and Analytics
shadow_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
pitch_control_surf = pygame.Surface((PITCH_WIDTH, PITCH_HEIGHT), pygame.SRCALPHA)

# Particle trail queue: [(x, y, radius, alpha)]
ball_trail = []

ball_radius = 8
ball_x = float(PITCH_X + PITCH_WIDTH // 2)
ball_y = float(PITCH_Y + PITCH_HEIGHT // 2)
ball_vx = 6.0
ball_vy = 3.0
ball_spin = 0.0

coach_blue = TacticalCoach("BLUE")
coach_red = TacticalCoach("RED")

rods = [
    Rod("B-Goalie",  "BLUE", PITCH_X + 40,  BLUE_PRIMARY, BLUE_DARK, BLUE_LIGHT, num_players=1, spacing=0,   max_speed=4.2, direction_facing=1,  accuracy=0.82),
    Rod("B-Defense", "BLUE", PITCH_X + 120, BLUE_PRIMARY, BLUE_DARK, BLUE_LIGHT, num_players=2, spacing=210, max_speed=4.0, direction_facing=1,  accuracy=0.88),
    Rod("R-Attack",  "RED",  PITCH_X + 205, RED_PRIMARY,  RED_DARK,  RED_LIGHT,  num_players=3, spacing=130, max_speed=4.2, direction_facing=-1, accuracy=0.90),
    Rod("B-Midfield","BLUE", PITCH_X + 295, BLUE_PRIMARY, BLUE_DARK, BLUE_LIGHT, num_players=3, spacing=140, max_speed=3.6, direction_facing=1,  accuracy=0.80),
    Rod("R-Midfield","RED",  PITCH_X + 405, RED_PRIMARY,  RED_DARK,  RED_LIGHT,  num_players=3, spacing=140, max_speed=3.6, direction_facing=-1, accuracy=0.80),
    Rod("B-Attack",  "BLUE", PITCH_X + 495, BLUE_PRIMARY, BLUE_DARK, BLUE_LIGHT, num_players=3, spacing=130, max_speed=4.2, direction_facing=1,  accuracy=0.90),
    Rod("R-Defense", "RED",  PITCH_X + 580, RED_PRIMARY,  RED_DARK,  RED_LIGHT,  num_players=2, spacing=210, max_speed=4.0, direction_facing=-1, accuracy=0.88),
    Rod("R-Goalie",  "RED",  PITCH_X + 660, RED_PRIMARY,  RED_DARK,  RED_LIGHT,  num_players=1, spacing=0,   max_speed=4.2, direction_facing=-1, accuracy=0.82),
]

score_blue = 0
score_red = 0
last_strategy_check = time.time()
last_kick_time = 0
selected_rod = None
is_paused = False

show_pitch_control = False
sim_speed = 1

match_over = False
match_winner = None
match_win_time = 0

# Typography
font_score = pygame.font.SysFont("Trebuchet MS", 28, bold=True)
font_match_banner = pygame.font.SysFont("Trebuchet MS", 30, bold=True)
font_series = pygame.font.SysFont("Lucida Console", 11, bold=True)
font_coach = pygame.font.SysFont("Trebuchet MS", 12, bold=True)
hud_font = pygame.font.SysFont("Courier New", 11, bold=True)
inspector_font = pygame.font.SysFont("Courier New", 12, bold=True)

blue_goalie_rod = rods[0]
red_goalie_rod = rods[7]
blue_midfield_rod = rods[3]
red_midfield_rod = rods[4]

# --- MAIN LOOP ---
running = True
while running:
    for _ in range(sim_speed):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not match_over:
                    is_paused = not is_paused
                elif event.key == pygame.K_v:
                    show_pitch_control = not show_pitch_control
                elif event.key == pygame.K_1:
                    sim_speed = 1
                elif event.key == pygame.K_2:
                    sim_speed = 2
                elif event.key == pygame.K_5:
                    sim_speed = 5
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                selected_rod = None
                for rod in rods:
                    rod.is_selected = False
                    if abs(mx - rod.x) < 18:
                        rod.is_selected = True
                        selected_rod = rod

        if match_over:
            if time.time() - match_win_time > 3.5:
                score_blue = 0
                score_red = 0
                match_over = False
                match_winner = None
                ball_x = float(PITCH_X + PITCH_WIDTH // 2)
                ball_y = float(PITCH_Y + PITCH_HEIGHT // 2)
                ball_vx = 6.0 * random.choice([-1, 1])
                ball_vy = random.choice([-3.0, 3.0])
                ball_spin = 0.0
                ball_trail.clear()
                for rod in rods:
                    rod.reset_for_new_match()
                print(f"\n[NEW MATCH STARTED] Match #{tournament.current_match_number} begun!")
            break

        if is_paused:
            break

        # 1. Macro Coach Periodic Check (every 10s)
        if time.time() - last_strategy_check > 10.0:
            coach_blue.trigger_strategy_eval(score_blue, score_red)
            coach_red.trigger_strategy_eval(score_red, score_blue)
            last_strategy_check = time.time()

        for rod in rods:
            multiplier = coach_blue.speed_multiplier if rod.team == "BLUE" else coach_red.speed_multiplier
            rod.apply_coach_strategy(multiplier)

        # 2. Inter-Agent Lane Clearing Protocol
        blue_clearing_forward = (ball_vx > 3 and ball_x < PITCH_X + 280)
        red_clearing_forward  = (ball_vx < -3 and ball_x > PITCH_X + PITCH_WIDTH - 280)

        for rod in rods:
            is_clearing = False
            if rod.team == "BLUE" and rod.name in ["B-Midfield", "B-Attack"] and blue_clearing_forward:
                is_clearing = True
            elif rod.team == "RED" and rod.name in ["R-Midfield", "R-Attack"] and red_clearing_forward:
                is_clearing = True

            bias = coach_blue.goalie_bias if rod.team == "BLUE" else coach_red.goalie_bias
            rod.update_agent(ball_x, ball_y, ball_vx, ball_vy, team_has_shot=is_clearing, coach_goalie_bias=bias)

        # 3. Physics & Magnus Spin Force
        ball_vx *= BALL_FRICTION
        ball_vy *= BALL_FRICTION
        ball_spin *= SPIN_DECAY

        magnus_ay = MAGNUS_COEFF * ball_spin * ball_vx
        ball_vy += magnus_ay

        current_speed = (ball_vx**2 + ball_vy**2)**0.5
        if current_speed < MIN_BALL_SPEED:
            ratio = MIN_BALL_SPEED / max(current_speed, 0.01)
            ball_vx *= ratio
            ball_vy *= ratio

        # Motion Blur Tracer Particles
        if current_speed > 7.0:
            ball_trail.append([ball_x, ball_y, 7, 180])

        actively_trapped = False
        for rod in rods:
            if rod.trapping_frames > 0:
                actively_trapped = True
                fig_center_y = rod.base_y + rod.target_figure_idx * rod.spacing + FIGURE_HEIGHT // 2
                ball_y = fig_center_y
                ball_x = rod.x + (14 if rod.direction == -1 else -14)
                ball_vx = 0
                ball_vy = 0
                ball_spin = 0

                if "Defense" in rod.name and random.random() < 0.65:
                    rod.is_passing_play = True
                    target_midfield = blue_midfield_rod if rod.team == "BLUE" else red_midfield_rod
                    target_fig_idx = random.randint(0, target_midfield.num_players - 1)
                    target_y = target_midfield.base_y + target_fig_idx * target_midfield.spacing + FIGURE_HEIGHT // 2
                    rod.aim_target_pos = (target_midfield.x, target_y)
                elif rod.name == "B-Attack":
                    goalie_y = red_goalie_rod.base_y
                    aim_y = GOAL_BOTTOM_Y - 20 if goalie_y < (GOAL_TOP_Y + GOAL_BOTTOM_Y) // 2 else GOAL_TOP_Y + 20
                    rod.aim_target_pos = (PITCH_X + PITCH_WIDTH, aim_y)
                    shot_analytics.record_shot_zone("BLUE", aim_y)
                elif rod.name == "R-Attack":
                    goalie_y = blue_goalie_rod.base_y
                    aim_y = GOAL_BOTTOM_Y - 20 if goalie_y < (GOAL_TOP_Y + GOAL_BOTTOM_Y) // 2 else GOAL_TOP_Y + 20
                    rod.aim_target_pos = (PITCH_X, aim_y)
                    shot_analytics.record_shot_zone("RED", aim_y)
                else:
                    rod.aim_target_pos = (rod.x + (160 * rod.direction), PITCH_Y + 40 if random.random() > 0.5 else PITCH_Y + PITCH_HEIGHT - 40)

                if rod.trapping_frames == 1:
                    rod.trigger_kick_animation()
                    if rod.is_passing_play:
                        shot_power = random.uniform(6.5, 8.0)
                        ball_spin = random.uniform(-0.5, 0.5)
                    elif "Attack" in rod.name:
                        shot_power = random.uniform(12.5, 15.0)
                        ball_spin = random.uniform(-3.5, 3.5)
                    else:
                        shot_power = random.uniform(8.5, 11.0)
                        ball_spin = random.uniform(-1.5, 1.5)

                    ball_vx = shot_power * (1 if rod.team == "BLUE" else -1)
                    dy = rod.aim_target_pos[1] - ball_y
                    dx = abs(rod.aim_target_pos[0] - ball_x)
                    ball_vy = (dy / max(dx, 1)) * shot_power
                    ball_vy = max(-6.5, min(ball_vy, 6.5))

                    audio.play_kick()
                    last_kick_time = pygame.time.get_ticks()
                    tournament.record_shot(rod.team)
                break

        if not actively_trapped:
            ball_x += ball_vx
            ball_y += ball_vy

        ball_rect = pygame.Rect(ball_x - ball_radius, ball_y - ball_radius, ball_radius * 2, ball_radius * 2)

        # 4. Collisions
        current_time = pygame.time.get_ticks()
        if not actively_trapped and current_time - last_kick_time > 180:
            for rod in rods:
                hit = False
                for idx, fig_rect in enumerate(rod.get_player_rects()):
                    if fig_rect.colliderect(ball_rect):
                        last_kick_time = current_time
                        hit = True
                        tournament.record_shot(rod.team)
                        rod.trigger_kick_animation()

                        rod_sliding_speed = rod.base_y - rod.prev_base_y

                        if random.random() < 0.38 and rod.trapping_frames == 0:
                            rod.trapping_frames = 18
                            rod.target_figure_idx = idx
                            audio.play_trap()
                        else:
                            audio.play_kick()
                            ball_spin = rod_sliding_speed * 1.8

                            if "Midfield" in rod.name and random.random() < 0.45:
                                shot_power = random.uniform(5.5, 7.5)
                            elif "Goalie" in rod.name or "Defense" in rod.name:
                                shot_power = random.uniform(10.0, 13.0)
                            else:
                                shot_power = random.uniform(7.5, 10.0)

                            ball_vx = -shot_power if ball_vx > 0 else shot_power

                            if random.random() < 0.28:
                                ball_vy = 5.0 if random.random() > 0.5 else -5.0
                            elif rod.name in ["B-Attack", "R-Attack"]:
                                goalie = red_goalie_rod if rod.name == "B-Attack" else blue_goalie_rod
                                target_vy = random.uniform(3.0, 4.5) if goalie.base_y < (GOAL_TOP_Y + GOAL_BOTTOM_Y) // 2 else random.uniform(-4.5, -3.0)
                                ball_vy = target_vy
                                shot_analytics.record_shot_zone(rod.team, goalie.base_y)
                            else:
                                hit_offset = (ball_y - fig_rect.centery) / (FIGURE_HEIGHT / 2)
                                ball_vy = hit_offset * random.uniform(3.5, 6.5)
                                if abs(ball_vy) < 1.8:
                                    ball_vy = 2.8 if random.random() > 0.5 else -2.8
                        break
                if hit:
                    break

        # 5. Table Walls & Goals
        if ball_y - ball_radius <= PITCH_Y:
            ball_y = PITCH_Y + ball_radius
            ball_vy = abs(ball_vy) * 0.95
            ball_spin = -ball_spin * 0.7
            audio.play_wall()
        elif ball_y + ball_radius >= PITCH_Y + PITCH_HEIGHT:
            ball_y = PITCH_Y + PITCH_HEIGHT - ball_radius
            ball_vy = -abs(ball_vy) * 0.95
            ball_spin = -ball_spin * 0.7
            audio.play_wall()

        # Left Goal (Red Scores)
        if ball_x - ball_radius <= PITCH_X:
            if GOAL_TOP_Y <= ball_y <= GOAL_BOTTOM_Y:
                score_red += 1
                audio.play_goal()
                ball_trail.clear()
                print(f"[GOAL!] RED SCORED! Score: BLUE {score_blue} - RED {score_red}")

                if score_red >= POINTS_TO_WIN:
                    match_over = True
                    match_winner = "RED"
                    match_win_time = time.time()
                    audio.play_match_win()
                    scout_summary = f"Red won {score_red}-{score_blue}. Attacked posts aggressively with {coach_red.active_tactic}."
                    tournament.complete_match("RED", score_blue, score_red, scout_summary)
                else:
                    ball_x = float(PITCH_X + PITCH_WIDTH // 2)
                    ball_y = float(PITCH_Y + PITCH_HEIGHT // 2)
                    ball_vx = 6.5
                    ball_vy = random.choice([-3.2, 3.2])
                    ball_spin = 0.0
                    coach_blue.trigger_strategy_eval(score_blue, score_red)
                    coach_red.trigger_strategy_eval(score_red, score_blue)
            else:
                ball_x = PITCH_X + ball_radius
                ball_vx = abs(ball_vx) * 0.95
                audio.play_wall()

        # Right Goal (Blue Scores)
        if ball_x + ball_radius >= PITCH_X + PITCH_WIDTH:
            if GOAL_TOP_Y <= ball_y <= GOAL_BOTTOM_Y:
                score_blue += 1
                audio.play_goal()
                ball_trail.clear()
                print(f"[GOAL!] BLUE SCORED! Score: BLUE {score_blue} - RED {score_red}")

                if score_blue >= POINTS_TO_WIN:
                    match_over = True
                    match_winner = "BLUE"
                    match_win_time = time.time()
                    audio.play_match_win()
                    scout_summary = f"Blue won {score_blue}-{score_red}. Dominated counter-attacks under {coach_blue.active_tactic}."
                    tournament.complete_match("BLUE", score_blue, score_red, scout_summary)
                else:
                    ball_x = float(PITCH_X + PITCH_WIDTH // 2)
                    ball_y = float(PITCH_Y + PITCH_HEIGHT // 2)
                    ball_vx = -6.5
                    ball_vy = random.choice([-3.2, 3.2])
                    ball_spin = 0.0
                    coach_blue.trigger_strategy_eval(score_blue, score_red)
                    coach_red.trigger_strategy_eval(score_red, score_blue)
            else:
                ball_x = PITCH_X + PITCH_WIDTH - ball_radius
                ball_vx = -abs(ball_vx) * 0.95
                audio.play_wall()

    # -------------------------------------------------------------
    # 6. ENHANCED RENDERING PIPELINE
    # -------------------------------------------------------------
    # Outer Beveled Wood Frame
    screen.fill(BORDER_WOOD_OUTER)
    pygame.draw.rect(screen, BORDER_WOOD_INNER, (PITCH_X - 18, PITCH_Y - 18, PITCH_WIDTH + 36, PITCH_HEIGHT + 36), border_radius=6)
    pygame.draw.rect(screen, (35, 20, 8), (PITCH_X - 6, PITCH_Y - 6, PITCH_WIDTH + 12, PITCH_HEIGHT + 12))

    # Two-Tone Lawn Stripes (Realistic Table Felt Pattern)
    stripe_w = 70
    for idx, sx in enumerate(range(PITCH_X, PITCH_X + PITCH_WIDTH, stripe_w)):
        color = PITCH_GREEN_LIGHT if idx % 2 == 0 else PITCH_GREEN_DARK
        pygame.draw.rect(screen, color, (sx, PITCH_Y, stripe_w, PITCH_HEIGHT))

    # Real-Time Spatial Pitch Control Overlay ('V' Toggle)
    if show_pitch_control:
        pitch_control_surf.fill((0, 0, 0, 0))
        for idx in range(0, PITCH_WIDTH, 20):
            world_x = PITCH_X + idx
            blue_dist = min([abs(world_x - r.x) for r in rods if r.team == "BLUE"])
            red_dist = min([abs(world_x - r.x) for r in rods if r.team == "RED"])

            if blue_dist < red_dist:
                color = (35, 115, 235, 45)
            else:
                color = (230, 45, 45, 45)
            pygame.draw.rect(pitch_control_surf, color, (idx, 0, 20, PITCH_HEIGHT))
        screen.blit(pitch_control_surf, (PITCH_X, PITCH_Y))

    # Pitch Markings
    pygame.draw.rect(screen, WHITE, (PITCH_X, PITCH_Y, PITCH_WIDTH, PITCH_HEIGHT), 2)
    center_x = PITCH_X + PITCH_WIDTH // 2
    pygame.draw.line(screen, WHITE, (center_x, PITCH_Y), (center_x, PITCH_Y + PITCH_HEIGHT), 2)
    pygame.draw.circle(screen, WHITE, (center_x, PITCH_Y + PITCH_HEIGHT // 2), 55, 2)
    pygame.draw.circle(screen, WHITE, (center_x, PITCH_Y + PITCH_HEIGHT // 2), 4)

    # Goal Boxes
    pygame.draw.rect(screen, BLACK, (PITCH_X - 25, GOAL_TOP_Y, 25, GOAL_HEIGHT))
    pygame.draw.rect(screen, (50, 50, 50), (PITCH_X - 25, GOAL_TOP_Y, 25, GOAL_HEIGHT), 2)
    pygame.draw.rect(screen, BLACK, (PITCH_X + PITCH_WIDTH, GOAL_TOP_Y, 25, GOAL_HEIGHT))
    pygame.draw.rect(screen, (50, 50, 50), (PITCH_X + PITCH_WIDTH, GOAL_TOP_Y, 25, GOAL_HEIGHT), 2)

    # Clear Shadow Layer
    shadow_surface.fill((0, 0, 0, 0))

    # Ball Shadow
    pygame.draw.ellipse(shadow_surface, BALL_SHADOW, (int(ball_x - ball_radius + 4), int(ball_y - ball_radius + 6), ball_radius * 2, ball_radius * 1.5))

    # Render Rod Shadows & Sculpted Figures
    for rod in rods:
        rod.draw(screen, shadow_surface)

    # Blit Dynamic Shadow Layer
    screen.blit(shadow_surface, (0, 0))

    # Draw Motion Blur Tracer Trails
    for p in ball_trail:
        trail_surf = pygame.Surface((p[2] * 2, p[2] * 2), pygame.SRCALPHA)
        pygame.draw.circle(trail_surf, (255, 255, 255, int(p[3])), (p[2], p[2]), p[2])
        screen.blit(trail_surf, (p[0] - p[2], p[1] - p[2]))
        p[3] -= 18  # Fade out
        p[2] = max(1, p[2] - 0.4)
    ball_trail[:] = [p for p in ball_trail if p[3] > 0]

    # Render Foosball (Molded Ball with Shading Highlight)
    pygame.draw.circle(screen, (210, 215, 225), (int(ball_x), int(ball_y)), ball_radius)
    pygame.draw.circle(screen, BALL_BASE, (int(ball_x - 1), int(ball_y - 1)), ball_radius - 1)
    pygame.draw.circle(screen, WHITE, (int(ball_x - 2), int(ball_y - 2)), 2)

    if abs(ball_spin) > 0.8:
        spin_offset = 4 if ball_spin > 0 else -4
        pygame.draw.circle(screen, (170, 175, 185), (int(ball_x), int(ball_y + spin_offset)), 2)

    # --- BROADCAST GLASS HUD (TOP BAR) ---
    hud_surf = pygame.Surface((WINDOW_WIDTH - 40, 58), pygame.SRCALPHA)
    pygame.draw.rect(hud_surf, HUD_GLASS, (0, 0, WINDOW_WIDTH - 40, 58), border_radius=10)
    pygame.draw.rect(hud_surf, (60, 75, 95), (0, 0, WINDOW_WIDTH - 40, 58), 1, border_radius=10)
    screen.blit(hud_surf, (20, 8))

    # Score Board
    score_surf = font_score.render(f"{score_blue}   :   {score_red}", True, WHITE)
    screen.blit(score_surf, (WINDOW_WIDTH // 2 - score_surf.get_width() // 2, 10))

    # Tournament Status
    series_text = f"MATCH #{tournament.current_match_number}  |  SERIES: BLUE {tournament.series_wins_blue} - {tournament.series_wins_red} RED  |  FIRST TO 10 PTS"
    series_surf = font_series.render(series_text, True, (255, 220, 110))
    screen.blit(series_surf, (WINDOW_WIDTH // 2 - series_surf.get_width() // 2, 42))

    # Tactical Directives
    b_surf = font_coach.render(f"BLUE [{coach_blue.active_tactic}]: \"{coach_blue.tactic_directive}\"", True, BLUE_LIGHT)
    screen.blit(b_surf, (36, 18))

    r_surf = font_coach.render(f"RED [{coach_red.active_tactic}]: \"{coach_red.tactic_directive}\"", True, RED_LIGHT)
    screen.blit(r_surf, (WINDOW_WIDTH - r_surf.get_width() - 36, 18))

    # Celebratory Match Banner
    if match_over:
        banner_bg = pygame.Rect(PITCH_X + 60, PITCH_Y + PITCH_HEIGHT // 2 - 50, PITCH_WIDTH - 120, 100)
        pygame.draw.rect(screen, (10, 15, 22), banner_bg, border_radius=10)
        pygame.draw.rect(screen, HIGHLIGHT_TEAL, banner_bg, 3, border_radius=10)

        win_color = BLUE_LIGHT if match_winner == "BLUE" else RED_LIGHT
        win_surf = font_match_banner.render(f"*** {match_winner} WINS MATCH #{tournament.matches_played}! ***", True, win_color)
        sub_surf = font_series.render(f"Memory saved to scouting_report.json. Next match starting... (Series: {tournament.series_wins_blue} - {tournament.series_wins_red})", True, WHITE)

        screen.blit(win_surf, (banner_bg.centerx - win_surf.get_width() // 2, banner_bg.centery - 22))
        screen.blit(sub_surf, (banner_bg.centerx - sub_surf.get_width() // 2, banner_bg.centery + 14))

    # Brain Inspector Glass Card
    if selected_rod is not None:
        insp_rect = pygame.Rect(WINDOW_WIDTH - 275, 75, 255, 160)
        insp_surf = pygame.Surface((insp_rect.width, insp_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(insp_surf, (15, 20, 30, 235), (0, 0, insp_rect.width, insp_rect.height), border_radius=8)
        pygame.draw.rect(insp_surf, HIGHLIGHT_TEAL, (0, 0, insp_rect.width, insp_rect.height), 2, border_radius=8)
        screen.blit(insp_surf, (insp_rect.x, insp_rect.y))

        pred_str = f"{int(selected_rod.predicted_y)}" if selected_rod.predicted_y else "N/A"
        lines = [
            f"[INSPECTING: {selected_rod.name}]",
            f"State     : {selected_rod.state}",
            f"Stamina   : {int(selected_rod.stamina)}%",
            f"Rot Angle : {int(selected_rod.rotation_angle)} deg",
            f"Kinematics: {selected_rod.current_max_speed:.1f} px/f",
            f"PredictedY: {pred_str}",
            f"Target Fig: #{selected_rod.target_figure_idx + 1}"
        ]
        for idx, text in enumerate(lines):
            color = HIGHLIGHT_TEAL if idx == 0 else WHITE
            t_surf = inspector_font.render(text, True, color)
            screen.blit(t_surf, (insp_rect.x + 10, insp_rect.y + 10 + idx * 20))

    # --- BOTTOM TELEMETRY DOCK ---
    bottom_surf = pygame.Surface((WINDOW_WIDTH - 40, 85), pygame.SRCALPHA)
    pygame.draw.rect(bottom_surf, HUD_GLASS, (0, 0, WINDOW_WIDTH - 40, 85), border_radius=10)
    pygame.draw.rect(bottom_surf, (60, 75, 95), (0, 0, WINDOW_WIDTH - 40, 85), 1, border_radius=10)
    screen.blit(bottom_surf, (20, WINDOW_HEIGHT - 95))

    if HAVE_GEMINI and not TacticalCoach.api_circuit_open:
        api_badge = "[GEMINI 3.8 FLASH LIVE]"
    else:
        api_badge = "[LOCAL HEURISTIC ENGINE]"

    title_text = hud_font.render(f"[KEYS: 'V' Pitch Control | '1/2/5' Speed | 'SPACE' Pause] {api_badge} | SPEED: {sim_speed}X", True, (0, 255, 180))
    screen.blit(title_text, (35, WINDOW_HEIGHT - 88))

    for idx, rod in enumerate(rods):
        state_color = (130, 135, 145)
        if rod.state == "TRACKING":
            state_color = (255, 230, 0)
        elif rod.state == "CLEAR_LANE":
            state_color = (0, 255, 255)
        elif rod.state == "AIMING":
            state_color = (0, 255, 220) if not rod.is_passing_play else (60, 255, 80)

        rod_info = f"{rod.name[:6]}:{rod.state[:5]}"
        status_surf = hud_font.render(rod_info, True, state_color)
        x_pos = 35 + (idx * 116)
        screen.blit(status_surf, (x_pos, WINDOW_HEIGHT - 60))

    if is_paused and not match_over:
        pause_overlay = font_score.render("[SIMULATION PAUSED]", True, (255, 220, 0))
        screen.blit(pause_overlay, (WINDOW_WIDTH // 2 - pause_overlay.get_width() // 2, WINDOW_HEIGHT // 2 - 20))

    pygame.display.flip()
    clock.tick(FPS)

# Export tournament telemetry on exit
tournament.export()
pygame.quit()
sys.exit()