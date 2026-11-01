import asyncio
import json
import os
import random
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
app = FastAPI(title="AutoFoos Real-Time Agent Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Gemini Client Initialization
api_key = os.environ.get("GEMINI_API_KEY")
HAVE_GEMINI = False
client = None
if api_key:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        HAVE_GEMINI = True
        print("[SUCCESS] Gemini API Key connected to server.")
    except Exception as e:
        print(f"[WARNING] Gemini init failed: {e}")

# Pitch Geometry
PITCH_WIDTH = 700
PITCH_HEIGHT = 480
GOAL_HEIGHT = 180
GOAL_TOP_Y = (PITCH_HEIGHT - GOAL_HEIGHT) // 2
GOAL_BOTTOM_Y = GOAL_TOP_Y + GOAL_HEIGHT

BALL_FRICTION = 0.993
MIN_BALL_SPEED = 2.2
ROD_ACCEL = 0.18
POINTS_TO_WIN = 10


class HeadlessRod:
    def __init__(self, name, team, x, num_players, spacing, max_speed, direction):
        self.name = name
        self.team = team
        self.x = x
        self.num_players = num_players
        self.spacing = spacing
        self.max_speed = max_speed
        self.direction = direction

        self.min_y = 10
        self.max_y = PITCH_HEIGHT - (num_players - 1) * spacing - 24 - 10
        self.base_y = float((self.min_y + self.max_y) // 2)
        self.target_y = self.base_y
        self.prev_y = self.base_y

        self.rotation = 0.0
        self.target_rotation = 0.0
        self.stamina = 100.0
        self.state = "IDLE"
        self.trapping_frames = 0
        self.target_figure_idx = 0

    def reset(self):
        self.base_y = float((self.min_y + self.max_y) // 2)
        self.target_y = self.base_y
        self.rotation = 0.0
        self.target_rotation = 0.0
        self.stamina = 100.0
        self.state = "IDLE"
        self.trapping_frames = 0

    def trigger_kick(self):
        self.rotation = -45.0
        self.target_rotation = 60.0

    def update(self, ball_x, ball_y, ball_vx, ball_vy):
        self.rotation += (self.target_rotation - self.rotation) * 0.3
        if abs(self.rotation - self.target_rotation) < 2.0 and self.target_rotation != 0.0:
            self.target_rotation = 0.0

        if self.trapping_frames > 0:
            self.trapping_frames -= 1
            self.state = "AIMING"
            self.rotation = -35.0
            return

        approaching = (self.direction == 1 and ball_vx < 0 and ball_x > self.x) or \
                      (self.direction == -1 and ball_vx > 0 and ball_x < self.x)

        if approaching:
            self.state = "TRACKING"
            frames = (self.x - ball_x) / (ball_vx if ball_vx != 0 else 0.01)
            pred_y = ball_y + ball_vy * frames

            while pred_y < 0 or pred_y > PITCH_HEIGHT:
                pred_y = -pred_y if pred_y < 0 else (2 * PITCH_HEIGHT - pred_y)

            best_diff = 9999
            for i in range(self.num_players):
                center = self.base_y + i * self.spacing + 12
                diff = pred_y - center
                if abs(diff) < abs(best_diff):
                    best_diff = diff
                    self.target_figure_idx = i

            self.target_y = self.base_y + best_diff
        else:
            self.state = "IDLE"
            self.target_y = (self.min_y + self.max_y) // 2

        stamina_factor = max(0.60, self.stamina / 100.0)
        max_v = self.max_speed * stamina_factor
        diff = self.target_y - self.base_y
        step = max(-max_v, min(diff * ROD_ACCEL, max_v))

        self.prev_y = self.base_y
        self.base_y = max(self.min_y, min(self.base_y + step, self.max_y))

        if abs(self.base_y - self.prev_y) > 1.8:
            self.stamina = max(10.0, self.stamina - 0.45)
        else:
            self.stamina = min(100.0, self.stamina + 0.35)

    def serialize(self):
        return {
            "name": self.name,
            "team": self.team,
            "x": self.x,
            "base_y": round(self.base_y, 1),
            "num_players": self.num_players,
            "spacing": self.spacing,
            "rotation": round(self.rotation, 1),
            "state": self.state,
            "stamina": int(self.stamina),
        }


class MatchEngine:
    def __init__(self):
        self.is_running = False
        self.sim_speed = 1  # Multiplier: 1x, 2x, 5x
        self.ball_x = PITCH_WIDTH / 2
        self.ball_y = PITCH_HEIGHT / 2
        self.ball_vx = 6.5
        self.ball_vy = 3.5
        self.ball_spin = 0.0

        self.score_blue = 0
        self.score_red = 0
        self.match_count = 1
        self.series_blue = 0
        self.series_red = 0

        self.tactic_blue = "HIGH_PRESS"
        self.tactic_red = "DEFENSIVE_WALL"
        self.directive_blue = "High press counter"
        self.directive_red = "Lock goal corridors"
        self.last_kick_time = 0

        self.rods = [
            HeadlessRod("B-Goalie", "BLUE", 40, 1, 0, 4.5, 1),
            HeadlessRod("B-Defense", "BLUE", 120, 2, 210, 4.2, 1),
            HeadlessRod("R-Attack", "RED", 205, 3, 130, 4.5, -1),
            HeadlessRod("B-Midfield", "BLUE", 295, 3, 140, 3.8, 1),
            HeadlessRod("R-Midfield", "RED", 405, 3, 140, 3.8, -1),
            HeadlessRod("B-Attack", "BLUE", 495, 3, 130, 4.5, 1),
            HeadlessRod("R-Defense", "RED", 580, 2, 210, 4.2, -1),
            HeadlessRod("R-Goalie", "RED", 660, 1, 0, 4.5, -1),
        ]

    def reset_match(self):
        self.score_blue = 0
        self.score_red = 0
        self.ball_x = PITCH_WIDTH / 2
        self.ball_y = PITCH_HEIGHT / 2
        self.ball_vx = 6.5 * random.choice([-1, 1])
        self.ball_vy = random.choice([-3.5, 3.5])
        self.ball_spin = 0.0
        for rod in self.rods:
            rod.reset()

    def tick_single(self):
        """Single physics tick (runs 1x, 2x, or 5x per render frame)."""
        self.ball_vx *= BALL_FRICTION
        self.ball_vy *= BALL_FRICTION
        self.ball_spin *= 0.985
        self.ball_vy += 0.015 * self.ball_spin * self.ball_vx

        speed = (self.ball_vx**2 + self.ball_vy**2)**0.5
        if speed < MIN_BALL_SPEED:
            ratio = MIN_BALL_SPEED / max(speed, 0.01)
            self.ball_vx *= ratio
            self.ball_vy *= ratio

        trapped = False
        for rod in self.rods:
            if rod.trapping_frames > 0:
                trapped = True
                fig_center_y = rod.base_y + rod.target_figure_idx * rod.spacing + 12
                self.ball_y = fig_center_y
                self.ball_x = rod.x + (14 if rod.direction == -1 else -14)
                self.ball_vx = 0
                self.ball_vy = 0

                if rod.trapping_frames == 1:
                    rod.trigger_kick()
                    power = random.uniform(12.0, 15.0) if "Attack" in rod.name else random.uniform(8.5, 11.0)
                    self.ball_vx = power * (1 if rod.team == "BLUE" else -1)
                    self.ball_vy = random.uniform(-4.5, 4.5)
                    self.last_kick_time = time.time()
                break

        if not trapped:
            self.ball_x += self.ball_vx
            self.ball_y += self.ball_vy

        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy = abs(self.ball_vy) * 0.95
        elif self.ball_y >= PITCH_HEIGHT:
            self.ball_y = PITCH_HEIGHT
            self.ball_vy = -abs(self.ball_vy) * 0.95

        now = time.time()
        for rod in self.rods:
            rod.update(self.ball_x, self.ball_y, self.ball_vx, self.ball_vy)

            if not trapped and (now - self.last_kick_time > 0.16):
                for i in range(rod.num_players):
                    fig_y = rod.base_y + i * rod.spacing
                    if abs(self.ball_x - rod.x) < 14 and fig_y <= self.ball_y <= fig_y + 24:
                        self.last_kick_time = now
                        if random.random() < 0.35:
                            rod.trapping_frames = 15
                            rod.target_figure_idx = i
                        else:
                            rod.trigger_kick()
                            shot_power = random.uniform(9.0, 13.0)
                            self.ball_vx = -shot_power if self.ball_vx > 0 else shot_power
                            self.ball_vy = random.uniform(-5.0, 5.0)
                            self.ball_spin = (rod.base_y - rod.prev_y) * 1.5
                        break

        # Goal Checks
        if self.ball_x <= 0:
            if GOAL_TOP_Y <= self.ball_y <= GOAL_BOTTOM_Y:
                self.score_red += 1
                self.reset_ball_center(7.0)
            else:
                self.ball_x = 0
                self.ball_vx = abs(self.ball_vx)

        elif self.ball_x >= PITCH_WIDTH:
            if GOAL_TOP_Y <= self.ball_y <= GOAL_BOTTOM_Y:
                self.score_blue += 1
                self.reset_ball_center(-7.0)
            else:
                self.ball_x = PITCH_WIDTH
                self.ball_vx = -abs(self.ball_vx)

        if self.score_blue >= POINTS_TO_WIN:
            self.series_blue += 1
            self.match_count += 1
            self.reset_match()
        elif self.score_red >= POINTS_TO_WIN:
            self.series_red += 1
            self.match_count += 1
            self.reset_match()

    def tick(self):
        """Executes sub-ticks based on selected speed multiplier without network lag."""
        if not self.is_running:
            return
        for _ in range(self.sim_speed):
            self.tick_single()

    def reset_ball_center(self, vx):
        self.ball_x = PITCH_WIDTH / 2
        self.ball_y = PITCH_HEIGHT / 2
        self.ball_vx = vx
        self.ball_vy = random.choice([-3.5, 3.5])
        self.ball_spin = 0.0

    def get_state(self):
        return {
            "is_running": self.is_running,
            "sim_speed": self.sim_speed,
            "ball": {"x": round(self.ball_x, 1), "y": round(self.ball_y, 1), "spin": round(self.ball_spin, 2)},
            "score": {"blue": self.score_blue, "red": self.score_red},
            "series": {"blue": self.series_blue, "red": self.series_red, "match": self.match_count},
            "tactics": {
                "blue": {"strategy": self.tactic_blue, "directive": self.directive_blue},
                "red": {"strategy": self.tactic_red, "directive": self.directive_red},
            },
            "rods": [r.serialize() for r in self.rods],
        }


engine = MatchEngine()
@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def command_listener():
        try:
            while True:
                msg_raw = await websocket.receive_text()
                msg = json.loads(msg_raw)
                action = msg.get("action")
                if action == "TOGGLE_PLAY":
                    engine.is_running = not engine.is_running
                elif action == "RESET":
                    engine.reset_match()
                    engine.is_running = False
                elif action == "SET_SPEED":
                    spd = int(msg.get("speed", 1))
                    if spd in [1, 2, 5]:
                        engine.sim_speed = spd
                        print(f"[SPEED] Engine simulation multiplier set to {spd}x")
        except WebSocketDisconnect:
            pass

    listener_task = asyncio.create_task(command_listener())

    try:
        while True:
            t0 = time.time()
            engine.tick()
            await websocket.send_text(json.dumps(engine.get_state()))
            # Maintain stable 60 FPS tick time
            elapsed = time.time() - t0
            sleep_time = max(0.001, (1 / 60) - elapsed)
            await asyncio.sleep(sleep_time)
    except WebSocketDisconnect:
        listener_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)