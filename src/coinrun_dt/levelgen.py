"""CoinRun のステージ生成（procgen/src/games/coinrun.cpp）を Python で再現し、生成器の内部の値を取り出す。

Procgen はステージごとに std::mt19937 を level seed で初期化し、そこから決まった順番で乱数を引いて
ステージを作る。同じ順番で乱数を引けば、info には出てこない生成器の難易度 dif（1〜3）や区間の数、
穴・回転のこぎり・木箱の数などがわかる。再現が正しいことは、実際の環境の地形（get_state の grid）と
一致することで確かめる（level_grid_from_env と tests/test_levelgen.py）。

    from coinrun_dt.levelgen import generate_level
    generate_level(123).dif
"""
from __future__ import annotations

import struct
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np

WIDTH = HEIGHT = 64
# object id（procgen の object-ids.h と coinrun.cpp）
SPACE, GOAL, SAW, ENEMY, CRATE = 100, 1, 2, 5, 20
WALL_MID, WALL_TOP, LAVA_MID, LAVA_TOP, ENEMY_BARRIER = 15, 16, 17, 18, 19
DANGER_TYPES = ["lava", "saw", "enemy"]   # 穴の中身
# ジャンプの物理（coinrun.cpp の game_reset: gravity 0.2, max_jump 1.5, maxspeed 0.5）
MAX_DY = int(1.5 * 1.5 / (2 * 0.2) - 0.5)        # 5
MAX_DX = int(0.5 * 2 * 1.5 / 0.2 - 0.5)          # 7


class MT19937:
    """std::mt19937（seed(int) は init_genrand と同じ）。"""

    def __init__(self, seed: int):
        mt = [0] * 624
        mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.mt, self.index = mt, 624
        self.count = 0   # 引いた回数

    def __call__(self) -> int:
        if self.index >= 624:
            mt = self.mt
            for i in range(624):
                y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7FFFFFFF)
                mt[i] = mt[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            self.index = 0
        y = self.mt[self.index]
        self.index += 1
        self.count += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y

    # procgen の RandGen と同じ使い方（randgen.cpp）
    def randn(self, high: int) -> int:
        return self() % high

    def rand01(self) -> float:
        return self() / 2 ** 32


@dataclass
class LevelInfo:
    level_seed: int
    dif: int                  # 生成器の難易度（1〜3）。区間の数・幅、穴とのこぎりの出やすさが増える
    danger_type: str          # 穴の中身（lava / saw / enemy）
    num_sections: int         # 実際に作った区間の数（予定の数でも、右端に近づくと打ち切る）
    num_pits: int
    pit_cells: int            # 穴の幅の合計
    num_saws: int             # 回転のこぎりの数（穴の中と地面の上）
    num_enemies: int          # 歩く敵の数（easy では穴の中だけ）
    num_crates: int
    goal_x: int               # コインの位置（ステージの長さ）
    goal_y: int
    total_climb: int          # 上りの段差の合計
    max_step_up: int          # いちばん高い上りの段差
    grid: np.ndarray = field(repr=False)        # (WIDTH, HEIGHT)。grid[x, y] が object id
    entities: List[Tuple[int, int, int]] = field(default_factory=list, repr=False)   # (type, x, y)

    def features(self) -> Dict[str, object]:
        d = asdict(self)
        d.pop("grid")
        d.pop("entities")
        return d


def generate_level(level_seed: int, distribution_mode: str = "easy") -> LevelInfo:
    if distribution_mode not in ("easy", "hard"):
        raise ValueError(f"easy と hard だけに対応: {distribution_mode}")
    easy = distribution_mode == "easy"
    rng = MT19937(level_seed)
    grid = np.full((WIDTH, HEIGHT), SPACE, dtype=np.int32)
    entities: List[Tuple[int, int, int]] = []

    def fill(x, y, dx, dy, v):
        grid[x:x + dx, y:y + dy] = v

    def fill_block_top(x, y, dx, dy, mid, top):
        fill(x, y, dx, dy - 1, mid)
        fill(x, y + dy - 1, dx, 1, top)

    def create_enemy(x, y):
        rng.randn(2)      # 向き
        rng()             # 見た目のテーマ（choose_random_theme）
        entities.append((ENEMY, x, y))

    def create_crate(x, y):
        rng()             # 見た目のテーマ
        entities.append((CRATE, x, y))

    # BasicAbstractGame::game_reset
    rng.rand01()          # bg_pct_x
    rng()                 # background_index（easy では後で 0 に上書きされるが、乱数は引く）
    rng.rand01()          # random_agent_start の x（後で上書き）
    rng.rand01()          # 同じく y
    # CoinRun::game_reset
    if not easy:
        rng()             # エージェントのテーマ
        rng()             # wall_theme
    fill(0, 0, WIDTH, 1, WALL_TOP)       # init_floor_and_walls
    fill(0, 0, 1, HEIGHT, WALL_MID)
    fill(WIDTH - 1, 0, 1, HEIGHT, WALL_MID)
    fill(0, HEIGHT - 1, WIDTH, 1, WALL_MID)

    # generate_coin_to_the_right
    dif = rng.randn(3) + 1
    planned = rng.randn(dif) + dif
    curr_x, curr_y = 5, 1
    danger_type = rng.randn(3)
    allow_monsters = not easy
    built = num_pits = pit_cells = total_climb = max_step_up = 0
    for _ in range(planned):
        if curr_x + 15 >= WIDTH:
            break
        dy = min(rng.randn(4) + 1 + dif // 3, MAX_DY)
        if curr_y >= 20:
            dy = -dy
        elif curr_y >= 5 and rng.randn(2) == 1:
            dy = -dy
        dx = rng.randn(2 * dif) + 3 + dif // 3
        prev_y = curr_y
        curr_y = max(1, curr_y + dy)
        total_climb += max(0, curr_y - prev_y)
        max_step_up = max(max_step_up, curr_y - prev_y)

        use_pit = dx > 7 and curr_y > 3 and rng.randn(20) >= dif
        if use_pit:
            x1 = rng.randn(3) + 1
            x2 = rng.randn(3) + 1
            pit_width = dx - x1 - x2
            if pit_width > MAX_DX:
                pit_width = MAX_DX
                x2 = dx - x1 - pit_width
            fill_block_top(curr_x, 0, x1, curr_y, WALL_MID, WALL_TOP)
            fill_block_top(curr_x + dx - x2, 0, x2, curr_y, WALL_MID, WALL_TOP)
            lava_height = rng.randn(curr_y - 3) + 1
            if danger_type == 0:
                fill_block_top(curr_x + x1, 1, pit_width, lava_height, LAVA_MID, LAVA_TOP)
            else:
                for ei in range(pit_width):
                    if danger_type == 1:
                        entities.append((SAW, curr_x + x1 + ei, 1))
                    else:
                        create_enemy(curr_x + x1 + ei, 1)
            if pit_width > 4:
                if pit_width == 5:
                    x3 = 1 + rng.randn(2)
                    w1 = 1 + rng.randn(2)
                elif pit_width == 6:
                    x3 = 2 + rng.randn(2)
                    w1 = 1 + rng.randn(2)
                else:
                    x3 = 2 + rng.randn(2)
                    x4 = 2 + rng.randn(2)
                    w1 = pit_width - x3 - x4
                fill_block_top(curr_x + x1 + x3, curr_y - 1, w1, 1, WALL_MID, WALL_TOP)
            num_pits += 1
            pit_cells += pit_width
        else:
            fill_block_top(curr_x, 0, dx, curr_y, WALL_MID, WALL_TOP)
            ob1_x = ob2_x = -1
            if rng.randn(10) < 2 * dif and dx > 3:
                ob1_x = curr_x + rng.randn(dx - 2) + 1
                entities.append((SAW, ob1_x, curr_y))
            if rng.randn(10) < dif and dx > 3 and MAX_DX >= 4 and allow_monsters:
                ob2_x = curr_x + rng.randn(dx - 2) + 1
                create_enemy(ob2_x, curr_y)
            for _ in range(2):
                crate_x = curr_x + rng.randn(dx - 2) + 1
                if rng.randn(2) == 1 and ob1_x != crate_x and ob2_x != crate_x:
                    for j in range(rng.randn(3) + 1):
                        create_crate(crate_x, curr_y + j)

        if grid[curr_x - 1, curr_y] not in (WALL_MID, WALL_TOP):
            grid[curr_x - 1, curr_y] = ENEMY_BARRIER
        curr_x += dx
        grid[curr_x, curr_y] = ENEMY_BARRIER
        built += 1

    grid[curr_x, curr_y] = GOAL
    fill_block_top(curr_x, 0, 1, curr_y, WALL_MID, WALL_TOP)
    fill(curr_x + 1, 0, WIDTH - curr_x - 1, HEIGHT, WALL_MID)

    kinds = [e[0] for e in entities]
    return LevelInfo(level_seed=level_seed, dif=dif, danger_type=DANGER_TYPES[danger_type],
                     num_sections=built, num_pits=num_pits, pit_cells=pit_cells,
                     num_saws=kinds.count(SAW), num_enemies=kinds.count(ENEMY),
                     num_crates=kinds.count(CRATE), goal_x=curr_x, goal_y=curr_y,
                     total_climb=total_climb, max_step_up=max_step_up, grid=grid, entities=entities)


def level_grid_from_env(state: bytes) -> np.ndarray:
    """ProcgenGym3Env.get_state() の1環境分から地形を取り出す（(WIDTH, HEIGHT)、grid[x, y]）。

    CoinRun の serialize は、BasicAbstractGame の最後に grid（w, h, 要素数, 要素…）を書き、
    そのあとに CoinRun 自身の値を数個書く（個数は procgen のバージョンで違う）ので、grid の先頭は
    末尾の近くにある (w, h, 要素数) の並びを探して決める。
    """
    n = WIDTH * HEIGHT
    header = struct.pack("<iii", WIDTH, HEIGHT, n)
    pos = state.rfind(header, max(0, len(state) - 4 * n - 256))
    if pos < 0 or pos + 12 + 4 * n > len(state):
        raise ValueError("state の中に地形が見つからない")
    data = np.frombuffer(state, dtype="<i4", count=n, offset=pos + 12)
    return data.reshape(HEIGHT, WIDTH).T.copy()   # data[y * w + x] → grid[x, y]
