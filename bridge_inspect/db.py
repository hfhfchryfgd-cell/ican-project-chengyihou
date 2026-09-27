"""SQLite 数据访问层。

所有落库动作都收在这里，页面层不直接写 SQL。表结构对应《作品说明书》§3.3 的图像输入
路径：images 表以 SD 卡离线导入写入的行为主，同时预留了机载图传通道写入所需的字段
（图传链路本身按 §3.3 未接入；影像由智能体工作台的“导入影像”入口接收）。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Iterable

from . import config as cfg

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT '巡检员',
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- 目标检测 / 分割量化 / 长周期监测
    batch       TEXT DEFAULT '',        -- 巡检批次 P1/P2/P3
    point_scope TEXT DEFAULT '',        -- 涉及点位范围，如 A01-A06
    image_count INTEGER DEFAULT 0,
    status      TEXT DEFAULT '已完成',
    owner       TEXT DEFAULT '',
    note        TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        INTEGER,
    path           TEXT NOT NULL,
    filename       TEXT NOT NULL,
    width          INTEGER DEFAULT 0,
    height         INTEGER DEFAULT 0,
    size_kb        REAL DEFAULT 0,
    source         TEXT DEFAULT 'SD卡导入',   -- SD卡导入 / 实时图传
    point_id       TEXT DEFAULT '',
    waypoint       TEXT DEFAULT '',
    batch          TEXT DEFAULT '',
    rtk_lon        REAL DEFAULT 0,
    rtk_lat        REAL DEFAULT 0,
    rtk_alt        REAL DEFAULT 0,
    rtk_status     TEXT DEFAULT '固定解',
    gimbal_yaw     REAL DEFAULT 0,
    gimbal_pitch   REAL DEFAULT 0,
    gimbal_roll    REAL DEFAULT 0,
    shoot_distance REAL DEFAULT 0,
    overlap        REAL DEFAULT 0,
    captured_at    TEXT DEFAULT '',
    imported_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id   INTEGER NOT NULL,
    task_id    INTEGER,
    point_id   TEXT DEFAULT '',
    cls_key    TEXT NOT NULL,
    cls_name   TEXT NOT NULL,
    conf       REAL NOT NULL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    gt_idx     INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id     INTEGER NOT NULL,
    task_id      INTEGER,
    point_id     TEXT DEFAULT '',
    cls_key      TEXT NOT NULL,
    cls_name     TEXT NOT NULL,
    batch        TEXT DEFAULT '',
    px_area      INTEGER DEFAULT 0,
    area_cm2     REAL DEFAULT 0,
    perimeter_px REAL DEFAULT 0,
    max_width_mm REAL DEFAULT 0,
    length_mm    REAL DEFAULT 0,   -- 裂缝长度，由掩膜面积与周长反解后按 GSD 换算
    avg_width_mm REAL DEFAULT 0,   -- 平均宽度 = 面积 ÷ 长度
    area_ratio   REAL DEFAULT 0,   -- 轮廓像素面积占 ROI 之比
    crack_count  INTEGER DEFAULT 0,-- 该 ROI 内连通裂缝条数
    conf         REAL DEFAULT 0,
    captured_at  TEXT DEFAULT '',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outputs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    INTEGER,
    kind       TEXT NOT NULL,       -- 标注图 / CSV / 报告 / 趋势图
    path       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_det_image ON detections(image_id);
CREATE INDEX IF NOT EXISTS idx_seg_point ON segments(point_id, cls_key, batch);
CREATE INDEX IF NOT EXISTS idx_img_point ON images(point_id, batch);
"""


def connect() -> sqlite3.Connection:
    """取全局连接（首次调用时建库建表）。"""
    global _conn
    with _lock:
        if _conn is None:
            _conn = sqlite3.connect(cfg.DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.executescript(SCHEMA)
            _migrate(_conn)
            _conn.commit()
        return _conn


def _migrate(conn: sqlite3.Connection) -> None:
    """把已存在的旧库补齐到当前表结构。

    CREATE TABLE IF NOT EXISTS 只对新建库生效，已经落过盘的库不会自动加列，
    所以新增字段要在这里显式补——否则老库升级后一查就报 no such column。
    """
    wanted = {
        "detections": {"gt_idx": "INTEGER DEFAULT 0"},
        # 量化口径扩到"长度 / 平均宽度 / 面积占比 / 裂缝条数"后新增的四列。
        # 旧库里这些列是空的（默认 0），不重算——重算要重新跑分割，而
        # 说明书的口径改一次、历史值跟着变一次，"趋势"就没法比了。
        "segments": {
            "length_mm": "REAL DEFAULT 0",
            "avg_width_mm": "REAL DEFAULT 0",
            "area_ratio": "REAL DEFAULT 0",
            "crack_count": "INTEGER DEFAULT 0",
        },
    }
    for table, cols in wanted.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    with _lock:
        return connect().execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    """执行写操作，返回 lastrowid。"""
    with _lock:
        conn = connect()
        cur = conn.execute(sql, tuple(params))
        conn.commit()
        return cur.lastrowid


def executemany(sql: str, rows: Iterable[Iterable[Any]]) -> None:
    with _lock:
        conn = connect()
        conn.executemany(sql, [tuple(r) for r in rows])
        conn.commit()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# 用户
# --------------------------------------------------------------------------
def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000
    ).hex()


def create_user(username: str, password: str, role: str = "巡检员") -> tuple[bool, str]:
    username = (username or "").strip()
    if len(username) < 2:
        return False, "用户名至少 2 个字符"
    if len(password or "") < 4:
        return False, "密码至少 4 位"
    if query_one("SELECT id FROM users WHERE username=?", (username,)):
        return False, "该用户名已被注册"
    salt = os.urandom(16).hex()
    execute(
        "INSERT INTO users(username,password_hash,salt,role,created_at) VALUES(?,?,?,?,?)",
        (username, _hash(password, salt), salt, role, now()),
    )
    return True, "注册成功"


def verify_user(username: str, password: str) -> sqlite3.Row | None:
    row = query_one("SELECT * FROM users WHERE username=?", ((username or "").strip(),))
    if row and _hash(password or "", row["salt"]) == row["password_hash"]:
        return row
    return None


def change_password(username: str, old: str, new: str) -> tuple[bool, str]:
    row = verify_user(username, old)
    if not row:
        return False, "原密码不正确"
    if len(new or "") < 4:
        return False, "新密码至少 4 位"
    salt = os.urandom(16).hex()
    execute(
        "UPDATE users SET password_hash=?, salt=? WHERE id=?",
        (_hash(new, salt), salt, row["id"]),
    )
    return True, "密码已更新"


def ensure_default_user() -> None:
    """首次启动播种一个演示账号，方便直接登录看效果。"""
    if not query_one("SELECT id FROM users LIMIT 1"):
        create_user("admin", "123456", "管理员")
        create_user("inspector", "123456", "巡检员")


# --------------------------------------------------------------------------
# 任务
# --------------------------------------------------------------------------
def create_task(name: str, kind: str, batch: str = "", point_scope: str = "",
                image_count: int = 0, owner: str = "", note: str = "",
                created_at: str = "") -> int:
    """建一条巡检任务。created_at 留空则取当前时刻。

    补录历史批次时要显式传该批次的日期：任务的创建时间若一律是"今天"，检测历史页
    会列出三条同一天建的任务，而它们的影像拍摄时间却分别在三、六、九月，两者的
    时间线对不上，读的人会以为数据串了批次。
    """
    return execute(
        "INSERT INTO tasks(name,kind,batch,point_scope,image_count,status,owner,note,created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        (name, kind, batch, point_scope, image_count, "已完成", owner, note,
         created_at or now()),
    )


def list_tasks(keyword: str = "", kind: str = "", limit: int = 500) -> list[sqlite3.Row]:
    sql = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    if kind:
        sql += " AND kind=?"
        params.append(kind)
    if keyword:
        sql += " AND (name LIKE ? OR point_scope LIKE ? OR note LIKE ?)"
        params += [f"%{keyword}%"] * 3
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return query(sql, params)


def list_detection_tasks(keyword: str = "", date_from: str = "", date_to: str = "",
                         limit: int = 500) -> list[sqlite3.Row]:
    """仅返回真正写入检测结果的任务。

    ``tasks`` 也会登记报告、复飞和已取消的空批处理；它们不能算作一次检测。
    此查询以 ``detections.task_id`` 为事实来源，并把检测时间、实际处理影像数和
    检出条数一并聚合，供「检测历史」页面显示与筛选。
    """
    sql = """
        SELECT t.*, MAX(d.created_at) AS detected_at,
               COUNT(d.id) AS detection_count,
               COUNT(DISTINCT d.image_id) AS detected_image_count,
               (SELECT COUNT(*) FROM segments s WHERE s.task_id=t.id) AS segment_count,
               (SELECT COUNT(*) FROM outputs o WHERE o.task_id=t.id) AS output_count
        FROM tasks t
        JOIN detections d ON d.task_id=t.id
        WHERE 1=1
    """
    params: list[Any] = []
    if keyword:
        sql += " AND (t.name LIKE ? OR t.point_scope LIKE ? OR t.note LIKE ?)"
        params += [f"%{keyword}%"] * 3
    if date_from:
        sql += " AND d.created_at >= ?"
        params.append(f"{date_from} 00:00:00")
    if date_to:
        sql += " AND d.created_at < datetime(?, '+1 day')"
        params.append(f"{date_to} 00:00:00")
    sql += " GROUP BY t.id ORDER BY detected_at DESC, t.id DESC LIMIT ?"
    params.append(limit)
    return query(sql, params)


def detection_rows_of_tasks(task_ids: Iterable[int]) -> list[sqlite3.Row]:
    """取一组真实检测任务的逐病害明细，供历史资料包导出。"""
    ids = [int(task_id) for task_id in task_ids]
    if not ids:
        return []
    marks = ",".join("?" for _ in ids)
    return query(
        "SELECT d.*, t.name AS task_name, t.kind AS task_kind, "
        "i.filename, i.path AS image_path, i.batch AS image_batch "
        "FROM detections d JOIN tasks t ON t.id=d.task_id "
        "LEFT JOIN images i ON i.id=d.image_id "
        f"WHERE d.task_id IN ({marks}) ORDER BY d.created_at DESC, d.id DESC",
        ids,
    )


def delete_task(task_id: int) -> None:
    with _lock:
        conn = connect()
        for tb in ("detections", "segments", "outputs"):
            conn.execute(f"DELETE FROM {tb} WHERE task_id=?", (task_id,))
        conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
        conn.commit()


# --------------------------------------------------------------------------
# 图片
# --------------------------------------------------------------------------
IMAGE_FIELDS = (
    "task_id", "path", "filename", "width", "height", "size_kb", "source",
    "point_id", "waypoint", "batch", "rtk_lon", "rtk_lat", "rtk_alt", "rtk_status",
    "gimbal_yaw", "gimbal_pitch", "gimbal_roll", "shoot_distance", "overlap",
    "captured_at", "imported_at",
)


def add_image(**kw: Any) -> int:
    kw.setdefault("imported_at", now())
    cols = [f for f in IMAGE_FIELDS if f in kw]
    placeholders = ",".join("?" for _ in cols)
    return execute(
        f"INSERT INTO images({','.join(cols)}) VALUES({placeholders})",
        [kw[c] for c in cols],
    )


def image_by_path(path: str) -> sqlite3.Row | None:
    return query_one("SELECT * FROM images WHERE path=?", (str(path),))


def get_image(image_id: int) -> sqlite3.Row | None:
    return query_one("SELECT * FROM images WHERE id=?", (image_id,))


def list_images(batch: str = "", limit: int = 500) -> list[sqlite3.Row]:
    if batch:
        return query("SELECT * FROM images WHERE batch=? ORDER BY point_id, id LIMIT ?",
                     (batch, limit))
    return query("SELECT * FROM images ORDER BY id DESC LIMIT ?", (limit,))


def image_count() -> int:
    row = query_one("SELECT COUNT(*) AS n FROM images")
    return row["n"] if row else 0


# --------------------------------------------------------------------------
# 检测记录
# --------------------------------------------------------------------------
def add_detections(rows: list[dict]) -> None:
    if not rows:
        return
    ts = now()
    executemany(
        "INSERT INTO detections(image_id,task_id,point_id,cls_key,cls_name,conf,"
        "x1,y1,x2,y2,gt_idx,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (r["image_id"], r.get("task_id"), r.get("point_id", ""), r["cls_key"],
             r["cls_name"], r["conf"], r["x1"], r["y1"], r["x2"], r["y2"],
             int(r.get("gt_idx", 0) or 0), ts)
            for r in rows
        ],
    )


def detections_of_image(image_id: int) -> list[sqlite3.Row]:
    return query("SELECT * FROM detections WHERE image_id=? ORDER BY conf DESC",
                 (image_id,))


def detections_of_task(task_id: int) -> list[sqlite3.Row]:
    return query("SELECT * FROM detections WHERE task_id=? ORDER BY id", (task_id,))


def detections_sig(image_id: int) -> tuple[int, int]:
    """一张影像的检测记录指纹 `(条数, 最大 id)`，给"要不要重读"做廉价判据。

    只有条数是不够的：换一个置信度阈值重跑一遍检测，框的数量有可能碰巧还是那么多，
    位置却全变了，页面摆出来的就是上一套框。`p_detect._store` 是**整体替换**（先
    DELETE 再 INSERT），id 只增不减，所以最大 id 变了就说明这套记录是新写的那份。
    """
    r = query_one("SELECT COUNT(*) AS n, COALESCE(MAX(id), 0) AS m "
                  "FROM detections WHERE image_id=?", (image_id,))
    return (int(r["n"]), int(r["m"])) if r else (0, 0)


# --------------------------------------------------------------------------
# 分割量化记录
# --------------------------------------------------------------------------
def add_segment(**kw: Any) -> int:
    kw.setdefault("created_at", now())
    cols = ["image_id", "task_id", "point_id", "cls_key", "cls_name", "batch",
            "px_area", "area_cm2", "perimeter_px", "max_width_mm",
            "length_mm", "avg_width_mm", "area_ratio", "crack_count", "conf",
            "captured_at", "created_at"]
    cols = [c for c in cols if c in kw]
    return execute(
        f"INSERT INTO segments({','.join(cols)}) VALUES({','.join('?' for _ in cols)})",
        [kw[c] for c in cols],
    )


def segments_of_task(task_id: int) -> list[sqlite3.Row]:
    return query("SELECT * FROM segments WHERE task_id=? ORDER BY id", (task_id,))


def distinct_points() -> list[str]:
    rows = query(
        "SELECT DISTINCT point_id FROM segments WHERE point_id<>'' ORDER BY point_id"
    )
    return [r["point_id"] for r in rows]


def timeseries(point_id: str, cls_key: str = "") -> list[sqlite3.Row]:
    """取某点位（可限定病害类别）的历次量化记录，按批次、时间排序。"""
    sql = ("SELECT * FROM segments WHERE point_id=?"
           " AND area_cm2>0")
    params: list[Any] = [point_id]
    if cls_key:
        sql += " AND cls_key=?"
        params.append(cls_key)
    sql += " ORDER BY batch, captured_at, id"
    return query(sql, params)


def timeseries_overview() -> list[sqlite3.Row]:
    """按 点位×病害类别×批次 聚合，取每批平均面积，供趋势总览图使用。

    `length` 一并聚合出来供趋势图切换「裂缝长度」用。取 MAX 与 `width` 同口径：
    同一批次的同一类别下可能有多条记录（多点位复拍），取该批次的最大值，比取平均
    更能反映"最不利的那一道缝"。area 用 AVG 是历史口径，没跟着改——改了历史曲线
    的数值会整体位移，跨批次比较就断了。

    `area_max` 是为报告的"最不利单处面积"补的：趋势图要的是批内平均值，报告要的
    是这一批里最大的那一道缝，两者不能混用。它是**追加列**，既有调用方按列名取数，
    不受影响。
    """
    return query(
        "SELECT point_id, cls_key, cls_name, batch,"
        " COUNT(*) AS n, AVG(area_cm2) AS area, MAX(area_cm2) AS area_max,"
        " MAX(max_width_mm) AS width, MAX(length_mm) AS length"
        " FROM segments WHERE area_cm2>0"
        " GROUP BY point_id, cls_key, batch ORDER BY point_id, cls_key, batch"
    )


def point_cls_detail() -> list[sqlite3.Row]:
    """按 点位×病害类别 汇总检测记录，供报告的点位明细表使用。

    **检出数与量测值要分两次取，不能 JOIN 成一条 SQL。** 两侧的粒度天然不同：

    · detections 决定"检出多少处"。一次检测能在一张图上标出多个框（演示库 B05
      渗水 3 张图 9 个框），而像素级分割只对裂缝做（见 cfg.SEGMENT_CLASSES），
      另外六类在 segments 里**一条记录都没有**——按 segments 数，这六类会整类
      归零，报告里写成"七类病害里六类未检出"，读的人只会以为漏检。
    · segments 决定"量出来多大"，且只取 MAX 不取 SUM：SUM 会把同一批次的多条
      记录叠加成"总面积"，而研判那边 _trend_of 取的是批内最大面积，两者口径
      不一致时，表格里的面积就和定级依据对不上。

    所以这里只聚合检测侧；量化侧由 report._latest_quant() 从 timeseries_overview()
    取最新批次，两边的批次先后都按 cfg.BATCHES 的顺序。

    detections 表**没有 batch 列**，批次只能经 images.batch 取，所以 LEFT JOIN 是
    必须的；用 LEFT 而不是 INNER，是为了让 image_id 指向已删影像的历史记录不至于
    整行消失。
    """
    return query(
        "SELECT d.point_id, d.cls_key, d.cls_name,"
        " COUNT(*) AS det_n,"
        " COUNT(DISTINCT d.image_id) AS img_n,"
        " COUNT(DISTINCT i.batch) AS n_batch,"
        " GROUP_CONCAT(DISTINCT i.batch) AS batches,"
        " AVG(d.conf) AS avg_conf, MAX(d.conf) AS max_conf"
        " FROM detections d LEFT JOIN images i ON i.id = d.image_id"
        " WHERE d.point_id <> ''"
        " GROUP BY d.point_id, d.cls_key, d.cls_name"
        " ORDER BY d.point_id, det_n DESC, d.cls_key"
    )


# --------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------
def add_output(task_id: int | None, kind: str, path: str) -> int:
    return execute(
        "INSERT INTO outputs(task_id,kind,path,created_at) VALUES(?,?,?,?)",
        (task_id, kind, str(path), now()),
    )


def outputs_of_task(task_id: int) -> list[sqlite3.Row]:
    return query("SELECT * FROM outputs WHERE task_id=? ORDER BY id DESC", (task_id,))


def list_outputs(limit: int = 200) -> list[sqlite3.Row]:
    return query("SELECT * FROM outputs ORDER BY id DESC LIMIT ?", (limit,))


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def stat_summary() -> dict:
    def n(sql: str) -> int:
        row = query_one(sql)
        return row[0] if row else 0

    return {
        "users": n("SELECT COUNT(*) FROM users"),
        "images": n("SELECT COUNT(*) FROM images"),
        "tasks": n("SELECT COUNT(*) FROM tasks"),
        "detections": n("SELECT COUNT(*) FROM detections"),
        "segments": n("SELECT COUNT(*) FROM segments"),
        "outputs": n("SELECT COUNT(*) FROM outputs"),
        # 「点位」有两种读法，两个都留着。images 里数出来的是**实拍点位**——总览页的
        # 「实拍点位」卡片、网页版的「覆盖点位」、系统管理的「巡检点位」要的都是它；
        # segments 里数出来的才是**有量化记录的点位**。像素级分割收窄到裂缝之后两者
        # 不再相等（8 比 3），继续混用一个数，报告里就会写出"量化记录 12 条，覆盖
        # 8 个点位"这种自相矛盾的句子。
        "points": n("SELECT COUNT(DISTINCT point_id) FROM images WHERE point_id<>''"),
        "seg_points": n("SELECT COUNT(DISTINCT point_id) FROM segments"
                        " WHERE point_id<>''"),
    }
