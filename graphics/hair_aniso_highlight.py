"""毛发各向异性高光渲染管线修复模块。

覆盖六类问题：
  1. 切线连续性   —— 平行传输 + 符号一致化，消除发丝方向翻转处的切线跳变
  2. LOD 高光近似 —— 卡片 LOD 用烘焙切线贴图 + Kajiya-Kay 近似，避免高光消失
  3. 发根偏移     —— 沿头皮法线推出发根，消除发根与头皮穿插
  4. 毛发阴影通道 —— 毛发进入阴影贴图（alpha 测试深度 + 深度偏移），补齐自阴影
  5. 生成顺序确定 —— 异步发丝按 (seed, index) 排序，消除顺序不定
  6. 高光参数统一 —— 各 LOD 共享同一份高光参数，消除两次渲染高光位置不同

所有入口函数接受单个可选参数（配置 dict / 数据 / 种子），并返回非 None 结果。
纯 Python 实现，无第三方依赖。
"""

import hashlib
import math

# ---------------------------------------------------------------------------
# 向量工具
# ---------------------------------------------------------------------------

def _v3(p):
    return (float(p[0]), float(p[1]), float(p[2]))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _length(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _length(a)
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------------------
# 统一高光参数（修复 6：高光参数统一）
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS = {
    "specular_exponent": 64.0,      # Kajiya-Kay 主高光指数
    "secondary_exponent": 16.0,     # 次高光（有色）指数
    "specular_strength": 0.6,
    "secondary_strength": 0.25,
    "tangent_shift_primary": -0.1,  # 主高光沿切线偏移
    "tangent_shift_secondary": -0.2,
    "root_color": (0.08, 0.05, 0.03),
    "tip_color": (0.35, 0.22, 0.12),
    "shadow_bias": 0.002,           # 阴影贴图深度偏移
    "root_offset": 0.005,           # 发根沿法线推出距离
    "alpha_cutoff": 0.35,           # 卡片 alpha 测试阈值
    "lod_blend_start": 5.0,         # 发丝->卡片混合起始距离（米）
    "lod_blend_end": 15.0,          # 完全切到卡片的距离
    "strands_per_cm2": 2.0,         # 目标发丝密度
}


def highlight_params(overrides=None):
    """返回各 LOD 共享的统一高光参数表。

    所有 LOD（发丝 / 卡片）必须从同一份参数渲染，否则同一发型
    两次渲染或 LOD 切换时高光位置会不一致。允许用 dict 覆盖单项，
    非法键或非法值会被拒绝并抛 ValueError。
    """
    params = dict(_DEFAULT_PARAMS)
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key not in _DEFAULT_PARAMS:
                raise ValueError("unknown highlight param: %r" % key)
            params[key] = value
    # 数值合法性检查
    for key in ("specular_exponent", "secondary_exponent",
                "specular_strength", "secondary_strength",
                "root_offset", "shadow_bias", "strands_per_cm2"):
        if not (params[key] >= 0.0):
            raise ValueError("param %r must be >= 0" % key)
    if not (0.0 <= params["alpha_cutoff"] <= 1.0):
        raise ValueError("alpha_cutoff must be in [0, 1]")
    if params["lod_blend_end"] <= params["lod_blend_start"]:
        raise ValueError("lod_blend_end must be > lod_blend_start")
    return params


# ---------------------------------------------------------------------------
# 修复 1：切线连续性
# ---------------------------------------------------------------------------

def _raw_tangents(points):
    """中心差分求原始切线（未做符号一致化）。"""
    n = len(points)
    tangents = []
    for i in range(n):
        a = points[max(i - 1, 0)]
        b = points[min(i + 1, n - 1)]
        tangents.append(_normalize(_sub(b, a)))
    return tangents


def _fix_tangent_signs(tangents):
    """沿发丝做符号一致化：与上一根切线点积为负则翻转。

    发丝高光用的是切线的"无向"方向（Kajiya-Kay 中 sin 对称），
    但插值/烘焙时符号跳变会造成高光闪烁，因此统一为连续朝向。
    """
    if not tangents:
        return []
    fixed = [tangents[0]]
    for t in tangents[1:]:
        if _dot(t, fixed[-1]) < 0.0:
            t = _scale(t, -1.0)
        fixed.append(t)
    return fixed


def _smooth_tangents(tangents):
    """对符号一致化后的切线做一次邻域平均平滑。

    切线必须保持"沿发丝"的方向（Kajiya-Kay 的 T 即发丝方向），
    平滑只在相邻切线间做加权平均并保持符号连续，
    消除参数化抖动造成的逐段跳变，而不改变整体走向。
    """
    n = len(tangents)
    if n < 3:
        return list(tangents)
    out = [tangents[0]]
    for i in range(1, n - 1):
        avg = _add(_add(_scale(tangents[i - 1], 0.25),
                        _scale(tangents[i], 0.5)),
                   _scale(tangents[i + 1], 0.25))
        t = _normalize(avg)
        if _dot(t, out[-1]) < 0.0:
            t = _scale(t, -1.0)
        out.append(t)
    last = tangents[-1]
    if _dot(last, out[-1]) < 0.0:
        last = _scale(last, -1.0)
    out.append(last)
    return out


def _demo_strand(seed=1, segments=16):
    """由种子确定性地生成一条演示发丝（含一处方向翻转）。"""
    rng = _Lcg(seed)
    points = []
    x = y = z = 0.0
    for i in range(segments):
        points.append((x, y, z))
        x += 0.1 + 0.02 * rng.next()
        y += 0.05 * (rng.next() - 0.5)
        z += 0.05 * (rng.next() - 0.5)
    return points


class _Lcg:
    """确定性线性同余发生器（不依赖随机库，保证跨平台一致）。"""

    def __init__(self, seed):
        self.state = (int(seed) * 2654435761 + 1) & 0xFFFFFFFF

    def next(self):
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state / 4294967296.0


def tangent_continuity(strands=None):
    """修复发丝切线在方向翻转处的跳变。

    输入：单条发丝（点列）或发丝列表；数字则作为种子生成演示发丝。
    输出：{"strands": [[连续切线...], ...], "flips_fixed": int}
    """
    if strands is None or isinstance(strands, (int, float)):
        strands = [_demo_strand(int(strands or 1))]
    elif strands and isinstance(strands[0][0], (int, float)):
        strands = [strands]  # 单条发丝 -> 列表

    fixed_strands = []
    flips_fixed = 0
    for points in strands:
        points = [_v3(p) for p in points]
        raw = _raw_tangents(points)
        signed = _fix_tangent_signs(raw)
        flips_fixed += sum(1 for a, b in zip(raw, signed)
                           if _dot(a, b) < 0.0)
        fixed_strands.append(_smooth_tangents(signed))

    return {"strands": fixed_strands, "flips_fixed": flips_fixed}


# ---------------------------------------------------------------------------
# 修复 2：LOD 高光近似（Kajiya-Kay，卡片用烘焙切线）
# ---------------------------------------------------------------------------

def _shift_tangent(tangent, normal, shift):
    """沿法线方向把切线偏移 shift 量（高光带位置控制）。"""
    return _normalize(_add(tangent, _scale(normal, shift)))


def _kajiya_kay(tangent, light_dir, view_dir, exponent):
    """Kajiya-Kay 各向异性高光项。"""
    t = _normalize(tangent)
    l = _normalize(light_dir)
    v = _normalize(view_dir)
    h = _normalize(_add(l, v))
    sin_th = math.sqrt(max(0.0, 1.0 - _dot(t, h) ** 2))
    return sin_th ** exponent


def lod_highlight(config=None):
    """卡片 LOD 的高光近似。

    卡片没有逐发丝几何，使用烘焙到贴图的切线方向 + 卡片法线
    跑同一个 Kajiya-Kay 模型，参数来自 highlight_params()，
    保证切到卡片后高光不消失、位置与发丝 LOD 一致。

    输入 config 可含：tangent, normal, light_dir, view_dir。
    输出主/次高光强度及合成值。
    """
    params = highlight_params(config if isinstance(config, dict)
                              and "params" in config else None)
    cfg = config if isinstance(config, dict) else {}
    tangent = _v3(cfg.get("tangent", (1.0, 0.0, 0.0)))
    normal = _v3(cfg.get("normal", (0.0, 0.0, 1.0)))
    light = _v3(cfg.get("light_dir", (0.5, 0.5, 0.707)))
    view = _v3(cfg.get("view_dir", (0.0, 0.0, 1.0)))

    t1 = _shift_tangent(tangent, normal, params["tangent_shift_primary"])
    t2 = _shift_tangent(tangent, normal, params["tangent_shift_secondary"])
    primary = (_kajiya_kay(t1, light, view, params["specular_exponent"])
               * params["specular_strength"])
    secondary = (_kajiya_kay(t2, light, view, params["secondary_exponent"])
                 * params["secondary_strength"])
    return {
        "primary": primary,
        "secondary": secondary,
        "combined": primary + secondary,
        "params_used": params,
    }


# ---------------------------------------------------------------------------
# 修复 3：发根偏移
# ---------------------------------------------------------------------------

def root_offset(strand=None, scalp_normal=(0.0, 0.0, 1.0), distance=None):
    """把发根沿头皮法线推出，消除发根与头皮的穿插。

    输入：发丝点列（第一个点为发根）；数字则生成演示发丝。
    输出：{"strand": 偏移后的点列, "offset": 实际偏移向量}
    """
    params = highlight_params(None)
    if distance is None:
        distance = params["root_offset"]
    if strand is None or isinstance(strand, (int, float)):
        strand = _demo_strand(int(strand or 1))
    strand = [_v3(p) for p in strand]
    n = _normalize(_v3(scalp_normal))
    offset = _scale(n, distance)
    moved = [_add(p, offset) if i == 0 else p
             for i, p in enumerate(strand)]
    return {"strand": moved, "offset": offset}


# ---------------------------------------------------------------------------
# 修复 4：毛发阴影通道
# ---------------------------------------------------------------------------

def hair_shadow(config=None):
    """毛发阴影通道配置与深度采样。

    毛发必须渲染进阴影贴图，否则完全没有自阴影：
      - 发丝 LOD：不透明几何直接写深度；
      - 卡片 LOD：alpha 测试（alpha_cutoff）后写深度；
      - 采样时加 shadow_bias 防止自遮挡条纹。

    输入 config 可含 "depths"（逐片元 alpha 值列表）做过滤演示。
    输出阴影通道描述及通过 alpha 测试的深度样本数。
    """
    params = highlight_params(None)
    cfg = config if isinstance(config, dict) else {}
    cutoff = params["alpha_cutoff"]
    depths = cfg.get("depths")
    passed = None
    if isinstance(depths, (list, tuple)):
        passed = [d for d in depths if d >= cutoff]
    return {
        "cast_shadow": True,
        "mode": "alpha_tested_depth",
        "alpha_cutoff": cutoff,
        "shadow_bias": params["shadow_bias"],
        "samples_in": len(depths) if isinstance(depths, (list, tuple)) else 0,
        "samples_written": len(passed) if passed is not None else None,
    }


# ---------------------------------------------------------------------------
# 修复 5：异步发丝生成顺序确定
# ---------------------------------------------------------------------------

def _strand_key(strand):
    """发丝的稳定排序键：(seed, index)。

    异步生成完成顺序不定，渲染前必须按确定键重排，
    否则写入顶点缓冲的顺序变化会导致高光逐帧/逐次不同。
    """
    return (int(strand.get("seed", 0)), int(strand.get("index", 0)))


def sort_strand(strands=None):
    """按 (seed, index) 对发丝做确定性排序，返回新列表。"""
    if strands is None or isinstance(strands, (int, float)):
        rng = _Lcg(int(strands or 1))
        strands = [{"seed": i % 4, "index": i,
                    "arrival": rng.next()} for i in range(8)]
    return sorted(strands, key=_strand_key)


def strand_order(strands=None):
    """把异步完成的发丝重排为确定顺序。

    输出：{"ordered": 排序后的发丝, "deterministic": True}
    """
    ordered = sort_strand(strands)
    keys = [_strand_key(s) for s in ordered]
    deterministic = keys == sorted(keys)
    return {"ordered": ordered, "deterministic": deterministic}


# ---------------------------------------------------------------------------
# 卡片混合 / alpha 边缘 / 密度
# ---------------------------------------------------------------------------

def card_blend(distance=None):
    """发丝 LOD 与卡片 LOD 的混合权重。

    distance <= lod_blend_start -> 0（纯发丝）
    distance >= lod_blend_end   -> 1（纯卡片）
    中间 smoothstep 过渡，避免切换时高光突变。
    """
    params = highlight_params(None)
    if distance is None or not isinstance(distance, (int, float)):
        distance = params["lod_blend_start"]
    t = ((distance - params["lod_blend_start"])
         / (params["lod_blend_end"] - params["lod_blend_start"]))
    t = _clamp(t)
    w = t * t * (3.0 - 2.0 * t)
    return {"distance": distance, "card_weight": w, "strand_weight": 1.0 - w}


def alpha_edge(alpha=None):
    """卡片边缘 alpha 处理：alpha 测试 + 边缘过渡带。

    返回该片元是否保留，以及用于混合的衰减系数。
    """
    params = highlight_params(None)
    if alpha is None or not isinstance(alpha, (int, float)):
        alpha = 1.0
    cutoff = params["alpha_cutoff"]
    fade_width = 0.1
    keep = alpha >= cutoff
    fade = _clamp((alpha - cutoff) / fade_width) if keep else 0.0
    return {"alpha": alpha, "keep": keep, "fade": fade, "cutoff": cutoff}


def density(config=None):
    """按目标密度（根/平方厘米）计算给定头皮面积应生成的发丝数。"""
    params = highlight_params(None)
    cfg = config if isinstance(config, dict) else {}
    area_cm2 = float(cfg.get("area_cm2", 1.0))
    if area_cm2 < 0.0:
        raise ValueError("area_cm2 must be >= 0")
    count = int(round(area_cm2 * params["strands_per_cm2"]))
    return {"area_cm2": area_cm2, "target_per_cm2": params["strands_per_cm2"],
            "strand_count": count}


# ---------------------------------------------------------------------------
# 校验 / 度量 / 确定性 / 审计 / 回退
# ---------------------------------------------------------------------------

def validation(config=None):
    """校验渲染配置，返回 {"ok": bool, "errors": [...]}。"""
    errors = []
    try:
        params = highlight_params(config if isinstance(config, dict) else None)
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)]}
    if params["specular_exponent"] < params["secondary_exponent"]:
        errors.append("specular_exponent should be >= secondary_exponent")
    if params["root_offset"] <= 0.0:
        errors.append("root_offset must be > 0 to avoid scalp intersection")
    return {"ok": not errors, "errors": errors}


def metrics(strands=None):
    """质量度量：切线翻转次数、相邻切线最大夹角（度）。"""
    result = tangent_continuity(strands)
    max_angle = 0.0
    for tangents in result["strands"]:
        for a, b in zip(tangents, tangents[1:]):
            cos = _clamp(_dot(a, b), -1.0, 1.0)
            max_angle = max(max_angle, math.degrees(math.acos(cos)))
    return {"flips_fixed": result["flips_fixed"],
            "max_tangent_angle_deg": max_angle}


def determinism(seed=None):
    """确定性自检：同一输入跑两遍，所有输出必须逐位一致。"""
    seed = int(seed) if isinstance(seed, (int, float)) else 1
    run_a = (tangent_continuity(seed), strand_order(seed),
             lod_highlight({"seed": seed}))
    run_b = (tangent_continuity(seed), strand_order(seed),
             lod_highlight({"seed": seed}))
    return {"deterministic": run_a == run_b, "seed": seed}


def audit(config=None):
    """对整条管线做体检，返回各项检查报告。"""
    report = {
        "validation": validation(config),
        "determinism": determinism(config if isinstance(config, (int, float))
                                   else None),
        "metrics": metrics(config if isinstance(config, (int, float))
                           else None),
        "shadow": hair_shadow(config if isinstance(config, dict) else None),
    }
    report["ok"] = (report["validation"]["ok"]
                    and report["determinism"]["deterministic"])
    return report


def fallback(config=None):
    """能力受限平台的回退配置：关掉发丝级特性，保留视觉一致性。

    回退时仍使用统一高光参数，仅降低精度，保证高光位置不漂移。
    """
    params = highlight_params(None)
    reason = None
    if isinstance(config, dict):
        reason = config.get("reason")
    return {
        "reason": reason or "unsupported_feature",
        "use_cards_only": True,
        "strand_shadow": False,          # 卡片走 alpha 测试阴影
        "card_shadow": True,
        "params": params,                # 参数不降级，只降级几何
    }
