"""Pure-python geo helpers: pixel -> bearing, ray -> footprint intersection, agreement."""
import math

EARTH_R = 6371000.0


def local_xy(lat, lng, lat0, lng0):
    """Metres east/north of (lat0,lng0). Good enough within a few hundred metres."""
    x = math.radians(lng - lng0) * EARTH_R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_R
    return x, y


def from_xy(x, y, lat0, lng0):
    lat = lat0 + math.degrees(y / EARTH_R)
    lng = lng0 + math.degrees(x / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lng


def dist_m(a, b):
    x, y = local_xy(b[0], b[1], a[0], a[1])
    return math.hypot(x, y)


def bearing_to(lat1, lng1, lat2, lng2):
    """Compass bearing (deg, 0=N, clockwise) from point 1 to point 2."""
    x, y = local_xy(lat2, lng2, lat1, lng1)
    return math.degrees(math.atan2(x, y)) % 360


def pixel_to_bearing(x_norm, heading, fov_deg):
    """Horizontal pixel position (0..1, 0=left) in a pinhole image of width fov -> compass bearing."""
    half = math.tan(math.radians(fov_deg) / 2)
    offset = math.degrees(math.atan((x_norm - 0.5) * 2 * half))
    return (heading + offset) % 360


def pano_pixel_to_bearing(x_norm, compass_angle):
    """Equirectangular 360 pano: x=0.5 looks along compass_angle."""
    return (compass_angle + (x_norm - 0.5) * 360) % 360


def ray_polygon_intersection(cam, bearing, polygon):
    """
    cam: (lat,lng). bearing: deg. polygon: list of (lat,lng).
    Returns nearest intersection (lat,lng, distance_m) or None.
    """
    lat0, lng0 = cam
    dx, dy = math.sin(math.radians(bearing)), math.cos(math.radians(bearing))
    pts = [local_xy(p[0], p[1], lat0, lng0) for p in polygon]
    best = None
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-9:
            continue
        # cam + t*d = p1 + u*e
        t = ((x1 * ey - y1 * ex)) / denom
        u = ((x1 * dy - y1 * dx)) / denom
        if t > 0 and 0 <= u <= 1:
            if best is None or t < best[2]:
                best = (t * dx, t * dy, t)
    if best is None:
        return None
    lat, lng = from_xy(best[0], best[1], lat0, lng0)
    return lat, lng, best[2]


def centroid(polygon):
    la = sum(p[0] for p in polygon) / len(polygon)
    ln = sum(p[1] for p in polygon) / len(polygon)
    return la, ln


def point_in_polygon(pt, polygon):
    x, y = pt[1], pt[0]
    inside = False
    n = len(polygon)
    for i in range(n):
        y1, x1 = polygon[i]
        y2, x2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def agree(points, tol_m=5.0):
    """points: list of (lat,lng). Returns (mean_point, max_spread_m) if all within tol, else None."""
    if len(points) < 2:
        return None
    c = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    spread = max(dist_m(c, p) for p in points)
    return (c, spread) if spread <= tol_m else None


def ray_ray_intersection(cam1, b1, cam2, b2):
    """Where two bearing rays cross. Returns (lat, lng, range1_m, range2_m) or None if parallel/behind."""
    lat0, lng0 = cam1
    x2, y2 = local_xy(cam2[0], cam2[1], lat0, lng0)
    d1 = (math.sin(math.radians(b1)), math.cos(math.radians(b1)))
    d2 = (math.sin(math.radians(b2)), math.cos(math.radians(b2)))
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-6:
        return None
    t1 = (x2 * d2[1] - y2 * d2[0]) / denom
    t2 = (x2 * d1[1] - y2 * d1[0]) / denom
    if t1 <= 0 or t2 <= 0:
        return None
    lat, lng = from_xy(t1 * d1[0], t1 * d1[1], lat0, lng0)
    return lat, lng, t1, t2


def ray_nearest_point(cam, bearing, target, max_m=15.0):
    """Point on the ray from cam along bearing that is closest to target. None if farther than max_m or behind."""
    tx, ty = local_xy(target[0], target[1], cam[0], cam[1])
    dx, dy = math.sin(math.radians(bearing)), math.cos(math.radians(bearing))
    t = tx * dx + ty * dy
    if t <= 0:
        return None
    px, py = t * dx, t * dy
    if math.hypot(px - tx, py - ty) > max_m:
        return None
    lat, lng = from_xy(px, py, cam[0], cam[1])
    return lat, lng, t


def snap_to_polygon_edge(pt, polygon, max_m=8.0):
    """Nearest point on the polygon outline to pt; returns (lat,lng,dist_m) or None if farther than max_m."""
    lat0, lng0 = pt
    best = None
    n = len(polygon)
    xy = [local_xy(p[0], p[1], lat0, lng0) for p in polygon]
    for i in range(n):
        (x1, y1), (x2, y2) = xy[i], xy[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0 if L2 == 0 else max(0, min(1, -(x1 * dx + y1 * dy) / L2))
        px, py = x1 + t * dx, y1 + t * dy
        d = math.hypot(px, py)
        if best is None or d < best[2]:
            best = (px, py, d)
    if best is None or best[2] > max_m:
        return None
    lat, lng = from_xy(best[0], best[1], lat0, lng0)
    return lat, lng, best[2]


def snap_to_facing_edge(pt, polygon, cams, max_m=10.0):
    """Like snap_to_polygon_edge, but only onto walls that FACE at least one of the cameras
    that saw the door (the camera is on the outside of that wall). A door seen from the street
    can't be on the far wall, even when the far wall happens to be nearer to a misplaced point.
    Returns (lat, lng, dist_m) or None."""
    lat0, lng0 = pt
    xy = [local_xy(p[0], p[1], lat0, lng0) for p in polygon]
    n = len(xy)
    area2 = sum(xy[i][0] * xy[(i + 1) % n][1] - xy[(i + 1) % n][0] * xy[i][1] for i in range(n))
    ccw = area2 > 0
    cxy = [local_xy(c[0], c[1], lat0, lng0) for c in cams]
    best = None
    for i in range(n):
        (x1, y1), (x2, y2) = xy[i], xy[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        L2 = ex * ex + ey * ey
        if L2 == 0:
            continue
        nx, ny = (ey, -ex) if ccw else (-ey, ex)          # outward normal
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        nl = math.hypot(nx, ny)
        if not any(((cx - mx) * nx + (cy - my) * ny) / (nl * (math.hypot(cx - mx, cy - my) or 1)) > 0.35
                   for cx, cy in cxy):
            continue                                       # wall faces away from (or edge-on to) every camera
        t = max(0.02, min(0.98, -(x1 * ex + y1 * ey) / L2))   # stay off the exact corners
        px, py = x1 + t * ex, y1 + t * ey
        d = math.hypot(px, py)
        if best is not None and d >= best[2]:
            continue
        plat, plng = from_xy(px, py, lat0, lng0)
        if not _seen_by_any((plat, plng), polygon, cams):
            continue                                       # the house itself blocks the view
        best = (px, py, d)
    if best is None or best[2] > max_m:
        return None
    lat, lng = from_xy(best[0], best[1], lat0, lng0)
    return lat, lng, best[2]


def _seen_by_any(pt, polygon, cams, tol_m=0.75):
    """True if some camera's line of sight to pt reaches it before hitting another wall."""
    for cam in cams:
        d = dist_m(cam, pt)
        hit = ray_polygon_intersection(cam, bearing_to(cam[0], cam[1], pt[0], pt[1]), polygon)
        if hit and abs(hit[2] - d) <= tol_m:
            return True
    return False
