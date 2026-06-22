from controller import Robot
import asyncio
import websockets
import json
import base64
import threading
import numpy as np
import cv2
import math

TIME_STEP = 32

MOTOR_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

GRIPPER_MOTORS = [
    "ROBOTIQ 2F-140 Gripper::left finger joint",
    "ROBOTIQ 2F-140 Gripper::right finger joint",
]

# ── POSICIONES DE CONTROL TRADICIONALES ────────────────
HOME_POSITION    = [0.0,   -1.5708, 0.0,  -1.5708, 0.0, 0.0]
OBSERVE_POSITION = [1.508, -0.377,  0.0,  -1.571,  0.0, 1.382]
DROP_POSITION    = [-1.5,  -0.8,    0.6,  -1.571,  0.0, 0.0]

GRIPPER_OPEN  = 0.0
GRIPPER_CLOSE = 0.55
MOTOR_SPEED   = 2.0   
GRIPPER_SPEED = 2.0

# ── PARÁMETROS ÓPTICOS DE LA CÁMARA ────────────────────
CAM_WIDTH  = 320
CAM_HEIGHT = 240
CAM_CX     = CAM_WIDTH  / 2.0
CAM_CY     = CAM_HEIGHT / 2.0

# Dimensiones cinemáticas estrictas del UR5e (Denavit-Hartenberg)
UR5E_D1 = 0.1625
UR5E_A2 = 0.425
UR5E_A3 = 0.3922
UR5E_D4 = 0.1333
UR5E_D5 = 0.0997
UR5E_D6 = 0.0996

ROBOT_BASE = np.array([0.04, 0.07, 0.72])

# Centro por defecto seguro en la mesa para realizar las figuras geométricas
# Se ubica a una altura Z fija (aprox sobre la mesa)
DRAW_CENTER_BASE = np.array([0.0, 0.85, 0.76])

# ── FILTROS HSV EXTRA RESTRINGIDOS CONTRA REFLEJOS DE LA MESA ──
BLUE_HSV_LOW  = np.array([100, 210, 140])
BLUE_HSV_HIGH = np.array([130, 255, 255])

RED_HSV_LOW1  = np.array([0,   210, 140])
RED_HSV_HIGH1 = np.array([7,   255, 255])
RED_HSV_LOW2  = np.array([173, 210, 140])
RED_HSV_HIGH2 = np.array([180, 255, 255])

MIN_AREA = 350  

robot_state = {
    "status": "idle",
    "camera_frame_b64": "",
    "detections": {},
}
command_queue = []
frame_lock = threading.Lock()
command_lock = threading.Lock()


# ── PROCESAMIENTO DE IMAGEN REFORZADO ──────────────────
def detect_objects(frame_bgr):
    if frame_bgr is None: return {}
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    detections = {}
    kernel = np.ones((5, 5), np.uint8)

    # Buscar Cilindro Azul
    mask_blue = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    cnts_b, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_b:
        c = max(cnts_b, key=cv2.contourArea)
        if cv2.contourArea(c) > MIN_AREA:
            M = cv2.moments(c)
            if M["m00"] > 0:
                detections["blue"] = {"x": int(M["m10"]/M["m00"]), "y": int(M["m01"]/M["m00"])}

    # Buscar Cubo Rojo
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, RED_HSV_LOW1, RED_HSV_HIGH1),
        cv2.inRange(hsv, RED_HSV_LOW2, RED_HSV_HIGH2)
    )
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    cnts_r, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_r:
        c = max(cnts_r, key=cv2.contourArea)
        if cv2.contourArea(c) > MIN_AREA:
            M = cv2.moments(c)
            if M["m00"] > 0:
                detections["red"] = {"x": int(M["m10"]/M["m00"]), "y": int(M["m01"]/M["m00"])}
                
    return detections

def draw_detections(frame_bgr, detections):
    if frame_bgr is None: return None
    vis = frame_bgr.copy()
    for color, label, col in [("blue", "CILINDRO AZUL", (255, 120, 10)), ("red", "CUBO ROJO", (20, 20, 240))]:
        if color in detections:
            d = detections[color]
            cv2.circle(vis, (d["x"], d["y"]), 5, col, -1)
            cv2.rectangle(vis, (d["x"]-12, d["y"]-12), (d["x"]+12, d["y"]+12), col, 2)
    return vis


# ── PLANO COORDENADO RE-CALIBRADO ──
def pixel_to_world_calibrated(px, py, color):
    if color == "red":
        base_x = 0.266421
        base_y = 0.86939
        base_z = 0.758924
        dx = (px - 213) * 0.00222
        dy = (py - 120) * 0.00225
        return np.array([base_x + dx, base_y - dy, base_z])
        
    elif color == "blue":
        base_x = -0.253759
        base_y = 0.872754
        base_z = 0.759471
        dx = (px - 106) * 0.00222
        dy = (py - 120) * 0.00225
        return np.array([base_x + dx, base_y - dy, base_z])
        
    return DRAW_CENTER_BASE.copy()


# ── CINEMÁTICA INVERSA CORREGIDA ──
def ik_ur5e(tx, ty, tz):
    dx = tx - ROBOT_BASE[0]
    dy = ty - ROBOT_BASE[1]
    dz = tz - ROBOT_BASE[2]

    pan = math.atan2(dy, dx)
    r = math.sqrt(dx**2 + dy**2)
    
    h = dz - UR5E_D1 - UR5E_D6
    reach = math.sqrt(r**2 + h**2)
    max_reach = UR5E_A2 + UR5E_A3 - 0.005 

    if reach > max_reach:
        scale = max_reach / reach
        r *= scale
        h *= scale

    cos_elbow = (r**2 + h**2 - UR5E_A2**2 - UR5E_A3**2) / (2 * UR5E_A2 * UR5E_A3)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow = math.acos(cos_elbow)

    alpha = math.atan2(h, r)
    beta = math.acos(max(-1.0, min(1.0, (r**2 + h**2 + UR5E_A2**2 - UR5E_A3**2) / (2 * UR5E_A2 * (max_reach if reach > max_reach else reach)))))

    lift = -(alpha + beta)
    w1 = -(math.pi/2 + lift + elbow)

    return [round(pan, 4), round(lift, 4), round(elbow, 4), round(w1, 4), round(-math.pi/2, 4), round(0.0, 4)]


# ── MOVIMIENTOS Y ACCIONES SÍNCRONAS ───────────────────
def wait_steps(n):
    for _ in range(n):
        robot.step(TIME_STEP)

def move_to_position(motors, target, steps=35):
    if target is None: return
    current = [m.getTargetPosition() for m in motors]
    for s in range(steps):
        t = (s + 1) / steps
        t = t * t * (3 - 2 * t)
        for i, m in enumerate(motors):
            m.setPosition(current[i] + (target[i] - current[i]) * t)
        robot.step(TIME_STEP)
    wait_steps(5)

def set_gripper(gm, pos):
    for m in gm:
        m.setPosition(pos)
    wait_steps(15)

# ── NUEVA FUNCIÓN PARA SEGUIMIENTO DE TRAYECTORIAS PUNTO A PUNTO ──
def follow_path(motors, path_points, steps_per_segment=8):
    """ Mueve el brazo robótico a través de una lista de coordenadas cartesianas """
    for pt in path_points:
        joints = ik_ur5e(pt[0], pt[1], pt[2])
        if joints:
            # Movimiento directo con pocos pasos para transiciones fluidas de trayectoria
            for i, m in enumerate(motors):
                m.setPosition(joints[i])
            wait_steps(steps_per_segment)


# ── NUEVAS FUNCIONES DE GENERACIÓN DE TRAYECTORIAS GEOMÉTRICAS ──
# ── NUEVAS FUNCIONES DE GENERACIÓN DE TRAYECTORIAS CON Z ESTRICTAMENTE FIJO ──
def execute_draw_circle(motors):
    print("[Draw] Iniciando trayectoria circular a Z fijo...")
    center = DRAW_CENTER_BASE.copy()
    radius = 0.06
    points = []
    
    # Generar 24 puntos del círculo en el plano X-Y manteniendo Z fijo
    for i in range(25):
        angle = (i * 2 * math.pi) / 24
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append([x, y, center[2]]) # Z fijo en el plano de la mesa
        
    # Posicionamiento inicial manteniendo la misma altura Z
    first_joints = ik_ur5e(points[0][0], points[0][1], points[0][2])
    move_to_position(motors, first_joints, steps=30)
    
    # Dibujar figura de forma continua
    follow_path(motors, points, steps_per_segment=6)
    print("[Draw] ¡Círculo completado!")

def execute_draw_square(motors):
    print("[Draw] Iniciando trayectoria cuadrada a Z fijo...")
    center = DRAW_CENTER_BASE.copy()
    side = 0.10
    half = side / 2.0
    
    # Esquinas del cuadrado en secuencia continua manteniendo Z fijo
    corners = [
        [center[0] - half, center[1] - half, center[2]],
        [center[0] + half, center[1] - half, center[2]],
        [center[0] + half, center[1] + half, center[2]],
        [center[0] - half, center[1] + half, center[2]],
        [center[0] - half, center[1] - half, center[2]]
    ]
    
    # Interpolar puntos entre esquinas para un avance lineal fluido
    points = []
    for i in range(len(corners)-1):
        p1, p2 = np.array(corners[i]), np.array(corners[i+1])
        for t in np.linspace(0, 1, 6):
            points.append(p1 + (p2 - p1) * t)
            
    first_joints = ik_ur5e(points[0][0], points[0][1], points[0][2])
    move_to_position(motors, first_joints, steps=30)
    
    follow_path(motors, points, steps_per_segment=10)
    print("[Draw] ¡Cuadrado completado!")

def execute_draw_star(motors):
    print("[Draw] Iniciando trayectoria de Estrella a Z fijo...")
    center = DRAW_CENTER_BASE.copy()
    r_outer = 0.07
    r_inner = 0.03
    points = []
    
    # Generar los 10 vértices de la estrella manteniendo Z fijo
    for i in range(11):
        angle = (i * math.pi) / 5 - (math.pi / 2) 
        r = r_outer if i % 2 == 0 else r_inner
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        points.append([x, y, center[2]])
        
    first_joints = ik_ur5e(points[0][0], points[0][1], points[0][2])
    move_to_position(motors, first_joints, steps=30)
    
    follow_path(motors, points, steps_per_segment=12)
    print("[Draw] ¡Estrella completada!")


def execute_pick(motors, gripper_motors, color):
    move_to_position(motors, OBSERVE_POSITION, steps=25)
    wait_steps(15)

    with frame_lock:
        dets = robot_state["detections"].copy()

    if color not in dets:
        print(f"[Pick] No se visualiza la pieza {color} en la mesa.")
        return False

    obj_world = pixel_to_world_calibrated(dets[color]["x"], dets[color]["y"], color)
    print(f"[Pick] Destino validado: X={round(obj_world[0],4)}, Y={round(obj_world[1],4)}, Z={round(obj_world[2],4)}")

    pre_target = obj_world.copy()
    pre_target[2] += 0.08  

    pre_joints = ik_ur5e(pre_target[0], pre_target[1], pre_target[2])
    grasp_joints = ik_ur5e(obj_world[0], obj_world[1], obj_world[2])

    if not pre_joints or not grasp_joints:
        print("[Pick] Error: El punto calculado excede los límites geométricos reales.")
        return False

    set_gripper(gripper_motors, GRIPPER_OPEN)
    move_to_position(motors, pre_joints, steps=35)     
    move_to_position(motors, grasp_joints, steps=18)   
    set_gripper(gripper_motors, GRIPPER_CLOSE)         
    move_to_position(motors, pre_joints, steps=20)     
    print(f"[Pick] ¡{color.upper()} tomado con éxito!")
    return True

def execute_place(motors, gripper_motors):
    move_to_position(motors, DROP_POSITION, steps=35)
    set_gripper(gripper_motors, GRIPPER_OPEN)
    move_to_position(motors, HOME_POSITION, steps=30)

def process_action(action, motors, gripper_motors):
    robot_state["status"] = "moving"
    if action == "pick_blue":
        execute_pick(motors, gripper_motors, "blue")
    elif action == "pick_red":
        execute_pick(motors, gripper_motors, "red")
    elif action == "place":
        execute_place(motors, gripper_motors)
    elif action == "observe":
        move_to_position(motors, OBSERVE_POSITION, steps=30)
    elif action == "home":
        set_gripper(gripper_motors, GRIPPER_OPEN)
        move_to_position(motors, HOME_POSITION, steps=30)
    # Habilitación de las nuevas acciones geométricas
    elif action == "draw_circle":
        execute_draw_circle(motors)
        move_to_position(motors, OBSERVE_POSITION, steps=25)
    elif action == "draw_square":
        execute_draw_square(motors)
        move_to_position(motors, OBSERVE_POSITION, steps=25)
    elif action == "draw_star":
        execute_draw_star(motors)
        move_to_position(motors, OBSERVE_POSITION, steps=25)
        
    robot_state["status"] = "idle"

def interpret_command(text):
    t = text.lower().strip()
    if any(w in t for w in ["circulo", "circle", "redondo"]): return "draw_circle"
    if any(w in t for w in ["cuadrado", "square", "rectangulo"]): return "draw_square"
    if any(w in t for w in ["estrella", "star"]): return "draw_star"
    if any(w in t for w in ["suelt", "deja", "soltar", "drop", "place"]): return "place"
    if any(w in t for w in ["inicio", "home", "reset"]): return "home"
    if any(w in t for w in ["azul", "blue"]): return "pick_blue"
    if any(w in t for w in ["rojo", "red"]): return "pick_red"
    return "observe"


# ── SERVIDOR WEBSOCKET ASÍNCRONO ───────────────────────
async def handle_client(websocket):
    try:
        async for message in websocket:
            data = json.loads(message)
            if data.get("type") == "command":
                text = data.get("text", "")
                action = interpret_command(text)
                await websocket.send(json.dumps({"type": "decision", "action": action, "status": "accepted"}))
                with command_lock:
                    if robot_state["status"] == "idle":
                        command_queue.append(action)
            elif data.get("type") == "request_frame":
                with frame_lock:
                    frame = robot_state["camera_frame_b64"]
                    dets = robot_state["detections"].copy()
                    status = robot_state["status"]
                await websocket.send(json.dumps({"type": "frame", "data": frame, "detections": dets, "status": status}))
    except websockets.exceptions.ConnectionClosed:
        pass

def start_websocket_server():
    loop = asyncio.new_event_loop()
    async def main():
        async with websockets.serve(handle_client, "localhost", 8765):
            await asyncio.Future()
    loop.run_until_complete(main())


# ── INICIALIZACIÓN DISPOSITIVOS WEBOTS ─────────────────
robot = Robot()
motors = [robot.getDevice(n) for n in MOTOR_NAMES]
gripper_mots = [robot.getDevice(n) for n in GRIPPER_MOTORS]
camera = robot.getDevice("wrist_camera")

for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(MOTOR_SPEED)
    ps = m.getPositionSensor()
    if ps: ps.enable(TIME_STEP)

for m in gripper_mots:
    m.setPosition(GRIPPER_OPEN)
    m.setVelocity(GRIPPER_SPEED)

camera.enable(TIME_STEP)

for i, m in enumerate(motors):
    m.setPosition(OBSERVE_POSITION[i])

ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
ws_thread.start()


# ── LOOP PRINCIPAL SEGURO ──────────────────────────────
while robot.step(TIME_STEP) != -1:
    try:
        raw_image = camera.getImage()
        if raw_image:
            w, h = camera.getWidth(), camera.getHeight()
            frame_rgba = np.frombuffer(raw_image, dtype=np.uint8).reshape((h, w, 4)).copy()
            bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)
            
            dets = detect_objects(bgr)
            vis = draw_detections(bgr, dets)
            
            if vis is not None:
                _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf).decode("utf-8")
                
                with frame_lock:
                    robot_state["camera_frame_b64"] = b64
                    robot_state["detections"] = dets
    except Exception as e:
        pass

    with command_lock:
        if command_queue and robot_state["status"] == "idle":
            action = command_queue.pop(0)
            threading.Thread(target=process_action, args=(action, motors, gripper_mots), daemon=True).start()