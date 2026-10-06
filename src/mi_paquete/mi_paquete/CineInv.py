import numpy as np
import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState


# ============================================================
# PARÁMETROS
# ============================================================

alpha_step = 0.15
iterations = 1000
tolerance = 1e-4


# ============================================================
# LÍMITES DE LAS ARTICULACIONES DEL TM5-900
# Según el URDF nominal
# ============================================================

joint_limits_lower = np.array([
    -4.71238898,   # joint_1
    -3.14159265,   # joint_2
    -2.70526034,   # joint_3
    -3.14159265,   # joint_4
    -3.14159265,   # joint_5
    -4.71238898    # joint_6
])

joint_limits_upper = np.array([
     4.71238898,
     3.14159265,
     2.70526034,
     3.14159265,
     3.14159265,
     4.71238898
])


# ============================================================
# MATRICES HOMOGÉNEAS
# ============================================================

def Rx(angle):

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0, 0, 1]
    ], dtype=float)


def Ry(angle):

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=float)


def Rz(angle):

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ], dtype=float)


def Tx(distance):

    T = np.eye(4)
    T[0, 3] = distance

    return T


def Ty(distance):

    T = np.eye(4)
    T[1, 3] = distance

    return T


def Tz(distance):

    T = np.eye(4)
    T[2, 3] = distance

    return T


# ============================================================
# ROTACIÓN RPY DEL URDF
# RPY = roll, pitch, yaw
# ============================================================

def RPY(roll, pitch, yaw):

    return Rx(roll) @ Ry(pitch) @ Rz(yaw)


# ============================================================
# CINEMÁTICA DIRECTA SEGÚN EL URDF DEL TM5-900
# ============================================================

def forward_kinematics_urdf(q):

    q1, q2, q3, q4, q5, q6 = q

    T = np.eye(4)

    # --------------------------------------------------------
    # joint_1
    # origin:
    # xyz = (0, 0, 0.1452)
    # rpy = (0, 0, 0)
    # axis = Z
    # --------------------------------------------------------

    T = T @ Tz(0.1452)
    T = T @ Rz(q1)


    # --------------------------------------------------------
    # joint_2
    # origin:
    # xyz = (0, 0, 0)
    # rpy = (-pi/2, -pi/2, 0)
    # axis = Z
    # --------------------------------------------------------

    T = T @ RPY(
        -np.pi / 2,
        -np.pi / 2,
        0.0
    )

    T = T @ Rz(q2)


    # --------------------------------------------------------
    # joint_3
    # origin:
    # xyz = (0.429, 0, 0)
    # rpy = (0, 0, 0)
    # axis = Z
    # --------------------------------------------------------

    T = T @ Tx(0.4290)
    T = T @ Rz(q3)


    # --------------------------------------------------------
    # joint_4
    # origin:
    # xyz = (0.4115, 0, -0.1223)
    # rpy = (0, 0, pi/2)
    # axis = Z
    # --------------------------------------------------------

    T = T @ Tx(0.4115)
    T = T @ Tz(-0.1223)

    T = T @ Rz(np.pi / 2)

    T = T @ Rz(q4)


    # --------------------------------------------------------
    # joint_5
    # origin:
    # xyz = (0, -0.106, 0)
    # rpy = (pi/2, 0, 0)
    # axis = Z
    # --------------------------------------------------------

    T = T @ Ty(-0.1060)

    T = T @ Rx(np.pi / 2)

    T = T @ Rz(q5)


    # --------------------------------------------------------
    # joint_6
    # origin:
    # xyz = (0, -0.11315, 0)
    # rpy = (pi/2, 0, 0)
    # axis = Z
    # --------------------------------------------------------

    T = T @ Ty(-0.11315)

    T = T @ Rx(np.pi / 2)

    T = T @ Rz(q6)


    return T


# ============================================================
# POSICIÓN DEL EFECTOR
# ============================================================

def get_position(q):

    T = forward_kinematics_urdf(q)

    return T[:3, 3]


# ============================================================
# JACOBIANO NUMÉRICO
# ============================================================

def numerical_jacobian(q):

    epsilon = 1e-6

    J = np.zeros((3, 6))

    current_position = get_position(q)

    for i in range(6):

        q_temp = q.copy()

        q_temp[i] += epsilon

        new_position = get_position(q_temp)

        J[:, i] = (
            new_position - current_position
        ) / epsilon

    return J


# ============================================================
# LIMITAR ARTICULACIONES
# ============================================================

def clamp_joints(q):

    return np.clip(
        q,
        joint_limits_lower,
        joint_limits_upper
    )


# ============================================================
# NODO ROS 2
# ============================================================

class InverseKinematics(Node):

    def __init__(self):

        super().__init__(
            'inverse_kinematics'
        )

        # ----------------------------------------------------
        # SUSCRIPTOR DEL OBJETIVO
        # ----------------------------------------------------

        self.sub = self.create_subscription(
            Point,
            '/target',
            self.solve,
            10
        )

        # ----------------------------------------------------
        # PUBLICADOR PARA RVIZ
        # ----------------------------------------------------

        self.pub = self.create_publisher(
            JointState,
            '/joint_states',
            10
        )

        # ----------------------------------------------------
        # CONFIGURACIÓN INICIAL
        # ----------------------------------------------------

        self.current_q_urdf = np.array([
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ])

        # ----------------------------------------------------
        # PUBLICACIÓN A 20 Hz
        # ----------------------------------------------------

        self.timer = self.create_timer(
            0.05,
            self.publish_joint_states
        )

        self.get_logger().info(
            'Nodo de Cinemática Inversa TM5-900 iniciado.'
        )


    # ========================================================
    # PUBLICAR JOINT STATES
    # ========================================================

    def publish_joint_states(self):

        msg = JointState()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.name = [
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6'
        ]

        msg.position = (
            self.current_q_urdf.tolist()
        )

        self.pub.publish(msg)


    # ========================================================
    # CINEMÁTICA INVERSA
    # ========================================================

    def solve(self, msg):

        target = np.array([
            msg.x,
            msg.y,
            msg.z
        ], dtype=float)

        self.get_logger().info(
            f'Objetivo recibido: '
            f'x={msg.x:.3f}, '
            f'y={msg.y:.3f}, '
            f'z={msg.z:.3f}'
        )


        # ----------------------------------------------------
        # SEMILLA INICIAL
        # ----------------------------------------------------

        q = self.current_q_urdf.copy()

        # Para un nuevo objetivo, si estamos cerca de cero,
        # utilizamos una posición inicial sencilla.
        if np.linalg.norm(q) < 1e-6:

            q = np.array([
                np.arctan2(msg.y, msg.x),
                0.3,
                1.0,
                1.5,
                1.5,
                0.0
            ])


        q = clamp_joints(q)

        reached = False


        # ====================================================
        # MÉTODO ITERATIVO DEL JACOBIANO
        # ====================================================

        for i in range(iterations):

            current_position = get_position(q)

            error_vector = (
                target - current_position
            )

            error = np.linalg.norm(
                error_vector
            )


            # ------------------------------------------------
            # CONVERGENCIA
            # ------------------------------------------------

            if error < tolerance:

                reached = True

                break


            # ------------------------------------------------
            # JACOBIANO
            # ------------------------------------------------

            J = numerical_jacobian(q)


            # ------------------------------------------------
            # PSEUDOINVERSA
            # ------------------------------------------------

            J_pinv = np.linalg.pinv(J)


            # ------------------------------------------------
            # ACTUALIZAR ARTICULACIONES
            # ------------------------------------------------

            dq = (
                alpha_step *
                (J_pinv @ error_vector)
            )

            q += dq


            # ------------------------------------------------
            # RESPETAR LÍMITES DEL URDF
            # ------------------------------------------------

            q = clamp_joints(q)


        # ====================================================
        # RESULTADO
        # ====================================================

        if reached:

            self.current_q_urdf = q.copy()

            final_position = get_position(q)

            final_error = np.linalg.norm(
                target - final_position
            )


            self.get_logger().info(
                '========================================'
            )

            self.get_logger().info(
                'OBJETIVO ALCANZADO'
            )

            self.get_logger().info(
                f'Objetivo: '
                f'x={target[0]:.4f}, '
                f'y={target[1]:.4f}, '
                f'z={target[2]:.4f}'
            )

            self.get_logger().info(
                f'FK-URDF: '
                f'x={final_position[0]:.4f}, '
                f'y={final_position[1]:.4f}, '
                f'z={final_position[2]:.4f}'
            )

            self.get_logger().info(
                f'Error: '
                f'{final_error:.6f} m'
            )

            self.get_logger().info(
                f'Joints URDF: '
                f'{q.tolist()}'
            )

            self.get_logger().info(
                '========================================'
            )


        else:

            final_position = get_position(q)

            final_error = np.linalg.norm(
                target - final_position
            )

            self.get_logger().warn(
                '========================================'
            )

            self.get_logger().warn(
                'OBJETIVO NO ALCANZADO'
            )

            self.get_logger().warn(
                f'Posición actual: '
                f'x={final_position[0]:.4f}, '
                f'y={final_position[1]:.4f}, '
                f'z={final_position[2]:.4f}'
            )

            self.get_logger().warn(
                f'Error final: '
                f'{final_error:.6f} m'
            )

            self.get_logger().warn(
                '========================================'
            )


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = InverseKinematics()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()