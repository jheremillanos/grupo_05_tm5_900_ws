import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sympy import Matrix, cos, pi, sin, symbols


class JointSubscriber(Node):

  def __init__(self):
    super().__init__('joint_subscriber')

    # Suscriptor al tópico de articulaciones
    self.subscription = self.create_subscription(
        JointState, 'joint_states', self.sub_callback, 10
    )

    # --- Variables simbólicas ---
    t, d, a, alpha = symbols('theta d a alpha')
    self.t1, self.t2, self.t3, self.t4, self.t5, self.t6 = symbols(
        'theta_1 theta_2 theta_3 theta_4 theta_5 theta_6'
    )

    # --- Matriz DH Genérica (Estándar) ---
    Rz = Matrix([
        [cos(t), -sin(t), 0, 0],
        [sin(t), cos(t), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    Tz = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])

    Tx = Matrix([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    Rx = Matrix([
        [1, 0, 0, 0],
        [0, cos(alpha), -sin(alpha), 0],
        [0, sin(alpha), cos(alpha), 0],
        [0, 0, 0, 1],
    ])

    A = Rz * Tz * Tx * Rx

    # --- Parámetros DH para Techman TM5-900 ---
    A01 = A.subs({t: self.t1, d: 0.1452, a: 0.0000, alpha: pi / 2})
    A12 = A.subs({t: self.t2, d: 0.0000, a: 0.4290, alpha: 0})
    A23 = A.subs({t: self.t3, d: 0.0000, a: 0.4115, alpha: 0})
    A34 = A.subs({t: self.t4, d: 0.1222, a: 0.0000, alpha: pi / 2})
    A45 = A.subs({t: self.t5, d: 0.1060, a: 0.0000, alpha: -pi / 2})
    A56 = A.subs({t: self.t6, d: 0.1131, a: 0.0000, alpha: 0})

    # Matriz de Transformación Total T0_6 (calculada una sola vez)
    self.T = A01 * A12 * A23 * A34 * A45 * A56

    self.get_logger().info(
        'Nodo de Cinemática Directa TM5-900 iniciado correctamente.'
    )

  def sub_callback(self, msg):
    if len(msg.position) >= 6:
      # 1. Posiciones articulares actuales de ROS 2 (URDF)
      q1 = msg.position[0]
      q2 = msg.position[1]
      q3 = msg.position[2]
      q4 = msg.position[3]
      q5 = msg.position[4]
      q6 = msg.position[5]

      # 2. Desfases angulares (Offsets) para alinear la postura vertical URDF con DH
      j1 = q1
      j2 = q2 + (pi / 2)  # Offset hombro
      j3 = q3
      j4 = q4 + (pi / 2)  # Offset muñeca 1 (corrige el eje Z de d5)
      j5 = q5
      j6 = q6

      # 3. Sustitución numérica en la matriz simbólica
      res = self.T.subs({
          self.t1: j1,
          self.t2: j2,
          self.t3: j3,
          self.t4: j4,
          self.t5: j5,
          self.t6: j6,
      })

      # 4. Extracción de coordenadas (x, y, z)
      x = float(res[0, 3])
      y = float(res[1, 3])
      z = float(res[2, 3])

      # 5. Extracción de orientación (Matriz 3x3 a Cuaternión)
      rot_matrix = np.array(res[0:3, 0:3], dtype=float)
      quat = R.from_matrix(rot_matrix).as_quat()

      # 6. Salida en terminal
      self.get_logger().info(
          f'x: {x:.4f} | y: {y:.4f} | z: {z:.4f} | quat: [{quat[0]:.4f},'
          f' {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]'
      )


def main(args=None):
  rclpy.init(args=args)
  node = JointSubscriber()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()