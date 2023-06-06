import numpy as np
from math import pi
from controller import Supervisor

class UR5:
    """
        This class defines the UR5 object and its functions
    """
    def __init__(self):
        print("Inicializando a classe UR5...")
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.supervisor.step(self.timestep)
        self.joints = None
        self.finger_joints = None
        self.finger_joint_limits = None
        self.init_handles()
        print('Pronto!')

    def setup_control_mode(self):
        for i, dev in enumerate(self.joints):
            dev.setPosition(float('inf'))
            dev.getPositionSensor().enable(self.timestep)

        for dev in self.finger_joints:
            dev.setVelocity(float(100))
            dev.getPositionSensor().enable(self.timestep)

    def init_handles(self):
        """
            This function initiates the nodes
        """
        print('Inicializando os nós e handles...')
        self.joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                       'wrist_3_joint']
        self.joint_sensors = ['shoulder_pan_joint_sensor', 'shoulder_lift_joint_sensor', 'elbow_joint_sensor',
                              'wrist_1_joint_sensor', 'wrist_2_joint_sensor',
                       'wrist_3_joint']
        self.finger_joints = ['finger_1_joint_1', 'finger_1_joint_2', 'finger_1_joint_3',
                              'finger_2_joint_1', 'finger_2_joint_2', 'finger_2_joint_3',
                              'finger_middle_joint_1', 'finger_middle_joint_2', 'finger_middle_joint_3']
        self.finger_joint_sensors = ['finger_1_joint_1_sensor', 'finger_1_joint_2_sensor', 'finger_1_joint_3_sensor',
                                     'finger_2_joint_1_sensor', 'finger_2_joint_2_sensor', 'finger_2_joint_3_sensor',
                                     'finger_middle_joint_1_sensor', 'finger_middle_joint_2_sensor',
                                     'finger_middle_joint_3_sensor']
        self.joints = [self.supervisor.getDevice(joint) for joint in self.joints]
        self.finger_joints = [self.supervisor.getDevice(joint) for joint in self.finger_joints]
        self.joint_sensors = [self.supervisor.getDevice(sensor) for sensor in self.joint_sensors]
        self.finger_joint_sensors = [self.supervisor.getDevice(s) for s in self.finger_joint_sensors]
        self.finger_joint_limits = [[0.0695, 0.8], [0.01, 1], [-0.8, -0.0723],
                               [0.0695, 0.8], [0.01, 1], [-0.8, -0.0723],
                               [0.0695, 0.8], [0.01, 1], [-0.8, -0.0723]]
        self.setup_control_mode()


    def get_joint_angles(self):
        angles = [joint.getPositionSensor().getValue() for joint in self.joints]
        angles[0] -= pi
        angles[1] += pi/2
        angles[3] += pi/2
        return np.array(angles)

    def get_finger_angles(self):
        return np.array([joint.getPositionSensor().getValue() for joint in self.finger_joints])

    def move_to_config(self, target, duration=5):
        t0 = self.supervisor.getTime()
        v0 = np.zeros(6)
        vf = np.zeros(6)
        q0 = self.get_joint_angles()
        qf = np.array(target)
        a0 = np.zeros(6)
        af = np.zeros(6)
        tf = t0 + duration
        A = np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
                      [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                      [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                      [1, tf, tf**2, tf**3, tf**4, tf**5],
                      [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                      [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])
        b = np.array([q0, v0, a0, qf, vf, af])
        x = [np.linalg.solve(A, b[:, i]) for i in range(6)]
        time0 = self.supervisor.getTime()
        iterations = 0
        vel_jacob = [[], [], [], [], [], []]
        pos = [[], [], [], [], [], []]
        vel = [[], [], [], [], [], []]
        acc = [[], [], [], [], [], []]
        jerk = [[], [], [], [], [], []]
        time_arr = [[], [], [], [], [], []]
        self.setup_control_mode()
        while self.supervisor.getTime() <= tf:
            t = self.supervisor.getTime()
            for idx, joint in enumerate(self.joints):
                joint.setVelocity(x[idx][1] + 2*x[idx][2]*t + 3*x[idx][3]*t**2 + 4*x[idx][4]*t**3 + 5*x[idx][5]*t**4)

                p = x[idx][0] + x[idx][1]*t + x[idx][2]*t**2 + x[idx][3]*t**3 + x[idx][4]*t**4 + x[idx][5]*t**5
                v = x[idx][1] + 2*x[idx][2]*t + 3*x[idx][3]*t**2 + 4*x[idx][4]*t**3 + 5*x[idx][5]*t**4
                a = 2*x[idx][2] + 6*x[idx][3]*t + 12*x[idx][4]*t**2 + 20*x[idx][5]*t**3
                j = 6*x[idx][3] + 24*x[idx][4]*t + 60*x[idx][5]*t**2
                time_arr[idx].append(t-time0)
                pos[idx].append(p)
                vel[idx].append(v)
                acc[idx].append(a)
                jerk[idx].append(j)
                v = x[idx][1] + 2*x[idx][2]*t + 3*x[idx][3]*t**2 + 4*x[idx][4]*t**3 + 5*x[idx][5]*t**4
                vel_jacob[idx].append(v)
            self.supervisor.step(self.timestep)
            iterations += 1
        for joint in self.joints:
            joint.setVelocity(0)
        timef = self.supervisor.getTime()
        error = np.abs(np.array(target) - self.get_joint_angles())*180/np.pi
        print('Iterações totais: ', iterations)
        elapsed = timef - time0
        return (elapsed, np.max(error), np.mean(error))

    def actuate_gripper(self, close=0):
        t0 = self.supervisor.getTime()
        v0 = np.zeros(9)
        vf = np.zeros(9)
        q0 = self.get_finger_angles()
        qf = np.hstack((np.array([lim[close] for lim in self.finger_joint_limits])))
        a0 = np.zeros(9)
        af = np.zeros(9)
        tf = t0 + 2
        A = np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
                      [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                      [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                      [1, tf, tf**2, tf**3, tf**4, tf**5],
                      [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                      [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])
        b = np.array([q0, v0, a0, qf, vf, af])
        x = [np.linalg.solve(A, b[:, i]) for i in range(9)]
        time0 = self.supervisor.getTime()
        iterations = 0
        self.setup_control_mode()
        while self.supervisor.getTime() <= tf:
            t = self.supervisor.getTime()
            for idx, joint in enumerate(self.finger_joints):
                joint.setPosition(x[idx][0] + x[idx][1]*t + x[idx][2]*t**2 + x[idx][3]*t**3 + x[idx][4]*t**4 + x[idx][5]*t**5)
            self.supervisor.step(self.timestep)
            iterations += 1
        for i, joint in enumerate(self.joints):
            joint.setPosition(self.finger_joint_limits[i][close])
        timef = self.supervisor.getTime()
        print('Iterações totais: ', iterations)
        print(timef-time0)