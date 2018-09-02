# -*- coding: utf-8 -*-

import copy
import numpy as np
from numpy import pi


class QubitWave:
    Sx = np.array([[0, 1. + 0.j], [1. + 0.j, 0]])
    Sy = np.array([[0, -1.0j], [1.0j, 0]])
    Sz = np.array([[1. + 0.j, 0], [0, -1. + 0.j]])
    Id = np.array([[1. + 0.j, 0], [0, 1. + 0.j]])

    def __init__(self, T_R, Amp_R, dt,
                 Amps=None, Phis=None, Deltas=None):
        # 衍生出来的R， Drive 等操作会生成新的对象。不在本对象中设置波形
        # hardware configure for experiment
        self.T_R = T_R  # Rabi Time
        self.Amp_R = Amp_R  # Amplitude to drive the Rabi flopping
        self.dt = dt
        # waveform parameters
        # 设置波形通过 SetWave 函数，
        # 注意R，Drive， Idle等操作不在本对象中设置波形，创建新对象然后设置波形
        self.SetWave(Amps, Phis, Deltas)
        # corresponding U operation
        self.U = None

    def SetWave(self, Amps, Phis, Deltas):
        self.Amps = Amps   # Amplitude list for wave
        self.Phis = Phis   # phi list for wave
        self.Deltas = Deltas  # delat list for wave

    def new(self):
        # generate a blank operation with same hardware config
        return QubitWave(self.T_R, self.Amp_R, self.dt)

    def Idle(self, N=None, Amp=0, Phi=0, Delta=-2e8):
        # generate a Idle with size length N
        # N is the Idle points,
        # use this function to
        #   1. creat new operation with same config, when N is None,
        #       return a blank operation
        #   2. creat Idle operation of AWG
        #   3. creat a path opetation  with constant amp, Phi and Delta
        Q = self.new()
        if N is None:
            return Q
        Q.Amps = np.ones(N) * Amp
        Q.Phis = np.ones(N) * Phi
        Q.Deltas = np.ones(N) * Delta
        return Q

    def R(self, theta, phi=0, delta=0, Amp=None):
        # if Amp is None , use self.Amp_R drive the qubit
        if Amp is None:
            Amp = self.Amp_R
        if theta < 0:
            theta += (np.abs(int(theta / (2 * pi))) + 1) * 2 * pi

        N = int(theta / (2 * pi) * self.T_R * self.Amp_R / Amp / self.dt)

        Amps = np.ones(N) * Amp
        Phis = np.zeros(N) + phi
        Deltas = np.zeros(N) + delta

        # corresponding U transform
        U = np.cos(theta / 2) * self.Id +\
            1.0j * (np.sin(theta / 2) * np.cos(phi) * self.Sx +
                    np.sin(theta / 2) * np.sin(phi) * self.Sy)
        Q = self.new()
        Q.SetWave(Amps, Phis, Deltas)
        Q.U = U  # set U
        return Q

    def H2U(self, H, omega_dt, U0=Id):
        # H 归一化 Hamilton量， 无量纲
        # omega_dt 计算的步长，无量纲
        # a general function, for theorically
        # Hamitonian to U operation
        # evolotion a Hamiltonian and get corressponding U operation
        # U_{n+1} = exp(-i*H_n*dt) * U0
        if np.abs(omega_dt) > 0.1:
            raise ValueError("evolution step large, omega*dt:%f" % omega_dt)

        U = U0
        H1 = copy.deepcopy(H)
        for Hti in H1:
            A0 = np.linalg.norm(Hti)
            sigma_n = (Hti[0] * self.Sx + Hti[1] *
                       self.Sy + Hti[2] * self.Sz) / A0
            dtheta = A0 * omega_dt
            if np.abs(dtheta) > 0.1:
                raise ValueError("evolution step large, dtheta:%f" % dtheta)
            dU = np.cos(dtheta) * self.Id - 1.0j * np.sin(dtheta) * sigma_n
            U = dU.dot(U)
        return U

    def eig(self, H):
        # 返回本征值和本征态，按照能量从低到高排列
        # H 传入矩阵，或者 Pauli矩阵x向量
        if len(H) == 3:  # 泡利矩阵的向量形式
            H = H[0] * self.Sx + H[1] * self.Sy + H[2] * self.Sz

        eigenValues, eigenVectors = np.linalg.eig(H)

        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        # 列向量是本征态， V[:,0] 是最低能态
        return (eigenValues, eigenVectors)

    def evo_H(self, H, omega_dt, psi0=np.array([[0. + 0.j], [1. + 0.j]])):
        # a general function, theorically
        # H 归一化 Hamilton量， 无量纲
        # 注意H是总的Hamilton，包含自旋
        # omega_dt 计算的步长，无量纲
        # evolotion a Hamiltonian and get states or U operations list with t
        # can be a initial state or a U operation
        # default psi0 is [0;1] is ground state
        if np.abs(omega_dt) > 0.1:
            raise ValueError("evolution step large, omega*dt:%f" % omega_dt)

        psi = psi0 / np.linalg.norm(psi0)
        states = []
        H1 = copy.deepcopy(H)
        for Hti in H1:
            A0 = np.linalg.norm(Hti)
            sigma_n = (Hti[0] * self.Sx + Hti[1] *
                       self.Sy + Hti[2] * self.Sz) / A0
            dtheta = A0 * omega_dt
            if np.abs(dtheta) > 0.1:
                raise ValueError("evolution step large, dtheta:%f" % dtheta)
            dU = np.cos(dtheta) * self.Id - 1.0j * np.sin(dtheta) * sigma_n
            psi = dU.dot(psi)
            psi /= np.linalg.norm(psi)
            states.append(psi)
        return states

    def Normlized_H_dt(self, Wave=None, spin=-1/2.0):
        # from the wave generate a normlizde Hamiltonian
        # 乘以 spin, default spin is 1/2：
        #      H = 1/2 * omega * n*sigma
        # if Wave is None, use self  wave
        # 数值计算归一化 dt =  omega * dt
        if Wave is None:
            Amps = self.Amps
            Phis = self.Phis
            Deltas = self.Deltas
        else:
            Amps, Phis, Deltas = Wave
        nxs = Amps * np.cos(Phis) / self.Amp_R
        nys = Amps * np.sin(Phis) / self.Amp_R
        nzs = Deltas / (2 * pi / self.T_R)

        H = zip(nxs, nys, nzs)
        dt = spin * 2 * pi / self.T_R * self.dt
        return (H, dt)

    def Drive(self, H, phi0=0):
        # H = [fx_t,fy_t,fz_t]
        # fx_t, fy_t 驱动电压
        # fz_t 是失谐大小，角频率
        # an experiment drive
        # Hi = (fx,fy,fz) to corresponding fx*sigmax+fy*singmay+fz*sigmaz
        # when norm(H_i) = Amp_R corresponding Rabi time
        # Note: 扫频时的扫描速率
        # 在 T_R 时间内, 失谐跑一个Rabi频率
        # 需要的扫描速率： v = Omega_R * Omega_R /(2*pi)
        # 因为： omega_R * Omega_R * T_R
        #       = omega_R * 2 * pi
        fxs, fys, fzs = H
        fxs /= self.Amp_R
        fys /= self.Amp_R

        Amps = self.Amp_R * np.sqrt(fxs**2 + fys**2)
        Phis = np.arctan2(fys, fxs) + phi0
        Deltas = fzs
        H1, dt = self.Normlized_H_dt((Amps, Phis, Deltas))
        U = self.H2U(H1, dt)

        Q = self.new()
        Q.SetWave(Amps, Phis, Deltas)
        Q.U = U
        return Q

    def Drive_Standard(self, H, spin=1/2.0, Amp=None, phi0=0):
            # 生成标准Hamilton量的驱动，(理论上归一化的Hamilton量)
            # 自动计算波形
            # H = (fx,fy,fz) to corresponding H = fx*sigmax+fy*singmay+fz*sigmaz

        if Amp is None:
            Amp = self.Amp_R
        Omega_R = 2 * pi / self.T_R
        Omega1 = Amp / self.Amp_R * Omega_R
        x, y, z = H

        fxs = Amp * x/spin
        fys = Amp * y/spin
        fzs = Omega1 * z/spin
        return self.Drive([fxs, fys, fzs], phi0)

    def __mul__(self, other):
        if self.dt != other.dt:
            raise ValueError('step is different of two wave')

        Q = self.new()
        Q.Amps = np.concatenate((other.Amps, self.Amps))
        Q.Phis = np.concatenate((other.Phis, self.Phis))
        Q.Deltas = np.concatenate((other.Deltas, self.Deltas))
        Q.U = self.U.dot(other.U)
        return Q

    def __sub__(self, other):
        if self.dt != other.dt:
            raise ValueError('step is different of two wave')

        Q = self.new()
        Q.Amps = other.Amps - self.Amps
        Q.Phis = other.Phis - self.Phis
        Q.Deltas = other.Deltas - self.Deltas
        Q.U = self.U - other.U
        return Q

    def cT(self):
        # 复共轭转置矩阵, conjuction Transpose
        # 酉矩阵的复共轭转置 等于矩阵的逆， 相当于求逆函数
        # 波形上将波形序列取反，相位取反，反演化即可
        Q = self.new()
        Q.Amps = self.Amps[::-1]
        Q.Phis = self.Phis[::-1] + pi
        Q.Deltas = self.Deltas[::-1]
        Q.U = self.U.conj().T
        return Q

    def __len__(self):
        if self.Amps is None:
            return None
        else:
            return len(self.Amps)

    def phi_0(self):
        # return the start phase
        if self.Phis is not None:
            return self.Phis[0]
        else:
            return None

    def phi_f(self):
        # return the final phase
        if self.Phis is not None:
            return self.Phis[-1]
        else:
            return None

    def AWG_Wave(self, w_c, fun=np.sin, dt=None):
        # if dt is not set use self.dt
        # if dt is set, generate wave from new dt by interplore
        t1 = np.arange(len(self.Amps)) * self.dt
        if dt is None:
            Amps = self.Amps
            Phis = self.Phis
            Deltas = self.Deltas
            dt = self.dt
            t = t1
        else:
            N = int(t1[-1] / dt)
            t = np.arange(N) * dt
            Amps = np.interp(t, t1, self.Amps)
            Phis = np.interp(t, t1, self.Phis)
            Deltas = np.interp(t, t1, self.Deltas)
        return Amps * fun(w_c * t - Phis - np.add.accumulate(Deltas) * dt)


class WaveArray:
    # array to represent a wave,
    # use Amp, Phi, Delta arrays to repsent the wave
    # Amp * sin( Phi+ np.add.accumulate(Delta)*dt )
    # default array: [], so the length is 0
    def __init__(self, dt, Amp=[], Phi=[], Delta=[]):
        # dt is time step, must be set
        self.dt = dt  # 设置时间步长
        self.setWave(Amp, Phi, Delta)

    def setWave(self, Amp, Phi, Delta):
        if len(Amp) == len(Phi) and len(Amp) == len(Delta):
            self.Amp = Amp
            self.Phi = Phi
            self.Delta = Delta
        else:
            raise ValueError("Amp,Phi, Delta length diff",
                             (len(Amp), len(Phi), len(Delta)))

    def setZeros(self, N):
        # set zeros wave
        # design for idle
        self.setWave(np.zeros(N), np.zeros(N), np.zeros(N))

    def append(self, other):
        # append other wave to this wave
        # Note! this wave will be change
        if self.dt != other.dt:
            raise ValueError('step is different of two wave')
        self.Amp = np.concatenate((self.Amp, other.Amp))
        self.Phi = np.concatenate((self.Phi, other.Phi))
        self.Delta = np.concatenate((self.Delta, other.Delta))

    def __add__(self, other):
        # concatenate two wave
        # Note! this wave will be change
        Q = WaveArray(self.dt)
        Q.Amp = np.concatenate((self.Amp, other.Amp))
        Q.Phi = np.concatenate((self.Phi, other.Phi))
        Q.Delta = np.concatenate((self.Delta, other.Delta))
        return Q

    def invert(self, dphi=pi):
        # designed to invert a evolution
        Q = WaveArray(self.dt)
        Q.Amp = self.Amp[::-1]
        Q.Phi = self.Phi[::-1] + dphi
        Q.Delta = self.Delta[::-1]
        return Q

    def __len__(self):
        if len(self.Amp) == len(self.Phi) and len(self.Amp) == len(self.Delta):
            return len(self.Amp)
        else:
            raise ValueError("Amp, Phi, Delta length diff",
                             (len(self.Amp), len(self.Phi), len(self.Delta)))
