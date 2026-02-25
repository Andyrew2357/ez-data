import numpy as np

from typing import Dict, List


class softPulsePair():
    def __init__(self, T: float):
        self.T = T

        self._enable = False
        self._T0 = 0.0
        self._T1 = T/2
        self._dT0 = 0.0
        self._dT1 = 0.0
        self._X = 0.0
        self._Y = 0.0

    def enable(self, on: bool | None = None) -> bool | None:
        if on is None:
            return self._enable
        self._enable = on

    def T0(self, t: float | None = None) -> float | None:
        if t is None:
            return self._T0
        self._T0 = t

    def T1(self, t: float | None = None) -> float | None:
        if t is None:
            return self._T1
        self._T1 = t

    def dT0(self, t: float | None = None) -> float | None:
        if t is None:
            return self._dT0
        self._dT0 = t

    def dT1(self, t: float | None = None) -> float | None:
        if t is None:
            return self._dT1
        self._dT1 = t

    def W(self, dt: float | None = None) -> float | None:
        if dt is None:
            return self._T1 - self._T0
        
        self.T1(self._T0 + dt)


    def X(self, v: float | None = None) -> float | None:
        if v is None:
            return self._X
        if self._X*v < 0:
            self._Y = -self._Y
        self._X = v

    def Y(self, v: float | None = None) -> float | None:
        if v is None:
            return self._Y
        if self._Y*v < 0:
            self._X = -self._X
        self._Y = v

class pulsedTunnelingBridge():
    """
    Computes the steady-state response of a lumped-element model tunneling 
    capacitor bridge to a collection of repeated square pulses. The model is
    fully analytic.

                    ┏━━━┓
                    ┃ X ┃
                    ┗━┳━┛
                      ┃
                    ╺━┻━╸ Cstd
                    ╺━┳━╸         GND
                      ┃            ┋ Rshunt
                      ┃            ┋          ┏━━━┓
                      ┣━━━━━━━━━━━━╋━━━━━━━━━━┫ Q ┃
                      ┃            ┃          ┗━━━┛
                      ┃          ╺━┻━╸Cpar
                   ┏━━┻━━┓       ╺━┳━╸
                   ┋    ━┻━        ┃
                   ┋ RT ━┳━ CT    GND
                   ┗━━┳━━┛
                      ┃
                    ┏━┻━┓
                    ┃ Y ┃
                    ┗━━━┛

    X and Y are inputs composed of time domain voltage pulses. Q is the output
    time domain waveform
                    
    """

    def __init__(self, Cstd: float, Cpar: float, Rshunt: float, G: float = 1.0, noise: float = 0.0):
        
        # Static Parameters Related to the Setup
        self.Cstd = Cstd
        self.Cpar = Cpar
        self.Rshunt = Rshunt
        self.G = G
        self.noise = noise

        # Tunneling Capacitor Params
        self.CT: float | None = None
        self.RT: float | None = None

        # Pulses
        self.pulses: List[softPulsePair] = []

    def set_tunneling_cap(self, CT: float, RT: float):
        self.CT = CT
        self.RT = RT
        self.tau = (self.Cstd + self.Cpar + self.CT)/(1/self.Rshunt + 1/self.RT)

    def set_measurement_window(self, dt: float, N: int):
        self.t = dt * np.arange(N)
        self.dt = dt
        self.N = N

    def add_pulse_pair(self, pulse: softPulsePair):
        self.pulses.append(pulse)

    def remove_pulse_pair(self, pulse: softPulsePair):
        self.pulses.remove(pulse)

    def _delta_resp(self, t0, T) -> np.ndarray:
        t = np.mod(self.t, T)
        O0 = np.heaviside(t - t0, 0.5)
        O1 = np.heaviside(t0 - t, 0.5)
        return np.exp((t0 - t)/self.tau)*(O0 + np.exp(-T/self.tau)*O1)/(1 - np.exp(-T/self.tau))
    
    def _box_resp(self, t0, t1, T) -> np.ndarray:
        t = np.mod(self.t, T)
        ea = np.exp((t0 - t)/self.tau)
        eb = np.exp((t1 - t)/self.tau)
        eT = np.exp(-T/self.tau)
        O0 = np.heaviside(t0 - t, 0.5)
        O1 = np.heaviside(t - t1, 0.5)
        OO = np.heaviside(t - t0, 0.5)*np.heaviside(t1 - t, 0.5)
        return (self.tau/(1 - eT))*((eb - ea)*(O1 + eT*O0) + OO*(1 - ea + eT*(eb - 1)))

    def _Y_resp(self) -> np.ndarray:
        K = self.Cstd / (self.Cstd + self.Cpar + self.CT)
        wf = np.zeros_like(self.t)
        for P in self.pulses:
            if not P._enable:
                continue

            wf += K*P._Y*(self._delta_resp(P._T0 + P._dT0, P.T) \
                        - self._delta_resp(P._T1 + P._dT1, P.T))
        
        return wf
            
    def _X_resp(self) -> np.ndarray:
        K = self.CT / (self.Cstd + self.Cpar + self.CT)
        G = (1/self.RT) / (self.Cstd + self.Cpar + self.CT)
        wf = np.zeros_like(self.t)
        for P in self.pulses:
            if not P._enable:
                continue

            wf += K*P._X*(self._delta_resp(P._T0, P.T) - self._delta_resp(P._T1, P.T)) + \
                    G*P._X*self._box_resp(P._T0, P._T1, P.T)
        
        return wf
    
    def __call__(self) -> np.ndarray:
        return self.G*(self._X_resp() + self._Y_resp()) + \
            self.noise*np.random.normal(0, 1, self.N)