import numpy as np


class MicrogridEnv:
    def __init__(self, episode_hours: int = 24, seed: int = None):
        self.episode_hours = episode_hours
        self.rng = np.random.default_rng(seed)

        # --- system parameters (sized so the battery is a REAL constraint,
        # not big enough to trivially cover the whole day) ---
        self.battery_capacity_kwh = 2.0
        self.battery_efficiency = 0.95
        self.solar_peak_w = 1000.0
        self.load_min_w = 100.0
        self.load_max_w = 800.0
        self.price_peak = 0.25       # $/kWh, 09:00-21:00
        self.price_offpeak = 0.10    # $/kWh, otherwise
        self.soc_min_safe = 0.20     # HARD critical floor - battery cannot discharge below this
        self.soc_max_safe = 0.95     # soft ceiling - discourage overcharging
        self.max_charge_rate_wh = 400.0  # per-hour cap on grid arbitrage charging

        # weather scenarios sampled once per episode (per simulated day):
        # clear (most common), partly cloudy, overcast
        self.weather_options = [1.0, 0.7, 0.35]
        self.weather_probs = [0.6, 0.25, 0.15]

        self.observation_space_shape = (6,)
        self.n_actions = 3

        self.hour = 0
        self.battery_soc = 0.5
        self.weather_factor = 1.0

    class _Box:
        def __init__(self, shape):
            self.shape = shape

    class _Discrete:
        def __init__(self, n):
            self.n = n

    @property
    def observation_space(self):
        return self._Box(self.observation_space_shape)
 
    @property
    def action_space(self):
        return self._Discrete(self.n_actions)

    # -----------------------------------------------------------------
    def _solar_output_w(self, hour: float) -> float:
        clear_sky_w = self.solar_peak_w * np.exp(-((hour - 12) ** 2) / (2 * 3.5 ** 2))
        return clear_sky_w * self.weather_factor

    def _load_w(self, hour: float) -> float:
        base = self.load_min_w
        morning = 300 * np.exp(-((hour - 8) ** 2) / (2 * 1.0 ** 2))
        lunch = 200 * np.exp(-((hour - 13) ** 2) / (2 * 1.0 ** 2))
        evening = 400 * np.exp(-((hour - 19.5) ** 2) / (2 * 1.5 ** 2))
        noise = self.rng.normal(0, 20)
        load = base + morning + lunch + evening + noise
        return float(np.clip(load, self.load_min_w, self.load_max_w))

    def _price(self, hour: float) -> float:
        return self.price_peak if 9 <= hour < 21 else self.price_offpeak

    def _get_state(self) -> np.ndarray:
        pv_w = self._solar_output_w(self.hour)
        load_w = self._load_w(self.hour)
        price = self._price(self.hour)

        pv_norm = pv_w / self.solar_peak_w
        load_norm = (load_w - self.load_min_w) / (self.load_max_w - self.load_min_w)
        price_norm = (price - self.price_offpeak) / (self.price_peak - self.price_offpeak)
        hour_sin = np.sin(2 * np.pi * self.hour / 24)
        hour_cos = np.cos(2 * np.pi * self.hour / 24)

        self._last_pv_w = pv_w
        self._last_load_w = load_w
        self._last_price = price

        return np.array(
            [pv_norm, load_norm, self.battery_soc, hour_sin, hour_cos, price_norm],
            dtype=np.float32,
        )