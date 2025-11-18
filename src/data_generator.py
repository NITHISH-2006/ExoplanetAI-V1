import numpy as np
from scipy import signal
import pandas as pd

class RealisticDataGenerator:
    """Generate physically accurate light curves WITHOUT external dependencies"""
    
    def __init__(self):
        self.kepler_like_params = {
            'cadence': 1765,  # Kepler's 30-min cadence over 80 days
            'duration_days': 80,
            'noise_level': 0.001  # Realistic Kepler noise
        }
    
    def generate_dataset(self, n_samples=500):
        """Generate balanced dataset of planet/no-planet light curves"""
        data, labels, periods, depths = [], [], [], []
        
        for i in range(n_samples):
            has_planet = np.random.random() > 0.6  # 40% have planets
            
            if has_planet:
                # PHYSICALLY REALISTIC PARAMETERS
                period = np.random.uniform(5, 100)
                rp_rs = np.random.uniform(0.02, 0.1)  # Planet/star radius ratio
                depth = rp_rs ** 2  # Actual transit depth formula
                duration = period * 0.1 * rp_rs  # Realistic duration
                
                time, flux = self._create_physical_transit(period, depth, duration)
                labels.append(1)
                periods.append(period)
                depths.append(depth)
            else:
                # Realistic false positives: eclipsing binaries, noise, etc.
                time, flux = self._create_false_positive()
                labels.append(0)
                periods.append(0.0)
                depths.append(0.0)
            
            data.append(flux)
        
        return {
            'time': time,
            'flux': np.array(data),
            'labels': np.array(labels),
            'periods': np.array(periods),
            'depths': np.array(depths)
        }
    
    def _create_physical_transit(self, period, depth, duration):
        """Create physically accurate transit light curve"""
        time = np.linspace(0, 80, 1765)
        
        # Start with stellar variability (real stars aren't flat!)
        flux = 1 + 0.005 * np.sin(2 * np.pi * time / 25)  # Stellar rotation
        flux += 0.002 * np.sin(2 * np.pi * time / 10)     # Short-term variations
        
        # Add multiple transits across observation period
        n_orbits = int(80 / period)
        for orbit in range(n_orbits):
            transit_center = (orbit + 0.5) * period
            
            # Realistic ingress/egress (not instant)
            transit_start = transit_center - duration/2
            transit_end = transit_center + duration/2
            
            # Full transit
            in_full_transit = (time >= transit_start + duration*0.2) & (time <= transit_end - duration*0.2)
            flux[in_full_transit] = 1 - depth
            
            # Ingress (entering)
            in_ingress = (time >= transit_start) & (time < transit_start + duration*0.2)
            ingress_progress = (time[in_ingress] - transit_start) / (duration * 0.2)
            flux[in_ingress] = 1 - depth * ingress_progress
            
            # Egress (exiting)
            in_egress = (time > transit_end - duration*0.2) & (time <= transit_end)
            egress_progress = 1 - (time[in_egress] - (transit_end - duration*0.2)) / (duration * 0.2)
            flux[in_egress] = 1 - depth * egress_progress
        
        # Add realistic noise (combination of white + red noise)
        white_noise = np.random.normal(0, self.kepler_like_params['noise_level'], len(time))
        red_noise = np.random.normal(0, self.kepler_like_params['noise_level'] * 0.3, len(time))
        red_noise = np.cumsum(red_noise)  # Random walk for red noise
        
        flux += white_noise + red_noise * 0.1
        
        return time, flux
    
    def _create_false_positive(self):
        """Create realistic false positives"""
        time = np.linspace(0, 80, 1765)
        flux_type = np.random.choice(['noise', 'binary', 'variable'])
        
        if flux_type == 'noise':
            # Pure noise
            flux = 1 + np.random.normal(0, 0.005, len(time))
        
        elif flux_type == 'binary':
            # Eclipsing binary - deeper, V-shaped
            period = np.random.uniform(2, 10)
            depth = np.random.uniform(0.1, 0.3)  # Much deeper than planets
            flux = np.ones(len(time))
            
            # Primary eclipse
            for orbit in range(int(80 / period)):
                eclipse_center = (orbit + 0.5) * period
                in_eclipse = np.abs(time - eclipse_center) < 0.2
                flux[in_eclipse] = 1 - depth
        
        else:  # variable
            # Stellar variability
            flux = 1 + 0.02 * np.sin(2 * np.pi * time / 15)
            flux += 0.01 * np.sin(2 * np.pi * time / 3)
            flux += np.random.normal(0, 0.003, len(time))
        
        return time, flux
    
    def generate_single_curve(self, period=20, depth=0.02, has_planet=True):
        """Generate single light curve for demo"""
        if has_planet:
            duration = period * 0.1 * np.sqrt(depth)  # Consistent with physics
            return self._create_physical_transit(period, depth, duration)
        else:
            return self._create_false_positive()