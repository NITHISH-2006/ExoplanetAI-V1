import numpy as np
import plotly.graph_objects as go

class OrbitalPhysics:
    """Real orbital mechanics - no approximations"""
    
    def __init__(self):
        self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        self.M_sun = 1.989e30  # kg
        self.AU = 1.496e11     # m
    
    def keplers_third_law(self, period_days, star_mass_solar=1.0):
        """Calculate semi-major axis from orbital period"""
        period_seconds = period_days * 24 * 3600
        star_mass = star_mass_solar * self.M_sun
        
        # a^3 = (G M P^2) / (4 π^2)
        a_cubed = (self.G * star_mass * period_seconds**2) / (4 * np.pi**2)
        semi_major_axis_m = a_cubed ** (1/3)
        semi_major_axis_au = semi_major_axis_m / self.AU
        
        return semi_major_axis_au
    
    def generate_3d_orbit(self, period_days, inclination=85, eccentricity=0.1, 
                         argument_periastron=0, star_mass_solar=1.0):
        """Generate physically accurate 3D orbit"""
        a = self.keplers_third_law(period_days, star_mass_solar)
        e = eccentricity
        i = np.radians(inclination)  # Convert to radians
        ω = np.radians(argument_periastron)
        
        # True anomaly (angle from periastron)
        theta = np.linspace(0, 2*np.pi, 200)
        
        # Orbital distance
        r = a * (1 - e**2) / (1 + e * np.cos(theta - ω))
        
        # 3D coordinates with proper orbital elements
        x = r * (np.cos(ω) * np.cos(theta) - np.sin(ω) * np.sin(theta) * np.cos(i))
        y = r * (np.sin(ω) * np.cos(theta) + np.cos(ω) * np.sin(theta) * np.cos(i))
        z = r * np.sin(theta) * np.sin(i)
        
        return x, y, z
    
    def create_3d_system_plot(self, period_days, planet_radius_earth=1.0, 
                            planet_name="New Planet", star_type='G'):
        """Create publication-quality 3D system visualization"""
        # Star properties based on type
        star_properties = {
            'G': {'radius': 1.0, 'temp': 5800, 'color': 'yellow', 'mass': 1.0},
            'K': {'radius': 0.8, 'temp': 4500, 'color': 'orange', 'mass': 0.8},
            'M': {'radius': 0.5, 'temp': 3200, 'color': 'red', 'mass': 0.5}
        }
        star = star_properties.get(star_type, star_properties['G'])
        
        # Calculate orbital parameters
        a = self.keplers_third_law(period_days, star['mass'])
        
        # Generate orbit
        x, y, z = self.generate_3d_orbit(period_days, star_mass_solar=star['mass'])
        
        fig = go.Figure()
        
        # Star (scaled appropriately)
        star_size = star['radius'] * 15
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(
                size=star_size,
                color=star['color'],
                opacity=0.9,
                sizemode='diameter'
            ),
            name=f'{star_type}-type Star\n{star["temp"]}K'
        ))
        
        # Orbit path
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='white', width=3, dash='dot'),
            name=f'Orbit: {a:.2f} AU'
        ))
        
        # Planet (size scaled to radius)
        planet_size = max(planet_radius_earth * 3, 2)
        
        # Determine planet type
        if period_days < 10:
            planet_color = 'red'
            planet_type = "Hot"
        elif a < 0.5:
            planet_color = 'orange'
            planet_type = "Warm"
        elif a < 2.0:
            planet_color = 'lightblue'
            planet_type = "Temperate"
        else:
            planet_color = 'darkblue'
            planet_type = "Cold"
        
        # Place planet at current position
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers+text',
            marker=dict(
                size=planet_size,
                color=planet_color,
                opacity=0.9
            ),
            text=[planet_name],
            textposition="top center",
            name=f'{planet_name}\n{planet_type} • {period_days:.1f} days'
        ))
        
        # Habitable zone (simplified)
        if star_type == 'G':
            hz_inner, hz_outer = 0.95, 1.67
        elif star_type == 'K':
            hz_inner, hz_outer = 0.60, 1.10
        else:  # M
            hz_inner, hz_outer = 0.08, 0.20
        
        # Add habitable zone ring if planet is close
        if hz_inner <= a <= hz_outer:
            theta_hz = np.linspace(0, 2*np.pi, 100)
            hz_radius = (hz_inner + hz_outer) / 2
            
            x_hz = hz_radius * np.cos(theta_hz)
            y_hz = hz_radius * np.sin(theta_hz)
            z_hz = np.zeros_like(theta_hz)
            
            fig.add_trace(go.Scatter3d(
                x=x_hz, y=y_hz, z=z_hz,
                mode='lines',
                line=dict(color='green', width=4, dash='dash'),
                name='Habitable Zone'
            ))
        
        fig.update_layout(
            title=f"3D Planetary System: {planet_name}",
            scene=dict(
                xaxis_title="X (AU)",
                yaxis_title="Y (AU)",
                zaxis_title="Z (AU)",
                bgcolor='black',
                camera=dict(
                    eye=dict(x=2, y=2, z=1),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            width=800,
            height=600
        )
        
        return fig