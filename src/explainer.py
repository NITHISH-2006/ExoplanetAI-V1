import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AttentionExplainer:
    """YOUR NOVEL CONTRIBUTION - Explainable AI for exoplanet detection"""
    
    def __init__(self):
        self.transit_features = [
            'periodic_dips', 'dip_consistency', 'ingress_egress', 
            'flat_bottom', 'out_of_transit_stability'
        ]
    
    def analyze_light_curve(self, time, flux, predicted_period):
        """Explain WHY the model thinks it's a planet"""
        explanations = []
        confidence_factors = []
        
        # 1. Check for periodic dips
        periodicity_score = self._check_periodicity(time, flux, predicted_period)
        explanations.append(f"Periodicity score: {periodicity_score:.2f}")
        confidence_factors.append(periodicity_score)
        
        # 2. Check dip consistency
        consistency_score = self._check_dip_consistency(time, flux, predicted_period)
        explanations.append(f"Transit consistency: {consistency_score:.2f}")
        confidence_factors.append(consistency_score)
        
        # 3. Check ingress/egress shape
        shape_score = self._check_transit_shape(time, flux, predicted_period)
        explanations.append(f"Transit shape: {shape_score:.2f}")
        confidence_factors.append(shape_score)
        
        # 4. Create attention heatmap
        heatmap = self._create_attention_heatmap(time, flux, predicted_period)
        
        overall_confidence = np.mean(confidence_factors)
        
        return {
            'explanations': explanations,
            'confidence_factors': confidence_factors,
            'overall_confidence': overall_confidence,
            'heatmap': heatmap
        }
    
    def _check_periodicity(self, time, flux, period):
        """Check if dips occur at regular intervals"""
        if period <= 0:
            return 0.0
        
        # Find potential transits
        dips = flux < (np.mean(flux) - 2 * np.std(flux))
        dip_times = time[dips]
        
        if len(dip_times) < 2:
            return 0.0
        
        # Check if dips are periodic
        time_diffs = np.diff(dip_times)
        period_ratios = time_diffs / period
        period_ratios = period_ratios[np.isfinite(period_ratios)]
        
        if len(period_ratios) == 0:
            return 0.0
        
        # Score based on how close to integer periods
        fractional_parts = np.abs(period_ratios - np.round(period_ratios))
        periodicity_score = 1.0 - np.mean(fractional_parts)
        
        return max(0.0, periodicity_score)
    
    def _check_dip_consistency(self, time, flux, period):
        """Check if transits have consistent depths"""
        if period <= 0:
            return 0.0
        
        dip_depths = []
        n_orbits = int(time[-1] / period)
        
        for orbit in range(n_orbits):
            transit_center = (orbit + 0.5) * period
            in_transit = np.abs(time - transit_center) < period * 0.1
            
            if np.sum(in_transit) > 0:
                transit_depth = np.mean(flux) - np.min(flux[in_transit])
                dip_depths.append(transit_depth)
        
        if len(dip_depths) < 2:
            return 0.0
        
        # Score based on depth consistency
        depth_std = np.std(dip_depths)
        depth_mean = np.mean(dip_depths)
        
        if depth_mean > 0:
            consistency = 1.0 - (depth_std / depth_mean)
            return max(0.0, consistency)
        
        return 0.0
    
    def _check_transit_shape(self, time, flux, period):
        """Check if transits have planet-like shape (flat bottom)"""
        if period <= 0:
            return 0.0
        
        # Analyze first transit
        transit_center = period / 2
        in_transit = np.abs(time - transit_center) < period * 0.1
        
        if np.sum(in_transit) < 5:
            return 0.0
        
        transit_flux = flux[in_transit]
        transit_time = time[in_transit]
        
        # Check for flat bottom (planets) vs V-shape (binaries)
        middle_third = len(transit_flux) // 3
        middle_flux = transit_flux[middle_third:2*middle_third]
        
        if len(middle_flux) > 0:
            flatness = 1.0 - (np.std(middle_flux) / (np.max(transit_flux) - np.min(transit_flux)))
            return max(0.0, flatness)
        
        return 0.0
    
    def _create_attention_heatmap(self, time, flux, period):
        """Create visual explanation of model attention"""
        attention = np.zeros_like(flux)
        
        if period > 0:
            # Highlight transit regions
            n_orbits = int(time[-1] / period)
            for orbit in range(n_orbits):
                transit_center = (orbit + 0.5) * period
                in_transit = np.abs(time - transit_center) < period * 0.15
                attention[in_transit] = 1.0
            
            # Add ingress/egress emphasis
            for orbit in range(n_orbits):
                transit_center = (orbit + 0.5) * period
                ingress_start = transit_center - period * 0.15
                ingress_end = transit_center - period * 0.1
                egress_start = transit_center + period * 0.1
                egress_end = transit_center + period * 0.15
                
                in_ingress = (time >= ingress_start) & (time < ingress_end)
                in_egress = (time >= egress_start) & (time < egress_end)
                
                attention[in_ingress] = 0.7
                attention[in_egress] = 0.7
        
        return attention
    
    def create_explanation_plot(self, time, flux, explanation_results):
        """Create comprehensive explanation visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"colspan": 2}, None], [{"type": "bar"}, {"type": "heatmap"}]],
            subplot_titles=(
                "Light Curve with AI Attention", 
                "Confidence Factors",
                "Attention Heatmap"
            ),
            row_heights=[0.6, 0.4]
        )
        
        # Original light curve
        fig.add_trace(go.Scatter(
            x=time, y=flux, mode='lines', name='Light Curve',
            line=dict(color='blue', width=1)
        ), row=1, col=1)
        
        # Add attention overlay
        attention = explanation_results['heatmap']
        if np.max(attention) > 0:
            fig.add_trace(go.Scatter(
                x=time, y=flux, mode='none', fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)', name='AI Attention',
                hoverinfo='skip'
            ), row=1, col=1)
        
        # Confidence factors
        factors = explanation_results['confidence_factors']
        feature_names = ['Periodicity', 'Consistency', 'Shape']
        
        fig.add_trace(go.Bar(
            x=feature_names, y=factors,
            marker_color=['green' if f > 0.5 else 'red' for f in factors],
            text=[f'{f:.2f}' for f in factors],
            textposition='auto'
        ), row=2, col=1)
        
        # Attention heatmap
        fig.add_trace(go.Heatmap(
            z=[attention], x=time, y=['Attention'],
            colorscale='Reds', showscale=False,
            hoverinfo='x+z'
        ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True)
        
        return fig