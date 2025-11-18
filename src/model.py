import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import os

class LightweightDetector:
    """Fast, accurate model that works on CPU and deploys easily"""
    
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.feature_names = None
        
    def extract_features(self, light_curves):
        """Extract domain-specific features from light curves"""
        features = []
        
        for lc in light_curves:
            feature_vector = []
            
            # Statistical features
            feature_vector.extend([
                np.mean(lc), np.std(lc), np.min(lc), np.max(lc),
                np.median(lc), np.percentile(lc, 5), np.percentile(lc, 95)
            ])
            
            # Periodogram-based features (simplified)
            autocorr = np.correlate(lc - np.mean(lc), lc - np.mean(lc), mode='same')
            feature_vector.extend([
                np.max(autocorr), np.mean(autocorr), np.std(autocorr)
            ])
            
            # Transit-like features
            dips = lc < (np.mean(lc) - 2 * np.std(lc))
            feature_vector.extend([
                np.sum(dips),  # Number of potential transits
                np.mean(lc[dips]) if np.sum(dips) > 0 else 0,  # Average dip depth
                np.std(lc[dips]) if np.sum(dips) > 0 else 0   # Dip consistency
            ])
            
            # Trend features
            x = np.arange(len(lc))
            slope = np.polyfit(x, lc, 1)[0]
            feature_vector.append(slope)
            
            features.append(feature_vector)
        
        self.feature_names = [
            'mean', 'std', 'min', 'max', 'median', 'p5', 'p95',
            'autocorr_max', 'autocorr_mean', 'autocorr_std',
            'n_dips', 'dip_depth_mean', 'dip_depth_std', 'trend_slope'
        ]
        
        return np.array(features)
    
    def train(self, light_curves, labels, periods):
        """Train the model"""
        print("🔄 Extracting features...")
        features = self.extract_features(light_curves)
        
        print("🌲 Training classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(features, labels)
        
        # Only train regressor on planet examples
        planet_mask = labels == 1
        if np.sum(planet_mask) > 10:
            print("🌲 Training period regressor...")
            self.regressor = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            self.regressor.fit(features[planet_mask], periods[planet_mask])
        
        # Evaluate
        preds = self.classifier.predict(features)
        accuracy = accuracy_score(labels, preds)
        print(f"✅ Training accuracy: {accuracy:.3f}")
        
        if self.regressor and np.sum(planet_mask) > 10:
            period_preds = self.regressor.predict(features[planet_mask])
            mae = mean_absolute_error(periods[planet_mask], period_preds)
            print(f"✅ Period MAE: {mae:.2f} days")
    
    def predict(self, light_curve):
        """Make prediction on single light curve"""
        features = self.extract_features([light_curve])
        
        planet_prob = self.classifier.predict_proba(features)[0][1]
        
        if self.regressor and planet_prob > 0.3:
            period_pred = self.regressor.predict(features)[0]
        else:
            period_pred = 10.0  # Default
        
        return {
            'planet_confidence': planet_prob,
            'predicted_period': max(1.0, period_pred),
            'features': dict(zip(self.feature_names, features[0]))
        }
    
    def save(self, filepath):
        """Save trained model"""
        model_data = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath):
        """Load trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.classifier = model_data['classifier']
            self.regressor = model_data['regressor']
            self.feature_names = model_data['feature_names']
            return True
        return False