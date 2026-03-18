# predict_attack.py
import os
import numpy as np
import json
import joblib

from .features.forensic_features import compute_features
from .cnn.cnn_extractor import extract_cnn_features

class AttackPredictor:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            BASE_DIR = os.path.dirname(__file__)

            model_path = os.path.join(BASE_DIR, "output", "hybrid_attack_model_compressed.pkl")
            scaler_path = os.path.join(BASE_DIR, "output", "forensic_scaler_compressed.pkl")
            features_path = os.path.join(BASE_DIR, "output", "top_features.json")

            # SAFETY CHECKS (VERY IMPORTANT)
            if not os.path.exists(model_path):
                raise Exception(f"Model not found at: {model_path}")

            if not os.path.exists(scaler_path):
                raise Exception(f"Scaler not found at: {scaler_path}")

            if not os.path.exists(features_path):
                raise Exception(f"top_features.json not found at: {features_path}")

            # Load model ONLY ONCE
            cls._instance.model = joblib.load(model_path)
            cls._instance.scaler = joblib.load(scaler_path)

            with open(features_path, "r") as f:
                cls._instance.top_features = json.load(f)

            cls._instance.forensic_cols = [
                "PSNR","SSIM","Entropy_Diff","Mean_Diff","Variance_Diff",
                "Noise_Variance","Median_Residual","Histogram_Corr",
                "Laplacian_Var","Edge_Density","Tenengrad_Score",
                "HF_Energy_Ratio","DCT_Variance","Wavelet_Energy_Change",
                "JPEG_Block_Strength","Scaling_Artifact_Score"
            ]

            cls._instance.attack_types = [
                "GAUSSIAN_NOISE",
                "SALT_PEPPER",
                "BLUR",
                "ROTATION",
                "CROP",
                "SCALING"
            ]

        return cls._instance

    # ---------------------------------------
    # MAIN PREDICT FUNCTION
    # ---------------------------------------
    def predict(self, original, attacked):

        # Extract features
        forensic = compute_features(original, attacked)
        cnn_feat = extract_cnn_features(attacked)

        feature_dict = {}

        # CNN features
        for i, val in enumerate(cnn_feat):
            feature_dict[f"cnn_{i}"] = val

        # Forensic features
        feature_dict.update(forensic)

        # Build feature vector(Ensure all forensic features exist)
        for f in self.forensic_cols:
            if f not in forensic:
                forensic[f] = 0

        # 🔹 Scale forensic features FIRST
        forensic_values = [forensic[f] for f in self.forensic_cols]
        scaled_forensic = self.scaler.transform([forensic_values])[0]

        scaled_dict = dict(zip(self.forensic_cols, scaled_forensic))

        # 🔹 Build final feature vector (CORRECT WAY)
        feature_vector = []

        for f in self.top_features:
            if f.startswith("cnn_"):
                feature_vector.append(feature_dict.get(f, 0))
            else:
                feature_vector.append(scaled_dict.get(f, 0))

        feature_vector = np.array(feature_vector).reshape(1, -1)

        # 🔹 Predict
        pred = self.model.predict(feature_vector)[0]
        prob = [
            est.predict_proba(feature_vector)[0][1]
            for est in self.model.estimators_
        ]

        detected = []

        for i, attack in enumerate(self.attack_types):
            if pred[i] == 1 and prob[i] > 0.6:
                detected.append({
                    "attack": attack,
                    "confidence": float(round(prob[i], 2))
                })

        if not detected:
            return []

        return detected