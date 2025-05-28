import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import lightgbm as lgbm
from xgboost.callback import TrainingCallback
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, roc_auc_score, 
                             confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_recall_curve, auc,
                             log_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from scipy.fftpack import dct
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = (256, 256)
LBP_PARAMS = {'P': 24, 'R': 3.0, 'method': 'uniform'}
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
COLOR_SPACE = 'ycrcb'  # More effective than RGB for recapture detection
RANDOM_STATE = 42

def load_images(folder):
    """Load and preprocess images with error handling"""
    images = []
    filenames = []
    for fname in os.listdir(folder):
        filepath = os.path.join(folder, fname)
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        try:
            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Failed to load {filepath}")
                continue
            if img.ndim != 3:
                print(f"Warning: Image {filepath} is not color, skipping.")
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) 
            images.append(img)
            filenames.append(fname)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    return images, filenames

#Feature Extraction

def compute_specular_features(img):
    """Multi-channel specularity analysis"""
    spec_features = []
    for channel in range(3):
        chan_img = img[:, :, channel]
        spec_map = cv2.subtract(chan_img, cv2.GaussianBlur(chan_img, (5,5), 0))
        spec_map = cv2.normalize(spec_map, None, 0, 255, cv2.NORM_MINMAX)
        hist = np.histogram(spec_map, bins=8, range=(0, 255))[0]
        spec_features.extend(hist / hist.sum())
    return spec_features

def compute_lbp_features(img):
    """Multi-channel LBP with color-space optimization"""
    lbp_features = []
    for channel in range(3):  # Y, Cr, Cb channels
        chan_img = img[:, :, channel]
        lbp = local_binary_pattern(chan_img, **LBP_PARAMS)
        hist, _ = np.histogram(lbp, bins=np.arange(0, LBP_PARAMS['P'] + 3), 
                              range=(0, LBP_PARAMS['P'] + 2))
        lbp_features.extend(hist / hist.sum())
    return lbp_features

def compute_glcm_features(img):
    """Multi-channel GLCM features"""
    glcm_features = []
    for channel in range(3):
        chan_img = img[:, :, channel]
        glcm = graycomatrix(chan_img, distances=[1], angles=GLCM_ANGLES,
                           levels=256, symmetric=True, normed=True)
        for prop in GLCM_PROPS:
            glcm_features.extend(graycoprops(glcm, prop).ravel())
    return glcm_features

def compute_color_moments(img):
    """Statistical color features in YCrCb space"""
    moments = []
    for channel in range(3):
        data = img[:, :, channel].flatten()
        moments.extend([
            np.mean(data),
            np.std(data),
            stats.skew(data),
            np.median(data),
            stats.kurtosis(data)
        ])
    return moments

def compute_blur_metric(img):
    """Multi-channel blur assessment"""
    blur_metrics = []
    for channel in range(3):
        chan_img = img[:, :, channel]
        blur_metrics.append(cv2.Laplacian(chan_img, cv2.CV_64F).var())
    return blur_metrics

def compute_noise_pattern(img):
    """Extract noise residuals that can reveal recapture artifacts"""
    noise_features = []
    for channel in range(3):
        chan_img = img[:, :, channel].astype(np.uint8)
        # Apply denoising filter
        denoised = cv2.fastNlMeansDenoising(chan_img, None, 10, 7, 21)
        # Extract noise pattern
        noise = chan_img.astype(np.float32) - denoised.astype(np.float32)
        # Compute statistics on noise
        noise_features.extend([
            np.mean(noise),
            np.std(noise),
            stats.skew(noise.flatten()),
            stats.kurtosis(noise.flatten())
        ])
    return noise_features

def compute_dct_features(img, block_size=8):
    """Extract DCT coefficients which can reveal compression artifacts"""
    dct_features = []
    for channel in range(3):
        chan_img = img[:, :, channel].astype(np.float32)
        h, w = chan_img.shape
        # Truncate to multiple of block_size
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        chan_img = chan_img[:h_blocks*block_size, :w_blocks*block_size]
        
        # Process blocks
        hist = np.zeros(8)  # Histogram of DC coefficients
        for i in range(0, chan_img.shape[0], block_size):
            for j in range(0, chan_img.shape[1], block_size):
                block = chan_img[i:i+block_size, j:j+block_size]
                coeffs = dct(dct(block.T, norm='ortho').T, norm='ortho')
                # Use DC coefficient (top-left) for histogram
                dc = coeffs[0, 0]
                # Simple binning 
                bin_idx = min(int(np.abs(dc) / 32), 7)
                hist[bin_idx] += 1
                
        # Normalize histogram
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        dct_features.extend(hist)
    
    return dct_features

def compute_edge_coherence(img):
    """Measure edge coherence patterns often affected by recapture"""
    edge_features = []
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    # Multiple edge detectors for robustness
    edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Compute gradient magnitude and direction
    gradient_magnitude = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
    gradient_direction = np.arctan2(edges_sobel_y, edges_sobel_x)
    
    # Direction histogram (sensitive to recapture alignment)
    direction_hist, _ = np.histogram(gradient_direction, 
                                   bins=18, range=(-np.pi, np.pi))
    direction_hist = direction_hist / direction_hist.sum() if direction_hist.sum() > 0 else direction_hist
    
    # Edge coherence features
    edge_features.extend(direction_hist)
    edge_features.extend([
        np.mean(gradient_magnitude),
        np.std(gradient_magnitude),
        np.mean(np.abs(edges_laplacian)),
        np.std(np.abs(edges_laplacian))
    ])
    
    return edge_features

def compute_chromatic_aberration(img):
    """Detect chromatic aberration differences (often stronger in recaptured images)"""
    # Convert to BGR for channel differences
    bgr_img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    b, g, r = cv2.split(bgr_img)
    
    # Channel differences (sensitive to chromatic aberration)
    r_g_diff = cv2.absdiff(r, g)
    r_b_diff = cv2.absdiff(r, b)
    g_b_diff = cv2.absdiff(g, b)
    
    # Statistics on channel differences
    aberration_features = []
    for diff in [r_g_diff, r_b_diff, g_b_diff]:
        aberration_features.extend([
            np.mean(diff),
            np.std(diff),
            np.percentile(diff, 90)  # High percentile captures strongest aberrations
        ])
    
    return aberration_features

def extract_features(img):
    """Integrated feature extraction pipeline with NaN handling"""
    features = []
    try:
        # Original features
        features.extend(compute_specular_features(img))
        features.extend(compute_lbp_features(img))
        features.extend(compute_glcm_features(img))
        features.extend(compute_color_moments(img))
        features.extend(compute_blur_metric(img))
        
        # New advanced features
        features.extend(compute_noise_pattern(img))
        features.extend(compute_dct_features(img))
        features.extend(compute_edge_coherence(img))
        features.extend(compute_chromatic_aberration(img))
        
        # Check for NaN values and replace them
        features = np.array(features)
        if np.isnan(features).any():
            # Replace NaN with feature means or zeros
            nan_indices = np.where(np.isnan(features))[0]
            for idx in nan_indices:
                features[idx] = 0.0  # Replace with zero as a safe value
            print(f"Warning: NaN values found and replaced in feature extraction")
            
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        # Return a feature vector of zeros as a fallback
        num_features = 247  # Expected number of features
        features = np.zeros(num_features)
        
    return list(features)

def extract_feature_names():
    """Generate feature names for interpretability"""
    feature_names = []
    
    # Specular features
    for channel in ['Y', 'Cr', 'Cb']:
        for i in range(8):
            feature_names.append(f'spec_{channel}_bin{i}')
    
    # LBP features
    for channel in ['Y', 'Cr', 'Cb']:
        for i in range(LBP_PARAMS['P'] + 2):
            feature_names.append(f'lbp_{channel}_bin{i}')
    
    # GLCM features
    for channel in ['Y', 'Cr', 'Cb']:
        for prop in GLCM_PROPS:
            for angle in ['0', '45', '90', '135']:
                feature_names.append(f'glcm_{channel}_{prop}_{angle}')
    
    # Color moments
    for channel in ['Y', 'Cr', 'Cb']:
        for moment in ['mean', 'std', 'skew', 'median', 'kurtosis']:
            feature_names.append(f'color_{channel}_{moment}')
    
    # Blur metrics
    for channel in ['Y', 'Cr', 'Cb']:
        feature_names.append(f'blur_{channel}')
    
    # Noise pattern features
    for channel in ['Y', 'Cr', 'Cb']:
        for stat in ['mean', 'std', 'skew', 'kurtosis']:
            feature_names.append(f'noise_{channel}_{stat}')
    
    # DCT features
    for channel in ['Y', 'Cr', 'Cb']:
        for i in range(8):
            feature_names.append(f'dct_{channel}_bin{i}')
    
    # Edge coherence features
    for i in range(18):
        feature_names.append(f'edge_dir_bin{i}')
    feature_names.extend(['edge_magnitude_mean', 'edge_magnitude_std', 
                          'edge_laplacian_mean', 'edge_laplacian_std'])
    
    # Chromatic aberration features
    for channels in ['RG', 'RB', 'GB']:
        for stat in ['mean', 'std', 'p90']:
            feature_names.append(f'aberration_{channels}_{stat}')
    
    return feature_names

#Visualising Training Metrics

class TrainingVisualizer:
    """Class to track and visualize training metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def add_metrics(self, model_name, train_loss, val_loss, iteration, 
                   train_auc=None, val_auc=None):
        """Add comprehensive training metrics"""
        if model_name not in self.metrics:
            self.metrics[model_name] = {
                'iterations': [],
                'train_loss': [],
                'val_loss': [],
                'train_auc': [],
                'val_auc': []
            }
        
        self.metrics[model_name]['iterations'].append(iteration)
        self.metrics[model_name]['train_loss'].append(train_loss)
        self.metrics[model_name]['val_loss'].append(val_loss)
        
        if train_auc is not None:
            self.metrics[model_name]['train_auc'].append(train_auc)
        if val_auc is not None:
            self.metrics[model_name]['val_auc'].append(val_auc)
    
    def plot_training_curves(self, model_name, save_path=None):
        """Plot comprehensive training visualization"""
        if model_name not in self.metrics:
            print(f"No training data found for {model_name}")
            return
        
        metrics = self.metrics[model_name]
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Training Analysis', fontsize=16)
        
        # Plot 1: Loss Curves
        ax1.plot(metrics['iterations'], metrics['train_loss'], 
                label='Training Loss', color='blue', linewidth=2)
        ax1.plot(metrics['iterations'], metrics['val_loss'], 
                label='Validation Loss', color='red', linewidth=2)
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: AUC Curves
        if len(metrics['train_auc']) > 0:
            ax2.plot(metrics['iterations'], metrics['train_auc'], 
                    label='Training AUC', color='blue', linewidth=2)
            ax2.plot(metrics['iterations'], metrics['val_auc'], 
                    label='Validation AUC', color='red', linewidth=2)
            ax2.set_title('AUC Curves')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('AUC')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Loss Difference (Overfitting Analysis)
        loss_diff = np.array(metrics['val_loss']) - np.array(metrics['train_loss'])
        ax3.plot(metrics['iterations'], loss_diff, color='purple', alpha=0.5)
        ax3.fill_between(metrics['iterations'], 0, loss_diff, 
                        where=(loss_diff > 0), color='red', alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_title('Overfitting Analysis')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Validation - Training Loss')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training Stability
        if len(metrics['train_loss']) > 1:
            train_stability = np.gradient(metrics['train_loss'])
            val_stability = np.gradient(metrics['val_loss'])
            ax4.plot(metrics['iterations'][1:], train_stability[1:], 
                    label='Training', color='blue', alpha=0.5)
            ax4.plot(metrics['iterations'][1:], val_stability[1:], 
                    label='Validation', color='red', alpha=0.5)
            ax4.set_title('Training Stability')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Loss Gradient')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

#Training functions

def train_xgboost_with_tracking(model, X_train, y_train, X_val, y_val, visualizer):
    """Train XGBoost with enhanced visualization tracking"""
    print("Training XGBoost with validation tracking...")
    
    evals_result = {}
    
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # Get evaluation results from the model's evals_result_ attribute
    if hasattr(model, 'evals_result_') and model.evals_result_:
        evals_result = model.evals_result_
        
        # Extract and store metrics after training
        iterations = len(evals_result['validation_0']['logloss'])
        for i in range(iterations):
            visualizer.add_metrics(
                'XGBoost',
                train_loss=evals_result['validation_0']['logloss'][i],
                val_loss=evals_result['validation_1']['logloss'][i],
                iteration=i,
                train_auc=evals_result['validation_0']['auc'][i] if 'auc' in evals_result['validation_0'] else None,
                val_auc=evals_result['validation_1']['auc'][i] if 'auc' in evals_result['validation_1'] else None
            )
    
    return model, evals_result

def train_lightgbm_with_tracking(model, X_train, y_train, X_val, y_val, visualizer):
    """Train LightGBM with robust validation tracking"""
    print("Training LightGBM with validation tracking...")
    
    try:
        # Train with evaluation sets
        model.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)],
                 callbacks=[lgbm.log_evaluation(0)])  # Silent training
        
        # Extract training history
        if hasattr(model, 'evals_result_'):
            history = model.evals_result_
            
            # Robust key extraction
            train_keys = [k for k in history.keys() if 'train' in k.lower()]
            val_keys = [k for k in history.keys() if 'val' in k.lower() or 'valid' in k.lower()]
            
            if train_keys and val_keys:
                train_key = train_keys[0]
                val_key = val_keys[0]
                
                train_metrics = history[train_key]
                val_metrics = history[val_key]
                
                # Find available metrics
                available_metrics = set(train_metrics.keys()) & set(val_metrics.keys())
                
                # Initialize variables for storing metrics
                train_loss_values = None
                val_loss_values = None
                train_auc_values = None
                val_auc_values = None
                
                # Extract metrics
                for metric in available_metrics:
                    if 'logloss' in metric or 'binary_logloss' in metric:
                        train_loss_values = train_metrics[metric]
                        val_loss_values = val_metrics[metric]
                    elif 'auc' in metric:
                        train_auc_values = train_metrics[metric]
                        val_auc_values = val_metrics[metric]
                
                # Add metrics to visualizer using the correct method
                if train_loss_values and val_loss_values:
                    iterations = len(train_loss_values)
                    for i in range(iterations):
                        visualizer.add_metrics(
                            'LightGBM',
                            train_loss=train_loss_values[i],
                            val_loss=val_loss_values[i],
                            iteration=i,
                            train_auc=train_auc_values[i] if train_auc_values else None,
                            val_auc=val_auc_values[i] if val_auc_values else None
                        )
        
    except Exception as e:
        print(f"Could not extract LightGBM training history: {e}")
        # Fallback: train without callbacks
        model.fit(X_train, y_train)
    
    return model

def plot_cross_validation_scores(cv_scores, model_name):
    """Visualize cross-validation scores"""
    plt.figure(figsize=(10, 6))
    
    # Box plot of CV scores
    plt.subplot(1, 2, 1)
    plt.boxplot(cv_scores, labels=[model_name])
    plt.ylabel('AUC Score')
    plt.title('Cross-Validation Score Distribution')
    plt.grid(True, alpha=0.3)
    
    # Individual fold scores
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=np.mean(cv_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(cv_scores):.4f}')
    plt.axhline(y=np.mean(cv_scores) + np.std(cv_scores), color='orange', 
                linestyle=':', alpha=0.7, label=f'±1 STD')
    plt.axhline(y=np.mean(cv_scores) - np.std(cv_scores), color='orange', 
                linestyle=':', alpha=0.7)
    plt.xlabel('Fold')
    plt.ylabel('AUC Score')
    plt.title('Per-Fold Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_cv_scores.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print(f"\n=== Cross-Validation Analysis for {model_name} ===")
    print(f"Mean AUC: {np.mean(cv_scores):.4f}")
    print(f"Std AUC: {np.std(cv_scores):.4f}")
    print(f"Min AUC: {np.min(cv_scores):.4f}")
    print(f"Max AUC: {np.max(cv_scores):.4f}")
    print(f"95% Confidence Interval: [{np.mean(cv_scores) - 1.96*np.std(cv_scores):.4f}, "
          f"{np.mean(cv_scores) + 1.96*np.std(cv_scores):.4f}]")
    
    # Variance analysis
    variance_ratio = np.std(cv_scores) / np.mean(cv_scores)
    if variance_ratio > 0.1:
        print("HIGH VARIANCE across folds - model may be unstable")
    elif variance_ratio > 0.05:
        print("MODERATE VARIANCE across folds")
    else:
        print("LOW VARIANCE across folds - stable model")

def plot_learning_curves(model, X_train, y_train, cv_folds=5):
    """Plot learning curves to analyze bias-variance tradeoff"""
    from sklearn.model_selection import learning_curve
    
    print("Computing learning curves...")
    
    # Define training sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train, 
        train_sizes=train_sizes,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='blue')
    
    plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('AUC Score')
    plt.title('Learning Curves - Bias vs Variance Analysis')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add analysis text
    final_gap = train_mean[-1] - val_mean[-1]
    plt.text(0.02, 0.02, f'Final Gap: {final_gap:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print(f"\n=== Learning Curve Analysis ===")
    print(f"Final training score: {train_mean[-1]:.4f} (±{train_std[-1]:.4f})")
    print(f"Final validation score: {val_mean[-1]:.4f} (±{val_std[-1]:.4f})")
    print(f"Bias-Variance gap: {final_gap:.4f}")
    
    if final_gap > 0.1:
        print("HIGH VARIANCE - Model is overfitting")
        print("   Recommendations: More data, regularization, or simpler model")
    elif final_gap > 0.05:
        print("MODERATE VARIANCE - Some overfitting")
        print("   Recommendations: Consider regularization or early stopping")
    elif val_mean[-1] < 0.8:
        print("HIGH BIAS - Model is underfitting")
        print("   Recommendations: More complex model or better features")
    else:
        print("GOOD BIAS-VARIANCE BALANCE")

# === Visualization Functions ===

def plot_feature_importance(model, feature_names, top_n=30):
    """Plot the most important features"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importances')
        
        # Plot top N features
        n = min(top_n, len(feature_names))
        plt.barh(range(n), importances[indices[:n]], align='center')
        plt.yticks(range(n), [feature_names[i] for i in indices[:n]])
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_misclassified(images, indices, y_true, y_pred, title, max_images=16):
    """Color visualization of misclassified samples"""
    n_images = min(len(indices), max_images)
    if n_images == 0:
        print(f"No images to display for {title}")
        return
    
    plt.figure(figsize=(15, 10))
    plt.suptitle(title, fontsize=16)
    rows = int(np.ceil(n_images / 4))
    
    for i, idx in enumerate(indices[:n_images]):
        plt.subplot(rows, 4, i+1)
        rgb_img = cv2.cvtColor(images[idx], cv2.COLOR_YCrCb2RGB)
        plt.imshow(rgb_img)
        plt.title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}", fontsize=8)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_pr_curve(y_test, y_proba):
    """Plot precision-recall curve"""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pr_auc

def evaluate_threshold(y_test, y_proba, threshold=0.5):
    """Evaluate model with custom threshold"""
    y_pred_threshold = (y_proba >= threshold).astype(int)
    print(f"\nEvaluation with threshold = {threshold}:")
    print(classification_report(y_test, y_pred_threshold, target_names=['Good', 'Fraud']))
    
    cm = confusion_matrix(y_test, y_pred_threshold)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Good', 'Fraud'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix (threshold={threshold})')
    plt.savefig(f'confusion_matrix_thresh_{threshold}.png', dpi=300, bbox_inches='tight')
    plt.show()

def ensemble_predictions(models, X_test):
    """Combine predictions from multiple models"""
    all_preds = np.zeros((len(X_test), len(models)))
    
    for i, model in enumerate(models):
        all_preds[:, i] = model.predict_proba(X_test)[:, 1]
    
    # Average predictions
    ensemble_proba = np.mean(all_preds, axis=1)
    return ensemble_proba

# === Main Training Pipeline ===

def main():
    # Load datasets
    print("Loading images...")
    good_images, good_filenames = load_images('/Users/aryangupta/Desktop/ReusableBagsFraudImageDetection/cleaned_dataset/good_cleaned')
    fraud_images, fraud_filenames = load_images('/Users/aryangupta/Desktop/ReusableBagsFraudImageDetection/cleaned_dataset/fraud_cleaned')
    all_images = good_images + fraud_images
    all_filenames = good_filenames + fraud_filenames
    
    print(f"Loaded {len(good_images)} good images and {len(fraud_images)} fraud images")
    
    # Create labels and features
    print("Extracting features...")
    X = [extract_features(img) for img in all_images]
    y = [0]*len(good_images) + [1]*len(fraud_images) 
    
    # Generate feature names for interpretability
    feature_names = extract_feature_names()
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature extraction complete. Feature vector size: {X.shape[1]}")
    
    # Check for NaN values
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values in features. Replacing with zeros.")
        X = np.nan_to_num(X, nan=0.0)
    
      # ========== NEW HOLD-OUT IMPLEMENTATION ==========
    # 1. Primary Split: 80% development, 20% final holdout
    indices = np.arange(len(X))
    X_dev, X_holdout, y_dev, y_holdout, idx_dev, idx_holdout = train_test_split(
        X, y, indices, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    # 2. Secondary Split: 80% train-val, 20% test from development set
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        X_dev, y_dev, idx_dev, test_size=0.2, stratify=y_dev, random_state=RANDOM_STATE
    )
    
    # 3. Tertiary Split: 75% train, 25% validation
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val, y_train_val, idx_train_val, 
        test_size=0.25, stratify=y_train_val, random_state=RANDOM_STATE
    )
    
    # Store holdout metadata
    holdout_images = [all_images[i] for i in idx_holdout]
    holdout_filenames = [all_filenames[i] for i in idx_holdout]
    # ========== END HOLD-OUT IMPLEMENTATION ==========
    
    test_images = [all_images[i] for i in idx_test]
    test_filenames = [all_filenames[i] for i in idx_test]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples") 
    print(f"Test set: {len(X_test)} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection using embedded method
    print("\nPerforming feature selection...")
    selector = SelectFromModel(XGBClassifier(n_estimators=100, random_state=RANDOM_STATE), threshold='median')
    selector.fit(X_train_scaled, y_train)
    
    # Get selected features
    selected_features = selector.get_support()
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]
    X_train_selected = selector.transform(X_train_scaled)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"Selected {len(selected_feature_names)} out of {len(feature_names)} features")
    
    # Class weight calculation
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    scale_pos_weight = class_weights[1]/class_weights[0]

    # Initialize training visualizer
    visualizer = TrainingVisualizer()

    # Create models with robust parameters
    print("\nInitializing models...")
    
    # XGBoost model - Using only stable parameters
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    
    # LightGBM model - Using only stable parameters
    lgb_model = lgbm.LGBMClassifier(
        n_estimators=300,
        num_leaves=50,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced',
        verbose=-1  # Silent training
    )
    
    # Apply SMOTE if needed
    if np.bincount(y_train)[1] / len(y_train) < 0.25:  # If minority class < 25%
        print("Applying SMOTE for class balance...")
        try:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, np.bincount(y_train)[1]-1))
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
            print(f"After SMOTE: {len(X_train_resampled)} samples")
        except Exception as e:
            print(f"SMOTE failed: {e}. Using original data.")
            X_train_resampled, y_train_resampled = X_train_selected, y_train
    else:
        X_train_resampled, y_train_resampled = X_train_selected, y_train
        
    print(f"Training data shape: {X_train_resampled.shape}")
    print(f"Class distribution: {np.bincount(y_train_resampled)}")
    
    # Train models with robust tracking
    print("\n=== Training Models ===")
    xgb_model, xgb_eval_results = train_xgboost_with_tracking(
        xgb_model, X_train_resampled, y_train_resampled, 
        X_val_selected, y_val, visualizer
    )
    
    lgb_model = train_lightgbm_with_tracking(
        lgb_model, X_train_resampled, y_train_resampled,
        X_val_selected, y_val, visualizer
    )
    
    # Plot training curves if available - FIXED: Use visualizer.metrics instead of train_metrics
    print("\n=== Plotting Training Analysis ===")
    if 'XGBoost' in visualizer.metrics:
        visualizer.plot_training_curves('XGBoost', 'xgboost_training_curves.png')
        # Add analyze_overfitting method if it exists in your visualizer class
        if hasattr(visualizer, 'analyze_overfitting'):
            visualizer.analyze_overfitting('XGBoost')
    
    if 'LightGBM' in visualizer.metrics:
        visualizer.plot_training_curves('LightGBM', 'lightgbm_training_curves.png')
        if hasattr(visualizer, 'analyze_overfitting'):
            visualizer.analyze_overfitting('LightGBM')
    
    # Cross-validation analysis
    print("\n=== Cross-Validation Analysis ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        XGBClassifier(**{k: v for k, v in xgb_model.get_params().items() 
                        if k not in ['n_jobs']}), 
        X_train_selected, y_train, 
        cv=cv, scoring='roc_auc', n_jobs=-1
    )
    
    plot_cross_validation_scores(cv_scores, 'XGBoost')
    
    # Learning curves analysis
    print("\n=== Learning Curves Analysis ===")
    # Use a simpler model for learning curves to speed up computation
    simple_xgb = XGBClassifier(
        n_estimators=50, 
        max_depth=6,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight
    )
    plot_learning_curves(simple_xgb, X_train_selected, y_train)
    
    # Model predictions
    print("\n=== Making Predictions ===")
    y_pred_xgb = xgb_model.predict(X_test_selected)
    y_proba_xgb = xgb_model.predict_proba(X_test_selected)[:, 1]
    
    y_pred_lgb = lgb_model.predict(X_test_selected)
    y_proba_lgb = lgb_model.predict_proba(X_test_selected)[:, 1]
    
    # Ensemble predictions
    models = [xgb_model, lgb_model]
    y_proba_ensemble = ensemble_predictions(models, X_test_selected)
    y_pred_ensemble = (y_proba_ensemble >= 0.5).astype(int)
    
    # Evaluation
    print("\n=== Model Performance ===")
    print("\n--- XGBoost Performance ---")
    print(classification_report(y_test, y_pred_xgb, target_names=['Good', 'Fraud']))
    print(f"ROC AUC (XGBoost): {roc_auc_score(y_test, y_proba_xgb):.4f}")
    
    print("\n--- LightGBM Performance ---")
    print(classification_report(y_test, y_pred_lgb, target_names=['Good', 'Fraud']))
    print(f"ROC AUC (LightGBM): {roc_auc_score(y_test, y_proba_lgb):.4f}")
    
    print("\n--- Ensemble Performance ---")
    print(classification_report(y_test, y_pred_ensemble, target_names=['Good', 'Fraud']))
    print(f"ROC AUC (Ensemble): {roc_auc_score(y_test, y_proba_ensemble):.4f}")
    
    # Validation set performance for final check
    y_val_pred_xgb = xgb_model.predict_proba(X_val_selected)[:, 1]
    y_val_pred_lgb = lgb_model.predict_proba(X_val_selected)[:, 1]
    y_val_ensemble = (y_val_pred_xgb + y_val_pred_lgb) / 2
    
    print(f"\n--- Validation Set Performance ---")
    print(f"XGBoost Validation AUC: {roc_auc_score(y_val, y_val_pred_xgb):.4f}")
    print(f"LightGBM Validation AUC: {roc_auc_score(y_val, y_val_pred_lgb):.4f}")
    print(f"Ensemble Validation AUC: {roc_auc_score(y_val, y_val_ensemble):.4f}")
    
    # Plot comprehensive model comparison
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: ROC AUC comparison
    plt.subplot(2, 2, 1)
    models_names = ['XGBoost', 'LightGBM', 'Ensemble']
    test_aucs = [roc_auc_score(y_test, y_proba_xgb), 
                 roc_auc_score(y_test, y_proba_lgb),
                 roc_auc_score(y_test, y_proba_ensemble)]
    val_aucs = [roc_auc_score(y_val, y_val_pred_xgb),
                roc_auc_score(y_val, y_val_pred_lgb),
                roc_auc_score(y_val, y_val_ensemble)]
    
    x = np.arange(len(models_names))
    width = 0.35
    
    plt.bar(x - width/2, test_aucs, width, label='Test AUC', alpha=0.8)
    plt.bar(x + width/2, val_aucs, width, label='Validation AUC', alpha=0.8)
    plt.ylabel('AUC Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Prediction distribution
    plt.subplot(2, 2, 2)
    plt.hist(y_proba_ensemble[y_test == 0], bins=30, alpha=0.7, label='Good Images', density=True)
    plt.hist(y_proba_ensemble[y_test == 1], bins=30, alpha=0.7, label='Fraud Images', density=True)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Calibration plot
    try:
        from sklearn.calibration import calibration_curve
        
        plt.subplot(2, 2, 3)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_proba_ensemble, n_bins=10
        )
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Ensemble")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel('Fraction of Positives')
        plt.xlabel('Mean Predicted Probability')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
    except ImportError:
        plt.subplot(2, 2, 3)
        plt.text(0.5, 0.5, 'Calibration plot\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Calibration Plot')
    
    # Subplot 4: Error analysis
    plt.subplot(2, 2, 4)
    errors = np.abs(y_test - y_proba_ensemble)
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot PR curve for ensemble
    pr_auc = plot_pr_curve(y_test, y_proba_ensemble)
    print(f"PR AUC (Ensemble): {pr_auc:.4f}")
    
    # Confusion matrix for ensemble
    cm = confusion_matrix(y_test, y_pred_ensemble)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Good', 'Fraud'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix (Ensemble)')
    plt.savefig('confusion_matrix_ensemble.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Try different thresholds
    evaluate_threshold(y_test, y_proba_ensemble, threshold=0.3)
    evaluate_threshold(y_test, y_proba_ensemble, threshold=0.7)
    
    # Visualize misclassified images
    false_positives = np.where((y_test == 0) & (y_pred_ensemble == 1))[0]
    false_negatives = np.where((y_test == 1) & (y_pred_ensemble == 0))[0]
    
    print(f"\nFound {len(false_positives)} false positives and {len(false_negatives)} false negatives")
    
    if len(test_images) > 0:
        plot_misclassified(test_images, false_positives, y_test, y_pred_ensemble, 
                          "False Positives (Good classified as Fraud)")
        plot_misclassified(test_images, false_negatives, y_test, y_pred_ensemble,
                          "False Negatives (Fraud classified as Good)")
    
    # Feature importance visualization
    print("\nGenerating feature importance plot...")
    plot_feature_importance(xgb_model, selected_feature_names)
    
    # Generate final training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset: {len(good_images)} good + {len(fraud_images)} fraud images")
    print(f"Features: {len(selected_feature_names)} selected from {len(feature_names)} total")
    print(f"Training samples: {len(X_train_resampled)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print("\nBest Model Performance:")
    print(f"  Validation AUC: {max(roc_auc_score(y_val, y_val_pred_xgb), roc_auc_score(y_val, y_val_pred_lgb), roc_auc_score(y_val, y_val_ensemble)):.4f}")
    print(f"  Test AUC: {roc_auc_score(y_test, y_proba_ensemble):.4f}")
    print(f"  Cross-validation AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

    # ========== NEW HOLD-OUT EVALUATION ==========
    # After final model training, evaluate on holdout
    print("\n=== Final Holdout Evaluation ===")
    
    # Transform holdout data using existing scaler/selector
    X_holdout_scaled = scaler.transform(X_holdout)
    X_holdout_selected = selector.transform(X_holdout_scaled)
    
    # Generate predictions
    y_holdout_proba = ensemble_predictions(models, X_holdout_selected)
    y_holdout_pred = (y_holdout_proba >= 0.5).astype(int)
    
    # Comprehensive evaluation
    print("Holdout Classification Report:")
    print(classification_report(y_holdout, y_holdout_pred, target_names=['Good', 'Fraud']))
    print(f"Holdout ROC AUC: {roc_auc_score(y_holdout, y_holdout_proba):.4f}")
    print(f"Holdout Log Loss: {log_loss(y_holdout, y_holdout_proba):.4f}")
    
    # Visualize errors
    plot_misclassified(holdout_images, 
                      np.where(y_holdout != y_holdout_pred)[0],
                      y_holdout, y_holdout_pred,
                      "Holdout Misclassifications")
    
    # Save holdout predictions
    holdout_results = pd.DataFrame({
        'filename': holdout_filenames,
        'true_label': y_holdout,
        'predicted_label': y_holdout_pred,
        'fraud_probability': y_holdout_proba
    })
    holdout_results.to_csv('holdout_predictions.csv', index=False)
    
    # Save models and pipeline components
    print("\nSaving models and preprocessing components...")
    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(selector, 'feature_selector.pkl')
    joblib.dump(xgb_model, 'xgb_classifier.pkl')
    joblib.dump(lgb_model, 'lgbm_classifier.pkl')
    
    # Save training history - FIXED: Use visualizer.metrics instead of train_metrics
    try:
        import pickle
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(visualizer.metrics, f)  # Changed from train_metrics to metrics
        print("Training history saved successfully.")
    except Exception as e:
        print(f"Could not save training history: {e}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("  - Models: xgb_classifier.pkl, lgbm_classifier.pkl")
    print("  - Preprocessing: feature_scaler.pkl, feature_selector.pkl")
    print("  - Training history: training_history.pkl")
    print("  - Visualizations: Various PNG files")
    print("\nCheck the generated plots for detailed training analysis!")


if __name__ == "__main__":
    main()