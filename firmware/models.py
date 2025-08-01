"""
Multi-Modal Machine Learning Models for Robotic Arm Control
Based on research from EMG-IMU fusion, LSTM networks, and reinforcement learning

Key features:
- Individual modality models (EMG-only, IMU-only, Visual-only)
- Multi-modal fusion models
- LSTM networks for temporal dependencies
- Reinforcement learning for continuous control
- Comprehensive evaluation and comparison
- Real-time inference pipeline
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import gym
from gym import spaces
import stable_baselines3 as sb3
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import optuna
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: str  # 'lstm', 'transformer', 'cnn_lstm', 'rf', 'gbm'
    input_modalities: List[str]  # ['emg', 'imu', 'visual']
    sequence_length: int = 50
    hidden_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10

class RoboticArmDataset(Dataset):
    """PyTorch dataset for multi-modal robotic arm data"""
    
    def __init__(self, data_samples, config: ModelConfig, target_type='joint_angles'):
        self.samples = data_samples
        self.config = config
        self.target_type = target_type
        self.sequence_length = config.sequence_length
        
        # Initialize scalers
        self.emg_scaler = StandardScaler()
        self.imu_scaler = StandardScaler()
        self.visual_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Prepare data
        self.sequences, self.targets = self._prepare_sequences()
        
    def _prepare_sequences(self):
        """Prepare sequential data for training"""
        sequences = []
        targets = []
        
        # Group samples by task/session for sequence creation
        task_groups = {}
        for sample in self.samples:
            task_key = f"{sample.task_info['task_name']}_{sample.timestamp // 60}"  # Group by minute
            if task_key not in task_groups:
                task_groups[task_key] = []
            task_groups[task_key].append(sample)
        
        # Create sequences from each task group
        for task_key, task_samples in task_groups.items():
            if len(task_samples) < self.sequence_length:
                continue
                
            # Sort by timestamp
            task_samples.sort(key=lambda x: x.timestamp)
            
            # Extract features and targets
            emg_features = []
            imu_features = []
            visual_features = []
            target_values = []
            
            for sample in task_samples:
                # EMG features
                emg_feat = [sample.emg_features[key] for key in sorted(sample.emg_features.keys())]
                emg_features.append(emg_feat)
                
                # IMU features (extract from processed data)
                imu_feat = []
                if 'accel' in sample.imu_data:
                    imu_feat.extend(sample.imu_data['accel'].flatten())
                if 'gyro' in sample.imu_data:
                    imu_feat.extend(sample.imu_data['gyro'].flatten())
                if 'accel_magnitude' in sample.imu_data:
                    imu_feat.append(sample.imu_data['accel_magnitude'])
                if 'gyro_magnitude' in sample.imu_data:
                    imu_feat.append(sample.imu_data['gyro_magnitude'])
                imu_features.append(imu_feat)
                
                # Visual features
                visual_feat = []
                if sample.visual_data and 'hand_landmarks' in sample.visual_data:
                    landmarks = sample.visual_data['hand_landmarks']
                    if landmarks:
                        # Flatten first hand landmarks
                        visual_feat.extend(landmarks[0].flatten())
                    else:
                        visual_feat.extend([0] * 63)  # 21 landmarks * 3 coordinates
                else:
                    visual_feat.extend([0] * 63)
                visual_features.append(visual_feat)
                
                # Target (simulate joint angles based on task)
                target = self._generate_target(sample)
                target_values.append(target)
            
            # Convert to numpy arrays
            emg_features = np.array(emg_features)
            imu_features = np.array(imu_features)
            visual_features = np.array(visual_features)
            target_values = np.array(target_values)
            
            # Create sliding window sequences
            for i in range(len(task_samples) - self.sequence_length + 1):
                end_idx = i + self.sequence_length
                
                seq_data = {}
                if 'emg' in self.config.input_modalities:
                    seq_data['emg'] = emg_features[i:end_idx]
                if 'imu' in self.config.input_modalities:
                    seq_data['imu'] = imu_features[i:end_idx]
                if 'visual' in self.config.input_modalities:
                    seq_data['visual'] = visual_features[i:end_idx]
                
                sequences.append(seq_data)
                targets.append(target_values[end_idx-1])  # Predict current state
        
        # Fit scalers and normalize data
        self._fit_scalers(sequences, targets)
        sequences = self._normalize_sequences(sequences)
        targets = self.target_scaler.fit_transform(np.array(targets).reshape(-1, 1)).flatten()
        
        return sequences, targets
    
    def _generate_target(self, sample):
        """Generate target values (joint angles) based on sample data"""
        # This is a simplified target generation - in practice, you'd have actual joint angle measurements
        # or inverse kinematics calculations based on hand position
        
        if sample.visual_data and sample.visual_data.get('hand_landmarks'):
            landmarks = sample.visual_data['hand_landmarks']
            if landmarks:
                # Calculate basic arm configuration from hand position
                palm_center = np.mean(landmarks[0][[0, 5, 9, 13, 17]], axis=0)
                
                # Simulate 6 DOF arm joint angles based on hand position and EMG
                joint_angles = np.array([
                    palm_center[0] * 180 - 90,  # Shoulder pan
                    palm_center[1] * 180 - 90,  # Shoulder tilt  
                    palm_center[2] * 90,        # Shoulder roll
                    np.sum(list(sample.emg_features.values())[:4]) * 10,  # Elbow
                    np.sum(list(sample.emg_features.values())[4:6]) * 15,  # Wrist pitch
                    np.sum(list(sample.emg_features.values())[6:8]) * 20   # Wrist roll
                ])
                
                return np.clip(joint_angles, -180, 180)
        
        # Fallback: generate from EMG patterns
        emg_values = list(sample.emg_features.values())
        return np.array([
            emg_values[0] * 45 if len(emg_values) > 0 else 0,
            emg_values[1] * 60 if len(emg_values) > 1 else 0,
            emg_values[2] * 30 if len(emg_values) > 2 else 0,
            emg_values[3] * 90 if len(emg_values) > 3 else 0,
            emg_values[4] * 45 if len(emg_values) > 4 else 0,
            emg_values[5] * 60 if len(emg_values) > 5 else 0,
        ])
    
    def _fit_scalers(self, sequences, targets):
        """Fit normalization scalers"""
        if 'emg' in self.config.input_modalities and sequences:
            emg_data = np.vstack([seq['emg'] for seq in sequences if 'emg' in seq])
            self.emg_scaler.fit(emg_data.reshape(-1, emg_data.shape[-1]))
            
        if 'imu' in self.config.input_modalities and sequences:
            imu_data = np.vstack([seq['imu'] for seq in sequences if 'imu' in seq])
            self.imu_scaler.fit(imu_data.reshape(-1, imu_data.shape[-1]))
            
        if 'visual' in self.config.input_modalities and sequences:
            visual_data = np.vstack([seq['visual'] for seq in sequences if 'visual' in seq])
            self.visual_scaler.fit(visual_data.reshape(-1, visual_data.shape[-1]))
    
    def _normalize_sequences(self, sequences):
        """Normalize sequence data"""
        normalized_sequences = []
        
        for seq in sequences:
            normalized_seq = {}
            
            if 'emg' in seq:
                emg_shape = seq['emg'].shape
                emg_normalized = self.emg_scaler.transform(seq['emg'].reshape(-1, emg_shape[-1]))
                normalized_seq['emg'] = emg_normalized.reshape(emg_shape)
                
            if 'imu' in seq:
                imu_shape = seq['imu'].shape
                imu_normalized = self.imu_scaler.transform(seq['imu'].reshape(-1, imu_shape[-1]))
                normalized_seq['imu'] = imu_normalized.reshape(imu_shape)
                
            if 'visual' in seq:
                visual_shape = seq['visual'].shape
                visual_normalized = self.visual_scaler.transform(seq['visual'].reshape(-1, visual_shape[-1]))
                normalized_seq['visual'] = visual_normalized.reshape(visual_shape)
            
            normalized_sequences.append(normalized_seq)
        
        return normalized_sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # Convert to tensors
        tensor_sequence = {}
        for modality, data in sequence.items():
            tensor_sequence[modality] = torch.FloatTensor(data)
        
        return tensor_sequence, torch.FloatTensor([target])

class MultiModalLSTM(nn.Module):
    """Multi-modal LSTM network for robotic arm control"""
    
    def __init__(self, config: ModelConfig, feature_dims: Dict[str, int]):
        super(MultiModalLSTM, self).__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.modalities = config.input_modalities
        
        # Modality-specific encoders
        self.modality_encoders = nn.ModuleDict()
        encoded_dims = {}
        
        for modality in self.modalities:
            if modality == 'emg':
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Linear(feature_dims[modality], config.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(config.hidden_size, config.hidden_size // 2)
                )
                encoded_dims[modality] = config.hidden_size // 2
                
            elif modality == 'imu':
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Linear(feature_dims[modality], config.hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
                )
                encoded_dims[modality] = config.hidden_size // 4
                
            elif modality == 'visual':
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Linear(feature_dims[modality], config.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(config.hidden_size, config.hidden_size // 2)
                )
                encoded_dims[modality] = config.hidden_size // 2
        
        # Fusion layer
        total_encoded_dim = sum(encoded_dims.values())
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_encoded_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers for joint angle prediction
        self.output_layers = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 6)  # 6 DOF arm
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout_rate
        )
    
    def forward(self, x):
        batch_size, seq_len = x[self.modalities[0]].shape[:2]
        
        # Encode each modality
        encoded_modalities = []
        for modality in self.modalities:
            if modality in x:
                # Reshape for processing
                modality_data = x[modality].view(-1, x[modality].shape[-1])
                encoded = self.modality_encoders[modality](modality_data)
                encoded = encoded.view(batch_size, seq_len, -1)
                encoded_modalities.append(encoded)
        
        # Fuse modalities
        if len(encoded_modalities) > 1:
            fused = torch.cat(encoded_modalities, dim=-1)
        else:
            fused = encoded_modalities[0]
        
        # Apply fusion layer
        fused = self.fusion_layer(fused.view(-1, fused.shape[-1]))
        fused = fused.view(batch_size, seq_len, -1)
        
        # Apply attention mechanism
        fused_permuted = fused.permute(1, 0, 2)  # (seq_len, batch, features)
        attended, _ = self.attention(fused_permuted, fused_permuted, fused_permuted)
        attended = attended.permute(1, 0, 2)  # Back to (batch, seq_len, features)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(attended)
        
        # Use last output for prediction
        final_output = lstm_out[:, -1, :]
        
        # Predict joint angles
        joint_angles = self.output_layers(final_output)
        
        return joint_angles

class SingleModalityLSTM(nn.Module):
    """Single modality LSTM for comparison"""
    
    def __init__(self, config: ModelConfig, input_dim: int, modality: str):
        super(SingleModalityLSTM, self).__init__()
        self.config = config
        self.modality = modality
        
        # Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 6)  # 6 DOF arm
        )
    
    def forward(self, x):
        # Process input
        processed = self.input_layer(x)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(processed)
        
        # Use last output for prediction
        final_output = lstm_out[:, -1, :]
        
        # Predict joint angles
        joint_angles = self.output_layers(final_output)
        
        return joint_angles

class TransformerModel(nn.Module):
    """Transformer-based model for comparison with LSTM"""
    
    def __init__(self, config: ModelConfig, feature_dims: Dict[str, int]):
        super(TransformerModel, self).__init__()
        self.config = config
        self.modalities = config.input_modalities
        
        # Modality encoders (similar to LSTM model)
        self.modality_encoders = nn.ModuleDict()
        for modality in self.modalities:
            self.modality_encoders[modality] = nn.Linear(
                feature_dims[modality], config.hidden_size
            )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(config.sequence_length, config.hidden_size)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_layers
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 6)
        )
    
    def forward(self, x):
        batch_size, seq_len = x[self.modalities[0]].shape[:2]
        
        # Encode and fuse modalities
        encoded_modalities = []
        for modality in self.modalities:
            if modality in x:
                encoded = self.modality_encoders[modality](x[modality])
                encoded_modalities.append(encoded)
        
        # Fuse modalities (simple average)
        if len(encoded_modalities) > 1:
            fused = torch.stack(encoded_modalities).mean(dim=0)
        else:
            fused = encoded_modalities[0]
        
        # Add positional encoding
        fused = fused + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer processing
        transformer_out = self.transformer(fused)
        
        # Use last output for prediction
        final_output = transformer_out[:, -1, :]
        
        # Predict joint angles
        joint_angles = self.output_layers(final_output)
        
        return joint_angles

class RoboticArmEnvironment(gym.Env):
    """Reinforcement Learning environment for robotic arm control"""
    
    def __init__(self, dataset, config: ModelConfig):
        super(RoboticArmEnvironment, self).__init__()
        self.dataset = dataset
        self.config = config
        self.current_idx = 0
        self.episode_length = 100
        self.current_step = 0
        
        # Action space: 6 DOF joint angle changes (in degrees)
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(6,), dtype=np.float32
        )
        
        # Observation space: EMG + IMU + Visual features
        obs_dim = self._calculate_observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Current state
        self.current_joint_angles = np.zeros(6)
        self.target_joint_angles = np.zeros(6)
        
    def _calculate_observation_dim(self):
        """Calculate total observation dimension"""
        dim = 0
        if 'emg' in self.config.input_modalities:
            dim += len(self.dataset.samples[0].emg_features)
        if 'imu' in self.config.input_modalities:
            dim += 10  # 3 accel + 3 gyro + 3 mag + 1 magnitude
        if 'visual' in self.config.input_modalities:
            dim += 63  # Hand landmarks
        dim += 6  # Current joint angles
        return dim
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_idx = np.random.randint(0, len(self.dataset.samples))
        self.current_step = 0
        self.current_joint_angles = np.random.uniform(-30, 30, 6)
        
        # Set target from dataset
        sample = self.dataset.samples[self.current_idx]
        self.target_joint_angles = self.dataset._generate_target(sample)
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        # Apply action (joint angle changes)
        self.current_joint_angles += action
        self.current_joint_angles = np.clip(self.current_joint_angles, -180, 180)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        self.current_step += 1
        done = (self.current_step >= self.episode_length) or (reward > 0.9)
        
        # Get next observation
        obs = self._get_observation()
        
        info = {
            'joint_angles': self.current_joint_angles.copy(),
            'target_angles': self.target_joint_angles.copy(),
            'joint_error': np.linalg.norm(self.current_joint_angles - self.target_joint_angles)
        }
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_idx >= len(self.dataset.samples):
            self.current_idx = 0
            
        sample = self.dataset.samples[self.current_idx]
        obs = []
        
        # EMG features
        if 'emg' in self.config.input_modalities:
            emg_features = [sample.emg_features[key] for key in sorted(sample.emg_features.keys())]
            obs.extend(emg_features)
        
        # IMU features
        if 'imu' in self.config.input_modalities:
            if 'accel' in sample.imu_data:
                obs.extend(sample.imu_data['accel'].flatten())
            if 'gyro' in sample.imu_data:
                obs.extend(sample.imu_data['gyro'].flatten())
            if 'accel_magnitude' in sample.imu_data:
                obs.append(sample.imu_data['accel_magnitude'])
            if 'gyro_magnitude' in sample.imu_data:
                obs.append(sample.imu_data['gyro_magnitude'])
        
        # Visual features
        if 'visual' in self.config.input_modalities:
            if sample.visual_data and sample.visual_data.get('hand_landmarks'):
                landmarks = sample.visual_data['hand_landmarks']
                if landmarks:
                    obs.extend(landmarks[0].flatten())
                else:
                    obs.extend([0] * 63)
            else:
                obs.extend([0] * 63)
        
        # Current joint angles
        obs.extend(self.current_joint_angles)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self):
        """Calculate reward based on joint angle error"""
        error = np.linalg.norm(self.current_joint_angles - self.target_joint_angles)
        max_error = 360  # Maximum possible error
        
        # Reward is inversely proportional to error
        reward = 1.0 - (error / max_error)
        
        # Bonus for being very close
        if error < 10:
            reward += 0.5
        
        # Penalty for large movements
        if np.any(np.abs(self.current_joint_angles) > 150):
            reward -= 0.2
        
        return max(0, reward)

class ModelTrainer:
    """Comprehensive model training and evaluation system"""
    
    def __init__(self, dataset, config: ModelConfig):
        self.dataset = dataset
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}
        
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self):
        """Prepare data for training"""
        # Create dataset
        pytorch_dataset = RoboticArmDataset(self.dataset, self.config)
        
        # Split data
        train_size = int(0.7 * len(pytorch_dataset))
        val_size = int(0.15 * len(pytorch_dataset))
        test_size = len(pytorch_dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            pytorch_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        # Calculate feature dimensions
        sample_batch = next(iter(self.train_loader))
        self.feature_dims = {}
        for modality in self.config.input_modalities:
            if modality in sample_batch[0]:
                self.feature_dims[modality] = sample_batch[0][modality].shape[-1]
        
        logger.info(f"Dataset prepared: Train={len(self.train_dataset)}, "
                   f"Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
        logger.info(f"Feature dimensions: {self.feature_dims}")
    
    def train_individual_modality_models(self):
        """Train individual modality models"""
        logger.info("Training individual modality models...")
        
        for modality in self.config.input_modalities:
            logger.info(f"Training {modality}-only model...")
            
            # Create single modality config
            single_config = ModelConfig(
                model_type='lstm',
                input_modalities=[modality],
                sequence_length=self.config.sequence_length,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                dropout_rate=self.config.dropout_rate,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                num_epochs=self.config.num_epochs
            )
            
            # Create model
            model = SingleModalityLSTM(
                single_config, 
                self.feature_dims[modality], 
                modality
            ).to(self.device)
            
            # Train model
            train_losses, val_losses = self._train_pytorch_model(model, single_config)
            
            # Evaluate model
            test_results = self._evaluate_pytorch_model(model)
            
            self.models[f'{modality}_only'] = model
            self.results[f'{modality}_only'] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_results': test_results
            }
            
            logger.info(f"{modality} model - Test MSE: {test_results['mse']:.4f}")
    
    def train_multimodal_models(self):
        """Train multi-modal fusion models"""
        logger.info("Training multi-modal models...")
        
        # LSTM model
        if 'lstm' in self.config.model_type:
            logger.info("Training Multi-Modal LSTM...")
            lstm_model = MultiModalLSTM(self.config, self.feature_dims).to(self.device)
            train_losses, val_losses = self._train_pytorch_model(lstm_model, self.config)
            test_results = self._evaluate_pytorch_model(lstm_model)
            
            self.models['multimodal_lstm'] = lstm_model
            self.results['multimodal_lstm'] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_results': test_results
            }
        
        # Transformer model
        logger.info("Training Multi-Modal Transformer...")
        transformer_model = TransformerModel(self.config, self.feature_dims).to(self.device)
        train_losses, val_losses = self._train_pytorch_model(transformer_model, self.config)
        test_results = self._evaluate_pytorch_model(transformer_model)
        
        self.models['multimodal_transformer'] = transformer_model
        self.results['multimodal_transformer'] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_results': test_results
        }
    
    def train_classical_ml_models(self):
        """Train classical ML models for comparison"""
        logger.info("Training classical ML models...")
        
        # Prepare data for sklearn
        X_train, y_train = self._prepare_sklearn_data(self.train_dataset)
        X_val, y_val = self._prepare_sklearn_data(self.val_dataset)
        X_test, y_test = self._prepare_sklearn_data(self.test_dataset)
        
        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_results = self._calculate_metrics(y_test, rf_pred)
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {'test_results': rf_results}
        
        # Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbm_model.fit(X_train, y_train)
        gbm_pred = gbm_model.predict(X_test)
        gbm_results = self._calculate_metrics(y_test, gbm_pred)
        
        self.models['gradient_boosting'] = gbm_model
        self.results['gradient_boosting'] = {'test_results': gbm_results}
        
        logger.info(f"Random Forest - Test MSE: {rf_results['mse']:.4f}")
        logger.info(f"Gradient Boosting - Test MSE: {gbm_results['mse']:.4f}")
    
    def train_reinforcement_learning_model(self):
        """Train RL model for continuous control"""
        logger.info("Training Reinforcement Learning models...")
        
        # Create environment
        env = RoboticArmEnvironment(self.dataset, self.config)
        check_env(env)
        
        # Train PPO
        logger.info("Training PPO...")
        ppo_model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
        ppo_model.learn(total_timesteps=50000)
        
        # Evaluate PPO
        ppo_results = self._evaluate_rl_model(ppo_model, env)
        
        self.models['ppo'] = ppo_model
        self.results['ppo'] = {'test_results': ppo_results}
        
        # Train SAC (if continuous)
        logger.info("Training SAC...")
        sac_model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=1
        )
        
        sac_model.learn(total_timesteps=50000)
        
        # Evaluate SAC
        sac_results = self._evaluate_rl_model(sac_model, env)
        
        self.models['sac'] = sac_model
        self.results['sac'] = {'test_results': sac_results}
        
        logger.info(f"PPO - Average Reward: {ppo_results['avg_reward']:.4f}")
        logger.info(f"SAC - Average Reward: {sac_results['avg_reward']:.4f}")
    
    def _train_pytorch_model(self, model, config):
        """Train PyTorch model with early stopping"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(config.num_epochs), desc="Training"):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                # Move data to device
                for modality in data:
                    data[modality] = data[modality].to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, targets in self.val_loader:
                    for modality in data:
                        data[modality] = data[modality].to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            # Average losses
            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pt'))
        
        return train_losses, val_losses
    
    def _evaluate_pytorch_model(self, model):
        """Evaluate PyTorch model on test set"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                for modality in data:
                    data[modality] = data[modality].to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(data)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        return self._calculate_metrics(targets, predictions)
    
    def _prepare_sklearn_data(self, dataset):
        """Prepare data for sklearn models"""
        X = []
        y = []
        
        for data, target in dataset:
            # Flatten all modality features
            features = []
            for modality in self.config.input_modalities:
                if modality in data:
                    # Take last timestep features
                    features.extend(data[modality][-1].numpy().flatten())
            
            X.append(features)
            y.append(target.numpy().flatten()[0])
        
        return np.array(X), np.array(y)
    
    def _evaluate_rl_model(self, model, env, n_episodes=100):
        """Evaluate RL model"""
        rewards = []
        episode_lengths = []
        joint_errors = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_errors = []
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                episode_errors.append(info['joint_error'])
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            joint_errors.extend(episode_errors)
        
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'avg_joint_error': np.mean(joint_errors),
            'success_rate': sum(1 for r in rewards if r > 0.8) / len(rewards)
        }
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Joint-wise metrics
        joint_errors = np.abs(y_true - y_pred)
        joint_mse = np.mean(joint_errors**2, axis=0) if y_true.ndim > 1 else [mse]
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'joint_mse': joint_mse.tolist() if hasattr(joint_mse, 'tolist') else joint_mse
        }
    
    def compare_models(self):
        """Compare all trained models"""
        logger.info("Comparing model performance...")
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if 'test_results' in results:
                test_results = results['test_results']
                comparison_data.append({
                    'Model': model_name,
                    'MSE': test_results.get('mse', test_results.get('avg_joint_error', 0)),
                    'MAE': test_results.get('mae', 0),
                    'R²': test_results.get('r2', 0),
                    'RMSE': test_results.get('rmse', 0)
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Visualize comparison
        self._plot_model_comparison(comparison_df)
        
        return comparison_df
    
    def _plot_model_comparison(self, comparison_df):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison')
        
        metrics = ['MSE', 'MAE', 'R²', 'RMSE']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            if metric in comparison_df.columns:
                bars = ax.bar(comparison_df['Model'], comparison_df[metric])
                ax.set_title(f'{metric} Comparison')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, save_dir='saved_models'):
        """Save all trained models"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = save_path / f"{model_name}.pt"
            
            if hasattr(model, 'state_dict'):  # PyTorch models
                torch.save(model.state_dict(), model_path)
            elif hasattr(model, 'save'):  # Stable Baselines models
                model.save(model_path)
            else:  # Sklearn models
                with open(model_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(model, f)
        
        # Save results
        results_path = save_path / "training_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in self.results.items():
                serializable_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        serializable_results[key][subkey] = subvalue.tolist()
                    elif isinstance(subvalue, dict):
                        serializable_results[key][subkey] = {}
                        for subsubkey, subsubvalue in subvalue.items():
                            if isinstance(subsubvalue, np.ndarray):
                                serializable_results[key][subkey][subsubkey] = subsubvalue.tolist()
                            else:
                                serializable_results[key][subkey][subsubkey] = subsubvalue
                    else:
                        serializable_results[key][subkey] = subvalue
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Models and results saved to {save_path}")

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, dataset, base_config: ModelConfig):
        self.dataset = dataset
        self.base_config = base_config
        
    def optimize_hyperparameters(self, n_trials=50):
        """Optimize hyperparameters for best model"""
        
        def objective(trial):
            # Define hyperparameter search space
            config = ModelConfig(
                model_type=self.base_config.model_type,
                input_modalities=self.base_config.input_modalities,
                sequence_length=trial.suggest_int('sequence_length', 30, 100),
                hidden_size=trial.suggest_categorical('hidden_size', [64, 128, 256]),
                num_layers=trial.suggest_int('num_layers', 1, 4),
                dropout_rate=trial.suggest_float('dropout_rate', 0.1, 0.5),
                learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),
                num_epochs=50,  # Reduced for faster optimization
                early_stopping_patience=5
            )
            
            # Train model with these hyperparameters
            trainer = ModelTrainer(self.dataset, config)
            trainer.prepare_data()
            
            # Train only multimodal LSTM for optimization
            model = MultiModalLSTM(config, trainer.feature_dims).to(trainer.device)
            train_losses, val_losses = trainer._train_pytorch_model(model, config)
            
            # Return validation loss as objective to minimize
            return min(val_losses)
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation loss: {study.best_value}")
        
        return study.best_params

# Main execution pipeline
def main():
    """Main execution pipeline for model training and evaluation"""
    
    # IMPORTANT: Replace 'dataset' with your actual dataset variable
    # Example of how to set your dataset:
    # dataset = your_loaded_dataset  # Replace with your actual dataset
    
    # Check if dataset is available
    try:
        # Replace this with your actual dataset loading
        # dataset = load_your_dataset()  # Your dataset loading function
        
        # For now, this will raise an error to remind you to set your dataset
        if 'dataset' not in locals():
            print("ERROR: Please set your dataset variable before running main()")
            print("Replace the dataset loading section in main() with your actual dataset")
            print("\nExample:")
            print("  dataset = your_dataset_variable")
            print("  # or")
            print("  dataset = load_dataset_from_file('path/to/your/dataset')")
            exit(1)
            
        # Run the main pipeline
        main()
        
        # Optional: Run inference example
        # Uncomment the following lines if you want to see an inference example
        # trainer = ModelTrainer(dataset, ModelConfig(
        #     model_type='lstm',
        #     input_modalities=['emg', 'imu', 'visual']
        # ))
        # run_inference_example(trainer)
        
    except NameError:
        print("ERROR: Dataset not found!")
        print("Please load your dataset first and assign it to the 'dataset' variable")
        print("\nExample:")
        print("  dataset = your_dataset  # Your existing dataset")
        print("  main()")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()