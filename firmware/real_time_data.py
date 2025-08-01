"""
Multi-Modal Data Collection System for Robotic Arm Control
Based on research from EMG-IMU fusion, multimodal sensor systems, and robotics applications

Key features:
- Synchronized EMG, IMU, and visual data collection
- Proper timestamping and alignment
- Data preprocessing and quality assessment
- Dataset construction for ML/LSTM/RL training
- Support for MyoWare EMG sensors, IMU, and cameras
"""

import numpy as np
import pandas as pd
import cv2
import threading
import time
import queue
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import serial
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataSample:
    """Single synchronized data sample across all modalities"""
    timestamp: float
    emg_data: np.ndarray  # Raw EMG signals from multiple channels
    emg_features: Dict[str, float]  # Extracted EMG features
    imu_data: Dict[str, np.ndarray]  # Accelerometer, gyroscope, magnetometer
    visual_data: Dict[str, Any]  # Frame, features, keypoints
    task_info: Dict[str, Any]  # Task state, target position, success/failure
    sync_quality: float  # Quality metric for temporal alignment

class EMGProcessor:
    """EMG signal processing and feature extraction"""
    
    def __init__(self, num_channels=8, sampling_rate=1000):
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.buffer_size = int(0.2 * sampling_rate)  # 200ms buffer
        self.buffer = deque(maxlen=self.buffer_size)
        
        # Bandpass filter for EMG (20-450 Hz typical range)
        self.b, self.a = signal.butter(4, [20, 450], btype='band', fs=sampling_rate)
        
        # Notch filter for 60Hz power line interference
        self.b_notch, self.a_notch = signal.butter(4, [58, 62], btype='bandstop', fs=sampling_rate)
        
    def preprocess_signal(self, raw_emg: np.ndarray) -> np.ndarray:
        """Apply filtering and preprocessing to raw EMG"""
        # Bandpass filter
        filtered = signal.filtfilt(self.b, self.a, raw_emg, axis=0)
        
        # Notch filter for power line noise
        filtered = signal.filtfilt(self.b_notch, self.a_notch, filtered, axis=0)
        
        # Rectification (absolute value)
        rectified = np.abs(filtered)
        
        # Envelope extraction using low-pass filter
        b_env, a_env = signal.butter(4, 10, btype='low', fs=self.sampling_rate)
        envelope = signal.filtfilt(b_env, a_env, rectified, axis=0)
        
        return envelope
    
    def extract_features(self, emg_window: np.ndarray) -> Dict[str, float]:
        """Extract time-domain and frequency-domain features from EMG window"""
        features = {}
        
        for ch in range(emg_window.shape[1]):
            ch_data = emg_window[:, ch]
            prefix = f'ch{ch}_'
            
            # Time domain features
            features[f'{prefix}rms'] = np.sqrt(np.mean(ch_data**2))
            features[f'{prefix}mav'] = np.mean(np.abs(ch_data))
            features[f'{prefix}var'] = np.var(ch_data)
            features[f'{prefix}wl'] = np.sum(np.abs(np.diff(ch_data)))
            features[f'{prefix}zc'] = self._zero_crossings(ch_data)
            features[f'{prefix}ssc'] = self._slope_sign_changes(ch_data)
            
            # Frequency domain features
            fft_data = np.fft.fft(ch_data)
            psd = np.abs(fft_data)**2
            freqs = np.fft.fftfreq(len(ch_data), 1/self.sampling_rate)
            
            # Mean and median frequency
            features[f'{prefix}mf'] = np.sum(freqs[:len(freqs)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
            features[f'{prefix}medf'] = self._median_frequency(psd[:len(psd)//2], freqs[:len(freqs)//2])
            
        return features
    
    def _zero_crossings(self, signal: np.ndarray, threshold: float = 0.01) -> int:
        """Count zero crossings in signal"""
        return np.sum(np.diff(np.sign(signal)) != 0)
    
    def _slope_sign_changes(self, signal: np.ndarray) -> int:
        """Count slope sign changes"""
        diff_signal = np.diff(signal)
        return np.sum(np.diff(np.sign(diff_signal)) != 0)
    
    def _median_frequency(self, psd: np.ndarray, freqs: np.ndarray) -> float:
        """Calculate median frequency"""
        cumsum_psd = np.cumsum(psd)
        median_idx = np.where(cumsum_psd >= cumsum_psd[-1] / 2)[0]
        return freqs[median_idx[0]] if len(median_idx) > 0 else 0.0

class IMUProcessor:
    """IMU data processing and feature extraction"""
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        self.gravity = 9.81
        self.calibration_samples = 100
        self.accel_offset = np.zeros(3)
        self.gyro_offset = np.zeros(3)
        self.is_calibrated = False
        
    def calibrate(self, accel_data: np.ndarray, gyro_data: np.ndarray):
        """Calibrate IMU sensors"""
        self.accel_offset = np.mean(accel_data, axis=0)
        self.accel_offset[2] -= self.gravity  # Account for gravity on Z-axis
        self.gyro_offset = np.mean(gyro_data, axis=0)
        self.is_calibrated = True
        logger.info("IMU calibration completed")
    
    def preprocess_imu(self, accel: np.ndarray, gyro: np.ndarray, mag: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess IMU data"""
        if self.is_calibrated:
            accel_cal = accel - self.accel_offset
            gyro_cal = gyro - self.gyro_offset
        else:
            accel_cal = accel
            gyro_cal = gyro
            
        # Apply low-pass filter to reduce noise
        b, a = signal.butter(4, 20, btype='low', fs=self.sampling_rate)
        accel_filtered = signal.filtfilt(b, a, accel_cal, axis=0)
        gyro_filtered = signal.filtfilt(b, a, gyro_cal, axis=0)
        mag_filtered = signal.filtfilt(b, a, mag, axis=0)
        
        return {
            'accel': accel_filtered,
            'gyro': gyro_filtered,
            'mag': mag_filtered,
            'accel_magnitude': np.linalg.norm(accel_filtered, axis=1),
            'gyro_magnitude': np.linalg.norm(gyro_filtered, axis=1)
        }
    
    def extract_features(self, imu_window: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract features from IMU window"""
        features = {}
        
        for sensor_type, data in imu_window.items():
            if sensor_type in ['accel', 'gyro', 'mag']:
                for axis, axis_name in enumerate(['x', 'y', 'z']):
                    axis_data = data[:, axis]
                    prefix = f'{sensor_type}_{axis_name}_'
                    
                    features[f'{prefix}mean'] = np.mean(axis_data)
                    features[f'{prefix}std'] = np.std(axis_data)
                    features[f'{prefix}max'] = np.max(axis_data)
                    features[f'{prefix}min'] = np.min(axis_data)
                    features[f'{prefix}range'] = np.max(axis_data) - np.min(axis_data)
                    features[f'{prefix}energy'] = np.sum(axis_data**2)
            
            elif sensor_type in ['accel_magnitude', 'gyro_magnitude']:
                prefix = f'{sensor_type}_'
                features[f'{prefix}mean'] = np.mean(data)
                features[f'{prefix}std'] = np.std(data)
                features[f'{prefix}max'] = np.max(data)
                features[f'{prefix}min'] = np.min(data)
        
        return features

class VisualProcessor:
    """Computer vision processing for hand/arm tracking"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30
        
        # Initialize MediaPipe for hand tracking
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mediapipe_available = True
        except ImportError:
            logger.warning("MediaPipe not available. Using basic computer vision.")
            self.mediapipe_available = False
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_id}")
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        logger.info(f"Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        return True
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame and extract features"""
        results = {
            'frame': frame,
            'hand_landmarks': [],
            'hand_confidence': [],
            'optical_flow': None,
            'motion_vectors': None
        }
        
        if self.mediapipe_available:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
                    results['hand_landmarks'].append(np.array(landmarks))
                    
                if hand_results.multi_handedness:
                    for handedness in hand_results.multi_handedness:
                        results['hand_confidence'].append(handedness.classification[0].score)
        
        return results
    
    def extract_features(self, visual_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from visual data"""
        features = {}
        
        # Hand landmark features
        if visual_data['hand_landmarks']:
            for i, landmarks in enumerate(visual_data['hand_landmarks']):
                prefix = f'hand{i}_'
                
                # Palm center
                palm_landmarks = landmarks[[0, 5, 9, 13, 17]]  # Wrist and finger bases
                palm_center = np.mean(palm_landmarks[:, :2], axis=0)
                features[f'{prefix}palm_x'] = palm_center[0]
                features[f'{prefix}palm_y'] = palm_center[1]
                
                # Finger distances from palm
                fingertips = landmarks[[4, 8, 12, 16, 20]]  # Fingertip landmarks
                for j, tip in enumerate(fingertips):
                    dist = np.linalg.norm(tip[:2] - palm_center)
                    features[f'{prefix}finger{j}_dist'] = dist
                
                # Hand span (distance between thumb and pinky)
                thumb_tip = landmarks[4]
                pinky_tip = landmarks[20]
                hand_span = np.linalg.norm(thumb_tip[:2] - pinky_tip[:2])
                features[f'{prefix}span'] = hand_span
                
                # Hand orientation (angle of palm)
                wrist = landmarks[0]
                middle_mcp = landmarks[9]
                hand_vector = middle_mcp[:2] - wrist[:2]
                hand_angle = np.arctan2(hand_vector[1], hand_vector[0])
                features[f'{prefix}angle'] = hand_angle
        
        # Motion features (if optical flow available)
        if visual_data['motion_vectors'] is not None:
            motion = visual_data['motion_vectors']
            features['motion_magnitude'] = np.mean(np.linalg.norm(motion, axis=2))
            features['motion_std'] = np.std(np.linalg.norm(motion, axis=2))
        
        return features
    
    def cleanup(self):
        """Clean up camera resources"""
        if self.cap is not None:
            self.cap.release()

class SynchronizedDataCollector:
    """Main data collection system with synchronized multi-modal recording"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_recording = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.threads = []
        
        # Initialize processors
        self.emg_processor = EMGProcessor(
            num_channels=config.get('emg_channels', 8),
            sampling_rate=config.get('emg_sampling_rate', 1000)
        )
        
        self.imu_processor = IMUProcessor(
            sampling_rate=config.get('imu_sampling_rate', 100)
        )
        
        self.visual_processor = VisualProcessor(
            camera_id=config.get('camera_id', 0)
        )
        
        # Data storage
        self.dataset = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(config.get('save_dir', 'data'))
        self.save_dir.mkdir(exist_ok=True)
        
        # Synchronization
        self.master_clock = time.time()
        self.sync_tolerance = config.get('sync_tolerance', 0.01)  # 10ms tolerance
        
        # Task tracking
        self.current_task = None
        self.task_start_time = None
        self.task_success = None
        
    def initialize_hardware(self) -> bool:
        """Initialize all hardware components"""
        logger.info("Initializing hardware...")
        
        # Initialize camera
        if not self.visual_processor.initialize_camera():
            return False
            
        # Initialize serial connections for EMG/IMU (placeholder)
        # In real implementation, you would set up serial connections here
        
        logger.info("Hardware initialization completed")
        return True
    
    def calibrate_sensors(self):
        """Calibrate IMU and establish EMG baselines"""
        logger.info("Starting sensor calibration...")
        
        # Collect calibration data (placeholder)
        # In real implementation, collect static samples for IMU calibration
        
        # Simulate calibration data
        accel_cal = np.random.normal(0, 0.1, (100, 3))
        accel_cal[:, 2] += 9.81  # Add gravity to z-axis
        gyro_cal = np.random.normal(0, 0.01, (100, 3))
        
        self.imu_processor.calibrate(accel_cal, gyro_cal)
        logger.info("Sensor calibration completed")
    
    def start_recording(self, task_name: str = "default"):
        """Start synchronized data recording"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
            
        self.is_recording = True
        self.current_task = task_name
        self.task_start_time = time.time()
        self.master_clock = time.time()
        
        logger.info(f"Starting recording for task: {task_name}")
        
        # Start data collection threads
        emg_thread = threading.Thread(target=self._emg_collection_loop)
        imu_thread = threading.Thread(target=self._imu_collection_loop)
        visual_thread = threading.Thread(target=self._visual_collection_loop)
        sync_thread = threading.Thread(target=self._synchronization_loop)
        
        self.threads = [emg_thread, imu_thread, visual_thread, sync_thread]
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()
    
    def stop_recording(self, task_success: bool = None):
        """Stop recording and save data"""
        if not self.is_recording:
            logger.warning("No recording in progress")
            return
            
        self.is_recording = False
        self.task_success = task_success
        
        logger.info("Stopping recording...")
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Process and save remaining data in queue
        self._process_remaining_data()
        
        logger.info(f"Recorded {len(self.dataset)} synchronized samples")
    
    def _emg_collection_loop(self):
        """EMG data collection thread"""
        while self.is_recording:
            try:
                # Simulate EMG data collection
                # In real implementation, read from serial/USB connection
                timestamp = time.time() - self.master_clock
                raw_emg = np.random.normal(0, 0.1, (50, 8))  # 50ms of 8-channel EMG at 1kHz
                
                # Add some realistic EMG-like characteristics
                for ch in range(8):
                    # Add some muscle activation patterns
                    activation = 0.5 * np.sin(2 * np.pi * 0.5 * timestamp + ch) + 0.5
                    raw_emg[:, ch] += activation * np.random.exponential(0.2, 50)
                
                processed_emg = self.emg_processor.preprocess_signal(raw_emg)
                features = self.emg_processor.extract_features(processed_emg)
                
                data_item = {
                    'type': 'emg',
                    'timestamp': timestamp,
                    'raw_data': raw_emg,
                    'processed_data': processed_emg,
                    'features': features
                }
                
                self.data_queue.put(data_item, timeout=0.1)
                time.sleep(0.05)  # 20Hz EMG feature extraction
                
            except queue.Full:
                logger.warning("EMG data queue full, dropping sample")
            except Exception as e:
                logger.error(f"Error in EMG collection: {e}")
    
    def _imu_collection_loop(self):
        """IMU data collection thread"""
        while self.is_recording:
            try:
                # Simulate IMU data collection
                timestamp = time.time() - self.master_clock
                
                # Simulate realistic arm movement
                accel = np.random.normal([0, 0, 9.81], [0.5, 0.5, 0.2])
                gyro = np.random.normal([0, 0, 0], [0.1, 0.1, 0.1])
                mag = np.random.normal([25, 0, -45], [2, 2, 2])  # Typical magnetic field
                
                # Add movement patterns based on task
                if self.current_task == "throw_ball":
                    # Simulate throwing motion
                    phase = (timestamp * 2) % (2 * np.pi)
                    accel[0] += 5 * np.sin(phase)  # Forward acceleration
                    gyro[1] += 2 * np.cos(phase)   # Rotation around Y-axis
                
                imu_data = {
                    'accel': accel.reshape(1, -1),
                    'gyro': gyro.reshape(1, -1),
                    'mag': mag.reshape(1, -1)
                }
                
                processed_imu = self.imu_processor.preprocess_imu(
                    imu_data['accel'], imu_data['gyro'], imu_data['mag']
                )
                features = self.imu_processor.extract_features(processed_imu)
                
                data_item = {
                    'type': 'imu',
                    'timestamp': timestamp,
                    'raw_data': imu_data,
                    'processed_data': processed_imu,
                    'features': features
                }
                
                self.data_queue.put(data_item, timeout=0.1)
                time.sleep(0.01)  # 100Hz IMU sampling
                
            except queue.Full:
                logger.warning("IMU data queue full, dropping sample")
            except Exception as e:
                logger.error(f"Error in IMU collection: {e}")
    
    def _visual_collection_loop(self):
        """Visual data collection thread"""
        prev_frame = None
        
        while self.is_recording:
            try:
                timestamp = time.time() - self.master_clock
                
                # Read frame from camera
                ret, frame = self.visual_processor.cap.read()
                if not ret:
                    logger.warning("Failed to read camera frame")
                    continue
                
                # Process frame
                visual_data = self.visual_processor.process_frame(frame)
                
                # Calculate optical flow if previous frame exists
                if prev_frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, None, None
                    )
                    visual_data['optical_flow'] = flow
                
                features = self.visual_processor.extract_features(visual_data)
                
                data_item = {
                    'type': 'visual',
                    'timestamp': timestamp,
                    'raw_data': visual_data,
                    'features': features
                }
                
                self.data_queue.put(data_item, timeout=0.1)
                prev_frame = frame.copy()
                time.sleep(1/30)  # 30 FPS
                
            except queue.Full:
                logger.warning("Visual data queue full, dropping sample")
            except Exception as e:
                logger.error(f"Error in visual collection: {e}")
    
    def _synchronization_loop(self):
        """Data synchronization and alignment thread"""
        emg_buffer = []
        imu_buffer = []
        visual_buffer = []
        
        while self.is_recording or not self.data_queue.empty():
            try:
                # Get data from queue
                data_item = self.data_queue.get(timeout=0.1)
                
                # Sort into appropriate buffer
                if data_item['type'] == 'emg':
                    emg_buffer.append(data_item)
                elif data_item['type'] == 'imu':
                    imu_buffer.append(data_item)
                elif data_item['type'] == 'visual':
                    visual_buffer.append(data_item)
                
                # Attempt synchronization
                self._attempt_synchronization(emg_buffer, imu_buffer, visual_buffer)
                
                # Clean old data from buffers
                current_time = time.time() - self.master_clock
                self._clean_buffers(emg_buffer, imu_buffer, visual_buffer, current_time)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in synchronization: {e}")
    
    def _attempt_synchronization(self, emg_buffer: List, imu_buffer: List, visual_buffer: List):
        """Attempt to create synchronized data samples"""
        if not (emg_buffer and imu_buffer and visual_buffer):
            return
        
        # Find the most recent timestamps from each modality
        latest_times = {
            'emg': emg_buffer[-1]['timestamp'],
            'imu': imu_buffer[-1]['timestamp'],  
            'visual': visual_buffer[-1]['timestamp']
        }
        
        # Find synchronization target (earliest of the latest timestamps)
        sync_target = min(latest_times.values())
        
        # Find closest samples to sync target
        emg_sample = self._find_closest_sample(emg_buffer, sync_target)
        imu_sample = self._find_closest_sample(imu_buffer, sync_target)
        visual_sample = self._find_closest_sample(visual_buffer, sync_target)
        
        if emg_sample and imu_sample and visual_sample:
            # Check synchronization quality
            timestamps = [emg_sample['timestamp'], imu_sample['timestamp'], visual_sample['timestamp']]
            sync_quality = self._calculate_sync_quality(timestamps, sync_target)
            
            if sync_quality > 0.5:  # Accept if sync quality is reasonable
                # Create synchronized sample
                sample = DataSample(
                    timestamp=sync_target,
                    emg_data=emg_sample['processed_data'],
                    emg_features=emg_sample['features'],
                    imu_data=imu_sample['processed_data'],
                    visual_data=visual_sample['raw_data'],
                    task_info={
                        'task_name': self.current_task,
                        'task_time': sync_target - (self.task_start_time - self.master_clock) if self.task_start_time else 0,
                        'success': self.task_success
                    },
                    sync_quality=sync_quality
                )
                
                self.dataset.append(sample)
                
                # Remove used samples from buffers
                emg_buffer.remove(emg_sample)
                imu_buffer.remove(imu_sample)
                visual_buffer.remove(visual_sample)
    
    def _find_closest_sample(self, buffer: List, target_time: float) -> Optional[Dict]:
        """Find sample closest to target time"""
        if not buffer:
            return None
            
        closest_sample = min(buffer, key=lambda x: abs(x['timestamp'] - target_time))
        
        if abs(closest_sample['timestamp'] - target_time) <= self.sync_tolerance:
            return closest_sample
        
        return None
    
    def _calculate_sync_quality(self, timestamps: List[float], target: float) -> float:
        """Calculate synchronization quality score (0-1)"""
        max_deviation = max(abs(t - target) for t in timestamps)
        return max(0, 1 - (max_deviation / self.sync_tolerance))
    
    def _clean_buffers(self, emg_buffer: List, imu_buffer: List, visual_buffer: List, current_time: float):
        """Remove old data from buffers"""
        cutoff_time = current_time - 1.0  # Keep 1 second of history
        
        for buffer in [emg_buffer, imu_buffer, visual_buffer]:
            buffer[:] = [item for item in buffer if item['timestamp'] > cutoff_time]
    
    def _process_remaining_data(self):
        """Process any remaining data in queue after recording stops"""
        remaining_data = []
        
        while not self.data_queue.empty():
            try:
                remaining_data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        
        logger.info(f"Processing {len(remaining_data)} remaining data items")
    
    def save_dataset(self, filename: Optional[str] = None):
        """Save collected dataset to disk"""
        if not self.dataset:
            logger.warning("No data to save")
            return
        
        if filename is None:
            filename = f"dataset_{self.session_id}.pkl"
        
        filepath = self.save_dir / filename
        
        # Convert dataset to serializable format
        serializable_data = []
        for sample in self.dataset:
            sample_dict = asdict(sample)
            # Convert numpy arrays to lists for JSON compatibility
            for key, value in sample_dict.items():
                if isinstance(value, np.ndarray):
                    sample_dict[key] = value.tolist()
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray):
                            value[sub_key] = sub_value.tolist()
            serializable_data.append(sample_dict)
        
        # Save as pickle for full data preservation
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_data, f)
        
        # Also save metadata as JSON
        metadata = {
            'session_id': self.session_id,
            'config': self.config,
            'num_samples': len(self.dataset),
            'duration': self.dataset[-1].timestamp - self.dataset[0].timestamp if self.dataset else 0,
            'tasks': list(set(sample.task_info['task_name'] for sample in self.dataset)),
            'sync_quality_stats': {
                'mean': np.mean([sample.sync_quality for sample in self.dataset]),
                'std': np.std([sample.sync_quality for sample in self.dataset]),
                'min': np.min([sample.sync_quality for sample in self.dataset]),
                'max': np.max([sample.sync_quality for sample in self.dataset])
            }
        }
        
        metadata_path = self.save_dir / f"metadata_{self.session_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dataset saved: {filepath}")
        logger.info(f"Metadata saved: {metadata_path}")
        
        return filepath
    
    def load_dataset(self, filepath: str) -> List[DataSample]:
        """Load dataset from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Convert back to DataSample objects
        dataset = []
        for sample_dict in data:
            # Convert lists back to numpy arrays where needed
            for key, value in sample_dict.items():
                if key in ['emg_data'] and isinstance(value, list):
                    sample_dict[key] = np.array(value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list) and sub_key in ['accel', 'gyro', 'mag']:
                            value[sub_key] = np.array(sub_value)
            
            dataset.append(DataSample(**sample_dict))
        
        logger.info(f"Loaded {len(dataset)} samples from {filepath}")
        return dataset
    
    def analyze_dataset(self, dataset: Optional[List[DataSample]] = None) -> Dict[str, Any]:
        """Analyze dataset quality and characteristics"""
        if dataset is None:
            dataset = self.dataset
        
        if not dataset:
            logger.warning("No dataset to analyze")
            return {}
        
        analysis = {
            'basic_stats': {
                'num_samples': len(dataset),
                'duration': dataset[-1].timestamp - dataset[0].timestamp,
                'sampling_rate': len(dataset) / (dataset[-1].timestamp - dataset[0].timestamp),
                'tasks': {}
            },
            'sync_quality': {
                'mean': np.mean([s.sync_quality for s in dataset]),
                'std': np.std([s.sync_quality for s in dataset]),
                'percentiles': np.percentile([s.sync_quality for s in dataset], [25, 50, 75, 90, 95])
            },
            'emg_analysis': {},
            'imu_analysis': {},
            'visual_analysis': {}
        }
        
        # Task distribution
        for sample in dataset:
            task = sample.task_info['task_name']
            if task not in analysis['basic_stats']['tasks']:
                analysis['basic_stats']['tasks'][task] = 0
            analysis['basic_stats']['tasks'][task] += 1
        
        # EMG analysis
        emg_features = [list(sample.emg_features.keys()) for sample in dataset[:10]]
        if emg_features:
            common_features = set(emg_features[0])
            for features in emg_features[1:]:
                common_features &= set(features)
            
            analysis['emg_analysis']['num_features'] = len(common_features)
            analysis['emg_analysis']['feature_names'] = list(common_features)
            
            # Calculate feature statistics
            feature_stats = {}
            for feature in common_features:
                values = [sample.emg_features[feature] for sample in dataset if feature in sample.emg_features]
                feature_stats[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range': [np.min(values), np.max(values)]
                }
            analysis['emg_analysis']['feature_stats'] = feature_stats
        
        # IMU analysis (similar structure)
        # Visual analysis (similar structure)
        
        return analysis
    
    def visualize_data(self, dataset: Optional[List[DataSample]] = None, save_plots: bool = True):
        """Create visualizations of the collected data"""
        if dataset is None:
            dataset = self.dataset
        
        if not dataset:
            logger.warning("No dataset to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Data Collection Analysis - Session {self.session_id}')
        
        # Timeline plot
        timestamps = [s.timestamp for s in dataset]
        sync_quality = [s.sync_quality for s in dataset]
        
        axes[0, 0].plot(timestamps, sync_quality, 'b-', alpha=0.7)
        axes[0, 0].set_title('Synchronization Quality Over Time')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Sync Quality')
        axes[0, 0].grid(True, alpha=0.3)
        
        # EMG feature heatmap (sample)
        if dataset and dataset[0].emg_features:
            sample_features = list(dataset[0].emg_features.keys())[:20]  # First 20 features
            feature_matrix = []
            for sample in dataset[::10]:  # Every 10th sample
                row = [sample.emg_features.get(f, 0) for f in sample_features]
                feature_matrix.append(row)
            
            if feature_matrix:
                im = axes[0, 1].imshow(np.array(feature_matrix).T, aspect='auto', cmap='viridis')
                axes[0, 1].set_title('EMG Features Over Time')
                axes[0, 1].set_xlabel('Time Samples')
                axes[0, 1].set_ylabel('Features')
                plt.colorbar(im, ax=axes[0, 1])
        
        # Task distribution
        task_counts = {}
        for sample in dataset:
            task = sample.task_info['task_name']
            task_counts[task] = task_counts.get(task, 0) + 1
        
        if task_counts:
            axes[1, 0].bar(task_counts.keys(), task_counts.values())
            axes[1, 0].set_title('Task Distribution')
            axes[1, 0].set_xlabel('Task')
            axes[1, 0].set_ylabel('Number of Samples')
        
        # Sync quality histogram
        axes[1, 1].hist(sync_quality, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Synchronization Quality Distribution')
        axes[1, 1].set_xlabel('Sync Quality')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.save_dir / f"analysis_{self.session_id}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Analysis plots saved: {plot_path}")
        
        plt.show()
    
    def cleanup(self):
        """Clean up resources"""
        self.visual_processor.cleanup()
        logger.info("Data collection system cleaned up")

class DatasetValidator:
    """Validate and assess dataset quality"""
    
    @staticmethod
    def validate_dataset(dataset: List[DataSample]) -> Dict[str, Any]:
        """Comprehensive dataset validation"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        if not dataset:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Check temporal consistency
        timestamps = [s.timestamp for s in dataset]
        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            validation_results['warnings'].append("Timestamps are not monotonically increasing")
        
        # Check sync quality
        sync_qualities = [s.sync_quality for s in dataset]
        low_quality_samples = sum(1 for sq in sync_qualities if sq < 0.5)
        low_quality_ratio = low_quality_samples / len(dataset)
        
        if low_quality_ratio > 0.1:
            validation_results['warnings'].append(f"High ratio of low-quality sync samples: {low_quality_ratio:.2%}")
        
        # Check data completeness
        incomplete_samples = 0
        for sample in dataset:
            if not sample.emg_data.size or not sample.imu_data or not sample.visual_data:
                incomplete_samples += 1
        
        if incomplete_samples > 0:
            validation_results['warnings'].append(f"Found {incomplete_samples} incomplete samples")
        
        # Check for missing features
        if dataset:
            first_sample_features = set(dataset[0].emg_features.keys())
            for i, sample in enumerate(dataset[1:], 1):
                if set(sample.emg_features.keys()) != first_sample_features:
                    validation_results['warnings'].append(f"Inconsistent features at sample {i}")
                    break
        
        validation_results['stats'] = {
            'total_samples': len(dataset),
            'duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'average_sync_quality': np.mean(sync_qualities),
            'low_quality_ratio': low_quality_ratio,
            'incomplete_samples': incomplete_samples
        }
        
        return validation_results

# Example usage and configuration
def main():
    """Example usage of the data collection system"""
    
    # Configuration
    config = {
        'emg_channels': 8,
        'emg_sampling_rate': 1000,
        'imu_sampling_rate': 100,
        'camera_id': 0,
        'save_dir': 'collected_data',
        'sync_tolerance': 0.02  # 20ms tolerance
    }
    
    # Initialize data collector
    collector = SynchronizedDataCollector(config)
    
    try:
        # Initialize hardware
        if not collector.initialize_hardware():
            logger.error("Failed to initialize hardware")
            return
        
        # Calibrate sensors
        collector.calibrate_sensors()
        
        # Record different tasks
        tasks = [
            ("baseline_rest", 10),
            ("hand_open_close", 15),
            ("throw_ball", 20),
            ("reach_target", 15)
        ]
        
        for task_name, duration in tasks:
            print(f"\nStarting task: {task_name}")
            print(f"Duration: {duration} seconds")
            input("Press Enter when ready to start recording...")
            
            collector.start_recording(task_name)
            
            # Simulate task execution
            time.sleep(duration)
            
            success = input("Was the task successful? (y/n): ").lower() == 'y'
            collector.stop_recording(task_success=success)
            
            print(f"Recorded {len(collector.dataset)} total samples so far")
        
        # Save dataset
        filepath = collector.save_dataset()
        
        # Analyze dataset
        analysis = collector.analyze_dataset()
        print("\nDataset Analysis:")
        print(json.dumps(analysis, indent=2, default=str))
        
        # Validate dataset
        validator = DatasetValidator()
        validation = validator.validate_dataset(collector.dataset)
        print("\nDataset Validation:")
        print(json.dumps(validation, indent=2, default=str))
        
        # Visualize data
        collector.visualize_data()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        raise
    finally:
        # Cleanup
        collector.cleanup()

if __name__ == "__main__":
    main()